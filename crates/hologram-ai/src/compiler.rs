//! Model compilation pipeline.
//!
//! Compiles AI models (ONNX, GGUF) into `.holo` archives via the hologram
//! O(1) LUT runtime. This crate is a **compiler** — it does not own inference
//! sessions or runtime state (see ADR-0016).

use anyhow::Context;
use hologram_ai_common::{
    lower, AiGraph, AiParam, LowerPhase, LoweringOptions, MemoryPlanner, OptPipeline, Pass,
};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use tracing::{debug, info, warn};

// ── Model source ──────────────────────────────────────────────────────────────

/// Source for a model to compile.
#[allow(clippy::large_enum_variant)]
pub enum ModelSource {
    /// Path to an ONNX model file.
    OnnxPath(PathBuf),
    /// Raw ONNX bytes.
    OnnxBytes(Vec<u8>),
    /// Path to a GGUF model file.
    GgufPath(PathBuf),
    /// Path to a GGML model file (legacy pre-GGUF format).
    GgmlPath(PathBuf),
    /// Pre-built `AiGraph` (bypass importer).
    AiGraph(AiGraph),
}

// ── Model metadata ────────────────────────────────────────────────────────────

/// High-level metadata extracted from the model.
pub struct ModelMetadata {
    pub arch: String,
    pub vocab_size: u32,
    pub context_len: u32,
    pub n_layers: u32,
    pub n_embd: u32,
}

// ── Compilation output ──────────────────────────────────────────────────────

/// Statistics from the compilation pipeline.
pub struct CompileStats {
    pub import_warnings: usize,
    pub validation_errors: usize,
    pub total_weight_bytes: u64,
    pub node_count: usize,
}

/// A compiled `.holo` archive ready to be saved or executed.
pub struct HoloArchive {
    /// The compiled archive bytes (single archive or pipeline archive).
    pub bytes: Vec<u8>,
    pub metadata: ModelMetadata,
    pub stats: CompileStats,
}

/// Debug mapping from source tensor names to compiled node indices.
///
/// Used by execution conformance testing to correlate ORT intermediate
/// tensors (keyed by ONNX name) with hologram executor buffers (keyed
/// by NodeId, which is derived from the builder index).
pub struct DebugMap {
    /// ONNX tensor name → builder node index in the compiled graph.
    pub name_to_idx: std::collections::HashMap<String, usize>,
}

impl HoloArchive {
    /// Write the compiled `.holo` archive to `path`.
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating output directory {parent:?}"))?;
            }
        }
        std::fs::write(path, &self.bytes)
            .with_context(|| format!("writing .holo archive to {path:?}"))
    }
}

// Backward-compatible type alias.
pub type CompiledModel = HoloArchive;

impl CompiledModel {
    /// Backward-compatible save method.
    pub fn save_archive(&self, path: &std::path::Path) -> anyhow::Result<()> {
        self.save(path)
    }
}

// ── Model compiler ────────────────────────────────────────────────────────────

/// Compiles a `ModelSource` through the full pipeline into a `HoloArchive`.
///
/// Pipeline:
///   import → optimize → validate → plan memory → lower → compile → embed weights
pub struct ModelCompiler {
    /// Use memory-mapping for weight loading when possible.
    pub mmap: bool,
}

impl Default for ModelCompiler {
    fn default() -> Self {
        Self { mmap: true }
    }
}

impl ModelCompiler {
    /// Compile a model source into a `.holo` archive.
    ///
    /// For LLM models (GGUF with transformer architecture), produces a pipeline
    /// archive with named layer entrypoints. For simpler models (ONNX), produces
    /// a single-graph archive.
    pub fn compile(&self, source: ModelSource) -> anyhow::Result<HoloArchive> {
        // Step 1 — import.
        let ai_graph = self.import(source)?;
        info!(nodes = ai_graph.nodes.len(), params = ai_graph.params.len(), "import complete");

        // Step 2 — optimize.
        let ai_graph = OptPipeline::mvp()
            .run(ai_graph)
            .context("optimization pass failed")?;
        info!(nodes = ai_graph.nodes.len(), "optimization complete");

        // Step 2b — concretize all symbolic/dynamic dims for compilation.
        // The runtime doesn't yet support deferred shape resolution, so we
        // bake in concrete values: Var dims → upper bounds, Dynamic → 1.
        let ai_graph = concretize_all_dims(ai_graph).context("shape concretization failed")?;

        // Step 2c — post-concretization shape repair.
        // After Var→concrete substitution, most shapes are already correct.
        // Only Dynamic dims remain (from broadcast_shape mismatches, etc.).
        // Clear stale known_i64_values so DataProp re-evaluates with the
        // now-concrete shapes, then run the aggressive pipeline to overwrite
        // remaining Dynamic dims with correct concrete values.
        let mut ai_graph = ai_graph;
        for info in ai_graph.tensor_info.values_mut() {
            info.known_i64_values = None;
        }
        // Run the pipeline in a fixpoint loop. Each iteration resolves more
        // Reshape targets as DataProp traces through newly-resolved shapes.
        // Shape-computation chains can be arbitrarily deep (e.g., Q/K/V each
        // depend on different Reshape chains that themselves depend on DataProp).
        // Post-concretization pipeline: uses AggressiveShapePropagation which
        // always overwrites shapes with inferred values. This is safe because
        // all dims are now concrete — no risk of overwriting good symbolic shapes
        // with weaker inferences. Run twice to handle chains where Reshape A
        // depends on DataProp, and Reshape B depends on Shape(A's output).
        let aggressive_pipeline = {
            use hologram_ai_common::{
                AggressiveShapePropagation, ConstantDeduplication,
                opt::{
                    const_eval::ConstantEvaluation, constant_fold::ConstantFolding,
                    data_prop::DataPropagation, dead_node::DeadNodeElimination,
                },
            };
            // Two DataProp passes handle multi-level shape dependencies:
            //
            //   Pass 1 (DataProp #1): evaluates shape subgraphs like Expand
            //     targets that depend on concrete input tensor shapes.
            //   Pass 2 (AggressiveProp #2): propagates DataProp #1 results
            //     to correctly shape intermediate tensors (e.g. K_intermediate).
            //   Pass 3 (DataProp #2): re-evaluates shape subgraphs that depend
            //     on K_intermediate's now-correct shape (e.g. K^T target).
            //     DataProp's re-materialization logic (computed_tids) ensures
            //     it overwrites any stale results from DataProp #1.
            //   Pass 4 (AggressiveProp #3): applies DataProp #2 results.
            OptPipeline::new(vec![
                Box::new(AggressiveShapePropagation),
                Box::new(DataPropagation),
                Box::new(AggressiveShapePropagation),
                Box::new(DataPropagation),
                Box::new(AggressiveShapePropagation),
                // Evaluate all-constant nodes (N-D broadcast, comparisons, etc.)
                Box::new(ConstantEvaluation),
                Box::new(ConstantFolding),
                Box::new(ConstantDeduplication),
                Box::new(DeadNodeElimination),
            ])
        };
        let mut ai_graph = ai_graph;
        for pass_num in 0..3 {
            ai_graph = aggressive_pipeline
                .run(ai_graph)
                .with_context(|| format!("post-concretization repair pass {pass_num} failed"))?;
            // Clear stale known_i64_values between iterations so DataProp
            // re-evaluates with the freshly-inferred shapes.
            for info in ai_graph.tensor_info.values_mut() {
                info.known_i64_values = None;
            }
        }

        // Replace any Dynamic or remaining Var dims introduced by the
        // aggressive pipeline (e.g., broadcast_shape returns Dynamic for
        // non-matching concrete dims).
        {
            use hologram_ai_common::Dim;
            for info in ai_graph.tensor_info.values_mut() {
                for dim in info.shape.iter_mut() {
                    if matches!(dim, Dim::Dynamic) {
                        *dim = Dim::Concrete(1);
                    } else if matches!(dim, Dim::Var(_)) {
                        *dim = Dim::Concrete(0);
                    }
                }
            }
        }

        // Step 2d — convert Slice ops to Gather (hologram has no native Slice).
        // Must run after concretization so dim values are known.
        let ai_graph = hologram_ai_common::SliceToGather
            .run(ai_graph)
            .context("slice-to-gather conversion failed")?;

        // Step 2e — shape healing: fill in any remaining empty shapes.
        let ai_graph = hologram_ai_common::ShapeHealing
            .run(ai_graph)
            .context("shape healing failed")?;

        // Diagnostic: report empty shapes and attention-dim issues after repair.
        {
            let empty_tensors: Vec<_> = ai_graph
                .tensor_info
                .iter()
                .filter(|(_, info)| info.shape.is_empty())
                .collect();
            if !empty_tensors.is_empty() {
                warn!(count = empty_tensors.len(), "tensors still have empty shapes after repair");
                let producers: std::collections::HashMap<u32, &hologram_ai_common::AiOp> = ai_graph
                    .nodes
                    .iter()
                    .flat_map(|n| n.outputs.iter().map(move |&tid| (tid, &n.op)))
                    .collect();
                for (&tid, info) in &empty_tensors {
                    let op_str = producers
                        .get(&tid)
                        .map(|op| format!("{op:?}"))
                        .unwrap_or_else(|| "input/param".into());
                    debug!(
                        tensor = tid,
                        dtype = ?info.logical_dtype,
                        producer = &op_str[..op_str.len().min(80)],
                        "empty shape"
                    );
                }
            }
            // Find the root cause of Dynamic dims: first node producing a
            // Dynamic-dim tensor where ALL inputs have concrete shapes.
            let mut found = 0u32;
            for node in &ai_graph.nodes {
                for &out_tid in &node.outputs {
                    let out_info = match ai_graph.tensor_info.get(&out_tid) {
                        Some(i) if i.shape.iter().any(|d| matches!(d, hologram_ai_common::Dim::Dynamic)) => i,
                        _ => continue,
                    };
                    // Check if all inputs have fully-concrete shapes (no Dynamic).
                    let all_inputs_concrete = node.inputs.iter().all(|&tid| {
                        ai_graph.tensor_info.get(&tid).map(|i| !i.shape.is_empty() && i.shape.iter().all(|d| d.as_concrete().is_some())).unwrap_or(false)
                    });
                    if all_inputs_concrete && found < 2 {
                        let input_shapes: Vec<_> = node.inputs.iter().map(|&t| {
                            let info = ai_graph.tensor_info.get(&t);
                            let shape = info.map(|i| format!("{:?}", i.shape.as_slice())).unwrap_or_default();
                            let kv = info.and_then(|i| i.known_i64_values.as_ref());
                            format!("T{t}:{shape} kv={kv:?}")
                        }).collect();
                        let prod_info: Vec<_> = node.inputs.iter().map(|&t| {
                            ai_graph.nodes.iter().find(|n| n.outputs.contains(&t)).map(|n| format!("T{t} <- node {} {:?}", n.id, format!("{:?}", &n.op).chars().take(50).collect::<String>())).unwrap_or_else(|| format!("T{t} <- input/param"))
                        }).collect();
                        warn!(
                            node_id = node.id,
                            output = out_tid,
                            shape = ?out_info.shape.as_slice(),
                            "Dynamic-dim root cause (all inputs concrete)"
                        );
                        for s in &input_shapes { debug!("  input: {s}"); }
                        for p in &prod_info { debug!("  {p}"); }
                        found += 1;
                    }
                }
            }
            // Check attention pattern: 4D tensors with [1, 32, X, Y] where Y=1
            // would indicate failed kv_seq_len resolution.
            let producers: std::collections::HashMap<u32, &hologram_ai_common::AiOp> = ai_graph
                .nodes
                .iter()
                .flat_map(|n| n.outputs.iter().map(move |&tid| (tid, &n.op)))
                .collect();
            let suspect: Vec<_> = ai_graph
                .tensor_info
                .iter()
                .filter(|(_, info)| {
                    info.shape.len() == 4
                        && info.shape[3].as_concrete() == Some(1)
                        && info.shape[1].as_concrete().map(|v| v > 1) == Some(true)
                })
                .take(5)
                .collect();
            if !suspect.is_empty() {
                warn!(
                    count = suspect.len(),
                    "4D tensors with last_dim=1 (possible kv_seq_len issue)"
                );
                for (&tid, info) in &suspect {
                    let op_str = producers
                        .get(&tid)
                        .map(|op| format!("{op:?}"))
                        .unwrap_or_else(|| "input/param".into());
                    debug!(
                        tensor = tid,
                        shape = ?info.shape.as_slice(),
                        producer = &op_str[..op_str.len().min(60)],
                        "suspect attention dim"
                    );
                }
            }
        }

        // Diagnostic: dump MatMul input shapes (first 5).
        {
            let mut matmul_count = 0u32;
            for node in &ai_graph.nodes {
                if matches!(node.op, hologram_ai_common::AiOp::MatMul | hologram_ai_common::AiOp::BatchMatMul) && matmul_count < 5 {
                    let input_shapes: Vec<_> = node.inputs.iter().map(|&t| {
                        ai_graph.tensor_info.get(&t).map(|i| format!("T{t}:{:?}", i.shape.as_slice())).unwrap_or_else(|| format!("T{t}:<?>"))
                    }).collect();
                    let out_shape = node.outputs.first().and_then(|&t| ai_graph.tensor_info.get(&t)).map(|i| format!("{:?}", i.shape.as_slice())).unwrap_or_default();
                    debug!(
                        node_id = node.id,
                        lhs = %input_shapes[0],
                        rhs = %input_shapes[1],
                        output = %out_shape,
                        "MatMul"
                    );
                    matmul_count += 1;
                }
            }
        }

        // Diagnostic: scan compiled params for inf/NaN (catches broken scale factors).
        {
            use hologram_ai_common::AiParam;
            use hologram_ai_common::DType;
            let mut nan_params = 0u32;
            for (&tid, param) in &ai_graph.params {
                if let AiParam::Inline { data, info } = param {
                    if info.logical_dtype == DType::F32 && !data.is_empty() && data.len() % 4 == 0 {
                        let floats: &[f32] = bytemuck::cast_slice(data);
                        let nan_count = floats.iter().filter(|f| f.is_nan()).count();
                        let inf_count = floats.iter().filter(|f| f.is_infinite()).count();
                        if nan_count > 0 || inf_count > 0 {
                            let shape = ai_graph.tensor_info.get(&tid)
                                .map(|i| format!("{:?}", i.shape.as_slice()))
                                .unwrap_or_default();
                            let producer = ai_graph.nodes.iter()
                                .find(|n| n.outputs.contains(&tid))
                                .map(|n| format!("{:?}", n.op))
                                .unwrap_or_else(|| "input/param".into());
                            warn!(tid, nan_count, inf_count, total=floats.len(), shape = %shape, producer = &producer[..producer.len().min(80)], "compiled f32 param has inf/NaN");
                            nan_params += 1;
                        }
                    }
                }
            }
            if nan_params > 0 {
                warn!(nan_params, "WARNING: inf/NaN scalar params detected — attention scale may be wrong!");
            }
        }

        // Diagnostic: total param data size.
        {
            let total_param_bytes: usize = ai_graph.params.values().map(|p| match p {
                hologram_ai_common::AiParam::Inline { data, .. } => data.len(),
                _ => 0,
            }).sum();
            info!(
                entries = ai_graph.params.len(),
                total_mb = format_args!("{:.1}", total_param_bytes as f64 / 1_048_576.0),
                "params"
            );
        }

        // Validate before lowering.
        let errs = ai_graph.validate();
        if !errs.is_empty() {
            anyhow::bail!("{} validation error(s): {}", errs.len(), errs[0].message);
        }
        info!("validation passed");

        // Shape consistency check: catch shape/weight mismatches before lowering.
        let shape_errors =
            hologram_ai_common::opt::shape_consistency::validate_shape_consistency(&ai_graph);
        if !shape_errors.is_empty() {
            warn!(
                count = shape_errors.len(),
                "shape consistency issues detected"
            );
            for err in &shape_errors {
                warn!(
                    node = err.node_name.as_deref().unwrap_or("-"),
                    "{}", err.message
                );
            }
        }

        // Step 3 — memory plan.
        let mem_plan = MemoryPlanner
            .plan(&ai_graph)
            .context("memory planning failed")?;

        // Extract metadata before lowering (borrows ai_graph).
        let metadata = extract_metadata(&ai_graph);
        let import_warnings = ai_graph.warnings.len();
        let node_count = ai_graph.nodes.len();
        let is_llm = metadata.arch != "unknown" && mem_plan.kv_cache_layout.n_layers > 0;

        info!(
            arch = %metadata.arch,
            is_llm,
            nodes = node_count,
            warnings = import_warnings,
            "starting compilation"
        );

        let archive_bytes = if is_llm {
            self.compile_llm_pipeline(&ai_graph, &mem_plan)?
        } else {
            self.compile_single_graph(&ai_graph, &mem_plan)?
        };

        // Collect total weight bytes for stats.
        let weight_blob = collect_weight_bytes(&ai_graph)?;
        let total_weight_bytes = weight_blob.len() as u64;

        Ok(HoloArchive {
            bytes: archive_bytes,
            metadata,
            stats: CompileStats {
                import_warnings,
                validation_errors: 0,
                total_weight_bytes,
                node_count,
            },
        })
    }

    /// Compile a model and return a debug map alongside the archive.
    ///
    /// The `DebugMap` maps ONNX tensor names → compiled builder node indices,
    /// enabling node-by-node comparison between ORT and hologram execution.
    ///
    /// Only meaningful for single-graph (non-LLM) models. LLM pipeline
    /// compilation does not produce a debug map.
    pub fn compile_with_debug_info(
        &self,
        source: ModelSource,
    ) -> anyhow::Result<(HoloArchive, DebugMap)> {
        // Reuse the full compile pipeline.
        let ai_graph = self.import(source)?;
        info!(nodes = ai_graph.nodes.len(), params = ai_graph.params.len(), "import complete (debug)");

        // Capture tensor_names before optimization passes (passes preserve it).
        let ai_graph = OptPipeline::mvp()
            .run(ai_graph)
            .context("optimization pass failed")?;
        let ai_graph = concretize_all_dims(ai_graph).context("shape concretization failed")?;

        // Post-concretization repair (same as compile()).
        let mut ai_graph = ai_graph;
        for info in ai_graph.tensor_info.values_mut() {
            info.known_i64_values = None;
        }
        let aggressive_pipeline = {
            use hologram_ai_common::{
                AggressiveShapePropagation, ConstantDeduplication,
                opt::{
                    const_eval::ConstantEvaluation, constant_fold::ConstantFolding,
                    data_prop::DataPropagation, dead_node::DeadNodeElimination,
                },
            };
            OptPipeline::new(vec![
                Box::new(AggressiveShapePropagation),
                Box::new(DataPropagation),
                Box::new(AggressiveShapePropagation),
                Box::new(DataPropagation),
                Box::new(AggressiveShapePropagation),
                Box::new(ConstantEvaluation),
                Box::new(ConstantFolding),
                Box::new(ConstantDeduplication),
                Box::new(DeadNodeElimination),
            ])
        };
        for pass_num in 0..3 {
            ai_graph = aggressive_pipeline
                .run(ai_graph)
                .with_context(|| format!("post-concretization repair pass {pass_num} failed"))?;
            for info in ai_graph.tensor_info.values_mut() {
                info.known_i64_values = None;
            }
        }
        {
            use hologram_ai_common::Dim;
            for info in ai_graph.tensor_info.values_mut() {
                for dim in info.shape.iter_mut() {
                    if matches!(dim, Dim::Dynamic) {
                        *dim = Dim::Concrete(1);
                    } else if matches!(dim, Dim::Var(_)) {
                        *dim = Dim::Concrete(0);
                    }
                }
            }
        }
        let ai_graph = hologram_ai_common::SliceToGather
            .run(ai_graph)
            .context("slice-to-gather conversion failed")?;
        let ai_graph = hologram_ai_common::ShapeHealing
            .run(ai_graph)
            .context("shape healing failed")?;

        // Validate.
        let errs = ai_graph.validate();
        if !errs.is_empty() {
            anyhow::bail!("{} validation error(s): {}", errs.len(), errs[0].message);
        }

        // Memory plan.
        let mem_plan = MemoryPlanner
            .plan(&ai_graph)
            .context("memory planning failed")?;

        let metadata = extract_metadata(&ai_graph);
        let import_warnings = ai_graph.warnings.len();
        let node_count = ai_graph.nodes.len();

        // Lower (single-graph only for debug mode).
        let lower_out = lower(
            &ai_graph,
            &mem_plan.kv_cache_layout,
            &LoweringOptions::default(),
            &LowerPhase::Forward,
        )
        .context("lowering failed")?;

        // Build debug map: compose tensor_names (TensorId→name) with tid_to_idx (TensorId→idx).
        let mut name_to_idx = std::collections::HashMap::new();
        for (tid, name) in &ai_graph.tensor_names {
            if let Some(&idx) = lower_out.tid_to_idx.get(tid) {
                name_to_idx.insert(name.clone(), idx);
            }
        }
        let debug_map = DebugMap { name_to_idx };
        info!(mapped = debug_map.name_to_idx.len(), "debug map built");

        // Compile graph.
        let compilation = hologram::compile(lower_out.graph).context("hologram::compile failed")?;
        let unpacked = unpack_archive(&compilation.archive)?;
        let layer_header = build_tensor_port_header(&unpacked.plan, &ai_graph);
        let weights = collect_weight_bytes(&ai_graph)?;
        let bundle = if lower_out.context.is_empty() {
            None
        } else {
            Some(&lower_out.context)
        };
        let archive_bytes = build_final_archive(
            unpacked,
            if weights.is_empty() { None } else { Some(weights.clone()) },
            Some(layer_header),
            bundle,
        )?;

        let archive = HoloArchive {
            bytes: archive_bytes,
            metadata,
            stats: CompileStats {
                import_warnings,
                validation_errors: 0,
                total_weight_bytes: weights.len() as u64,
                node_count,
            },
        };

        Ok((archive, debug_map))
    }

    /// Compile a non-LLM model into a single-graph archive.
    fn compile_single_graph(
        &self,
        ai_graph: &AiGraph,
        mem_plan: &hologram_ai_common::MemoryPlan,
    ) -> anyhow::Result<Vec<u8>> {
        let lower_out = lower(
            ai_graph,
            &mem_plan.kv_cache_layout,
            &LoweringOptions::default(),
            &LowerPhase::Forward,
        )
        .context("lowering failed")?;

        let compilation = hologram::compile(lower_out.graph).context("hologram::compile failed")?;
        debug!(archive_bytes = compilation.archive.len(), "hologram::compile complete");

        // Single unpack → modify → repack cycle.
        let unpacked = unpack_archive(&compilation.archive)?;
        let layer_header = build_tensor_port_header(&unpacked.plan, ai_graph);
        let weights = collect_weight_bytes(ai_graph)?;
        let bundle = if lower_out.context.is_empty() {
            None
        } else {
            Some(&lower_out.context)
        };

        let archive = build_final_archive(
            unpacked,
            if weights.is_empty() { None } else { Some(weights) },
            Some(layer_header),
            bundle,
        )?;
        info!(archive_bytes = archive.len(), "single-graph archive assembled");
        Ok(archive)
    }

    /// Compile an LLM into a pipeline archive with prefill + decode sub-archives.
    fn compile_llm_pipeline(
        &self,
        ai_graph: &AiGraph,
        mem_plan: &hologram_ai_common::MemoryPlan,
    ) -> anyhow::Result<Vec<u8>> {
        use hologram::hologram_archive::writer::pipeline_writer::PipelineWriter;

        let opts = LoweringOptions::default();
        let weights = collect_weight_bytes(ai_graph)?;
        let extra_weights = if weights.is_empty() { None } else { Some(weights) };
        info!(
            weight_mb = format_args!("{:.1}", extra_weights.as_ref().map_or(0, |w| w.len()) as f64 / 1_048_576.0),
            "compiling LLM pipeline (prefill + decode)"
        );

        // Lower + compile + single-pass assemble for prefill graph.
        let prefill_out = lower(
            ai_graph,
            &mem_plan.kv_cache_layout,
            &opts,
            &LowerPhase::Prefill,
        )
        .context("lowering prefill graph failed")?;
        debug!(graph_nodes = prefill_out.graph.node_count(), "prefill lowered");
        let prefill_compiled =
            hologram::compile(prefill_out.graph).context("compiling prefill graph failed")?;
        debug!(archive_bytes = prefill_compiled.archive.len(), "prefill compiled");
        let prefill_unpacked = unpack_archive(&prefill_compiled.archive)?;
        let prefill_lh = build_tensor_port_header(&prefill_unpacked.plan, ai_graph);
        let prefill_bundle = if prefill_out.context.is_empty() {
            None
        } else {
            Some(&prefill_out.context)
        };
        let prefill_archive = build_final_archive(
            prefill_unpacked,
            extra_weights.clone(),
            Some(prefill_lh),
            prefill_bundle,
        )?;
        info!(archive_bytes = prefill_archive.len(), "prefill archive assembled");

        // Lower + compile + single-pass assemble for decode graph.
        let decode_out = lower(
            ai_graph,
            &mem_plan.kv_cache_layout,
            &opts,
            &LowerPhase::Decode,
        )
        .context("lowering decode graph failed")?;
        debug!(graph_nodes = decode_out.graph.node_count(), "decode lowered");
        let decode_compiled =
            hologram::compile(decode_out.graph).context("compiling decode graph failed")?;
        debug!(archive_bytes = decode_compiled.archive.len(), "decode compiled");
        let decode_unpacked = unpack_archive(&decode_compiled.archive)?;
        let decode_lh = build_tensor_port_header(&decode_unpacked.plan, ai_graph);
        let decode_bundle = if decode_out.context.is_empty() {
            None
        } else {
            Some(&decode_out.context)
        };
        let decode_archive = build_final_archive(
            decode_unpacked,
            extra_weights,
            Some(decode_lh),
            decode_bundle,
        )?;
        info!(archive_bytes = decode_archive.len(), "decode archive assembled");

        // Bundle into pipeline.
        let pipeline = PipelineWriter::new()
            .add_model("lm.prefill", prefill_archive)
            .add_model("lm.decode", decode_archive)
            .build()
            .map_err(|e| anyhow::anyhow!("building pipeline archive: {e}"))?;
        info!(archive_bytes = pipeline.len(), "pipeline archive built");
        Ok(pipeline)
    }

    fn import(&self, source: ModelSource) -> anyhow::Result<AiGraph> {
        match source {
            ModelSource::OnnxPath(path) => {
                hologram_ai_onnx::import_onnx_path(&path, Default::default())
                    .with_context(|| format!("importing ONNX from {path:?}"))
            }
            ModelSource::OnnxBytes(bytes) => {
                hologram_ai_onnx::import_onnx(&bytes, Default::default())
                    .context("importing ONNX from bytes")
            }
            ModelSource::GgufPath(path) => hologram_ai_gguf::import_gguf(
                &path,
                hologram_ai_gguf::GgufImportOptions {
                    mmap: self.mmap,
                    arch_override: None,
                },
            )
            .with_context(|| format!("importing GGUF from {path:?}")),
            ModelSource::GgmlPath(path) => {
                let bytes =
                    std::fs::read(&path).with_context(|| format!("reading GGML file {path:?}"))?;
                hologram_ai_ggml::import_ggml(&bytes)
                    .with_context(|| format!("importing GGML from {path:?}"))
            }
            ModelSource::AiGraph(g) => Ok(g),
        }
    }
}

// ── Single-pass archive assembly ─────────────────────────────────────────────

/// Raw components extracted from a compiled archive via a single
/// `load_from_bytes` call. Avoids repeated deserialization/decompression.
struct UnpackedArchive {
    /// Compressed graph bytes (passed through as-is to `set_graph_bytes`).
    graph_bytes: Vec<u8>,
    /// Existing weight bytes from the archive.
    weight_bytes: Vec<u8>,
    /// Existing sections (kind, raw bytes).
    sections: Vec<(u32, Vec<u8>)>,
    /// The loaded plan — used to read layer_header, etc.
    plan: hologram::LoadedPlan,
}

/// Unpack a compiled archive into its raw components with a single
/// `load_from_bytes` call.
fn unpack_archive(archive: &[u8]) -> anyhow::Result<UnpackedArchive> {
    let plan = hologram::load_from_bytes(archive).context("unpacking archive")?;
    let h = plan.header();
    let graph_bytes =
        archive[h.graph_offset as usize..(h.graph_offset + h.graph_size) as usize].to_vec();
    let weight_bytes = plan.weights().to_vec();

    let mut sections = Vec::new();
    for entry in &plan.sections().entries {
        let offset = entry.offset as usize;
        let size = entry.size as usize;
        if offset + size <= archive.len() {
            sections.push((entry.kind, archive[offset..offset + size].to_vec()));
        }
    }

    debug!(
        archive_bytes = archive.len(),
        graph_bytes = graph_bytes.len(),
        weight_bytes = weight_bytes.len(),
        section_count = sections.len(),
        "unpacked archive"
    );

    Ok(UnpackedArchive {
        graph_bytes,
        weight_bytes,
        sections,
        plan,
    })
}

/// Build a final archive from unpacked components plus all modifications,
/// using a single `HoloWriter::build()` call.
///
/// - `extra_weights`: if `Some`, replaces the archive's existing weight bytes.
/// - `layer_header`: if `Some`, replaces/adds the LayerHeader section.
/// - `bundle`: if `Some`, merges bundle sections (replacing matching kinds).
fn build_final_archive(
    unpacked: UnpackedArchive,
    extra_weights: Option<Vec<u8>>,
    layer_header: Option<hologram::hologram_archive::entrypoint::schedule::LayerHeader>,
    bundle: Option<&hologram_ai_common::ContextBundle>,
) -> anyhow::Result<Vec<u8>> {
    use hologram::hologram_archive::section::{EmbeddableSection, SECTION_LAYER_HEADER};

    let weights = extra_weights.unwrap_or(unpacked.weight_bytes);
    let mut writer = hologram::HoloWriter::new()
        .set_graph_bytes(unpacked.graph_bytes)
        .set_weights(weights);

    // Determine which section kinds will be replaced.
    let layer_header_kind = layer_header
        .as_ref()
        .map(|lh| lh.section_kind());
    let bundle_kinds: Vec<u32> = bundle
        .map(|b| b.iter().map(|(k, _)| k).collect())
        .unwrap_or_default();

    // Carry forward existing sections, skipping those we're about to replace.
    for (kind, bytes) in unpacked.sections {
        if layer_header_kind == Some(kind) {
            continue;
        }
        if bundle_kinds.contains(&kind) {
            continue;
        }
        writer = writer.add_raw_section(kind, bytes);
    }

    // Add the new LayerHeader section.
    if let Some(ref lh) = layer_header {
        writer = writer.add_raw_section(SECTION_LAYER_HEADER, lh.to_bytes());
    }

    // Add all bundle sections.
    if let Some(bundle) = bundle {
        for (kind, bytes) in bundle.iter() {
            writer = writer.add_raw_section(kind, bytes.to_vec());
        }
    }

    writer
        .build()
        .map_err(|e| anyhow::anyhow!("building final archive: {e}"))
}

/// Build a corrected `LayerHeader` with proper TensorPorts from the AiGraph.
///
/// Pure function — reads from the already-loaded plan, no archive I/O.
fn build_tensor_port_header(
    plan: &hologram::LoadedPlan,
    ai_graph: &AiGraph,
) -> hologram::hologram_archive::entrypoint::schedule::LayerHeader {
    use hologram::hologram_archive::entrypoint::schedule::LayerHeader;
    use hologram::hologram_archive::entrypoint::{
        LayerDescriptor, LayerEntrypoint, LayerId, TensorPort,
    };

    let input_ports: Vec<TensorPort> = ai_graph
        .inputs
        .iter()
        .enumerate()
        .map(|(i, &tid)| {
            let name = ai_graph.input_name(i);
            let (shape, dtype) = tensor_port_info(tid, &ai_graph.tensor_info);
            TensorPort { name, shape, dtype }
        })
        .collect();

    let output_ports: Vec<TensorPort> = ai_graph
        .outputs
        .iter()
        .enumerate()
        .map(|(i, &tid)| {
            let name = ai_graph.output_name(i);
            let (shape, dtype) = tensor_port_info(tid, &ai_graph.tensor_info);
            TensorPort { name, shape, dtype }
        })
        .collect();

    if let Some(old_lh) = plan.layer_header() {
        let mut new_layers = old_lh.layers.clone();
        for layer in &mut new_layers {
            layer.inputs = input_ports.clone();
            layer.outputs = output_ports.clone();
        }
        LayerHeader {
            layers: new_layers,
            schedule: old_lh.schedule.clone(),
        }
    } else {
        let layer = LayerDescriptor {
            id: LayerId(0),
            name: "forward".into(),
            entrypoint: LayerEntrypoint::Graph,
            inputs: input_ports,
            outputs: output_ports,
            group: 0,
            plan_offset: 0,
            plan_size: 0,
        };
        LayerHeader {
            layers: vec![layer],
            schedule: vec![vec![LayerId(0)]],
        }
    }
}

/// Extract shape and dtype for a TensorPort from tensor_info.
fn tensor_port_info(
    tid: hologram_ai_common::TensorId,
    tensor_info: &std::collections::HashMap<
        hologram_ai_common::TensorId,
        hologram_ai_common::TensorInfo,
    >,
) -> (Vec<u64>, hologram::hologram_archive::weight::WeightDType) {
    use hologram::hologram_archive::weight::WeightDType;

    if let Some(info) = tensor_info.get(&tid) {
        let shape: Vec<u64> = info
            .shape
            .iter()
            .map(|dim| match dim {
                hologram_ai_common::Dim::Concrete(n) => *n,
                _ => 0, // symbolic dim → 0 (dynamic)
            })
            .collect();
        let dtype = ai_dtype_to_weight_dtype(&info.logical_dtype);
        (shape, dtype)
    } else {
        (vec![1], WeightDType::U8) // fallback placeholder
    }
}

/// Convert hologram-ai DType to archive WeightDType.
fn ai_dtype_to_weight_dtype(
    dtype: &hologram_ai_common::DType,
) -> hologram::hologram_archive::weight::WeightDType {
    use hologram::hologram_archive::weight::WeightDType;
    use hologram_ai_common::DType;
    match dtype {
        DType::F32 => WeightDType::F32,
        DType::F64 => WeightDType::F32, // F64 → F32 at weight serialization
        DType::F16 => WeightDType::F16,
        DType::BF16 => WeightDType::BF16,
        DType::INT8 => WeightDType::I8,
        DType::INT4 => WeightDType::I4,
        DType::U8 => WeightDType::U8,
        DType::INT16 => WeightDType::I32, // INT16 widened to I32
        DType::INT32 => WeightDType::I32,
        DType::INT64 => WeightDType::I64,
        DType::BOOL => WeightDType::U8,
    }
}

fn extract_metadata(graph: &AiGraph) -> ModelMetadata {
    use hologram_ai_common::MetaValue;

    let arch = match graph.metadata.get("arch") {
        Some(MetaValue::Str(s)) => s.clone(),
        _ => "unknown".into(),
    };
    let vocab_size = meta_u32(graph, "vocab_size").unwrap_or(0);
    let context_len = meta_u32(graph, "context_length").unwrap_or(0);
    let n_layers = meta_u32(graph, "n_layers").unwrap_or(0);
    let n_embd = meta_u32(graph, "n_embd").unwrap_or(0);

    ModelMetadata {
        arch,
        vocab_size,
        context_len,
        n_layers,
        n_embd,
    }
}

/// Concretize all symbolic and dynamic dimensions in the graph.
///
/// - `DimExpr::Var` dims → substituted with a concrete value
/// - `DimExpr::Dynamic` dims → replaced with `Concrete(1)`
///
/// For Var dims with an upper bound (e.g., GGUF models), uses the upper bound.
/// For Var dims without an upper bound (e.g., ONNX models), uses heuristic
/// defaults: batch-like dims → 1, sequence-like dims → 2048, others → 1.
///
/// The `ShapeRecipeSection` is still embedded for future runtime resolution.
fn concretize_all_dims(mut graph: AiGraph) -> anyhow::Result<AiGraph> {
    use hologram_ai_common::Dim;

    // Set upper bounds for ONNX dim vars that don't have them,
    // using heuristic defaults based on the variable name.
    //
    // Sequence-like vars are fixed to 0 (sentinel) rather than a concrete value.
    // The executor resolves them at runtime from actual input buffer sizes,
    // enabling dynamic-length inference without padding to a fixed seq_len.
    for (_, entry) in graph.dim_vars.iter_mut() {
        if entry.fixed.is_some() {
            continue;
        }
        if entry.upper.is_none() {
            let name_lower = entry.name.to_lowercase();
            if name_lower.contains("seq") || name_lower.contains("length") {
                // Emit as 0-sentinel: runtime resolves from actual token count.
                entry.fixed = Some(0);
            } else {
                entry.upper = Some(1u64);
            }
        }
    }

    // Concretize Var dims to their upper bounds.
    let _ = graph.dim_vars.concretize_to_upper(); // ok if no vars
    let subs = graph.dim_vars.fixed_substitutions();

    // Diagnostic: count Dynamic dims before concretization.
    {
        let mut dyn_count = 0u32;
        let mut dyn_tensors = Vec::new();
        let producers: std::collections::HashMap<u32, &hologram_ai_common::AiOp> = graph
            .nodes
            .iter()
            .flat_map(|n| n.outputs.iter().map(move |&tid| (tid, &n.op)))
            .collect();
        for (&tid, info) in &graph.tensor_info {
            for (di, dim) in info.shape.iter().enumerate() {
                if matches!(dim, Dim::Dynamic) {
                    dyn_count += 1;
                    if dyn_tensors.len() < 20 {
                        let op_str = producers
                            .get(&tid)
                            .map(|op| format!("{op:?}"))
                            .unwrap_or_else(|| "input/param".into());
                        dyn_tensors.push(format!(
                            "  T{tid}[{di}]: shape={:?} (from {})",
                            info.shape.as_slice(),
                            &op_str[..op_str.len().min(60)]
                        ));
                    }
                }
            }
        }
        if dyn_count > 0 {
            warn!(count = dyn_count, "Dynamic dims remain after shape propagation");
            for line in &dyn_tensors {
                debug!("{line}");
            }
        }
    }

    // Apply Var substitutions and replace Dynamic→Concrete(1).
    // After this, all dims are concrete. The post-concretization aggressive
    // pipeline will re-infer correct shapes from these concrete anchors.
    for info in graph.tensor_info.values_mut() {
        for dim in info.shape.iter_mut() {
            for (var_id, replacement) in &subs {
                *dim = dim.substitute(*var_id, replacement);
            }
            if let Some(v) = dim.evaluate() {
                *dim = Dim::Concrete(v);
            }
            if matches!(dim, Dim::Dynamic) {
                *dim = Dim::Concrete(1);
            } else if matches!(dim, Dim::Var(_)) {
                // Safety net: any remaining Var (e.g. seq-like fixed=0) that
                // wasn't substituted above gets the 0-sentinel treatment.
                *dim = Dim::Concrete(0);
            }
        }
    }

    Ok(graph)
}

fn meta_u32(graph: &AiGraph, key: &str) -> Option<u32> {
    use hologram_ai_common::MetaValue;
    match graph.metadata.get(key) {
        Some(MetaValue::Int(i)) => Some(*i as u32),
        Some(MetaValue::Float(f)) => Some(*f as u32),
        _ => None,
    }
}

/// Collect weight bytes from all Mmap params in TensorId-sorted order.
///
/// The ordering must match builder.rs which assigns cumulative byte offsets
/// as `source_id` in `ConstantData::Deferred` using the same sorted order.
fn collect_weight_bytes(ai_graph: &AiGraph) -> anyhow::Result<Vec<u8>> {
    let mut sorted: Vec<_> = ai_graph
        .params
        .iter()
        .filter(|(_, p)| matches!(p, AiParam::Mmap { .. }))
        .collect();
    if sorted.is_empty() {
        return Ok(Vec::new());
    }
    sorted.sort_by_key(|(&tid, _)| tid);

    let total_size: u64 = sorted
        .iter()
        .map(|(_, p)| match p {
            AiParam::Mmap { len, .. } => *len,
            _ => 0,
        })
        .sum();
    let mut blob = Vec::with_capacity(total_size as usize);

    for (_, param) in &sorted {
        if let AiParam::Mmap {
            path, offset, len, ..
        } = param
        {
            let mut f = std::fs::File::open(path)
                .with_context(|| format!("opening weight file {path:?}"))?;
            f.seek(SeekFrom::Start(*offset))?;
            let mut buf = vec![0u8; *len as usize];
            f.read_exact(&mut buf)
                .with_context(|| format!("reading {} bytes from {path:?}", len))?;
            blob.extend_from_slice(&buf);
        }
    }

    Ok(blob)
}

/// Rebuild a compiled archive adding an extra section.
///
/// Preserves all existing sections from the source archive so that
/// layer headers, model metadata, tokenizer data, etc. are not lost.
/// Uses a single unpack/repack cycle internally.
pub fn rebuild_archive_with_section(
    archive: &[u8],
    section: &dyn hologram::hologram_archive::section::EmbeddableSection,
) -> anyhow::Result<Vec<u8>> {
    let unpacked = unpack_archive(archive)?;

    // Filter out the section kind we're replacing.
    let new_kind = section.section_kind();
    let mut writer = hologram::HoloWriter::new()
        .set_graph_bytes(unpacked.graph_bytes)
        .set_weights(unpacked.weight_bytes);

    for (kind, bytes) in unpacked.sections {
        if kind == new_kind {
            continue;
        }
        writer = writer.add_raw_section(kind, bytes);
    }

    writer = writer.add_section(section);

    writer
        .build()
        .map_err(|e| anyhow::anyhow!("rebuilding archive with section: {e}"))
}

