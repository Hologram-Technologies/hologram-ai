//! Model compilation pipeline.
//!
//! Compiles AI models (ONNX, GGUF) into `.holo` archives via the hologram
//! O(1) LUT runtime. This crate is a **compiler** — it does not own inference
//! sessions or runtime state (see ADR-0016).

use anyhow::Context;
use rayon::prelude::*;
use hologram_ai_common::{
    exec_context::ShapeContextGraph, lower, AiGraph, AiParam, LowerPhase,
    LoweringOptions, MemoryPlanner, OptPipeline, Pass,
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
    /// Pre-built `AiGraph` (bypass importer).
    AiGraph(AiGraph),
    /// Multiple ONNX files forming a multi-component model.
    ///
    /// Each component is independently imported, optimized, and compiled.
    /// The result is a pipeline archive with a `MetaSection` describing
    /// component roles and connections.
    MultiOnnx {
        components: Vec<ComponentInput>,
        connections: Vec<hologram_ai_common::sections::meta::ComponentConnection>,
    },
}

/// Input specification for a single component in a multi-ONNX model.
pub struct ComponentInput {
    /// Pipeline key (e.g., "ae.encoder", "backbone").
    pub name: String,
    /// Path to the ONNX file for this component.
    pub path: PathBuf,
    /// What role this component plays.
    pub role: hologram_ai_common::sections::meta::ComponentRole,
    /// Components sharing this value share weights.
    pub weight_group: String,
}

// ── Model metadata ────────────────────────────────────────────────────────────

/// High-level metadata extracted from the model.
pub struct ModelMetadata {
    pub arch: String,
    pub vocab_size: u32,
    pub context_len: u32,
    pub n_layers: u32,
    pub n_embd: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
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

// ── Multi-component compilation ──────────────────────────────────────────────

/// Specification for a single component in a multi-component pipeline.
///
/// Used by [`ModelCompiler::compile_components`] to compile N independent
/// graphs into a single pipeline archive via `PipelineWriter`.
pub struct ComponentSpec<'a> {
    /// Pipeline key (e.g., "lm.prefill", "ae.encoder").
    pub name: String,
    /// What role this component plays in the pipeline.
    pub role: hologram_ai_common::sections::meta::ComponentRole,
    /// Components sharing this value share weights. Used by weight
    /// deduplication to avoid storing duplicate blobs.
    pub weight_group: String,
    /// Which optimization passes to run.
    pub opt_profile: hologram_ai_common::OptProfile,
    /// The component's graph (already imported, possibly pre-optimized).
    pub graph: &'a AiGraph,
    /// Memory plan for this component. Use `MemoryPlan::empty()` for
    /// components without attention / KV-cache.
    pub mem_plan: &'a hologram_ai_common::MemoryPlan,
    /// Lowering phase — determines layer name in the archive.
    pub phase: LowerPhase,
    /// Weight bytes. Components sharing a weight group should pass the same
    /// blob to the first component and `None` to the rest.
    pub weights: Option<Vec<u8>>,
}

// ── Model compiler ────────────────────────────────────────────────────────────

/// Compiles a `ModelSource` through the full pipeline into a `HoloArchive`.
///
/// Pipeline:
///   import → optimize → validate → plan memory → lower → compile → embed weights
pub struct ModelCompiler {
    /// Use memory-mapping for weight loading when possible.
    pub mmap: bool,
    /// Override the sequence length for compilation.
    /// When `Some(n)`, all seq-like dims are set to `n` instead of the
    /// model's `context_length`. Use `None` to auto-detect from metadata.
    pub seq_len_override: Option<u64>,
    /// Force single-graph compilation (skip LLM pipeline detection).
    /// Used internally for the decode sub-graph in LLM pipelines.
    pub force_single_graph: bool,
}

impl Default for ModelCompiler {
    fn default() -> Self {
        Self {
            mmap: true,
            seq_len_override: None,
            force_single_graph: false,
        }
    }
}

impl ModelCompiler {
    /// Compile a model source into a `.holo` archive.
    ///
    /// For LLM models (GGUF with transformer architecture), produces a pipeline
    /// archive with named layer entrypoints. For simpler models (ONNX), produces
    /// a single-graph archive.
    pub fn compile(&self, source: ModelSource) -> anyhow::Result<HoloArchive> {
        // Multi-component models have their own compilation path.
        if let ModelSource::MultiOnnx { components, connections } = source {
            return self.compile_multi_onnx(components, connections);
        }

        // Step 1 — import.
        let ai_graph = self.import(source)?;
        info!(nodes = ai_graph.nodes.len(), params = ai_graph.params.len(), "import complete");

        // Step 2 — optimize.
        let mut ai_graph = OptPipeline::mvp()
            .run(ai_graph)
            .context("optimization pass failed")?;
        info!(nodes = ai_graph.nodes.len(), "optimization complete");

        // Step 2a — infer LLM metadata from fused attention nodes.
        // ONNX models don't carry arch/n_layers/n_kv_heads metadata natively.
        // After AttentionFusion, we can extract these from GroupedQueryAttention
        // nodes so the MemoryPlanner and LLM pipeline detection work correctly.
        infer_llm_metadata_from_graph(&mut ai_graph);

        // Save the MVP-optimized graph before concretization so the LLM
        // pipeline can re-concretize it at seq=1 for the decode graph,
        // avoiding a full re-import + re-optimization from disk.
        let pre_concretized = ai_graph.clone();

        // Step 2b — concretize all symbolic/dynamic dims for compilation.
        // The runtime doesn't yet support deferred shape resolution, so we
        // bake in concrete values: Var dims → upper bounds, Dynamic → 1.
        let ai_graph = concretize_all_dims(ai_graph, self.seq_len_override).context("shape concretization failed")?;

        // Step 2c — post-concretization shape repair (fixpoint loop with
        // aggressive shape propagation, slice→gather, shape healing).
        let ai_graph = post_concretization_repair(ai_graph)?;

        // Diagnostic: report empty shapes, attention dims, MatMul shapes,
        // inf/NaN params after repair.
        log_post_repair_diagnostics(&ai_graph);

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

        let archive_bytes = if is_llm && !self.force_single_graph {
            self.compile_llm_pipeline(&ai_graph, &mem_plan, pre_concretized)?
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
        let mut ai_graph = OptPipeline::mvp()
            .run(ai_graph)
            .context("optimization pass failed")?;

        // Infer LLM metadata from GQA nodes (same as compile()).
        infer_llm_metadata_from_graph(&mut ai_graph);

        let ai_graph = concretize_all_dims(ai_graph, self.seq_len_override).context("shape concretization failed")?;

        // Post-concretization repair (same as compile()).
        let ai_graph = post_concretization_repair(ai_graph)?;

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

    /// Compile a model and return both the debug map and the `ShapeContextGraph`.
    ///
    /// Like [`compile_with_debug_info`](Self::compile_with_debug_info) but also
    /// returns the compile-time shape projection map so callers can verify that
    /// `walk_shape_context()` produces correct shapes for given runtime inputs.
    pub fn compile_with_shape_context(
        &self,
        source: ModelSource,
    ) -> anyhow::Result<(HoloArchive, DebugMap, Option<ShapeContextGraph>)> {
        // Reuse the full pipeline by calling compile_with_debug_info, but we
        // also need the ShapeContextGraph from the LoweringOutput.  Rather
        // than duplicating the whole pipeline, we re-run the lower step once
        // more to extract the context. This is acceptable for testing code.
        let ai_graph = self.import(source)?;
        let ai_graph = OptPipeline::mvp()
            .run(ai_graph)
            .context("optimization pass failed")?;
        let ai_graph = concretize_all_dims(ai_graph, self.seq_len_override).context("shape concretization failed")?;

        let ai_graph = post_concretization_repair(ai_graph)?;

        let errs = ai_graph.validate();
        if !errs.is_empty() {
            anyhow::bail!("{} validation error(s): {}", errs.len(), errs[0].message);
        }

        let mem_plan = MemoryPlanner
            .plan(&ai_graph)
            .context("memory planning failed")?;

        let metadata = extract_metadata(&ai_graph);
        let import_warnings = ai_graph.warnings.len();
        let node_count = ai_graph.nodes.len();

        let lower_out = lower(
            &ai_graph,
            &mem_plan.kv_cache_layout,
            &LoweringOptions::default(),
            &LowerPhase::Forward,
        )
        .context("lowering failed")?;

        // Extract ShapeContextGraph before the context is consumed.
        let shape_ctx = lower_out
            .context
            .get::<ShapeContextGraph>()
            .ok()
            .flatten();

        // Build debug map.
        let mut name_to_idx = std::collections::HashMap::new();
        for (tid, name) in &ai_graph.tensor_names {
            if let Some(&idx) = lower_out.tid_to_idx.get(tid) {
                name_to_idx.insert(name.clone(), idx);
            }
        }
        let debug_map = DebugMap { name_to_idx };

        // Compile and assemble archive.
        let compilation =
            hologram::compile(lower_out.graph).context("hologram::compile failed")?;
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

        Ok((archive, debug_map, shape_ctx))
    }

    /// Compile a non-LLM model into a single-graph archive.
    ///
    /// Collects weight bytes from mmap params and passes them as `extra_weights`
    /// so that `ConstantData::Deferred` offsets resolve correctly at runtime.
    fn compile_single_graph(
        &self,
        ai_graph: &AiGraph,
        mem_plan: &hologram_ai_common::MemoryPlan,
    ) -> anyhow::Result<Vec<u8>> {
        let weights = collect_weight_bytes(ai_graph)?;
        let extra_weights = if weights.is_empty() { None } else { Some(weights) };
        let archive = compile_one_component(
            ai_graph,
            &mem_plan.kv_cache_layout,
            &LoweringOptions::default(),
            &LowerPhase::Forward,
            extra_weights,
        )?;
        info!(archive_bytes = archive.len(), "single-graph archive assembled");
        Ok(archive)
    }

    /// Compile an LLM into a pipeline archive with prefill + decode sub-archives.
    ///
    /// Prefill: compiled at the configured seq_len (full prompt).
    /// Decode: compiled at seq=1 (single token per step).
    ///
    /// Delegates to [`compile_components`](Self::compile_components) with two
    /// `ComponentSpec` entries (prefill + decode).
    fn compile_llm_pipeline(
        &self,
        ai_graph: &AiGraph,
        mem_plan: &hologram_ai_common::MemoryPlan,
        pre_concretized: AiGraph,
    ) -> anyhow::Result<Vec<u8>> {
        let weights = collect_weight_bytes(ai_graph)?;
        let extra_weights = if weights.is_empty() { None } else { Some(weights) };
        info!(
            weight_mb = format_args!("{:.1}", extra_weights.as_ref().map_or(0, |w| w.len()) as f64 / 1_048_576.0),
            "compiling LLM pipeline (prefill + decode)"
        );

        // Re-concretize the pre-optimized graph at seq=1 for decode,
        // avoiding a full re-import + re-optimization from disk.
        info!("compiling decode graph (seq=1) from pre-optimized graph");
        let decode_graph = concretize_all_dims(pre_concretized, Some(1))
            .context("concretizing decode graph (seq=1) failed")?;
        let decode_graph = post_concretization_repair(decode_graph)?;

        use hologram_ai_common::sections::meta::{ComponentConnection, ComponentRole};

        // Both components share weight_group="lm" — WeightStore deduplicates
        // by group, so only one copy is stored in the shared blob.
        self.compile_components(
            vec![
                ComponentSpec {
                    name: "lm.prefill".into(),
                    role: ComponentRole::Prefill,
                    weight_group: "lm".into(),
                    opt_profile: hologram_ai_common::OptProfile::Llm,
                    graph: ai_graph,
                    mem_plan,
                    phase: LowerPhase::Prefill,
                    weights: extra_weights.clone(),
                },
                ComponentSpec {
                    name: "lm.decode".into(),
                    role: ComponentRole::Decode,
                    weight_group: "lm".into(),
                    opt_profile: hologram_ai_common::OptProfile::Llm,
                    graph: &decode_graph,
                    mem_plan,
                    phase: LowerPhase::Decode,
                    weights: extra_weights,
                },
            ],
            vec![ComponentConnection {
                from_component: "lm.prefill".into(),
                from_output: "kv_cache".into(),
                to_component: "lm.decode".into(),
                to_input: "kv_cache".into(),
            }],
        )
    }

    /// Compile N component specs into a single pipeline archive.
    ///
    /// Each component is independently lowered, compiled, and assembled into a
    /// sub-archive. All sub-archives are bundled via `PipelineWriter` into a
    /// single `.holo` pipeline archive with a `MetaSection` describing
    /// component roles, weight groups, and connections.
    pub fn compile_components(
        &self,
        specs: Vec<ComponentSpec<'_>>,
        connections: Vec<hologram_ai_common::sections::meta::ComponentConnection>,
    ) -> anyhow::Result<Vec<u8>> {
        use hologram::hologram_archive::section::EmbeddableSection;
        use hologram::hologram_archive::writer::pipeline_writer::PipelineWriter;
        use hologram::hologram_archive::WeightStore;
        use hologram_ai_common::sections::meta::{ComponentDescriptor, MetaSection};

        let n = specs.len();
        info!(components = n, "compiling multi-component pipeline");

        let mut descriptors = Vec::with_capacity(n);
        let mut writer = PipelineWriter::new();
        let mut weight_store = WeightStore::new();

        // Track which weight groups have already been seen so we can
        // record weight_source for deduplication hints and skip duplicate
        // weight embedding in sub-archives.
        let mut weight_group_owners: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        let mut total_weight_bytes_before: u64 = 0;

        for spec in specs {
            let _is_first_in_group = !weight_group_owners.contains_key(&spec.weight_group);
            let weight_source = if let Some(owner) = weight_group_owners.get(&spec.weight_group) {
                Some(owner.clone())
            } else {
                weight_group_owners.insert(spec.weight_group.clone(), spec.name.clone());
                None
            };

            // Register weights in the content-addressable store for dedup.
            // The shared blob is built later — sub-archives embed NO weights.
            // At load time, the pipeline loader resolves each component's
            // weights from the shared blob via WeightDedupIndex.
            if let Some(ref w) = spec.weights {
                total_weight_bytes_before += w.len() as u64;
                weight_store.insert(&spec.name, &spec.weight_group, w);
            }
            // Embed weights directly in each sub-archive. Shared weight dedup
            // via build_with_shared_weights has an offset alignment bug that
            // causes NaN in the embedding layer. Until fixed, embed weights
            // per-component (doubles archive size but produces correct output).
            let weights_for_component = spec.weights.clone();

            descriptors.push(ComponentDescriptor {
                name: spec.name.clone(),
                role: spec.role,
                weight_group: spec.weight_group,
                weight_source,
            });

            let opts = LoweringOptions::default();
            let archive = compile_one_component(
                spec.graph,
                &spec.mem_plan.kv_cache_layout,
                &opts,
                &spec.phase,
                weights_for_component,
            )
            .with_context(|| format!("compiling component '{}'", spec.name))?;
            info!(
                component = %spec.name,
                archive_bytes = archive.len(),
                "component compiled"
            );
            writer = writer.add_model(&spec.name, archive);
        }

        // Embed MetaSection in the pipeline wrapper archive.
        let meta = MetaSection::new(descriptors, connections);
        let meta_section = meta.to_bytes();
        writer = writer.add_section(meta.section_kind(), meta_section);

        // Build shared weight blob + dedup index. Sub-archives have no
        // embedded weights — the loader resolves them from the shared blob.
        let dedup_bytes = weight_store.total_bytes();
        let pipeline = if dedup_bytes > 0 {
            let savings = if total_weight_bytes_before > 0 && dedup_bytes < total_weight_bytes_before {
                (1.0 - dedup_bytes as f64 / total_weight_bytes_before as f64) * 100.0
            } else {
                0.0
            };
            let (shared_blob, dedup_index) = weight_store.build();
            info!(
                before_mb = format_args!("{:.1}", total_weight_bytes_before as f64 / 1_048_576.0),
                after_mb = format_args!("{:.1}", dedup_bytes as f64 / 1_048_576.0),
                savings_pct = format_args!("{:.0}", savings),
                "shared weight blob built (deduplicated)"
            );
            writer
                .build_with_shared_weights(shared_blob, &dedup_index)
                .map_err(|e| anyhow::anyhow!("building pipeline archive: {e}"))?
        } else {
            writer
                .build()
                .map_err(|e| anyhow::anyhow!("building pipeline archive: {e}"))?
        };
        info!(
            archive_bytes = pipeline.len(),
            components = n,
            "multi-component pipeline archive built"
        );
        Ok(pipeline)
    }

    /// Compile multiple ONNX files into a multi-component pipeline archive.
    ///
    /// Each component is independently imported, optimized, concretized,
    /// and compiled. The result is a single `.holo` pipeline archive with
    /// a `MetaSection` describing component roles and connections.
    ///
    /// This is the generic path for any multi-component ONNX model:
    /// CALM, Whisper, Stable Diffusion, encoder-decoder, etc.
    fn compile_multi_onnx(
        &self,
        components: Vec<ComponentInput>,
        connections: Vec<hologram_ai_common::sections::meta::ComponentConnection>,
    ) -> anyhow::Result<HoloArchive> {
        use hologram_ai_common::sections::meta::ComponentRole;

        info!(
            components = components.len(),
            connections = connections.len(),
            "compiling multi-ONNX pipeline"
        );

        // Track first-seen weight groups for dedup (computed from input order,
        // does not depend on compilation results).
        let mut weight_group_first: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (idx, comp) in components.iter().enumerate() {
            weight_group_first
                .entry(comp.weight_group.clone())
                .or_insert(idx);
        }

        // Import, optimize, concretize, and plan memory for each component
        // in parallel. Each component is fully independent at this stage.
        let seq_len = self.seq_len_override;
        let results: Vec<anyhow::Result<(AiGraph, hologram_ai_common::MemoryPlan)>> = components
            .par_iter()
            .map(|comp| {
                // Import.
                let ai_graph = hologram_ai_onnx::import_onnx_path(&comp.path, Default::default())
                    .with_context(|| format!("importing ONNX component '{}' from {:?}", comp.name, comp.path))?;
                info!(
                    component = %comp.name,
                    nodes = ai_graph.nodes.len(),
                    "imported component"
                );

                // Optimize with appropriate profile.
                let is_transformer = matches!(
                    comp.role,
                    ComponentRole::Prefill | ComponentRole::Decode | ComponentRole::Backbone
                );
                let ai_graph = if is_transformer {
                    OptPipeline::mvp().run(ai_graph).with_context(|| {
                        format!("optimizing component '{}' (Llm profile)", comp.name)
                    })?
                } else {
                    OptPipeline::generic().run(ai_graph).with_context(|| {
                        format!("optimizing component '{}' (Generic profile)", comp.name)
                    })?
                };

                // Concretize.
                let ai_graph = concretize_all_dims(ai_graph, seq_len)
                    .with_context(|| format!("concretizing component '{}'", comp.name))?;
                let ai_graph = post_concretization_repair(ai_graph)?;

                // Memory plan.
                let mem_plan = if is_transformer {
                    hologram_ai_common::MemoryPlanner
                        .plan(&ai_graph)
                        .with_context(|| format!("planning memory for component '{}'", comp.name))?
                } else {
                    hologram_ai_common::MemoryPlan::empty()
                };

                Ok((ai_graph, mem_plan))
            })
            .collect();

        // Unpack parallel results, preserving original component order.
        let mut graphs: Vec<AiGraph> = Vec::with_capacity(components.len());
        let mut mem_plans: Vec<hologram_ai_common::MemoryPlan> = Vec::with_capacity(components.len());
        for (idx, result) in results.into_iter().enumerate() {
            let (graph, mem_plan) = result
                .with_context(|| format!("compiling component '{}'", components[idx].name))?;
            graphs.push(graph);
            mem_plans.push(mem_plan);
        }

        // Collect weights and build ComponentSpecs.
        let mut specs: Vec<ComponentSpec<'_>> = Vec::with_capacity(graphs.len());

        for (i, comp) in components.iter().enumerate() {
            let graph = &graphs[i];
            let is_first_in_group = weight_group_first.get(&comp.weight_group) == Some(&i);

            // Only the first component in a weight group collects weights.
            // Subsequent components pass None — compile_components registers
            // them under the same group via WeightStore's group-reuse path.
            let weights = if is_first_in_group {
                let w = collect_weight_bytes(graph)?;
                if !w.is_empty() { Some(w) } else { None }
            } else {
                None
            };

            specs.push(ComponentSpec {
                name: comp.name.clone(),
                role: comp.role.clone(),
                weight_group: comp.weight_group.clone(),
                opt_profile: hologram_ai_common::OptProfile::Generic,
                graph,
                mem_plan: &mem_plans[i],
                phase: LowerPhase::Named(comp.name.clone()),
                weights,
            });
        }

        let total_weight_bytes: u64 = specs.iter().filter_map(|s| s.weights.as_ref()).map(|w| w.len() as u64).sum();
        let total_nodes: usize = graphs.iter().map(|g| g.nodes.len()).sum();

        let archive_bytes = self.compile_components(specs, connections)?;

        Ok(HoloArchive {
            bytes: archive_bytes,
            metadata: ModelMetadata {
                arch: "multi-onnx".into(),
                vocab_size: 0,
                context_len: self.seq_len_override.unwrap_or(0) as u32,
                n_layers: 0,
                n_embd: 0,
                n_kv_heads: 0,
                head_dim: 0,
            },
            stats: CompileStats {
                import_warnings: 0,
                validation_errors: 0,
                total_weight_bytes,
                node_count: total_nodes,
            },
        })
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
            ModelSource::AiGraph(g) => Ok(g),
            ModelSource::MultiOnnx { .. } => {
                unreachable!("MultiOnnx is handled before import()")
            }
        }
    }
}

// ── Single-pass archive assembly ─────────────────────────────────────────────

/// Raw components extracted from a compiled archive via a single
/// `load_from_bytes` call. Avoids repeated deserialization/decompression.
struct UnpackedArchive {
    /// Raw graph bytes (uncompressed rkyv, passed to `set_graph_bytes_uncompressed`).
    graph_bytes: Vec<u8>,
    /// Existing weight bytes from the archive.
    weight_bytes: Vec<u8>,
    /// Existing sections (kind, raw bytes).
    sections: Vec<(u32, Vec<u8>)>,
    /// The loaded plan — used to read layer_header, etc.
    plan: hologram::LoadedPlan,
}

/// Unpack a compiled archive into its raw components with a single
/// `load_from_bytes` call.  Transparently decompresses if the archive
/// was written with `compress_graph()` / `compress_weights()`.
fn unpack_archive(archive: &[u8]) -> anyhow::Result<UnpackedArchive> {
    use hologram::hologram_archive::{decompress_archive, is_compressed};

    let effective: std::borrow::Cow<'_, [u8]> = if is_compressed(archive) {
        std::borrow::Cow::Owned(
            decompress_archive(archive).context("decompressing archive for unpack")?
                .context("decompress returned None")?,
        )
    } else {
        std::borrow::Cow::Borrowed(archive)
    };

    let plan =
        hologram::load_from_bytes(&effective).context("unpacking archive")?;
    let h = plan.header();
    let graph_bytes =
        effective[h.graph_offset as usize..(h.graph_offset + h.graph_size) as usize].to_vec();
    let weight_bytes = plan.weights().to_vec();

    let mut sections = Vec::new();
    for entry in &plan.sections().entries {
        let offset = entry.offset as usize;
        let size = entry.size as usize;
        if offset + size <= effective.len() {
            sections.push((entry.kind, effective[offset..offset + size].to_vec()));
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
    // Uncompressed by default — enables zero-copy mmap via
    // `load_from_bytes_zero_copy()` at runtime. Compression can be
    // opted in via HologramConfig for distribution builds.
    let mut writer = hologram::HoloWriter::new()
        .set_graph_bytes_uncompressed(unpacked.graph_bytes)
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

/// Compile a single AiGraph into a sub-archive ready for PipelineWriter.
///
/// Encapsulates the lower → compile → unpack → assemble pipeline that was
/// previously duplicated across `compile_single_graph` and
/// `compile_llm_pipeline` (prefill/decode paths).
fn compile_one_component(
    ai_graph: &AiGraph,
    kv_layout: &hologram_ai_common::mem::KvCacheLayout,
    opts: &LoweringOptions,
    phase: &LowerPhase,
    extra_weights: Option<Vec<u8>>,
) -> anyhow::Result<Vec<u8>> {
    let phase_name = phase.layer_name();

    let lower_out = lower(ai_graph, kv_layout, opts, phase)
        .with_context(|| format!("lowering {phase_name} graph"))?;
    debug!(graph_nodes = lower_out.graph.node_count(), phase = phase_name, "lowered");

    let compiled = hologram::compile(lower_out.graph)
        .with_context(|| format!("compiling {phase_name} graph"))?;
    debug!(archive_bytes = compiled.archive.len(), phase = phase_name, "compiled");

    let unpacked = unpack_archive(&compiled.archive)?;
    let layer_header = build_tensor_port_header(&unpacked.plan, ai_graph);
    let bundle = if lower_out.context.is_empty() {
        None
    } else {
        Some(&lower_out.context)
    };

    build_final_archive(unpacked, extra_weights, Some(layer_header), bundle)
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
    let n_kv_heads = meta_u32(graph, "n_kv_heads").unwrap_or(0);
    let head_dim = meta_u32(graph, "head_dim").unwrap_or(0);

    ModelMetadata {
        arch,
        vocab_size,
        context_len,
        n_layers,
        n_embd,
        n_kv_heads,
        head_dim,
    }
}

/// Infer LLM architecture metadata from fused GroupedQueryAttention nodes.
///
/// ONNX models don't carry `arch`, `n_layers`, `n_kv_heads`, etc. natively.
/// After AttentionFusion, we can extract these from the GQA nodes so that
/// `compute_kv_layout` and `is_llm` detection work correctly.
/// Only sets metadata fields that are not already present (GGUF sets them
/// during import, so this is a no-op for GGUF models).
fn infer_llm_metadata_from_graph(graph: &mut AiGraph) {
    use hologram_ai_common::ir::op::AiOp;
    use hologram_ai_common::MetaValue;

    let gqa_params: Vec<(u32, u32, u32)> = graph
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            AiOp::GroupedQueryAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => Some((*num_heads, *num_kv_heads, *head_dim)),
            _ => None,
        })
        .collect();

    if gqa_params.is_empty() {
        return;
    }

    let n_layers = gqa_params.len() as u32;

    // Only infer LLM metadata for models with multiple attention layers.
    // Single-layer fixtures/tests should compile as single-graph, not pipeline.
    if n_layers < 2 {
        return;
    }
    let (num_heads, n_kv_heads, head_dim) = gqa_params[0];
    let n_embd = num_heads * head_dim;

    graph
        .metadata
        .entry("arch".into())
        .or_insert(MetaValue::Str("llama".into()));
    graph
        .metadata
        .entry("n_layers".into())
        .or_insert(MetaValue::Int(n_layers as i64));
    graph
        .metadata
        .entry("n_kv_heads".into())
        .or_insert(MetaValue::Int(n_kv_heads as i64));
    graph
        .metadata
        .entry("head_dim".into())
        .or_insert(MetaValue::Int(head_dim as i64));
    graph
        .metadata
        .entry("n_embd".into())
        .or_insert(MetaValue::Int(n_embd as i64));
    graph
        .metadata
        .entry("context_length".into())
        .or_insert(MetaValue::Int(2048));

    info!(
        n_layers,
        n_kv_heads,
        head_dim,
        n_embd,
        "inferred LLM metadata from {} GQA nodes",
        gqa_params.len()
    );
}

/// Log diagnostic information about the compiled graph after repair.
///
/// Reports empty shapes, Dynamic-dim root causes, suspect attention dims,
/// MatMul shapes, inf/NaN params, and total param size. All tracing-based
/// (no println).
fn log_post_repair_diagnostics(ai_graph: &AiGraph) {
    // Empty shapes.
    let empty_count = ai_graph
        .tensor_info
        .values()
        .filter(|info| info.shape.is_empty())
        .count();
    if empty_count > 0 {
        warn!(count = empty_count, "tensors still have empty shapes after repair");
    }

    // Dynamic-dim root causes.
    let mut dynamic_roots = 0u32;
    for node in &ai_graph.nodes {
        for &out_tid in &node.outputs {
            let has_dynamic = ai_graph
                .tensor_info
                .get(&out_tid)
                .map(|i| i.shape.iter().any(|d| matches!(d, hologram_ai_common::Dim::Dynamic)))
                .unwrap_or(false);
            if !has_dynamic {
                continue;
            }
            let all_inputs_concrete = node.inputs.iter().all(|&tid| {
                ai_graph
                    .tensor_info
                    .get(&tid)
                    .map(|i| !i.shape.is_empty() && i.shape.iter().all(|d| d.as_concrete().is_some()))
                    .unwrap_or(false)
            });
            if all_inputs_concrete && dynamic_roots < 2 {
                warn!(
                    node_id = node.id,
                    output = out_tid,
                    "Dynamic-dim root cause (all inputs concrete)"
                );
                dynamic_roots += 1;
            }
        }
    }

    // MatMul shapes (first 5).
    let mut matmul_count = 0u32;
    for node in &ai_graph.nodes {
        if matches!(node.op, hologram_ai_common::AiOp::MatMul | hologram_ai_common::AiOp::BatchMatMul)
            && matmul_count < 5
        {
            let input_shapes: Vec<_> = node
                .inputs
                .iter()
                .map(|&t| {
                    ai_graph
                        .tensor_info
                        .get(&t)
                        .map(|i| format!("T{t}:{:?}", i.shape.as_slice()))
                        .unwrap_or_else(|| format!("T{t}:<?>"))
                })
                .collect();
            if input_shapes.len() >= 2 {
                debug!(
                    node_id = node.id,
                    lhs = %input_shapes[0],
                    rhs = %input_shapes[1],
                    "MatMul"
                );
            }
            matmul_count += 1;
        }
    }

    // inf/NaN params.
    let mut nan_params = 0u32;
    for (&tid, param) in &ai_graph.params {
        if let hologram_ai_common::AiParam::Inline { data, info } = param {
            if info.logical_dtype == hologram_ai_common::DType::F32
                && !data.is_empty()
                && data.len() % 4 == 0
            {
                let floats: &[f32] = bytemuck::cast_slice(data);
                let bad = floats.iter().any(|f| !f.is_finite());
                if bad {
                    warn!(tid, "compiled f32 param has inf/NaN");
                    nan_params += 1;
                }
            }
        }
    }
    if nan_params > 0 {
        warn!(nan_params, "inf/NaN scalar params detected");
    }

    // Total param data size.
    let total_param_bytes: usize = ai_graph
        .params
        .values()
        .map(|p| match p {
            hologram_ai_common::AiParam::Inline { data, .. } => data.len(),
            _ => 0,
        })
        .sum();
    info!(
        entries = ai_graph.params.len(),
        total_mb = format_args!("{:.1}", total_param_bytes as f64 / 1_048_576.0),
        "params"
    );
}

/// Post-concretization shape repair.
///
/// After `concretize_all_dims`, most shapes are correct but some Dynamic dims
/// remain (from `broadcast_shape` mismatches, etc.). This function:
///
/// 1. Clears stale `known_i64_values` so DataProp re-evaluates.
/// 2. Runs the aggressive pipeline in a fixpoint loop (up to 3 iterations,
///    with early exit when no more Dynamic dims are resolved).
/// 3. Replaces any remaining Dynamic/Var dims with `Concrete(1)`.
/// 4. Converts Slice→Gather (hologram has no native Slice).
/// 5. Runs shape healing to fill any remaining empty shapes.
fn post_concretization_repair(mut ai_graph: AiGraph) -> anyhow::Result<AiGraph> {
    use hologram_ai_common::{
        AggressiveShapePropagation, ConstantDeduplication, Dim,
        opt::{
            const_eval::ConstantEvaluation, constant_fold::ConstantFolding,
            data_prop::DataPropagation, dead_node::DeadNodeElimination,
        },
    };

    // Clear stale known_i64_values so DataProp re-evaluates with the
    // now-concrete shapes.
    for info in ai_graph.tensor_info.values_mut() {
        info.known_i64_values = None;
    }

    // Post-concretization pipeline: uses AggressiveShapePropagation which
    // always overwrites shapes with inferred values. This is safe because
    // all dims are now concrete — no risk of overwriting good symbolic shapes
    // with weaker inferences.
    //
    // Two DataProp passes handle multi-level shape dependencies:
    //   Pass 1 (DataProp #1): evaluates shape subgraphs like Expand
    //     targets that depend on concrete input tensor shapes.
    //   Pass 2 (AggressiveProp #2): propagates DataProp #1 results
    //     to correctly shape intermediate tensors (e.g. K_intermediate).
    //   Pass 3 (DataProp #2): re-evaluates shape subgraphs that depend
    //     on K_intermediate's now-correct shape (e.g. K^T target).
    //   Pass 4 (AggressiveProp #3): applies DataProp #2 results.
    // ForceConcretize: replaces any remaining Dynamic/Var dims with
    // Concrete values. Inserted before ConstantEvaluation so that Shape
    // nodes can be folded (they need fully-concrete input shapes).
    struct ForceConcretize;
    impl Pass for ForceConcretize {
        fn name(&self) -> &str { "ForceConcretize" }
        fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
            let subs = graph.dim_vars.fixed_substitutions();
            for info in graph.tensor_info.values_mut() {
                for dim in info.shape.iter_mut() {
                    for (var_id, replacement) in &subs {
                        *dim = dim.substitute(*var_id, replacement);
                    }
                    if let Some(v) = dim.evaluate() {
                        *dim = Dim::Concrete(v);
                    }
                    if matches!(dim, Dim::Dynamic | Dim::Var(_)) {
                        *dim = Dim::Concrete(1);
                    }
                }
            }
            Ok(graph)
        }
    }

    let aggressive_pipeline = OptPipeline::new(vec![
        Box::new(AggressiveShapePropagation),
        Box::new(DataPropagation),
        Box::new(AggressiveShapePropagation),
        Box::new(DataPropagation),
        Box::new(AggressiveShapePropagation),
        Box::new(ForceConcretize),
        Box::new(ConstantEvaluation),
        Box::new(ConstantFolding),
        Box::new(ConstantDeduplication),
        Box::new(DeadNodeElimination),
    ]);

    // Fixpoint loop: each iteration resolves more Reshape targets as DataProp
    // traces through newly-resolved shapes. Early exit when no progress is made.
    let mut prev_dynamic_count = usize::MAX;
    for pass_num in 0..3 {
        ai_graph = aggressive_pipeline
            .run(ai_graph)
            .with_context(|| format!("post-concretization repair pass {pass_num} failed"))?;

        // Count remaining non-concrete dims for convergence check.
        let dynamic_count = ai_graph
            .tensor_info
            .values()
            .flat_map(|info| info.shape.iter())
            .filter(|dim| matches!(dim, Dim::Dynamic | Dim::Var(_)))
            .count();

        if dynamic_count >= prev_dynamic_count {
            info!(
                pass_num,
                dynamic_count,
                "post-concretization repair converged early"
            );
            break;
        }
        prev_dynamic_count = dynamic_count;

        // Clear stale known_i64_values between iterations so DataProp
        // re-evaluates with the freshly-inferred shapes.
        for info in ai_graph.tensor_info.values_mut() {
            info.known_i64_values = None;
        }
    }

    // Replace any Dynamic or remaining Var dims with Concrete(1).
    // concretize_all_dims already set seq-like dims to the correct
    // context_length; these are edge cases from the aggressive pipeline
    // introducing new Dynamic dims.
    for info in ai_graph.tensor_info.values_mut() {
        for dim in info.shape.iter_mut() {
            if matches!(dim, Dim::Dynamic | Dim::Var(_)) {
                *dim = Dim::Concrete(1);
            }
        }
    }

    // Convert Slice→Gather (hologram has no native Slice).
    // Must run after concretization so dim values are known.
    let ai_graph = hologram_ai_common::SliceToGather
        .run(ai_graph)
        .context("slice-to-gather conversion failed")?;

    // Shape healing: fill in any remaining empty shapes.
    let ai_graph = hologram_ai_common::ShapeHealing
        .run(ai_graph)
        .context("shape healing failed")?;

    Ok(ai_graph)
}

/// Concretize all symbolic and dynamic dimensions in the graph.
///
/// ALL dims become concrete at compile time. No 0-sentinels, no runtime
/// shape resolution needed. The executor simply dispatches with baked shapes.
///
/// - `DimExpr::Var` (sequence-like) → `context_length` from model metadata
/// - `DimExpr::Var` (batch-like) → 1
/// - `DimExpr::Dynamic` → inferred from context or defaulted
///
/// For LLM pipeline archives, the caller compiles prefill (seq=context_len)
/// and decode (seq=1) as separate graphs — both fully concrete.
fn concretize_all_dims(mut graph: AiGraph, seq_len_override: Option<u64>) -> anyhow::Result<AiGraph> {
    use hologram_ai_common::Dim;

    // Use override if provided, otherwise extract from model metadata.
    let context_len = seq_len_override
        .unwrap_or_else(|| meta_u32(&graph, "context_length").unwrap_or(2048) as u64);
    info!(context_len, "concretizing all dims (fully static shapes)");

    // Set concrete values for all dim vars based on their name.
    // Sequence-like dims get context_length; batch-like dims get 1.
    for (_, entry) in graph.dim_vars.iter_mut() {
        if entry.fixed.is_some() {
            continue;
        }
        let name_lower = entry.name.to_lowercase();
        if name_lower.contains("seq") || name_lower.contains("length") || name_lower.contains("position") {
            debug!(var = %entry.name, value = context_len, "concretizing seq-like dim");
            entry.fixed = Some(context_len);
        } else {
            debug!(var = %entry.name, value = 1, "concretizing batch-like dim");
            entry.upper = Some(1u64);
        }
    }

    // Concretize Var dims to their fixed/upper values.
    let _ = graph.dim_vars.concretize_to_upper();
    let subs = graph.dim_vars.fixed_substitutions();

    // Apply substitutions and replace any remaining non-concrete dims.
    for info in graph.tensor_info.values_mut() {
        for dim in info.shape.iter_mut() {
            for (var_id, replacement) in &subs {
                *dim = dim.substitute(*var_id, replacement);
            }
            if let Some(v) = dim.evaluate() {
                *dim = Dim::Concrete(v);
            }
            // Any remaining Dynamic or Var dims get context_length (seq-like)
            // rather than 0-sentinel. This ensures all shapes are fully concrete.
            if matches!(dim, Dim::Dynamic | Dim::Var(_)) {
                *dim = Dim::Concrete(context_len);
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
    // Single pre-allocated buffer — read directly into the blob without
    // intermediate per-weight Vec allocations.
    let mut blob = vec![0u8; total_size as usize];
    let mut write_offset = 0usize;

    for (_, param) in &sorted {
        if let AiParam::Mmap {
            path, offset, len, ..
        } = param
        {
            let n = *len as usize;
            let mut f = std::fs::File::open(path)
                .with_context(|| format!("opening weight file {path:?}"))?;
            f.seek(SeekFrom::Start(*offset))?;
            f.read_exact(&mut blob[write_offset..write_offset + n])
                .with_context(|| format!("reading {n} bytes from {path:?}"))?;
            write_offset += n;
        }
    }

    Ok(blob)
}

/// Rebuild a compiled archive adding an extra section.
///
/// Preserves all existing sections from the source archive so that
/// layer headers, model metadata, tokenizer data, etc. are not lost.
/// Uses a single unpack/repack cycle internally.
/// Pre-loaded archive ready for repeated shape-aware execution.
///
/// Supports both single-graph archives (non-LLM models) and pipeline archives
/// (LLM with prefill + decode sub-models). For pipeline archives, the first
/// `execute()` call runs the prefill model; subsequent calls run the decode model
/// (when KV cache is wired up — currently both use the prefill model).
///
/// Load once with [`HoloRunner::from_bytes`], then call [`HoloRunner::execute`]
/// many times with different inputs.
/// Owned archive storage — either heap-allocated Vec or memory-mapped file.
enum ArchiveStorage {
    Owned(Vec<u8>),
    Mmap(memmap2::Mmap),
}

impl AsRef<[u8]> for ArchiveStorage {
    fn as_ref(&self) -> &[u8] {
        match self {
            ArchiveStorage::Owned(v) => v,
            ArchiveStorage::Mmap(m) => m,
        }
    }
}

pub struct HoloRunner {
    /// Backing storage: mmap or heap. MUST be listed first so it's dropped last
    /// (LoadedPlan borrows from it).
    _storage: ArchiveStorage,
    /// For pipeline: the prefill sub-archive bytes (copied from storage).
    /// For single-graph: empty (plan borrows directly from _storage).
    _sub_archive_bytes: Vec<u8>,
    /// Single-graph plan (non-pipeline) or the prefill sub-model.
    plan: hologram::LoadedPlan,
    /// Decode sub-model plan (pipeline only, compiled at seq=1).
    decode_plan: Option<hologram::LoadedPlan>,
    /// Decode sub-archive bytes (for section lookups).
    _decode_bytes: Option<Vec<u8>>,
    shape_ctx: Option<ShapeContextGraph>,
    /// True if the archive is a pipeline (prefill + decode).
    is_pipeline: bool,
    /// Pre-compiled execution tape for the prefill graph.
    tape: hologram::hologram_exec::tape::EnumTape,
    /// Pre-compiled execution tape for the decode graph (pipeline only).
    decode_tape: Option<hologram::hologram_exec::tape::EnumTape>,
}

impl HoloRunner {
    /// Load a runner from raw archive bytes (heap-allocated).
    pub fn from_bytes(bytes: Vec<u8>) -> anyhow::Result<Self> {
        Self::from_storage(ArchiveStorage::Owned(bytes))
    }

    /// Load a runner from a `.holo` file on disk using memory-mapping.
    ///
    /// This avoids reading the entire archive (often multi-GB) into heap.
    /// Weights are accessed on-demand via page faults, so RSS stays low.
    ///
    /// If the archive is compressed, decompresses to a cache file for instant
    /// loading on subsequent runs. Cache location is controlled by `cache_dir`:
    /// - `None` — falls back to `HologramConfig` then caches next to the archive
    /// - `Some(dir)` — cache in the given directory (e.g., `~/.hologram/cache/`)
    pub fn from_path(
        path: &std::path::Path,
        cache_dir: Option<&std::path::Path>,
        config_path: Option<&std::path::Path>,
    ) -> anyhow::Result<Self> {
        // Load config: explicit path > standard search.
        let config = match config_path {
            Some(p) => hologram::config::HologramConfig::load_file(p)
                .unwrap_or_default(),
            None => hologram::config::HologramConfig::load(),
        };
        // CLI cache_dir > config cache.dir > default (next to archive).
        let config_cache = config.cache_dir();
        let cache_dir = cache_dir.or(config_cache.as_deref());
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening archive {}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("memory-mapping archive {}", path.display()))?;

        // If compressed, decompress to a cache file for instant loading.
        if hologram::hologram_archive::is_compressed(&mmap) {
            let cache_path = match cache_dir {
                Some(dir) => {
                    std::fs::create_dir_all(dir)
                        .with_context(|| format!("creating cache dir {}", dir.display()))?;
                    let stem = path.file_name().unwrap_or_default();
                    dir.join(format!("{}.cache", stem.to_string_lossy()))
                }
                None => path.with_extension("holo.cache"),
            };

            if cache_path.exists() {
                let cache_file = std::fs::File::open(&cache_path)
                    .with_context(|| format!("opening cache {}", cache_path.display()))?;
                let cache_mmap = unsafe { memmap2::Mmap::map(&cache_file) }
                    .with_context(|| format!("mmap cache {}", cache_path.display()))?;
                return Self::from_storage(ArchiveStorage::Mmap(cache_mmap));
            }

            eprintln!("decompressing to {} (one-time)...", cache_path.display());
            if let Some(uncompressed) = hologram::hologram_archive::decompress_archive(&mmap)
                .with_context(|| "decompressing archive")?
            {
                std::fs::write(&cache_path, &uncompressed)
                    .with_context(|| format!("writing cache {}", cache_path.display()))?;
                let cache_file = std::fs::File::open(&cache_path)?;
                let cache_mmap = unsafe { memmap2::Mmap::map(&cache_file) }?;
                return Self::from_storage(ArchiveStorage::Mmap(cache_mmap));
            }
        }

        Self::from_storage(ArchiveStorage::Mmap(mmap))
    }

    fn from_storage(storage: ArchiveStorage) -> anyhow::Result<Self> {
        let bytes: &[u8] = storage.as_ref();

        // Zero-copy probe: detect pipeline by checking section table.
        // SAFETY: storage outlives all plans created here.
        let probe = unsafe { hologram::load_from_bytes_zero_copy(bytes) }
            .map_err(|e| anyhow::anyhow!("loading archive: {e}"))?;
        let is_pipeline = probe.sections().entries.iter().any(|e| {
            e.kind == hologram::hologram_archive::section::SECTION_PIPELINE
        });

        if is_pipeline {
            // Pipeline: sub-archives are embedded in wrapper weights region.
            // Parse pipeline header from the section bytes, then use its
            // model offsets to slice directly from mmap — no copy.
            let weights_start = probe.header().weights_offset as usize;

            let pipeline_entry = probe.sections()
                .find(hologram::hologram_archive::section::SECTION_PIPELINE)
                .ok_or_else(|| anyhow::anyhow!("pipeline section missing"))?;
            let ps = pipeline_entry.offset as usize;
            let pe = ps + pipeline_entry.size as usize;
            let ph: hologram::hologram_archive::writer::pipeline_writer::PipelineHeader =
                rkyv::from_bytes::<hologram::hologram_archive::writer::pipeline_writer::PipelineHeader, rkyv::rancor::Error>(&bytes[ps..pe])
                    .map_err(|e| anyhow::anyhow!("parsing pipeline header: {e}"))?;

            tracing::debug!(
                weights_start,
                file_len = bytes.len(),
                n_models = ph.models.len(),
                "pipeline header parsed"
            );
            for entry in &ph.models {
                tracing::debug!(
                    name = %entry.name,
                    offset = entry.offset,
                    size = entry.size,
                    "pipeline model entry"
                );
            }

            let find_model = |name: &str| -> anyhow::Result<&[u8]> {
                let entry = ph.models.iter()
                    .find(|m| m.name == name)
                    .ok_or_else(|| anyhow::anyhow!("pipeline has no model '{name}'"))?;
                let start = weights_start + entry.offset as usize;
                let end = start + entry.size as usize;
                if end > bytes.len() {
                    anyhow::bail!("sub-archive '{name}' out of bounds: {start}..{end} > {}", bytes.len());
                }
                Ok(&bytes[start..end])
            };

            let prefill_slice = find_model("lm.prefill")?;
            let decode_slice = find_model("lm.decode")?;

            // Parse optional weight dedup index for shared-weight pipelines.
            let dedup_index = probe.sections()
                .find(hologram::hologram_archive::section::SECTION_WEIGHT_DEDUP)
                .and_then(|entry| {
                    let s = entry.offset as usize;
                    let e = s + entry.size as usize;
                    if e <= bytes.len() {
                        hologram::hologram_archive::WeightDedupIndex::from_bytes(&bytes[s..e]).ok()
                    } else {
                        None
                    }
                });

            // Zero-copy load both sub-archives.
            // SAFETY: storage (mmap) outlives these plans.
            let mut plan = unsafe { hologram::load_from_bytes_zero_copy(prefill_slice) }
                .map_err(|e| anyhow::anyhow!("loading prefill plan: {e}"))?;
            let mut decode_plan = unsafe { hologram::load_from_bytes_zero_copy(decode_slice) }
                .map_err(|e| anyhow::anyhow!("loading decode plan: {e}"))?;

            // Resolve shared weights: if sub-archives have no embedded weights,
            // graft from the wrapper's weight region via the dedup index.
            let wrapper_weights = probe.weights();
            if let Some(ref idx) = dedup_index {
                for (name, model_plan) in [("lm.prefill", &mut plan), ("lm.decode", &mut decode_plan)] {
                    if model_plan.weights().is_empty() {
                        if let Some(entry) = idx.find_component(name) {
                            let w_start = entry.offset as usize;
                            let w_end = w_start + entry.size as usize;
                            if w_end <= wrapper_weights.len() {
                                // SAFETY: wrapper_weights borrows from storage (mmap),
                                // which outlives these plans.
                                unsafe { model_plan.set_weights_borrowed(&wrapper_weights[w_start..w_end]); }
                            }
                        }
                    }
                }
            }

            let shape_ctx = read_shape_context_from_plan(&plan, prefill_slice)?;

            let tape = hologram::build_tape_from_plan(&plan)
                .map_err(|e| anyhow::anyhow!("building prefill tape: {e}"))?;
            let decode_tape = hologram::build_tape_from_plan(&decode_plan)
                .map_err(|e| anyhow::anyhow!("building decode tape: {e}"))?;

            Ok(Self {
                _storage: storage,
                _sub_archive_bytes: Vec::new(),
                plan,
                decode_plan: Some(decode_plan),
                _decode_bytes: None,
                shape_ctx,
                is_pipeline: true,
                tape,
                decode_tape: Some(decode_tape),
            })
        } else {
            // Single-graph: the entire archive IS the model.
            // Reuse the probe as the plan (already loaded zero-copy).
            let shape_ctx = read_shape_context_from_plan(&probe, bytes)?;

            let tape = hologram::build_tape_from_plan(&probe)
                .map_err(|e| anyhow::anyhow!("building execution tape: {e}"))?;

            Ok(Self {
                _storage: storage,
                _sub_archive_bytes: Vec::new(),
                plan: probe,
                decode_plan: None,
                _decode_bytes: None,
                shape_ctx,
                is_pipeline: false,
                tape,
                decode_tape: None,
            })
        }
    }

    /// Execute the compiled graph with the given inputs.
    ///
    /// Uses the EnumTape executor for zero-overhead dispatch with GPU backend
    /// support. Dynamic sizes are resolved via `resolve_size()` at execution time.
    pub fn execute(&self, inputs: &hologram::GraphInputs) -> anyhow::Result<hologram::GraphOutputs> {
        hologram::execute_tape(&self.tape, &self.plan, inputs)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Access the underlying loaded plan (for layer headers, weights, etc.).
    #[must_use]
    pub fn plan(&self) -> &hologram::LoadedPlan {
        &self.plan
    }

    /// Effective archive bytes (prefill sub-archive for pipeline, or full archive).
    #[must_use]
    pub fn archive_bytes(&self) -> &[u8] {
        if self.is_pipeline {
            &self._sub_archive_bytes
        } else {
            self._storage.as_ref()
        }
    }

    /// Raw top-level archive bytes (for section lookups on pipeline wrapper).
    /// For single-graph archives, returns the effective bytes.
    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        self._storage.as_ref()
    }

    /// Whether this archive has a `ShapeContextGraph` for variable seq_len support.
    #[must_use]
    pub fn has_shape_context(&self) -> bool {
        self.shape_ctx.is_some()
    }

    /// Whether this is a pipeline archive (prefill + decode sub-models).
    #[must_use]
    pub fn is_pipeline(&self) -> bool {
        self.is_pipeline
    }

    /// Execute with a mutable KV cache state for autoregressive generation.
    ///
    /// Uses the prefill graph for step 0, decode graph (seq=1) for subsequent steps.
    /// Uses the EnumTape executor for all execution (17.5x faster than legacy path).
    pub fn execute_with_kv(
        &self,
        inputs: &hologram::GraphInputs,
        kv_state: &mut hologram::KvCacheState,
    ) -> anyhow::Result<hologram::GraphOutputs> {
        let (tape, plan) = if kv_state.write_pos() > 0 {
            (
                self.decode_tape
                    .as_ref()
                    .expect("pipeline archive must have decode tape"),
                self.decode_plan
                    .as_ref()
                    .expect("pipeline archive must have decode plan"),
            )
        } else {
            (&self.tape, &self.plan)
        };
        hologram::execute_tape_with_kv(tape, plan, inputs, kv_state)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

}

/// Extract a named sub-archive's raw bytes from a pipeline archive.
///
/// Uses `LoadedPipeline` to parse the pipeline header, then extracts the
/// Read the [`ShapeContextGraph`] from an already-loaded plan + raw archive bytes.
///
/// Avoids re-deserializing the archive — uses the plan's section table to
/// find the shape context section, then reads it from the raw bytes.
fn read_shape_context_from_plan(
    plan: &hologram::LoadedPlan,
    archive_bytes: &[u8],
) -> anyhow::Result<Option<ShapeContextGraph>> {
    use hologram_ai_common::exec_context::{ExecContext, SECTION_SHAPE_CONTEXT};
    let entry = match plan.sections().find(SECTION_SHAPE_CONTEXT) {
        Some(e) => e,
        None => return Ok(None),
    };
    let start = entry.offset as usize;
    let end = start + entry.size as usize;
    if end > archive_bytes.len() {
        anyhow::bail!(
            "ShapeContextGraph section out of bounds: offset={} size={} archive_len={}",
            start,
            entry.size,
            archive_bytes.len()
        );
    }
    let ctx = ShapeContextGraph::from_context_bytes(&archive_bytes[start..end])?;
    Ok(Some(ctx))
}

/// Read the [`ShapeContextGraph`] embedded in a compiled `.holo` archive.
///
/// Returns `None` if the archive was compiled without a shape context section
/// (older archives or models compiled with shape context disabled).
pub fn read_shape_context_from_archive(archive_bytes: &[u8]) -> anyhow::Result<Option<ShapeContextGraph>> {
    // SAFETY: plan is dropped at the end of this function; archive_bytes outlives it.
    let plan = unsafe { hologram::load_from_bytes_zero_copy(archive_bytes) }?;
    read_shape_context_from_plan(&plan, archive_bytes)
}

/// Execute a compiled archive with variable-length shape support.
///
/// Builds a one-shot tape and executes via the EnumTape path.
/// Dynamic sizes are resolved at execution time via `resolve_size()`
/// and `infer_matmul_k()` in the tape executor.
///
/// For repeated execution, prefer [`HoloRunner`] which builds the tape once.
pub fn run_with_shape_context(
    archive: &HoloArchive,
    inputs: &hologram::GraphInputs,
) -> anyhow::Result<hologram::GraphOutputs> {
    let plan = hologram::load_from_bytes(&archive.bytes)?;
    let tape = hologram::build_tape_from_plan(&plan)
        .map_err(|e| anyhow::anyhow!("building execution tape: {e}"))?;
    hologram::execute_tape(&tape, &plan, inputs)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

pub fn rebuild_archive_with_section(
    archive: &[u8],
    section: &dyn hologram::hologram_archive::section::EmbeddableSection,
) -> anyhow::Result<Vec<u8>> {
    let unpacked = unpack_archive(archive)?;

    // Filter out the section kind we're replacing.
    let new_kind = section.section_kind();
    let mut writer = hologram::HoloWriter::new()
        .set_graph_bytes_uncompressed(unpacked.graph_bytes)
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

