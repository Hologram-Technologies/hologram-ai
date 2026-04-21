//! Model compilation pipeline.
//!
//! Compiles ONNX models into `.holo` archives via the hologram O(1) LUT
//! runtime. This crate is a **compiler** — it does not own inference
//! sessions or runtime state (see ADR-0016).

use anyhow::Context;
use hologram_ai_common::{
    exec_context::ShapeContextGraph, lower, AiGraph, AiParam, LowerPhase, LoweringOptions,
    MemoryPlanner, OptPipeline, Pass,
};
use rayon::prelude::*;
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
    /// The compiled archive bytes. Empty when `path` is set.
    pub bytes: Vec<u8>,
    /// Path to the archive on disk. Set for streaming compilation of
    /// large models — the archive was written directly to disk without
    /// ever loading it into memory.
    pub path: Option<std::path::PathBuf>,
    pub metadata: ModelMetadata,
    pub stats: CompileStats,
}

/// Sections to embed in the archive during the single build pass.
///
/// Collected by the CLI before compilation so that the compiler can
/// include them in `build_final_archive_to_file` — no post-processing
/// `rebuild_archive_with_section` round-trips needed.
#[derive(Default)]
pub struct ArchiveSections {
    sections: Vec<(u32, Vec<u8>)>,
}

impl ArchiveSections {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a section from an `EmbeddableSection` implementor.
    pub fn add(&mut self, section: &dyn hologram::hologram_archive::section::EmbeddableSection) {
        self.sections
            .push((section.section_kind(), section.to_bytes()));
    }

    /// Add a raw section (kind + pre-serialized bytes).
    pub fn add_raw(&mut self, kind: u32, bytes: Vec<u8>) {
        self.sections.push((kind, bytes));
    }

    /// Whether any sections have been added.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// Consume and return the collected sections.
    pub fn into_inner(self) -> Vec<(u32, Vec<u8>)> {
        self.sections
    }
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
    /// Weight bytes (borrowed). Components sharing a weight group should pass
    /// the same slice to the first component and `None` to the rest.
    pub weights: Option<&'a [u8]>,
    /// Per-tensor weight offset index for this component.
    pub weight_index: Option<hologram::hologram_archive::weight::index::WeightIndex>,
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
    /// Weight quantization strategy for LUT-GEMM acceleration.
    /// When set to `Q4_0`, f32 MatMul weights are quantized at compile time
    /// to 4-bit centroid indices, enabling the LUT-GEMM execution path.
    pub quant_strategy: hologram_ai_common::lower::QuantStrategy,
    /// Scale factor for spatial dimensions (H, W) in 4D tensors.
    /// When `Some(n)`, input spatial dims are divided by `n` before
    /// compilation. E.g., `spatial_scale: Some(4)` compiles a 512×512
    /// model at 128×128 resolution, using 16× less activation memory.
    /// Shape propagation derives all downstream dims from the scaled input.
    pub spatial_scale: Option<u32>,
    /// Patch budget ratio for ViT models (PixelPrune).
    /// When `Some(ratio)`, the compiler inserts a patch pruning pass that
    /// rewrites the ViT graph to accept a reduced token sequence of
    /// `ceil(grid_h × grid_w × ratio)` patches. A runtime pre-processing
    /// kernel selects the most informative patches via 2D predictive coding
    /// before the compiled graph runs. Positions are preserved for correct
    /// positional encoding.
    ///
    /// Range: `(0.0, 1.0]`. Default: `Some(0.75)`.
    /// Set to `None` to disable patch pruning entirely.
    pub patch_budget_ratio: Option<f32>,
}

impl Default for ModelCompiler {
    fn default() -> Self {
        Self {
            mmap: true,
            seq_len_override: None,
            spatial_scale: None,
            quant_strategy: hologram_ai_common::lower::QuantStrategy::Auto,
            patch_budget_ratio: Some(0.75),
        }
    }
}

impl ModelCompiler {
    /// Build `LoweringOptions` from this compiler's configuration.
    fn lowering_options(&self) -> LoweringOptions {
        LoweringOptions {
            quant_strategy: self.quant_strategy,
        }
    }

    /// Compile a model source into a `.holo` archive.
    ///
    /// For LLM models (GGUF with transformer architecture), produces a pipeline
    /// archive with named layer entrypoints. For simpler models (ONNX), produces
    /// a single-graph archive.
    ///
    /// `extra_sections`: optional sections (model_meta, tokenizer, host) to
    /// embed in the archive during the single build pass. Avoids the costly
    /// rebuild_archive_with_section post-processing that unpacks + repacks the
    /// entire archive per section (3× for a 14 GB SDXL UNet archive = 42 GB
    /// of wasted allocations).
    pub fn compile(&self, source: ModelSource) -> anyhow::Result<HoloArchive> {
        self.compile_with_sections(source, ArchiveSections::new())
    }

    /// Compile with embedded sections.
    ///
    /// Sections (model_meta, tokenizer, host) are included in the single
    /// archive build pass. No post-processing `rebuild_archive_with_section`
    /// round-trips — critical for large models where each rebuild unpacks +
    /// repacks the entire archive.
    pub fn compile_with_sections(
        &self,
        source: ModelSource,
        sections: ArchiveSections,
    ) -> anyhow::Result<HoloArchive> {
        let extra_sections = sections.into_inner();
        // Multi-component models have their own compilation path.
        if let ModelSource::MultiOnnx {
            components,
            connections,
        } = source
        {
            return self.compile_multi_onnx(components, connections, &extra_sections);
        }

        // Extract model directory for companion file discovery (config.json).
        let model_dir = match &source {
            ModelSource::OnnxPath(p) => p.parent().map(|d| d.to_path_buf()),
            _ => None,
        };

        // Step 1 — import.
        let mut ai_graph = {
            let _span = tracing::info_span!("compile_import").entered();
            self.import(source)?
        };
        info!(
            nodes = ai_graph.nodes.len(),
            params = ai_graph.params.len(),
            "import complete"
        );

        // Step 1a — read companion config.json for architecture metadata.
        // HuggingFace models include config.json with model_type, rope_theta,
        // vocab_size, etc. Pre-set these on the graph so that
        // infer_llm_metadata_from_graph uses them instead of defaults.
        if let Some(dir) = &model_dir {
            seed_metadata_from_config_json(&mut ai_graph, dir);
        }

        // Step 1b — apply spatial scaling if requested.
        if let Some(scale) = self.spatial_scale {
            apply_spatial_scale(&mut ai_graph, scale);
        }

        // Step 2 — optimize.
        // Choose pipeline based on input signature: models with `input_ids`
        // are text models that benefit from attention fusion + KV injection.
        // Vision/diffusion models (sample, pixel_values, etc.) use the
        // generic pipeline to avoid incorrectly injecting KV cache ops.
        let has_input_ids = ai_graph.input_names.iter().any(|n| n == "input_ids");
        // Only use LLM pipeline (attention fusion + KV cache) for causal
        // language models. Encoder-only models (BERT, CLIP) have input_ids
        // but should NOT get KV injection. Heuristic: causal LMs output
        // "logits"; encoders output "last_hidden_state" or similar.
        let looks_like_causal_lm = has_input_ids
            && ai_graph
                .output_names
                .iter()
                .any(|n| n == "logits" || n == "output");
        // Detect ViT topology: image-like 4D input feeding a Conv2d with
        // kernel == stride (patch embedding). Use the ViT pipeline which
        // includes patch pruning when a budget ratio is configured.
        let looks_like_vit = !looks_like_causal_lm
            && ai_graph
                .input_names
                .iter()
                .any(|n| n == "pixel_values" || n == "image" || n == "input_image" || n == "x");
        let pipeline = if looks_like_causal_lm {
            OptPipeline::mvp()
        } else if looks_like_vit {
            if let Some(ratio) = self.patch_budget_ratio {
                info!(
                    budget_ratio = ratio,
                    "using ViT pipeline with patch pruning"
                );
                OptPipeline::vit(ratio)
            } else {
                info!("using ViT pipeline (patch pruning disabled)");
                OptPipeline::vit(1.0)
            }
        } else {
            info!("using generic optimization pipeline (no input_ids detected)");
            OptPipeline::generic()
        };
        let mut ai_graph = {
            let _span = tracing::info_span!("compile_optimize").entered();
            pipeline.run(ai_graph).context("optimization pass failed")?
        };
        info!(nodes = ai_graph.nodes.len(), "optimization complete");

        // Step 2a — infer LLM metadata from fused attention nodes.
        // ONNX models don't carry arch/n_layers/n_kv_heads metadata natively.
        // After AttentionFusion, we can extract these from GroupedQueryAttention
        // nodes so the MemoryPlanner and LLM pipeline detection work correctly.
        if looks_like_causal_lm {
            infer_llm_metadata_from_graph(&mut ai_graph);
        }

        // Step 2b — concretize + compile.
        // For LLM models: compile two graphs (prefill at prompt seq_len, decode
        // at seq=1) as a pipeline archive. The decode graph is ~Nx faster because
        // all seq-dependent MatMul/Attention dimensions are 1 instead of N.
        // For non-LLM models: compile one graph at the specified seq_len.

        // Extract metadata from the pre-concretized graph (before cloning).
        let pre_metadata = extract_metadata(&ai_graph);
        let import_warnings = ai_graph.warnings.len();
        // Determine if this is an LLM needing prefill+decode pipeline.
        // Check: detected as causal LM AND has fused attention layers (from AttentionFusion).
        let prefill_mem = MemoryPlanner
            .plan(&ai_graph)
            .context("memory planning failed")?;
        let is_llm = looks_like_causal_lm && prefill_mem.kv_cache_layout.n_layers > 0;

        use hologram_ai_common::sections::meta::ComponentRole;

        // Collect weight bytes. For large models (>256 MB of Mmap weights),
        // stream to a temp file to avoid a multi-GB Vec<u8> allocation.
        let total_mmap_bytes: u64 = ai_graph
            .params
            .values()
            .filter_map(|p| match p {
                AiParam::Mmap { len, .. } => Some(*len),
                _ => None,
            })
            .sum();
        let use_streaming = total_mmap_bytes > 256 * 1024 * 1024; // 256 MB threshold

        let (weights, weight_source, weight_index) = {
            let _span = tracing::info_span!("compile_collect_weights", streaming = use_streaming).entered();
            if use_streaming {
                let (source, idx) = collect_weight_bytes_streaming(&ai_graph)?;
                (Vec::new(), Some(source), idx)
            } else {
                let (w, idx) = collect_weight_bytes(&ai_graph)?;
                (w, None, idx)
            }
        };
        let total_weight_bytes = if use_streaming {
            total_mmap_bytes
        } else {
            weights.len() as u64
        };
        let extra_weights: Option<&[u8]> = if weights.is_empty() && !use_streaming {
            None
        } else if !weights.is_empty() {
            Some(&weights)
        } else {
            // Streaming path: weights are in temp file, not in memory.
            // Pass empty slice — the archive will be built via build_to_file.
            Some(&[])
        };

        let archive_bytes = if is_llm {
            // ── LLM pipeline: prefill (seq=N) + decode (seq=1) ──────────────
            info!("compiling LLM pipeline: prefill + decode");

            // Clone the optimized graph for decode and verify paths (before concretization).
            let decode_ai_graph = ai_graph.clone();
            let verify_ai_graph = ai_graph.clone();

            // Prepare all three LLM graphs in parallel.
            // Each is fully independent after cloning: concretize → repair →
            // validate → memory plan. Uses std::thread::scope (not rayon) to
            // avoid work-stealing contention with inner parallelism.
            let verify_seq = 8u64;
            let seq_override = self.seq_len_override;
            let (prefill, decode, verify) = std::thread::scope(|s| {
                let h_prefill = s.spawn(|| {
                    prepare_llm_component(ai_graph, seq_override, "prefill", true)
                });
                let h_decode = s.spawn(|| {
                    prepare_llm_component(decode_ai_graph, Some(1), "decode", false)
                });
                let h_verify = s.spawn(|| {
                    prepare_llm_component(verify_ai_graph, Some(verify_seq), "verify", false)
                });
                // Join all threads — propagate panics.
                let prefill = h_prefill.join().expect("prefill thread panicked");
                let decode = h_decode.join().expect("decode thread panicked");
                let verify = h_verify.join().expect("verify thread panicked");
                (prefill, decode, verify)
            });
            let prefill = prefill.context("prefill preparation failed")?;
            let decode = decode.context("decode preparation failed")?;
            let verify = verify.context("verify preparation failed")?;

            let prefill_nodes = prefill.node_count;
            let decode_nodes = decode.node_count;
            let verify_nodes = verify.node_count;

            info!(
                prefill_nodes,
                decode_nodes,
                verify_nodes,
                verify_seq,
                "LLM pipeline graphs ready (prefill + decode + verify)"
            );

            // Compile all three as a pipeline via streaming writer.
            // The first sub-archive (prefill) gets the real weights via
            // add_model_streaming. The other two get graph-only sub-archives.
            if let Some(ws) = weight_source {
                // Streaming path: build pipeline archive to a temp file.
                use hologram::hologram_archive::section::EmbeddableSection;
                use hologram::hologram_archive::writer::pipeline_writer::PipelineWriter;
                use hologram_ai_common::sections::meta::{
                    ComponentConnection, ComponentDescriptor, MetaSection,
                };

                let tmp_output = tempfile::NamedTempFile::new()
                    .context("creating temp output file for LLM pipeline")?;
                let output_path = tmp_output
                    .into_temp_path()
                    .keep()
                    .map_err(|e| anyhow::anyhow!("persisting temp output: {e}"))?;
                let scratch = tempfile::NamedTempFile::new()
                    .context("creating scratch file for LLM pipeline assembly")?;
                let scratch_path = scratch.path().to_path_buf();

                // Mmap the weight file for the quantize pass.
                let weight_mmap: Option<memmap2::Mmap> = match &ws {
                    hologram::hologram_archive::WeightSource::File { path, .. } => {
                        let f = std::fs::File::open(path)
                            .context("opening weight file for quantize mmap")?;
                        Some(unsafe { memmap2::Mmap::map(&f) }
                            .context("mmap weight file for quantize pass")?)
                    }
                    _ => None,
                };
                let wm_slice: Option<&[u8]> = weight_mmap.as_deref();

                // Compile sub-archives in parallel (graph + sections, no weights).
                let lowering_opts = self.lowering_options();
                let (prefill_archive, decode_archive, verify_archive) =
                    std::thread::scope(|s| {
                        let h_p = s.spawn(|| {
                            compile_one_component(
                                &prefill.graph,
                                &prefill.mem_plan.kv_cache_layout,
                                &lowering_opts,
                                &LowerPhase::Forward,
                                Some(&[]),
                                Some(&weight_index),
                                total_weight_bytes / 4,
                                wm_slice,
                            )
                        });
                        let h_d = s.spawn(|| {
                            compile_one_component(
                                &decode.graph,
                                &decode.mem_plan.kv_cache_layout,
                                &lowering_opts,
                                &LowerPhase::Forward,
                                None,
                                None,
                                total_weight_bytes / 4,
                                wm_slice,
                            )
                        });
                        let h_v = s.spawn(|| {
                            compile_one_component(
                                &verify.graph,
                                &verify.mem_plan.kv_cache_layout,
                                &lowering_opts,
                                &LowerPhase::Forward,
                                None,
                                None,
                                total_weight_bytes / 4,
                                wm_slice,
                            )
                        });
                        let p = h_p.join().expect("prefill compile thread panicked");
                        let d = h_d.join().expect("decode compile thread panicked");
                        let v = h_v.join().expect("verify compile thread panicked");
                        (p, d, v)
                    });
                let prefill_result = prefill_archive
                    .context("compiling prefill component")?;
                let decode_result = decode_archive
                    .context("compiling decode component")?;
                let verify_result = verify_archive
                    .context("compiling verify component")?;

                // Build MetaSection.
                let meta = MetaSection::new(
                    vec![
                        ComponentDescriptor {
                            name: "prefill".into(),
                            role: ComponentRole::Prefill,
                            weight_group: "model".into(),
                            weight_source: None,
                        },
                        ComponentDescriptor {
                            name: "decode".into(),
                            role: ComponentRole::Decode,
                            weight_group: "model".into(),
                            weight_source: Some("prefill".into()),
                        },
                        ComponentDescriptor {
                            name: "verify".into(),
                            role: ComponentRole::Prefill,
                            weight_group: "model".into(),
                            weight_source: Some("prefill".into()),
                        },
                    ],
                    vec![] as Vec<ComponentConnection>,
                );

                // If all weights were encoded (Q4/Q8), the f32 weight source
                // is no longer needed — all data is in Bytes constants inside
                // the sub-archive. Skip streaming to avoid bloated archives.
                let mut writer = if prefill_result.all_weights_encoded {
                    tracing::info!(
                        "all weights encoded — skipping f32 weight stream"
                    );
                    PipelineWriter::new()
                        .add_model("prefill", prefill_result.archive)
                        .add_model("decode", decode_result.archive)
                        .add_model("verify", verify_result.archive)
                        .add_section(meta.section_kind(), meta.to_bytes())
                } else {
                    PipelineWriter::new()
                        .add_model_streaming("prefill", prefill_result.archive, ws)
                        .add_model("decode", decode_result.archive)
                        .add_model("verify", verify_result.archive)
                        .add_section(meta.section_kind(), meta.to_bytes())
                };

                for (kind, bytes) in &extra_sections {
                    writer = writer.add_section(*kind, bytes.clone());
                }

                // Add model_meta section for LLM detection at load time.
                {
                    use hologram::hologram_archive::section::EmbeddableSection;
                    let model_meta =
                        hologram::hologram_archive::section::model_meta::ModelMetaSection {
                            kind:
                                hologram::hologram_archive::section::model_meta::ModelKind::TextLlm,
                            arch: pre_metadata.arch.clone(),
                            description: pre_metadata.arch.clone(),
                            max_seq_len: pre_metadata.context_len,
                            supports_prompt: true,
                            n_layers: pre_metadata.n_layers,
                            n_kv_heads: pre_metadata.n_kv_heads,
                            head_dim: pre_metadata.head_dim,
                            kv_k_bits: 0,
                            kv_v_bits: 0,
                            kv_boundary_layers: 2,
                            kv_wht: false,
                        };
                    writer = writer.add_section(model_meta.section_kind(), model_meta.to_bytes());
                }

                writer
                    .build_to_file(&output_path, &scratch_path)
                    .map_err(|e| anyhow::anyhow!("building streaming LLM pipeline: {e}"))?;

                let _ = std::fs::remove_file(&scratch_path);

                let archive_size = std::fs::metadata(&output_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                info!(
                    archive_mb = archive_size / (1024 * 1024),
                    "streaming LLM pipeline archive: {} MiB",
                    archive_size / (1024 * 1024),
                );

                return Ok(HoloArchive {
                    bytes: Vec::new(),
                    path: Some(output_path),
                    metadata: pre_metadata,
                    stats: CompileStats {
                        import_warnings,
                        validation_errors: 0,
                        total_weight_bytes,
                        node_count: prefill_nodes + decode_nodes + verify_nodes,
                    },
                });
            }

            // Non-streaming fallback (small models < 256 MB).
            self.compile_components_with_sections(
                vec![
                    ComponentSpec {
                        name: "prefill".into(),
                        role: ComponentRole::Prefill,
                        weight_group: "model".into(),
                        opt_profile: hologram_ai_common::OptProfile::Llm,
                        graph: &prefill.graph,
                        mem_plan: &prefill.mem_plan,
                        phase: LowerPhase::Forward,
                        weights: extra_weights,
                        weight_index: Some(weight_index.clone()),
                    },
                    ComponentSpec {
                        name: "decode".into(),
                        role: ComponentRole::Decode,
                        weight_group: "model".into(),
                        opt_profile: hologram_ai_common::OptProfile::Llm,
                        graph: &decode.graph,
                        mem_plan: &decode.mem_plan,
                        phase: LowerPhase::Forward,
                        weights: extra_weights,
                        weight_index: Some(weight_index.clone()),
                    },
                    ComponentSpec {
                        name: "verify".into(),
                        role: ComponentRole::Prefill,
                        weight_group: "model".into(),
                        opt_profile: hologram_ai_common::OptProfile::Llm,
                        graph: &verify.graph,
                        mem_plan: &verify.mem_plan,
                        phase: LowerPhase::Forward,
                        weights: extra_weights,
                        weight_index: Some(weight_index),
                    },
                ],
                vec![],
                &extra_sections,
            )?
        } else {
            // ── Non-LLM: single graph ───────────────────────────────────────
            let (ai_graph, seq_dim_positions) =
                concretize_all_dims(ai_graph, self.seq_len_override)
                    .context("shape concretization failed")?;
            let mut ai_graph = post_concretization_repair(ai_graph)?;
            zero_seq_dims_for_lowering(&mut ai_graph, &seq_dim_positions);
            log_post_repair_diagnostics(&ai_graph);
            let errs = ai_graph.validate();
            if !errs.is_empty() {
                anyhow::bail!("{} validation error(s): {}", errs.len(), errs[0].message);
            }

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

            let mem_plan = MemoryPlanner
                .plan(&ai_graph)
                .context("memory planning failed")?;

            if let Some(ws) = weight_source {
                // Streaming path: build a pipeline archive directly to a
                // temp file, streaming weights from the collection temp file.
                // Uses PipelineWriter so the output is a proper pipeline
                // archive (SECTION_PIPELINE header), enabling partial
                // recompilation of individual components.
                use hologram::hologram_archive::section::EmbeddableSection;
                use hologram::hologram_archive::writer::pipeline_writer::PipelineWriter;
                use hologram_ai_common::sections::meta::{
                    ComponentConnection, ComponentDescriptor, ComponentRole, MetaSection,
                };

                let tmp_output =
                    tempfile::NamedTempFile::new().context("creating temp output file")?;
                let output_path = tmp_output
                    .into_temp_path()
                    .keep()
                    .map_err(|e| anyhow::anyhow!("persisting temp output: {e}"))?;

                let scratch = tempfile::NamedTempFile::new()
                    .context("creating scratch file for pipeline assembly")?;
                let scratch_path = scratch.path().to_path_buf();

                // Compile the sub-archive (graph + sections, empty weights).
                let result = compile_one_component(
                    &ai_graph,
                    &mem_plan.kv_cache_layout,
                    &self.lowering_options(),
                    &LowerPhase::Forward,
                    Some(&[]),
                    Some(&weight_index),
                    total_weight_bytes / 4,
                    if weights.is_empty() { None } else { Some(&weights) },
                )?;

                // Build MetaSection for the pipeline wrapper.
                let meta = MetaSection::new(
                    vec![ComponentDescriptor {
                        name: "model".into(),
                        role: ComponentRole::Backbone,
                        weight_group: "model".into(),
                        weight_source: None,
                    }],
                    vec![] as Vec<ComponentConnection>,
                );

                let writer = if result.all_weights_encoded {
                    tracing::info!("all weights encoded — skipping f32 weight stream");
                    PipelineWriter::new()
                        .add_model("model", result.archive)
                        .add_section(meta.section_kind(), meta.to_bytes())
                } else {
                    PipelineWriter::new()
                        .add_model_streaming("model", result.archive, ws)
                        .add_section(meta.section_kind(), meta.to_bytes())
                };

                writer
                    .build_to_file(&output_path, &scratch_path)
                    .map_err(|e| anyhow::anyhow!("building streaming pipeline archive: {e}"))?;

                // Clean up scratch file.
                let _ = std::fs::remove_file(&scratch_path);

                let archive_size = std::fs::metadata(&output_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                info!(
                    archive_mb = archive_size / (1024 * 1024),
                    "streaming pipeline archive: {} MiB",
                    archive_size / (1024 * 1024),
                );

                return Ok(HoloArchive {
                    bytes: Vec::new(),
                    path: Some(output_path),
                    metadata: pre_metadata,
                    stats: CompileStats {
                        import_warnings,
                        validation_errors: 0,
                        total_weight_bytes,
                        node_count: 0,
                    },
                });
            } else {
                self.compile_components_with_sections(
                    vec![ComponentSpec {
                        name: "model".into(),
                        role: ComponentRole::Backbone,
                        weight_group: "model".into(),
                        opt_profile: hologram_ai_common::OptProfile::Generic,
                        graph: &ai_graph,
                        mem_plan: &mem_plan,
                        phase: LowerPhase::Forward,
                        weights: extra_weights,
                        weight_index: Some(weight_index.clone()),
                    }],
                    vec![],
                    &extra_sections,
                )?
            }
        };

        let metadata = pre_metadata;
        let node_count = 0; // TODO: sum prefill + decode nodes

        Ok(HoloArchive {
            bytes: archive_bytes,
            path: None,
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
        info!(
            nodes = ai_graph.nodes.len(),
            params = ai_graph.params.len(),
            "import complete (debug)"
        );

        // Capture tensor_names before optimization passes (passes preserve it).
        let mut ai_graph = OptPipeline::mvp()
            .run(ai_graph)
            .context("optimization pass failed")?;

        // Infer LLM metadata from GQA nodes (same as compile()).
        infer_llm_metadata_from_graph(&mut ai_graph);

        let (ai_graph, seq_dim_positions) = concretize_all_dims(ai_graph, self.seq_len_override)
            .context("shape concretization failed")?;

        // Post-concretization repair (same as compile()).
        let mut ai_graph = post_concretization_repair(ai_graph)?;
        zero_seq_dims_for_lowering(&mut ai_graph, &seq_dim_positions);

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
        let lowering_opts = self.lowering_options();
        let lower_out = lower(
            &ai_graph,
            &mem_plan.kv_cache_layout,
            &lowering_opts,
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
        let (weights, weight_index) = collect_weight_bytes(&ai_graph)?;
        let bundle = if lower_out.context.is_empty() {
            None
        } else {
            Some(&lower_out.context)
        };
        let total_weight_bytes = weights.len() as u64;
        let wi = if weight_index.entries.is_empty() {
            None
        } else {
            Some(&weight_index)
        };
        let archive_bytes = build_final_archive(
            unpacked,
            if weights.is_empty() {
                None
            } else {
                Some(&weights)
            },
            Some(layer_header),
            bundle,
            wi,
            &[],
        )?;

        let archive = HoloArchive {
            bytes: archive_bytes,
            path: None,
            metadata,
            stats: CompileStats {
                import_warnings,
                validation_errors: 0,
                total_weight_bytes,
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
        let (ai_graph, seq_dim_positions) = concretize_all_dims(ai_graph, self.seq_len_override)
            .context("shape concretization failed")?;

        let mut ai_graph = post_concretization_repair(ai_graph)?;
        zero_seq_dims_for_lowering(&mut ai_graph, &seq_dim_positions);

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

        let lowering_opts = self.lowering_options();
        let mut lower_out = lower(
            &ai_graph,
            &mem_plan.kv_cache_layout,
            &lowering_opts,
            &LowerPhase::Forward,
        )
        .context("lowering failed")?;

        // Extract ShapeContextGraph before the context is consumed.
        let mut shape_ctx = lower_out.context.get::<ShapeContextGraph>().ok().flatten();

        // Build debug map.
        let mut name_to_idx = std::collections::HashMap::new();
        for (tid, name) in &ai_graph.tensor_names {
            if let Some(&idx) = lower_out.tid_to_idx.get(tid) {
                name_to_idx.insert(name.clone(), idx);
            }
        }
        let debug_map = DebugMap { name_to_idx };

        // Compile and assemble archive.
        let compilation = hologram::compile(lower_out.graph).context("hologram::compile failed")?;
        let unpacked = unpack_archive(&compilation.archive)?;

        // Prune shape context entries referencing nodes removed by fusion,
        // then update the context bundle so the archive embeds the pruned version.
        if let Some(ref mut ctx) = shape_ctx {
            let live_ids: std::collections::HashSet<u32> = unpacked
                .plan
                .graph()
                .nodes
                .iter()
                .map(|n| n.id.index())
                .collect();
            ctx.retain_live_nodes(&live_ids);
            lower_out.context.insert(ctx);
        }
        let layer_header = build_tensor_port_header(&unpacked.plan, &ai_graph);
        let (weights, weight_index) = collect_weight_bytes(&ai_graph)?;
        let bundle = if lower_out.context.is_empty() {
            None
        } else {
            Some(&lower_out.context)
        };
        let total_weight_bytes = weights.len() as u64;
        let wi = if weight_index.entries.is_empty() {
            None
        } else {
            Some(&weight_index)
        };
        let archive_bytes = build_final_archive(
            unpacked,
            if weights.is_empty() {
                None
            } else {
                Some(&weights)
            },
            Some(layer_header),
            bundle,
            wi,
            &[],
        )?;

        let archive = HoloArchive {
            bytes: archive_bytes,
            path: None,
            metadata,
            stats: CompileStats {
                import_warnings,
                validation_errors: 0,
                total_weight_bytes,
                node_count,
            },
        };

        Ok((archive, debug_map, shape_ctx))
    }

    /// Compile a non-LLM model into a single-graph archive.
    ///
    /// Collects weight bytes from mmap params and passes them as `extra_weights`
    /// so that `ConstantData::Deferred` offsets resolve correctly at runtime.
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
        self.compile_components_with_sections(specs, connections, &[])
    }

    /// Compile components with additional sections (tokenizer, host meta, etc.)
    /// embedded in the pipeline wrapper archive.
    pub fn compile_components_with_sections(
        &self,
        specs: Vec<ComponentSpec<'_>>,
        connections: Vec<hologram_ai_common::sections::meta::ComponentConnection>,
        extra_sections: &[(u32, Vec<u8>)],
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

            if let Some(w) = spec.weights {
                total_weight_bytes_before += w.len() as u64;
                // Weights go into the shared store for later dedup.
                // We'll skip building the shared blob if all components share
                // the same weight_group (checked after the loop).
                weight_store.insert(&spec.name, &spec.weight_group, w);
            }
            // For single-component: embed weights in the sub-archive directly.
            // For multi-component with shared weight_group (LLM prefill+decode):
            // embed in the FIRST component only. Both graphs reference the same
            // constants in the same order, so weight offsets are identical.
            // The second component resolves weights from the first via dedup index.
            let is_first_in_group = weight_source.is_none();
            let weights_for_component: Option<&[u8]> = if n == 1 || is_first_in_group {
                spec.weights
            } else {
                None
            };
            let wi_for_component = if n == 1 || is_first_in_group {
                spec.weight_index.as_ref()
            } else {
                None
            };

            descriptors.push(ComponentDescriptor {
                name: spec.name.clone(),
                role: spec.role,
                weight_group: spec.weight_group,
                weight_source,
            });

            let lowering_opts = self.lowering_options();
            let result = compile_one_component(
                spec.graph,
                &spec.mem_plan.kv_cache_layout,
                &lowering_opts,
                &spec.phase,
                weights_for_component,
                wi_for_component,
                total_weight_bytes_before / 4,
                None, // multi-component: weights are in-memory via spec.weights
            )
            .with_context(|| format!("compiling component '{}'", spec.name))?;
            info!(
                component = %spec.name,
                archive_bytes = result.archive.len(),
                "component compiled"
            );
            writer = writer.add_model(&spec.name, result.archive);
        }

        // Embed MetaSection in the pipeline wrapper archive.
        let meta = MetaSection::new(descriptors, connections);
        let meta_section = meta.to_bytes();
        writer = writer.add_section(meta.section_kind(), meta_section);

        // Embed extra sections (tokenizer, host metadata, etc.) in the wrapper.
        for (kind, bytes) in extra_sections {
            writer = writer.add_section(*kind, bytes.clone());
        }

        // Build shared weight blob + dedup index for multi-component models
        // with DIFFERENT weight groups (e.g., SD text_encoder + unet).
        // For LLM pipeline (prefill+decode sharing one weight_group), skip the
        // shared blob — weights are embedded in the first sub-archive and the
        // decode component borrows them at load time via set_weights_borrowed().
        let distinct_groups = weight_group_owners.len();
        let dedup_bytes = weight_store.total_bytes();
        let needs_shared_blob = dedup_bytes > 0 && n > 1 && distinct_groups > 1;
        let pipeline = if needs_shared_blob {
            let savings =
                if total_weight_bytes_before > 0 && dedup_bytes < total_weight_bytes_before {
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
        _extra_sections: &[(u32, Vec<u8>)],
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
                    .with_context(|| {
                    format!(
                        "importing ONNX component '{}' from {:?}",
                        comp.name, comp.path
                    )
                })?;
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
                let (ai_graph, seq_dim_positions) = concretize_all_dims(ai_graph, seq_len)
                    .with_context(|| format!("concretizing component '{}'", comp.name))?;
                let mut ai_graph = post_concretization_repair(ai_graph)?;
                zero_seq_dims_for_lowering(&mut ai_graph, &seq_dim_positions);

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
        let mut mem_plans: Vec<hologram_ai_common::MemoryPlan> =
            Vec::with_capacity(components.len());
        for (idx, result) in results.into_iter().enumerate() {
            let (graph, mem_plan) = result
                .with_context(|| format!("compiling component '{}'", components[idx].name))?;
            graphs.push(graph);
            mem_plans.push(mem_plan);
        }

        // Collect weights and build ComponentSpecs.
        let mut specs: Vec<ComponentSpec<'_>> = Vec::with_capacity(graphs.len());

        // Collect weight blobs and indexes for each weight group.
        let mut weight_blobs: Vec<Vec<u8>> = Vec::new();
        let mut weight_indexes: Vec<hologram::hologram_archive::weight::index::WeightIndex> =
            Vec::new();
        let mut weight_blob_indices: Vec<Option<usize>> = Vec::new();

        for (i, comp) in components.iter().enumerate() {
            let graph = &graphs[i];
            let is_first_in_group = weight_group_first.get(&comp.weight_group) == Some(&i);

            if is_first_in_group {
                let (w, wi) = collect_weight_bytes(graph)?;
                if !w.is_empty() {
                    weight_blob_indices.push(Some(weight_blobs.len()));
                    weight_blobs.push(w);
                    weight_indexes.push(wi);
                } else {
                    weight_blob_indices.push(None);
                }
            } else {
                weight_blob_indices.push(None);
            }
        }

        for (i, comp) in components.iter().enumerate() {
            let graph = &graphs[i];
            let weights: Option<&[u8]> =
                weight_blob_indices[i].map(|idx| weight_blobs[idx].as_slice());
            let weight_index = weight_blob_indices[i].map(|idx| weight_indexes[idx].clone());

            specs.push(ComponentSpec {
                name: comp.name.clone(),
                role: comp.role.clone(),
                weight_group: comp.weight_group.clone(),
                opt_profile: hologram_ai_common::OptProfile::Generic,
                graph,
                mem_plan: &mem_plans[i],
                phase: LowerPhase::Named(comp.name.clone()),
                weights,
                weight_index,
            });
        }

        let total_weight_bytes: u64 = specs
            .iter()
            .filter_map(|s| s.weights.as_ref())
            .map(|w| w.len() as u64)
            .sum();
        let total_nodes: usize = graphs.iter().map(|g| g.nodes.len()).sum();

        let archive_bytes = self.compile_components(specs, connections)?;

        Ok(HoloArchive {
            bytes: archive_bytes,
            path: None,
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
            decompress_archive(archive)
                .context("decompressing archive for unpack")?
                .context("decompress returned None")?,
        )
    } else {
        std::borrow::Cow::Borrowed(archive)
    };

    let plan = hologram::load_from_bytes(&effective).context("unpacking archive")?;
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
    extra_weights: Option<&[u8]>,
    layer_header: Option<hologram::hologram_archive::entrypoint::schedule::LayerHeader>,
    bundle: Option<&hologram_ai_common::ContextBundle>,
    weight_index: Option<&hologram::hologram_archive::weight::index::WeightIndex>,
    extra_sections: &[(u32, Vec<u8>)],
) -> anyhow::Result<Vec<u8>> {
    use hologram::hologram_archive::section::{
        EmbeddableSection, SECTION_LAYER_HEADER, SECTION_WEIGHT_INDEX,
    };

    let weights = match extra_weights {
        Some(w) => w.to_vec(),
        None => unpacked.weight_bytes,
    };
    // Uncompressed by default — enables zero-copy mmap via
    // `load_from_bytes_zero_copy()` at runtime. Compression can be
    // opted in via HologramConfig for distribution builds.
    let mut writer = hologram::HoloWriter::new()
        .set_graph_bytes_uncompressed(unpacked.graph_bytes)
        .set_weights(weights);

    // Determine which section kinds will be replaced.
    let layer_header_kind = layer_header.as_ref().map(|lh| lh.section_kind());
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
        if weight_index.is_some() && kind == SECTION_WEIGHT_INDEX {
            continue;
        }
        writer = writer.add_raw_section(kind, bytes);
    }

    // Add the new LayerHeader section.
    if let Some(ref lh) = layer_header {
        writer = writer.add_raw_section(SECTION_LAYER_HEADER, lh.to_bytes());
    }

    // Add weight offset index section.
    if let Some(wi) = weight_index {
        writer = writer.add_raw_section(SECTION_WEIGHT_INDEX, wi.to_bytes());
    }

    // Add all bundle sections.
    if let Some(bundle) = bundle {
        for (kind, bytes) in bundle.iter() {
            writer = writer.add_raw_section(kind, bytes.to_vec());
        }
    }

    // Add caller-provided sections (model_meta, tokenizer, host_meta).
    for (kind, bytes) in extra_sections {
        writer = writer.add_raw_section(*kind, bytes.clone());
    }

    writer
        .build()
        .map_err(|e| anyhow::anyhow!("building final archive: {e}"))
}

/// Build a final archive to a file, streaming weights from a `WeightSource`.
///
/// Unlike `build_final_archive` which holds all weights in memory, this writes
/// the archive directly to disk. Peak memory: graph + sections (~tens of MB).
fn _build_final_archive_to_file(
    unpacked: UnpackedArchive,
    weight_source: hologram::hologram_archive::WeightSource,
    layer_header: Option<hologram::hologram_archive::entrypoint::schedule::LayerHeader>,
    bundle: Option<&hologram_ai_common::ContextBundle>,
    weight_index: Option<&hologram::hologram_archive::weight::index::WeightIndex>,
    extra_sections: &[(u32, Vec<u8>)],
    output_path: &std::path::Path,
) -> anyhow::Result<()> {
    use hologram::hologram_archive::section::{
        EmbeddableSection, SECTION_LAYER_HEADER, SECTION_WEIGHT_INDEX,
    };

    let mut writer = hologram::HoloWriter::new()
        .set_graph_bytes_uncompressed(unpacked.graph_bytes)
        .set_weight_source(weight_source);

    let layer_header_kind = layer_header.as_ref().map(|lh| lh.section_kind());
    let bundle_kinds: Vec<u32> = bundle
        .map(|b| b.iter().map(|(k, _)| k).collect())
        .unwrap_or_default();

    for (kind, bytes) in unpacked.sections {
        if layer_header_kind == Some(kind) {
            continue;
        }
        if bundle_kinds.contains(&kind) {
            continue;
        }
        if weight_index.is_some() && kind == SECTION_WEIGHT_INDEX {
            continue;
        }
        writer = writer.add_raw_section(kind, bytes);
    }

    if let Some(ref lh) = layer_header {
        writer = writer.add_raw_section(SECTION_LAYER_HEADER, lh.to_bytes());
    }

    if let Some(wi) = weight_index {
        writer = writer.add_raw_section(SECTION_WEIGHT_INDEX, wi.to_bytes());
    }

    if let Some(bundle) = bundle {
        for (kind, bytes) in bundle.iter() {
            writer = writer.add_raw_section(kind, bytes.to_vec());
        }
    }

    // Add caller-provided sections (model_meta, tokenizer, host_meta).
    for (kind, bytes) in extra_sections {
        writer = writer.add_raw_section(*kind, bytes.clone());
    }

    writer
        .build_to_file(output_path)
        .map_err(|e| anyhow::anyhow!("building final archive to file: {e}"))
}

/// Result of preparing a single LLM component (concretize + repair + validate + plan).
struct PreparedComponent {
    graph: AiGraph,
    mem_plan: hologram_ai_common::MemoryPlan,
    node_count: usize,
}

/// Prepare a single LLM component: concretize dims, repair, validate, plan memory.
///
/// This is the common pipeline shared by prefill, decode, and verify graph
/// preparation. Extracted to enable parallel preparation of all three.
fn prepare_llm_component(
    ai_graph: AiGraph,
    seq_len: Option<u64>,
    component_name: &str,
    zero_seq_dims: bool,
) -> anyhow::Result<PreparedComponent> {
    let _span =
        tracing::info_span!("compile_prepare", component = component_name).entered();
    let (graph, seq_dim_positions) = concretize_all_dims(ai_graph, seq_len)
        .with_context(|| format!("{component_name} concretization failed"))?;
    let mut graph = post_concretization_repair(graph)?;
    if zero_seq_dims {
        zero_seq_dims_for_lowering(&mut graph, &seq_dim_positions);
        log_post_repair_diagnostics(&graph);
    }
    let errs = graph.validate();
    if !errs.is_empty() {
        anyhow::bail!(
            "{component_name}: {} validation error(s): {}",
            errs.len(),
            errs[0].message
        );
    }
    let mem_plan = MemoryPlanner
        .plan(&graph)
        .with_context(|| format!("{component_name} memory planning failed"))?;
    let node_count = graph.nodes.len();
    Ok(PreparedComponent {
        graph,
        mem_plan,
        node_count,
    })
}

/// Result from compiling a single component.
struct ComponentResult {
    /// The sub-archive bytes.
    archive: Vec<u8>,
    /// True if all Deferred (f32) weight constants were replaced by
    /// quantized Bytes constants. When true, the streaming weight source
    /// is no longer needed for this component.
    all_weights_encoded: bool,
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
    extra_weights: Option<&[u8]>,
    weight_index: Option<&hologram::hologram_archive::weight::index::WeightIndex>,
    total_params: u64,
    weight_mmap: Option<&[u8]>,
) -> anyhow::Result<ComponentResult> {
    let phase_name = phase.layer_name();
    let _span = tracing::info_span!("compile_one_component", phase = phase_name).entered();

    let mut lower_out = {
        let _span = tracing::info_span!("lower", phase = phase_name).entered();
        lower(ai_graph, kv_layout, opts, phase)
            .with_context(|| format!("lowering {phase_name} graph"))?
    };
    debug!(
        graph_nodes = lower_out.graph.node_count(),
        phase = phase_name,
        "lowered"
    );

    // Validate: check all Gemm/MatMul nodes' weight inputs are valid constants.
    validate_matmul_constants(&lower_out.graph, extra_weights);

    // UOR encoding resolution (Plan 077): convert eligible Float(MatMul)/Gemm
    // to MatMulLut4/8 with content-addressed constants.
    let (content_entries, all_weights_encoded) = {
        let _span = tracing::info_span!("resolve_encodings", phase = phase_name).entered();
        let mut quant_cache = std::collections::HashMap::new();
        let stats = hologram_ai_common::lower::resolve_encodings(
            &mut lower_out.graph,
            opts.quant_strategy,
            total_params,
            &mut quant_cache,
            weight_mmap,
        )?;
        // Check if any Deferred constants remain in the lowered graph.
        // If none, the streaming weight source is no longer needed.
        let store = lower_out.graph.constant_store();
        let has_deferred = (0..store.len()).any(|i| {
            store
                .get(hologram::ConstantId::new(i as u32))
                .map_or(false, |c| c.is_deferred())
        });
        let all_weights_encoded = stats.encoded > 0 && !has_deferred;
        if stats.encoded > 0 {
            tracing::info!(
                encoded = stats.encoded,
                skipped = stats.skipped,
                saved_mb = stats.bytes_saved / (1024 * 1024),
                content_entries = stats.content_entries.len(),
                all_weights_encoded,
                "UOR encoding resolution"
            );
        }
        (stats.content_entries, all_weights_encoded)
    };

    // Trim the streaming weight blob to contain only non-quantized Deferred
    // constants. This replaces the 4+ GB f32 blob with a much smaller blob
    // (~400 MB for TinyLlama) containing just embeddings, biases, and norms.
    let trimmed_weights: Option<Vec<u8>> = if !all_weights_encoded {
        weight_mmap.and_then(|wm| {
            hologram_ai_common::lower::trim_weight_blob(&mut lower_out.graph, wm)
        })
    } else {
        None
    };

    let compiled = {
        let _span = tracing::info_span!("hologram_compile", phase = phase_name).entered();
        hologram::compile(lower_out.graph)
            .with_context(|| format!("compiling {phase_name} graph"))?
    };
    debug!(
        archive_bytes = compiled.archive.len(),
        phase = phase_name,
        "compiled"
    );

    let unpacked = unpack_archive(&compiled.archive)?;

    // Prune shape context entries referencing nodes removed by fusion.
    if let Some(mut ctx) = lower_out.context.get::<ShapeContextGraph>().ok().flatten() {
        let live_ids: std::collections::HashSet<u32> = unpacked
            .plan
            .graph()
            .nodes
            .iter()
            .map(|n| n.id.index())
            .collect();
        ctx.retain_live_nodes(&live_ids);
        lower_out.context.insert(&ctx);
    }

    // Embed PatchPruneContext if the graph has patch pruning metadata.
    if let Some(hologram_ai_common::MetaValue::Int(max_kept)) =
        ai_graph.metadata.get("patch_prune_budget")
    {
        let total_patches = ai_graph
            .metadata
            .get("patch_prune_grid_patches")
            .and_then(|v| {
                if let hologram_ai_common::MetaValue::Int(n) = v {
                    Some(*n as u32)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let embed_dim = ai_graph
            .metadata
            .get("patch_prune_embed_dim")
            .and_then(|v| {
                if let hologram_ai_common::MetaValue::Int(n) = v {
                    Some(*n as u32)
                } else {
                    None
                }
            })
            .unwrap_or(0);

        // Find the kept_indices input index.
        let kept_idx = ai_graph
            .input_names
            .iter()
            .position(|n| n == "kept_indices")
            .unwrap_or(ai_graph.inputs.len().saturating_sub(1)) as u32;

        // Find the pixel input index (first image-like input).
        let pixel_idx = ai_graph
            .input_names
            .iter()
            .position(|n| n == "pixel_values" || n == "image" || n == "input_image" || n == "x")
            .unwrap_or(0) as u32;

        // Infer patch size from the Conv2d in the graph.
        let (patch_h, patch_w) = ai_graph
            .nodes
            .iter()
            .find_map(|n| match &n.op {
                hologram_ai_common::AiOp::Conv {
                    kernel_shape,
                    strides,
                    ..
                } if kernel_shape == strides && kernel_shape.len() == 2 => {
                    Some((kernel_shape[0] as u32, kernel_shape[1] as u32))
                }
                _ => None,
            })
            .unwrap_or((16, 16));

        // Infer channels from pixel input shape.
        let channels = ai_graph
            .tensor_info
            .get(&ai_graph.inputs[pixel_idx as usize])
            .and_then(|info| info.shape.get(1)?.as_concrete())
            .unwrap_or(3) as u32;

        let prune_ctx = hologram_ai_common::PatchPruneContext {
            kept_indices_input: kept_idx,
            pixel_input: pixel_idx,
            channels,
            patch_h,
            patch_w,
            total_patches,
            max_kept: *max_kept as u32,
        };
        lower_out.context.insert(&prune_ctx);
        let _ = embed_dim; // used for logging context, not stored
        info!(
            max_kept,
            total_patches,
            patch_h,
            patch_w,
            channels,
            kept_indices_input = kept_idx,
            "embedded PatchPruneContext in archive"
        );
    }

    let layer_header = build_tensor_port_header(&unpacked.plan, ai_graph);
    let bundle = if lower_out.context.is_empty() {
        None
    } else {
        Some(&lower_out.context)
    };

    // Build ContentAddressIndex from encoding resolution entries (Plan 077).
    let mut extra_sections: Vec<(u32, Vec<u8>)> = Vec::new();
    if !content_entries.is_empty() {
        use hologram::hologram_archive::section::EmbeddableSection;
        use hologram::hologram_archive::weight::content_addr::ContentAddressIndex;
        let mut index = ContentAddressIndex::with_capacity(content_entries.len());
        // Offset tracking: we don't know the exact blob offsets here since they
        // depend on the archive layout. Use byte_size as placeholder offsets —
        // the ContentAddressIndex primarily serves as a digest registry for now.
        // Full offset resolution comes when PipelineWriter is content-address-aware.
        let mut running_offset: u64 = 0;
        for entry in &content_entries {
            index.insert(entry.digest, running_offset, entry.byte_size);
            running_offset += entry.byte_size;
        }
        index.sort();
        extra_sections.push((index.section_kind(), index.to_bytes()));
        tracing::info!(
            entries = content_entries.len(),
            "emitted ContentAddressIndex section"
        );
    }

    // Use trimmed weight blob if available (quantized streaming path),
    // otherwise fall back to the original extra_weights.
    let effective_weights: Option<&[u8]> = if let Some(ref tw) = trimmed_weights {
        Some(tw.as_slice())
    } else {
        extra_weights
    };

    let archive = build_final_archive(
        unpacked,
        effective_weights,
        Some(layer_header),
        bundle,
        weight_index,
        &extra_sections,
    )?;
    Ok(ComponentResult {
        archive,
        all_weights_encoded: all_weights_encoded || trimmed_weights.is_some(),
    })
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

/// Read companion `config.json` (HuggingFace format) and seed metadata on the
/// graph. This lets `infer_llm_metadata_from_graph` use the correct arch name
/// and context length instead of defaulting to "llama"/2048.
fn seed_metadata_from_config_json(graph: &mut AiGraph, model_dir: &std::path::Path) {
    use hologram_ai_common::MetaValue;

    let config_path = model_dir.join("config.json");
    let data = match std::fs::read_to_string(&config_path) {
        Ok(d) => d,
        Err(_) => return,
    };
    let json: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(_) => return,
    };

    // model_type → arch (e.g. "qwen2", "llama", "mistral", "gemma", "phi")
    if let Some(model_type) = json.get("model_type").and_then(|v| v.as_str()) {
        graph
            .metadata
            .entry("arch".into())
            .or_insert(MetaValue::Str(model_type.into()));
    }

    // vocab_size
    if let Some(vocab) = json.get("vocab_size").and_then(|v| v.as_i64()) {
        graph
            .metadata
            .entry("vocab_size".into())
            .or_insert(MetaValue::Int(vocab));
    }

    info!(
        config = %config_path.display(),
        "seeded metadata from config.json"
    );
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
        warn!(
            count = empty_count,
            "tensors still have empty shapes after repair"
        );
        for (&tid, info) in &ai_graph.tensor_info {
            if info.shape.is_empty() {
                let name = ai_graph
                    .tensor_names
                    .get(&tid)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                // Find which node produces this tensor.
                let producer = ai_graph.nodes.iter().find(|n| n.outputs.contains(&tid));
                let op_name = producer
                    .map(|n| format!("{:?}", n.op))
                    .unwrap_or_else(|| "input/param".into());
                warn!(tid, name, op = %op_name, "empty shape tensor");
            }
        }
    }

    // Dynamic-dim root causes.
    let mut dynamic_roots = 0u32;
    for node in &ai_graph.nodes {
        for &out_tid in &node.outputs {
            let has_dynamic = ai_graph
                .tensor_info
                .get(&out_tid)
                .map(|i| {
                    i.shape
                        .iter()
                        .any(|d| matches!(d, hologram_ai_common::Dim::Dynamic))
                })
                .unwrap_or(false);
            if !has_dynamic {
                continue;
            }
            let all_inputs_concrete = node.inputs.iter().all(|&tid| {
                ai_graph
                    .tensor_info
                    .get(&tid)
                    .map(|i| {
                        !i.shape.is_empty() && i.shape.iter().all(|d| d.as_concrete().is_some())
                    })
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
        if matches!(
            node.op,
            hologram_ai_common::AiOp::MatMul | hologram_ai_common::AiOp::BatchMatMul
        ) && matmul_count < 5
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
        opt::{
            const_eval::ConstantEvaluation, constant_fold::ConstantFolding,
            data_prop::DataPropagation, dead_node::DeadNodeElimination,
        },
        AggressiveShapePropagation, ConstantDeduplication, Dim,
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
        fn name(&self) -> &str {
            "ForceConcretize"
        }
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
        Box::new(DataPropagation),
        Box::new(ForceConcretize),
        Box::new(ConstantEvaluation),
        Box::new(ConstantFolding),
        // Extra shape pass after ConstEval: newly-folded constants may
        // enable shape inference that was blocked by unresolved shape
        // subgraphs (common in cross-attention Reshape chains).
        Box::new(AggressiveShapePropagation),
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
                dynamic_count, "post-concretization repair converged early"
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

/// Zero seq-dependent dimensions in shape tensor constants for lowering.
///
/// After `post_concretization_repair` all dims are concrete, which means
/// Reshape/Expand ops bake the compiled seq length into their target shapes.
/// This pass zeroes seq-dependent values in `known_i64_values` so the
/// lowering emits 0-sentinels that the runtime resolves from buffer sizes.
///
/// Two categories:
/// - **Reshape/Flatten outputs**: zero `known_i64_values[axis]` so
///   `infer_reshape_shape` emits 0-sentinels in the target shape.
/// - **Expand shape inputs**: zero `known_i64_values[axis]` on input\[1\]
///   so `resolve_op` emits 0-sentinels in `target_shape`.
///
/// MatMul/Softmax/RmsNorm/LayerNorm are NOT modified here — their baked
/// sizes are overridden at runtime via ShapeContextGraph + `input_metas`.
/// Setting their tensor shapes to Dynamic would remove them from the shape
/// context graph seeds (builder.rs filters `shape.contains(&0)`), breaking
/// runtime shape resolution.
fn zero_seq_dims_for_lowering(
    graph: &mut AiGraph,
    seq_dim_positions: &std::collections::HashSet<(hologram_ai_common::TensorId, usize)>,
) {
    use hologram_ai_common::AiOp;

    if seq_dim_positions.is_empty() {
        return;
    }

    // ── Collect pass: gather mutations without borrowing tensor_info mutably ──
    let mut reshape_zeros: Vec<(hologram_ai_common::TensorId, usize)> = Vec::new();
    let mut expand_zeros: Vec<(hologram_ai_common::TensorId, usize)> = Vec::new();

    for node in &graph.nodes {
        match &node.op {
            AiOp::Reshape { .. } | AiOp::Flatten { .. } => {
                if let Some(&out_tid) = node.outputs.first() {
                    for &(tid, axis) in seq_dim_positions {
                        if tid == out_tid {
                            reshape_zeros.push((tid, axis));
                        }
                    }
                }
            }
            AiOp::Expand => {
                let out_tid = node.outputs.first().copied();
                let shape_tid = node.inputs.get(1).copied();
                if let (Some(out), Some(shape_in)) = (out_tid, shape_tid) {
                    for &(tid, axis) in seq_dim_positions {
                        if tid == out {
                            expand_zeros.push((shape_in, axis));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // ── Apply pass: mutate tensor_info ──────────────────────────────────────
    let mut zeroed_i64 = 0usize;

    for (tid, axis) in reshape_zeros {
        if let Some(info) = graph.tensor_info.get_mut(&tid) {
            if let Some(ref mut vals) = info.known_i64_values {
                if axis < vals.len() {
                    vals[axis] = Some(0);
                    zeroed_i64 += 1;
                }
            }
        }
    }

    for (tid, axis) in expand_zeros {
        if let Some(info) = graph.tensor_info.get_mut(&tid) {
            if let Some(ref mut vals) = info.known_i64_values {
                if axis < vals.len() {
                    vals[axis] = Some(0);
                    zeroed_i64 += 1;
                }
            }
        }
    }

    // ── Zero seq dims in tensor shapes ────────────────────────────────────
    // For ALL seq-dependent tensors, set the shape dim to 0 (sentinel).
    // This ensures that downstream ops (Slice, Reshape, etc.) see the
    // seq axis as dynamic at lowering time, producing 0-sentinel parameters
    // that the runtime resolves from actual buffer sizes.
    // Zero seq dims in activation tensor shapes only.
    // Skip constants/weights — their shapes must stay concrete for correct lowering.
    let constant_tids: std::collections::HashSet<hologram_ai_common::TensorId> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, hologram_ai_common::AiOp::Constant { .. }))
        .flat_map(|n| n.outputs.iter().copied())
        .collect();

    let mut zeroed_shapes = 0usize;
    for &(tid, axis) in seq_dim_positions {
        // Skip weight/constant tensors.
        if constant_tids.contains(&tid) {
            continue;
        }
        if let Some(info) = graph.tensor_info.get_mut(&tid) {
            if axis < info.shape.len() {
                if let hologram_ai_common::Dim::Concrete(v) = &info.shape[axis] {
                    if *v > 0 {
                        info.shape[axis] = hologram_ai_common::Dim::Concrete(0);
                        zeroed_shapes += 1;
                    }
                }
            }
        }
    }

    debug!(
        zeroed_i64,
        zeroed_shapes, "zero_seq_dims_for_lowering complete"
    );
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
///
/// Returns the concretized graph and a set of `(TensorId, axis_index)` pairs
/// identifying which tensor dimensions were originally seq-dependent DimVars.
/// The lowering pass uses this to emit 0-sentinels for those dims so the
/// runtime can resolve them from actual buffer sizes.
fn concretize_all_dims(
    mut graph: AiGraph,
    seq_len_override: Option<u64>,
) -> anyhow::Result<(
    AiGraph,
    std::collections::HashSet<(hologram_ai_common::TensorId, usize)>,
)> {
    use hologram_ai_common::Dim;

    // Use override if provided, otherwise extract from model metadata.
    let context_len = seq_len_override
        .unwrap_or_else(|| meta_u32(&graph, "context_length").unwrap_or(2048) as u64);
    info!(context_len, "concretizing all dims (fully static shapes)");

    // Identify seq-like DimVarIds BEFORE concretization.
    let mut seq_var_ids = std::collections::HashSet::new();
    for (id, entry) in graph.dim_vars.iter() {
        if entry.fixed.is_some() {
            continue;
        }
        let name_lower = entry.name.to_lowercase();
        if name_lower.contains("seq")
            || name_lower.contains("length")
            || name_lower.contains("position")
        {
            seq_var_ids.insert(id);
        }
    }

    // Scan tensor shapes to find which (TensorId, axis) pairs reference
    // a seq-dependent DimVar. This must happen BEFORE substitution.
    let mut seq_dim_positions = std::collections::HashSet::new();
    for (&tid, info) in &graph.tensor_info {
        for (axis, dim) in info.shape.iter().enumerate() {
            let is_seq = dim.free_vars().iter().any(|v| seq_var_ids.contains(v));
            if is_seq {
                seq_dim_positions.insert((tid, axis));
            }
        }
    }
    debug!(
        n_seq_positions = seq_dim_positions.len(),
        "tagged seq-dependent tensor dimensions"
    );

    // Set concrete values for all dim vars based on their name.
    for (_, entry) in graph.dim_vars.iter_mut() {
        if entry.fixed.is_some() {
            continue;
        }
        let name_lower = entry.name.to_lowercase();
        if name_lower.contains("seq")
            || name_lower.contains("length")
            || name_lower.contains("position")
        {
            debug!(var = %entry.name, value = context_len, "concretizing seq-like dim");
            entry.fixed = Some(context_len);
        } else if name_lower.contains("height") || name_lower.contains("width") {
            // Spatial dims for diffusion models: default to 128 (1024×1024
            // image ÷ 8 VAE downscale). Overridden by spatial_scale if set.
            let spatial_default = 128u64;
            debug!(var = %entry.name, value = spatial_default, "concretizing spatial dim");
            entry.fixed = Some(spatial_default);
        } else if name_lower.contains("channel") {
            // Channel dim for diffusion latent space: typically 4.
            debug!(var = %entry.name, value = 4, "concretizing channel dim");
            entry.fixed = Some(4);
        } else if name_lower == "steps" {
            // Timestep steps: always 1 (single timestep per forward pass).
            debug!(var = %entry.name, value = 1, "concretizing steps dim");
            entry.fixed = Some(1);
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

    Ok((graph, seq_dim_positions))
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
/// Log basic stats about the lowered graph for debugging.
fn validate_matmul_constants(
    _graph: &hologram::hologram_graph::graph::Graph,
    weight_bytes: Option<&[u8]>,
) {
    let weight_len = weight_bytes.map(|w| w.len()).unwrap_or(0);
    tracing::debug!(weight_len, "lowered graph weight blob size");
}

/// The ordering must match builder.rs which assigns cumulative byte offsets
/// as `source_id` in `ConstantData::Deferred` using the same sorted order.
///
/// Returns the weight blob and a [`WeightIndex`] mapping each tensor to its
/// byte range and layer group within the blob.
fn collect_weight_bytes(
    ai_graph: &AiGraph,
) -> anyhow::Result<(
    Vec<u8>,
    hologram::hologram_archive::weight::index::WeightIndex,
)> {
    collect_weight_bytes_filtered(ai_graph, &std::collections::HashSet::new())
}

/// Streaming version of `collect_weight_bytes`: writes weights to a temp file
/// instead of a `Vec<u8>`. Returns a `WeightSource::File` for the archive writer.
///
/// Peak memory: one weight's I/O buffer (256 KB) + index entries (~1 KB each).
/// For SDXL UNet with 10 GB of Mmap weights, this uses ~1 MB instead of 10 GB.
fn collect_weight_bytes_streaming(
    ai_graph: &AiGraph,
) -> anyhow::Result<(
    hologram::hologram_archive::WeightSource,
    hologram::hologram_archive::weight::index::WeightIndex,
)> {
    use hologram::hologram_archive::weight::index::{
        derive_layer_group, WeightIndex, WeightIndexEntry,
    };
    use std::io::{Seek, Write};

    let mut sorted: Vec<_> = ai_graph
        .params
        .iter()
        .filter(|(_, p)| matches!(p, AiParam::Mmap { .. }))
        .collect();
    if sorted.is_empty() {
        return Ok((
            hologram::hologram_archive::WeightSource::Bytes(Vec::new()),
            WeightIndex { entries: vec![] },
        ));
    }
    sorted.sort_by_key(|(&tid, _)| tid);

    let tmp = tempfile::NamedTempFile::new()
        .context("creating temp file for streaming weight collection")?;
    let mut writer = std::io::BufWriter::with_capacity(256 * 1024, tmp.as_file().try_clone()?);
    let mut entries = Vec::with_capacity(sorted.len());
    let mut write_offset = 0u64;

    for (&tid, param) in &sorted {
        let AiParam::Mmap {
            path, offset, len, ..
        } = param
        else {
            continue;
        };
        let n = *len as usize;
        let mut f =
            std::fs::File::open(path).with_context(|| format!("opening weight file {path:?}"))?;

        // Disable OS page caching for this file — prevents the OS from
        // keeping 10 GB of weight pages in the page cache (which inflates
        // RSS even though the data isn't in our heap).
        #[cfg(target_os = "macos")]
        {
            use std::os::unix::io::AsRawFd;
            extern "C" {
                fn fcntl(fd: std::ffi::c_int, cmd: std::ffi::c_int, ...) -> std::ffi::c_int;
            }
            // F_NOCACHE = 48 on macOS (Darwin)
            const F_NOCACHE: std::ffi::c_int = 48;
            unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1 as std::ffi::c_int) };
        }

        f.seek(std::io::SeekFrom::Start(*offset))?;

        // Stream from source to temp file in 256 KB chunks.
        let mut remaining = n;
        let mut buf = vec![0u8; 256 * 1024];
        while remaining > 0 {
            let to_read = remaining.min(buf.len());
            std::io::Read::read_exact(&mut f, &mut buf[..to_read])
                .with_context(|| format!("reading {n} bytes from {path:?}"))?;
            writer.write_all(&buf[..to_read])?;
            remaining -= to_read;
        }

        let tensor_name = ai_graph
            .tensor_names
            .get(&tid)
            .cloned()
            .unwrap_or_else(|| format!("tensor_{tid}"));
        entries.push(WeightIndexEntry {
            tensor_name: tensor_name.clone(),
            group: derive_layer_group(&tensor_name),
            offset: write_offset,
            size: n as u64,
        });

        // 4-byte align.
        let aligned = (n as u64).div_ceil(4) * 4;
        let padding = aligned - n as u64;
        if padding > 0 {
            writer.write_all(&vec![0u8; padding as usize])?;
        }
        write_offset += aligned;
    }
    writer.flush()?;

    let total_len = write_offset;
    // Persist the temp file so it survives until the archive is built.
    // The caller is responsible for cleanup (or the OS cleans /tmp on reboot).
    let path = tmp.into_temp_path();
    let persisted = path
        .keep()
        .map_err(|e| anyhow::anyhow!("persisting temp weight file: {e}"))?;
    tracing::info!(
        total_mb = total_len / (1024 * 1024),
        n_weights = entries.len(),
        path = %persisted.display(),
        "streamed weights to temp file ({} MiB)",
        total_len / (1024 * 1024),
    );

    Ok((
        hologram::hologram_archive::WeightSource::File {
            path: persisted,
            len: total_len,
        },
        WeightIndex { entries },
    ))
}

/// Predict which weight TIDs will be quantized during lowering.
///
/// Mirrors the gating logic in `try_convert_f32_to_lut4` in the builder:
/// - Weight is input[1] of a MatMul node
/// - Weight is a 2D parameter with both dims ≥ 256
/// - Output dim doesn't match attention head sizes (not Q/K/V/O projection)
#[allow(dead_code)] // TODO: use once constant offset remapping is implemented
fn predict_quantized_weight_tids(
    ai_graph: &AiGraph,
) -> std::collections::HashSet<hologram_ai_common::ir::TensorId> {
    use hologram_ai_common::ir::{AiOp, Dim};
    let mut quantized = std::collections::HashSet::new();

    // Collect attention dimensions for gating.
    let attn_dims: Vec<usize> = ai_graph
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            AiOp::GroupedQueryAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => Some(vec![
                *num_heads as usize * *head_dim as usize,
                *num_kv_heads as usize * *head_dim as usize,
            ]),
            _ => None,
        })
        .flatten()
        .collect();

    for node in &ai_graph.nodes {
        if !matches!(node.op, AiOp::MatMul) {
            continue;
        }
        // Weight is input[1].
        let weight_tid = match node.inputs.get(1) {
            Some(&tid) => tid,
            None => continue,
        };
        // Must be a parameter.
        if !ai_graph.params.contains_key(&weight_tid) {
            continue;
        }
        // Must have 2D concrete shape with both dims ≥ 256.
        let info = match ai_graph.tensor_info.get(&weight_tid) {
            Some(info) => info,
            None => continue,
        };
        let dims: Vec<usize> = info
            .shape
            .iter()
            .filter_map(|d| match d {
                Dim::Concrete(n) => Some(*n as usize),
                _ => None,
            })
            .collect();
        if dims.len() != 2 || dims[0] < 256 || dims[1] < 256 {
            continue;
        }
        // Skip attention projections (output dim matches head sizes).
        if attn_dims.contains(&dims[1]) {
            continue;
        }
        quantized.insert(weight_tid);
    }

    quantized
}

/// Collect weight bytes, skipping TIDs in `skip_tids`.
///
/// Used for Q4 quantization: weights that are quantized to LUT-GEMM format
/// during lowering become dead — they're replaced by graph constants.
/// Skipping them avoids embedding redundant f32 originals in the archive.
fn collect_weight_bytes_filtered(
    ai_graph: &AiGraph,
    skip_tids: &std::collections::HashSet<hologram_ai_common::ir::TensorId>,
) -> anyhow::Result<(
    Vec<u8>,
    hologram::hologram_archive::weight::index::WeightIndex,
)> {
    use hologram::hologram_archive::weight::index::{
        derive_layer_group, WeightIndex, WeightIndexEntry,
    };

    let mut sorted: Vec<_> = ai_graph
        .params
        .iter()
        .filter(|(&tid, p)| matches!(p, AiParam::Mmap { .. }) && !skip_tids.contains(&tid))
        .collect();
    if sorted.is_empty() {
        return Ok((Vec::new(), WeightIndex { entries: vec![] }));
    }
    sorted.sort_by_key(|(&tid, _)| tid);

    let total_size: u64 = sorted
        .iter()
        .map(|(_, p)| match p {
            AiParam::Mmap { len, .. } => {
                // Account for 4-byte alignment padding per tensor.
                (*len).div_ceil(4) * 4
            }
            _ => 0,
        })
        .sum();
    // Single pre-allocated buffer — read directly into the blob without
    // intermediate per-weight Vec allocations.
    let mut blob = vec![0u8; total_size as usize];
    let mut write_offset = 0usize;
    let mut entries = Vec::with_capacity(sorted.len());

    for (&tid, param) in &sorted {
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

            let tensor_name = ai_graph
                .tensor_names
                .get(&tid)
                .cloned()
                .unwrap_or_else(|| format!("tensor_{tid}"));
            entries.push(WeightIndexEntry {
                tensor_name: tensor_name.clone(),
                group: derive_layer_group(&tensor_name),
                offset: write_offset as u64,
                size: n as u64,
            });

            write_offset += n;
            // Pad to 4-byte alignment so f32 cast_slice never fails.
            let aligned = (write_offset + 3) & !3;
            write_offset = aligned;
        }
    }

    Ok((blob, WeightIndex { entries }))
}

/// Scale spatial dimensions (H, W) of 4D input tensors by dividing by `scale`.
///
/// For vision models (Conv2d, Resize), this reduces the compiled resolution
/// and proportionally reduces activation memory. Shape propagation derives
/// all downstream dims from the scaled inputs.
///
/// Only affects tensors referenced by graph inputs with 4D shapes (NCHW).
fn apply_spatial_scale(graph: &mut AiGraph, scale: u32) {
    use hologram_ai_common::Dim;

    if scale <= 1 {
        return;
    }
    let s = scale as u64;

    // Scale ONLY input tensor shapes — let shape propagation derive all
    // intermediate shapes from the (now-scaled) inputs. This avoids the
    // bug where pre-scaling intermediates conflicts with shape propagation's
    // own inference (e.g., ForceConcretize sets Dynamic→1, then Resize
    // computes 1×scale=2 instead of the correct scaled value).
    for &input_tid in &graph.inputs {
        if let Some(info) = graph.tensor_info.get_mut(&input_tid) {
            if info.shape.len() == 4 {
                for dim in info.shape[2..].iter_mut() {
                    if let Some(v) = dim.as_concrete() {
                        *dim = Dim::Concrete((v / s).max(1));
                    }
                }
                tracing::info!(tid = input_tid, shape = ?info.shape, "spatial scale applied");
            }
        }
    }

    // Scale ALL non-param 4D tensor shapes proportionally.
    // Weight tensors (in graph.params) are never scaled — their kernel sizes
    // are architecture-fixed. All other 4D tensors are activations whose
    // spatial dims should scale with the input.
    let param_tids: std::collections::HashSet<_> = graph.params.keys().copied().collect();
    for (&tid, info) in graph.tensor_info.iter_mut() {
        if param_tids.contains(&tid) || graph.inputs.contains(&tid) {
            continue;
        }
        if info.shape.len() >= 4 {
            for dim in info.shape[2..].iter_mut() {
                if let Some(v) = dim.as_concrete() {
                    if v > 1 {
                        *dim = Dim::Concrete((v / s).max(1));
                    }
                }
            }
        }
    }
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
    /// The prefill model plan (first component, or only component for non-LLM).
    plan: hologram::LoadedPlan,
    shape_ctx: Option<ShapeContextGraph>,
    /// Pre-compiled execution tape for prefill.
    tape: hologram::hologram_exec::tape::EnumTape,
    /// Optional decode model (second component in LLM pipeline archives).
    /// When present, `execute_with_kv` switches to this after step 0.
    decode_plan: Option<hologram::LoadedPlan>,
    decode_tape: Option<hologram::hologram_exec::tape::EnumTape>,
    /// Pre-computed shape overrides for decode (seq=1). Extracted from the
    /// compiled node_shapes at load time — no walk_shape_context needed per
    /// step. Provides input_metas that Q4 LUT-GEMM kernels require.
    decode_shape_map: std::collections::HashMap<u32, Vec<usize>>,
    /// Optional verification model (third component in LLM pipeline archives).
    /// Compiled at seq=N for batch speculative decoding verification.
    verify_plan: Option<hologram::LoadedPlan>,
    verify_tape: Option<hologram::hologram_exec::tape::EnumTape>,
    /// Persistent weight cache for LUT-GEMM. Deserialized quantized weights
    /// are cached here across execution calls, avoiding per-step rkyv overhead.
    weight_cache: parking_lot::RwLock<hologram::WeightCache>,
    /// Optional patch pruning configuration (ViT models with PixelPrune).
    /// When present, `execute()` preprocesses the pixel input to produce
    /// `kept_indices` before feeding the compiled graph.
    patch_prune: Option<hologram_ai_common::PatchPruneContext>,
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
            Some(p) => hologram::config::HologramConfig::load_file(p).unwrap_or_default(),
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

        // SAFETY: storage outlives all plans created here.
        let probe = unsafe { hologram::load_from_bytes_zero_copy(bytes) }
            .map_err(|e| anyhow::anyhow!("loading archive: {e}"))?;

        // Check if this is a pipeline archive (has SECTION_PIPELINE header).
        let is_pipeline = probe
            .sections()
            .entries
            .iter()
            .any(|e| e.kind == hologram::hologram_archive::section::SECTION_PIPELINE);

        if is_pipeline {
            // Pipeline archive: load the first (or only) model component.
            let weights_start = probe.header().weights_offset as usize;

            let pipeline_entry = probe
                .sections()
                .find(hologram::hologram_archive::section::SECTION_PIPELINE)
                .ok_or_else(|| anyhow::anyhow!("pipeline section missing"))?;
            let ps = pipeline_entry.offset as usize;
            let pe = ps + pipeline_entry.size as usize;
            let ph: hologram::hologram_archive::writer::pipeline_writer::PipelineHeader =
                rkyv::from_bytes::<
                    hologram::hologram_archive::writer::pipeline_writer::PipelineHeader,
                    rkyv::rancor::Error,
                >(&bytes[ps..pe])
                .map_err(|e| anyhow::anyhow!("parsing pipeline header: {e}"))?;

            // Load the first model component.
            let first = ph
                .models
                .first()
                .ok_or_else(|| anyhow::anyhow!("pipeline has no models"))?;
            let model_start = weights_start + first.offset as usize;
            let model_end = model_start + first.size as usize;
            if model_end > bytes.len() {
                anyhow::bail!("sub-archive out of bounds");
            }
            let model_slice = &bytes[model_start..model_end];

            let mut plan = unsafe { hologram::load_from_bytes_zero_copy(model_slice) }
                .map_err(|e| anyhow::anyhow!("loading model plan: {e}"))?;

            // Resolve shared weights via dedup index if available.
            let dedup_index = probe
                .sections()
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

            if let Some(ref idx) = dedup_index {
                if plan.weights().is_empty() {
                    let wrapper_weights = probe.weights();
                    if let Some(entry) = idx.find_component(&first.name) {
                        let w_start = entry.offset as usize;
                        let w_end = w_start + entry.size as usize;
                        if w_end <= wrapper_weights.len() {
                            unsafe {
                                plan.set_weights_borrowed(&wrapper_weights[w_start..w_end]);
                            }
                        }
                    }
                }
            }

            let shape_ctx = read_shape_context_from_plan(&plan, model_slice)?;
            let tape = hologram::build_tape_from_plan(&plan)
                .map_err(|e| anyhow::anyhow!("building prefill tape: {e}"))?;

            // Load decode model (second component) if present.
            let (decode_plan, decode_tape) = if ph.models.len() >= 2 {
                let second = &ph.models[1];
                let d_start = weights_start + second.offset as usize;
                let d_end = d_start + second.size as usize;
                if d_end > bytes.len() {
                    anyhow::bail!("decode sub-archive out of bounds");
                }
                let d_slice = &bytes[d_start..d_end];
                let mut d_plan = unsafe { hologram::load_from_bytes_zero_copy(d_slice) }
                    .map_err(|e| anyhow::anyhow!("loading decode plan: {e}"))?;

                // Share weights from prefill → decode. Both components were compiled
                // from the same AiGraph with identical constant ordering, so weight
                // offsets are the same. Just borrow the prefill's weight buffer.
                if d_plan.weights().is_empty() && !plan.weights().is_empty() {
                    unsafe {
                        d_plan.set_weights_borrowed(plan.weights());
                    }
                    info!(
                        decode_weights = plan.weights().len(),
                        "decode shares prefill weights"
                    );
                }

                let d_tape = hologram::build_tape_from_plan(&d_plan)
                    .map_err(|e| anyhow::anyhow!("building decode tape: {e}"))?;
                info!("loaded decode model (seq=1) for LLM pipeline");
                (Some(d_plan), Some(d_tape))
            } else {
                (None, None)
            };

            // Load verify model (third component) if present.
            let (verify_plan, verify_tape) = if ph.models.len() >= 3 {
                let third = &ph.models[2];
                let v_start = weights_start + third.offset as usize;
                let v_end = v_start + third.size as usize;
                if v_end > bytes.len() {
                    anyhow::bail!("verify sub-archive out of bounds");
                }
                let v_slice = &bytes[v_start..v_end];
                let mut v_plan = unsafe { hologram::load_from_bytes_zero_copy(v_slice) }
                    .map_err(|e| anyhow::anyhow!("loading verify plan: {e}"))?;
                if v_plan.weights().is_empty() && !plan.weights().is_empty() {
                    unsafe {
                        v_plan.set_weights_borrowed(plan.weights());
                    }
                }
                let v_tape = hologram::build_tape_from_plan(&v_plan)
                    .map_err(|e| anyhow::anyhow!("building verify tape: {e}"))?;
                info!("loaded verify model (seq=8) for speculative decoding");
                (Some(v_plan), Some(v_tape))
            } else {
                (None, None)
            };

            // Pre-compute decode shape map from the compiled graph's node_shapes.
            // The decode graph has fully concrete shapes (seq=1), so these
            // are exact — no runtime resolution needed.
            let decode_shape_map = decode_plan
                .as_ref()
                .map(|dp| {
                    dp.graph()
                        .node_shapes
                        .iter()
                        .map(|(nid, shape)| (nid.index(), shape.clone()))
                        .collect()
                })
                .unwrap_or_default();

            let patch_prune = read_patch_prune_from_plan(&plan, model_slice);

            let runner = Self {
                _storage: storage,
                plan,
                shape_ctx,
                tape,
                decode_plan,
                decode_tape,
                decode_shape_map,
                verify_plan,
                verify_tape,
                weight_cache: parking_lot::RwLock::new(hologram::WeightCache::new()),
                patch_prune,
            };
            // Pre-warm dequant cache: populate f32 expansion for all Q4 constants
            // so decode steps never pay the dequant overhead.
            #[cfg(target_os = "macos")]
            {
                let sg = runner.plan.graph();
                let mut wc = runner.weight_cache.write();
                wc.prewarm_q4(&runner.tape, &sg.constants, runner.plan.weights());
                wc.prewarm_q8(&runner.tape, &sg.constants, runner.plan.weights());
                if let Some(ref dt) = runner.decode_tape {
                    // Decode plan shares weights with prefill plan.
                    wc.prewarm_q4(dt, &sg.constants, runner.plan.weights());
                    wc.prewarm_q8(dt, &sg.constants, runner.plan.weights());
                }
            }
            Ok(runner)
        } else {
            // Legacy single-graph archive (backward compat).
            let shape_ctx = read_shape_context_from_plan(&probe, bytes)?;
            let tape = hologram::build_tape_from_plan(&probe)
                .map_err(|e| anyhow::anyhow!("building tape: {e}"))?;

            let patch_prune = read_patch_prune_from_plan(&probe, bytes);

            let runner = Self {
                _storage: storage,
                plan: probe,
                shape_ctx,
                tape,
                decode_plan: None,
                decode_tape: None,
                decode_shape_map: std::collections::HashMap::new(),
                verify_plan: None,
                verify_tape: None,
                weight_cache: parking_lot::RwLock::new(hologram::WeightCache::new()),
                patch_prune,
            };
            #[cfg(target_os = "macos")]
            {
                let sg = runner.plan.graph();
                let mut wc = runner.weight_cache.write();
                wc.prewarm_q4(&runner.tape, &sg.constants, runner.plan.weights());
                wc.prewarm_q8(&runner.tape, &sg.constants, runner.plan.weights());
            }
            Ok(runner)
        }
    }

    /// Execute the compiled graph with the given inputs.
    ///
    /// When a `ShapeContextGraph` is available, projects runtime input shapes
    /// through the graph to produce correct per-node shapes. This enables
    /// variable-length execution (runtime seq_len != compiled seq_len).
    ///
    /// When patch pruning is configured (ViT models compiled with
    /// `PatchPruneInjection`), automatically preprocesses the pixel input
    /// to produce `kept_indices` before feeding the compiled graph.
    pub fn execute(
        &self,
        inputs: &hologram::GraphInputs,
    ) -> anyhow::Result<hologram::GraphOutputs> {
        // If patch pruning is configured, preprocess the pixel input.
        let inputs = if let Some(ref prune) = self.patch_prune {
            self.preprocess_patch_prune(inputs, prune)?
        } else {
            std::borrow::Cow::Borrowed(inputs)
        };

        if let Some(ref ctx) = self.shape_ctx {
            let shape_map = self.resolve_shapes(ctx, &self.plan, &inputs);
            hologram::execute_tape_with_shapes(&self.tape, &self.plan, &inputs, &shape_map)
                .map_err(|e| anyhow::anyhow!("{e}"))
        } else {
            hologram::execute_tape(&self.tape, &self.plan, &inputs)
                .map_err(|e| anyhow::anyhow!("{e}"))
        }
    }

    /// Run the PatchPrune kernel on the pixel input and inject `kept_indices`.
    fn preprocess_patch_prune(
        &self,
        inputs: &hologram::GraphInputs,
        prune: &hologram_ai_common::PatchPruneContext,
    ) -> anyhow::Result<std::borrow::Cow<'_, hologram::GraphInputs>> {
        let pixel_bytes = inputs.get(prune.pixel_input).ok_or_else(|| {
            anyhow::anyhow!(
                "PatchPrune: pixel input at index {} not found",
                prune.pixel_input
            )
        })?;

        // Interpret pixel bytes as f32.
        let pixels: &[f32] = bytemuck::cast_slice(pixel_bytes);

        // Infer image dimensions from pixel count and channel count.
        let channels = prune.channels as usize;
        let total_pixels = pixels.len();
        let spatial_pixels = total_pixels / channels;
        // Assume square image if no shape info available.
        let img_side = (spatial_pixels as f64).sqrt() as usize;
        let (img_h, img_w) = if let Some(shape) = inputs.shape(prune.pixel_input) {
            // Shape is [N, C, H, W] or [C, H, W].
            if shape.len() == 4 {
                (shape[2], shape[3])
            } else if shape.len() == 3 {
                (shape[1], shape[2])
            } else {
                (img_side, img_side)
            }
        } else {
            (img_side, img_side)
        };

        let params = hologram::hologram_exec::PatchPruneParams {
            channels,
            img_h,
            img_w,
            patch_h: prune.patch_h as usize,
            patch_w: prune.patch_w as usize,
            tau: 0.0, // lossless by default
            max_kept: prune.max_kept as usize,
        };

        let result = hologram::hologram_exec::patch_prune(pixels, &params);

        // Build new inputs with kept_indices injected.
        let mut new_inputs = inputs.clone();
        let indices_bytes =
            hologram::hologram_exec::patch_prune::indices_to_bytes(&result.kept_indices);
        new_inputs.set_with_shape(
            prune.kept_indices_input,
            indices_bytes,
            vec![prune.max_kept as usize],
        );

        tracing::debug!(
            n_kept = result.n_kept,
            max_kept = prune.max_kept,
            "PatchPrune preprocessor: selected {}/{} patches",
            result.n_kept,
            prune.total_patches,
        );

        Ok(std::borrow::Cow::Owned(new_inputs))
    }

    /// Access the underlying loaded plan (for layer headers, weights, etc.).
    #[must_use]
    pub fn plan(&self) -> &hologram::LoadedPlan {
        &self.plan
    }

    /// Archive bytes (the full pipeline archive).
    #[must_use]
    pub fn archive_bytes(&self) -> &[u8] {
        self._storage.as_ref()
    }

    /// Raw top-level archive bytes (same as archive_bytes for unified format).
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

    /// Project runtime input shapes through the `ShapeContextGraph` to produce
    /// per-node shape overrides for the executor.
    fn resolve_shapes(
        &self,
        ctx: &ShapeContextGraph,
        plan: &hologram::LoadedPlan,
        inputs: &hologram::GraphInputs,
    ) -> std::collections::HashMap<u32, Vec<usize>> {
        let mut runtime_inputs = std::collections::HashMap::new();
        let sg = plan.graph();

        // Map graph input names to their node indices and inject runtime shapes.
        for (slot, name) in sg.input_names.iter().enumerate() {
            if let Some(shape) = inputs.shape(slot as u32) {
                for node in &sg.nodes {
                    if matches!(node.op, hologram::hologram_graph::graph::GraphOp::Input)
                        && node.id.index() == slot as u32
                    {
                        runtime_inputs.insert(node.id.index(), shape.to_vec());
                        break;
                    }
                }
                runtime_inputs
                    .entry(slot as u32)
                    .or_insert_with(|| shape.to_vec());
            }
            let _ = name;
        }

        let mut shape_map = std::collections::HashMap::new();
        hologram_ai_common::walk_shape_context(
            ctx,
            &runtime_inputs,
            &std::collections::HashMap::new(),
            &mut shape_map,
        );
        shape_map
    }

    /// Execute with a mutable KV cache state for autoregressive generation.
    ///
    /// For LLM pipeline archives: uses the prefill tape for step 0 (write_pos == 0)
    /// and the decode tape for subsequent steps. The decode graph is compiled at
    /// seq=1, making each decode step ~Nx faster than running the full prefill graph.
    ///
    /// For single-graph archives: uses the same tape for all steps.
    ///
    /// Single execution path for all steps (prefill + decode).
    ///
    /// Always uses `execute_tape_with_kv_shapes_cached` with shape overrides.
    /// - **Prefill**: resolves shapes at runtime via `walk_shape_context` for
    ///   variable-length prompt support.
    /// - **Decode**: uses pre-computed `decode_shape_map` (constant at seq=1,
    ///   no walk needed). Provides input_metas that LUT-GEMM kernels require.
    pub fn execute_with_kv(
        &self,
        inputs: &hologram::GraphInputs,
        kv_state: &mut hologram::KvCacheState,
    ) -> anyhow::Result<hologram::GraphOutputs> {
        let is_decode = kv_state.write_pos() > 0;
        let (tape, plan) = if is_decode {
            if let (Some(ref dt), Some(ref dp)) = (&self.decode_tape, &self.decode_plan) {
                (dt, dp)
            } else {
                (&self.tape, &self.plan)
            }
        } else {
            (&self.tape, &self.plan)
        };

        // Prefill: walk shape context for variable-length support.
        // Decode: use pre-computed shape map (no walk overhead).
        let shape_map = if !is_decode {
            if let Some(ref ctx) = self.shape_ctx {
                self.resolve_shapes(ctx, plan, inputs)
            } else {
                std::collections::HashMap::new()
            }
        } else {
            self.decode_shape_map.clone()
        };

        hologram::execute_tape_with_kv_shapes_cached(
            tape,
            plan,
            inputs,
            kv_state,
            &shape_map,
            &self.weight_cache,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Whether this runner has a separate decode model for fast autoregressive generation.
    #[must_use]
    pub fn has_decode_model(&self) -> bool {
        self.decode_tape.is_some()
    }

    /// Whether this runner has a verification model for batch speculative decoding.
    #[must_use]
    pub fn has_verify_model(&self) -> bool {
        self.verify_tape.is_some()
    }

    /// Whether this runner has patch pruning configured (ViT models).
    #[must_use]
    pub fn has_patch_prune(&self) -> bool {
        self.patch_prune.is_some()
    }

    /// Access the patch pruning config (for diagnostics/testing).
    #[must_use]
    pub fn patch_prune_config(&self) -> Option<&hologram_ai_common::PatchPruneContext> {
        self.patch_prune.as_ref()
    }

    /// Execute a batch verification forward pass using the verify tape (seq=N).
    ///
    /// Used by speculative decoding: draft N tokens with decode tape (seq=1),
    /// then verify all N in one forward pass through the verify tape (seq=N).
    /// BLAS amortizes the weight read across N output tokens → N× throughput.
    pub fn execute_verify(
        &self,
        inputs: &hologram::GraphInputs,
        kv_state: &mut hologram::KvCacheState,
    ) -> anyhow::Result<hologram::GraphOutputs> {
        let (tape, plan) =
            if let (Some(ref vt), Some(ref vp)) = (&self.verify_tape, &self.verify_plan) {
                (vt, vp)
            } else {
                // No verify tape — fall back to prefill tape.
                (&self.tape, &self.plan)
            };
        hologram::execute_tape_with_kv_cached(tape, plan, inputs, kv_state, &self.weight_cache)
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
/// Read a [`PatchPruneContext`] from an already-loaded plan + raw archive bytes.
///
/// Returns `None` if the archive has no patch pruning section (non-ViT models).
fn read_patch_prune_from_plan(
    plan: &hologram::LoadedPlan,
    archive_bytes: &[u8],
) -> Option<hologram_ai_common::PatchPruneContext> {
    use hologram_ai_common::exec_context::{ExecContext, SECTION_PATCH_PRUNE};
    let entry = plan.sections().find(SECTION_PATCH_PRUNE)?;
    let start = entry.offset as usize;
    let end = start + entry.size as usize;
    if end > archive_bytes.len() {
        tracing::warn!("PatchPruneContext section out of bounds, ignoring");
        return None;
    }
    match hologram_ai_common::PatchPruneContext::from_context_bytes(&archive_bytes[start..end]) {
        Ok(ctx) => {
            tracing::info!(
                max_kept = ctx.max_kept,
                total_patches = ctx.total_patches,
                "loaded PatchPruneContext from archive"
            );
            Some(ctx)
        }
        Err(e) => {
            tracing::warn!("failed to deserialize PatchPruneContext: {e}");
            None
        }
    }
}

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
pub fn read_shape_context_from_archive(
    archive_bytes: &[u8],
) -> anyhow::Result<Option<ShapeContextGraph>> {
    // SAFETY: plan is dropped at the end of this function; archive_bytes outlives it.
    let plan = unsafe { hologram::load_from_bytes_zero_copy(archive_bytes) }?;
    read_shape_context_from_plan(&plan, archive_bytes)
}

/// Execute a compiled archive with variable-length input support.
///
/// Builds a one-shot tape and executes via the EnumTape path.
/// Dynamic sizes are resolved at execution time via `resolve_size()`
/// and `infer_matmul_k()` in the tape executor.
///
/// If the archive was compiled from a model with attention layers
/// (`n_layers > 0`), a fresh `KvCacheState` is initialised automatically
/// so that KvWrite/KvRead ops succeed during the forward pass.
///
/// For repeated execution, prefer [`HoloRunner`] which builds the tape once.
pub fn run_with_shape_context(
    archive: &HoloArchive,
    inputs: &hologram::GraphInputs,
) -> anyhow::Result<hologram::GraphOutputs> {
    let runner = HoloRunner::from_bytes(archive.bytes.clone())?;
    let m = &archive.metadata;

    if m.n_layers > 0 {
        let mut kv = hologram::KvCacheState::new(
            m.n_layers,
            m.n_kv_heads,
            m.head_dim,
            m.context_len as usize,
        );
        runner.execute_with_kv(inputs, &mut kv)
    } else {
        runner.execute(inputs)
    }
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
