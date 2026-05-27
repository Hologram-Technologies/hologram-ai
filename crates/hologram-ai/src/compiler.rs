//! Model compilation pipeline.
//!
//! Compiles ONNX models into `.holo` archives via the hologram O(1) LUT
//! runtime. This crate is a **compiler** — it does not own inference
//! sessions or runtime state (see ADR-0016).

use anyhow::Context;
use hologram_ai_common::{lower, AiGraph, LowerPhase, LoweringOptions, OptPipeline, Pass};
use std::path::PathBuf;
use tracing::{debug, info};

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
    /// The source model's uor-addr κ-label (`<axis>:<hex>`) — its canonical
    /// content identity for dedup / warm-start (architecture §8, class MA).
    /// `None` when compiled from a pre-built `AiGraph` (no source bytes).
    pub kappa_label: Option<String>,
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

/// Archive extension key for the canonicalized tokenizer JSON (uor-addr JCS).
pub const TOKENIZER_EXT: &str = "tokenizer.json";
/// Archive extension key for the tokenizer's uor-addr κ-label (integrity check).
pub const TOKENIZER_KAPPA_EXT: &str = "tokenizer.kappa";

/// Open **extension sections** (key → bytes) to embed in the `.holo` during the
/// single build pass — the runtime carries them opaquely and exposes them via
/// `session.extension(key)`. hologram-ai uses this to bake the model's tokenizer
/// (canonicalized via uor-addr) into the archive so it is self-describing; the
/// platform also uses it for generation config, labels, provenance, etc.
#[derive(Default)]
pub struct ArchiveSections {
    extensions: Vec<(String, Vec<u8>)>,
}

impl ArchiveSections {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach an extension section under `key`. A later add with the same key
    /// shadows the earlier one (last-wins).
    pub fn add_extension(&mut self, key: impl Into<String>, bytes: Vec<u8>) {
        self.extensions.push((key.into(), bytes));
    }

    /// Whether the given extension key is already present.
    pub fn contains(&self, key: &str) -> bool {
        self.extensions.iter().any(|(k, _)| k == key)
    }

    /// Whether any extension has been added.
    pub fn is_empty(&self) -> bool {
        self.extensions.is_empty()
    }

    /// Consume and return the collected extensions.
    pub fn into_inner(self) -> Vec<(String, Vec<u8>)> {
        self.extensions
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
    ///
    /// Handles both archive modes:
    /// - In-memory (`bytes` populated): writes bytes to `path`.
    /// - Streaming (`path` set, `bytes` empty): copies the streamed file to `path`.
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating output directory {parent:?}"))?;
            }
        }
        if let Some(src) = &self.path {
            std::fs::copy(src, path)
                .with_context(|| format!("copying streamed archive {src:?} to {path:?}"))?;
            return Ok(());
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
    /// Mint the model's uor-addr κ-label (MA dedup/warm-start identity) during
    /// compile. **Off by default**: it canonicalizes the *entire* model via
    /// uor-addr, which is prohibitively slow for large models (minutes for a
    /// 0.5 GB ONNX) and is not needed for a working `.holo`. Enable only when the
    /// κ-label is actually required (addressing/dedup tooling).
    pub address_model: bool,
}

impl Default for ModelCompiler {
    fn default() -> Self {
        Self {
            mmap: true,
            seq_len_override: None,
            spatial_scale: None,
            quant_strategy: hologram_ai_common::lower::QuantStrategy::Auto,
            patch_budget_ratio: Some(0.75),
            address_model: false,
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

    /// Compile a model source into a `.holo` archive — the UOR-native pipeline:
    /// import → optimize → concretize → lower → compile. Weights flow through the
    /// canonical graph's constant store; hologram's compiler owns weight layout,
    /// scheduling, fusion, and content addressing (architecture §5, §7).
    pub fn compile_with_sections(
        &self,
        source: ModelSource,
        mut sections: ArchiveSections,
    ) -> anyhow::Result<HoloArchive> {
        use hologram_compiler::{compile, BackendKind};

        let model_dir = match &source {
            ModelSource::OnnxPath(p) => p.parent().map(|d| d.to_path_buf()),
            _ => None,
        };

        // Address the source model to its κ-label before import consumes it
        // (architecture §8 — the model's canonical dedup / warm-start identity).
        // Opt-in: full-model uor-addr canonicalization is too slow for large
        // models to sit on the compile critical path.
        let kappa_label = if self.address_model {
            source_kappa_label(&source)
        } else {
            None
        };

        // Step 1 — import.
        let mut ai_graph = self.import(source)?;
        if let Some(dir) = &model_dir {
            seed_metadata_from_config_json(&mut ai_graph, dir);
        }

        // Step 2 — optimize (pipeline chosen by model topology).
        let pipeline = select_pipeline(&ai_graph, self.patch_budget_ratio);
        let mut ai_graph = pipeline.run(ai_graph).context("optimization pass failed")?;
        infer_llm_metadata_from_graph(&mut ai_graph);

        // Step 3 — concretize every dim so the canonical graph carries concrete
        // shapes; hologram's compiler derives op params from them (architecture §5.1).
        let (ai_graph, _zeroed) =
            concretize_all_dims(ai_graph, self.seq_len_override, self.spatial_scale)
                .context("dimension concretization failed")?;
        let ai_graph =
            post_concretization_repair(ai_graph).context("post-concretization repair failed")?;

        let mut metadata = extract_metadata(&ai_graph);
        metadata.kappa_label = kappa_label;
        let import_warnings = ai_graph.warnings.len();
        let node_count = ai_graph.nodes.len();

        // Step 4 — lower to a canonical hologram graph.
        let mut lowered = lower(&ai_graph, &self.lowering_options(), &LowerPhase::Forward)
            .context("lowering to canonical graph failed")?;

        // Step 4b — bake the model's tokenizer into the archive as an open
        // extension, canonicalized via uor-addr (JCS-RFC8785 + NFC) so it is a
        // deterministic, content-addressed artifact — the `.holo` is then
        // self-describing and `run --prompt` needs no external tokenizer file.
        if let Some(dir) = &model_dir {
            let tok = dir.join("tokenizer.json");
            if tok.exists() && !sections.contains(TOKENIZER_EXT) {
                let raw = std::fs::read(&tok)
                    .with_context(|| format!("reading tokenizer {tok:?}"))?;
                let canonical = uor_addr::json::canonicalize(&raw)
                    .map_err(|e| anyhow::anyhow!("tokenizer.json is not valid JSON: {e:?}"))?;
                // Content address (κ-label) of the canonical form, for integrity
                // verification + dedup at load (class MA), stored alongside it.
                let kappa = uor_addr::json::address(&canonical)
                    .map_err(|e| anyhow::anyhow!("addressing tokenizer.json: {e:?}"))?
                    .address
                    .as_str()
                    .to_string();
                sections.add_extension(TOKENIZER_EXT, canonical);
                sections.add_extension(TOKENIZER_KAPPA_EXT, kappa.into_bytes());
            }
        }
        for (key, bytes) in sections.into_inner() {
            lowered.graph.add_extension(key, bytes);
        }

        // Step 5 — compile to a `.holo` archive.
        let out = compile(lowered.graph, BackendKind::Cpu, hologram_witt_level())
            .map_err(|e| anyhow::anyhow!("hologram compile failed: {e:?}"))?;

        Ok(HoloArchive {
            bytes: out.archive,
            path: None,
            metadata,
            stats: CompileStats {
                import_warnings,
                validation_errors: 0,
                total_weight_bytes: 0,
                node_count,
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
        }
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
        kappa_label: None,
    }
}

/// Mint the source model's uor-addr κ-label (class MA, architecture §8).
/// Returns `None` for an in-memory `AiGraph` (no source bytes) or when the
/// bytes are unreadable / unaddressable — the label is identity metadata, so a
/// failure to mint it never fails the compile.
fn source_kappa_label(source: &ModelSource) -> Option<String> {
    let label = match source {
        ModelSource::OnnxBytes(bytes) => crate::address::model_kappa_label(bytes),
        ModelSource::OnnxPath(path) => {
            let bytes = std::fs::read(path).ok()?;
            crate::address::model_kappa_label(&bytes)
        }
        ModelSource::AiGraph(_) => return None,
    };
    match label {
        Ok(l) => Some(l),
        Err(e) => {
            tracing::debug!("model κ-label unavailable: {e:#}");
            None
        }
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
    spatial_scale: Option<u32>,
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
            // image ÷ 8 VAE downscale). When spatial_scale is set, divide by it
            // so a 4× scale gives 32, a 8× scale gives 16, etc.
            let spatial_default = 128u64 / spatial_scale.unwrap_or(1) as u64;
            let spatial_default = spatial_default.max(1);
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
        for (axis, dim) in info.shape.iter_mut().enumerate() {
            for (var_id, replacement) in &subs {
                *dim = dim.substitute(*var_id, replacement);
            }
            if let Some(v) = dim.evaluate() {
                *dim = Dim::Concrete(v);
            }
            // A dim that is still unresolved lost its symbolic var to `Dynamic`
            // somewhere in shape inference. Concretize it by position: axis 0 is
            // the batch dim (LLM prefill batch = 1, matching the `Var(batch-like)
            // → 1` policy above); any inner unresolved dim is sequence-like →
            // `context_length`. Forcing *every* remaining dim to `context_length`
            // (the old behavior) made a leading batch dim that lost its var take
            // the sequence length, desyncing every downstream broadcast (e.g. an
            // activation `[Dynamic, Dynamic, hidden]` became `[seq, seq, hidden]`
            // instead of `[1, seq, hidden]`).
            if matches!(dim, Dim::Dynamic | Dim::Var(_)) {
                *dim = Dim::Concrete(if axis == 0 { 1 } else { context_len });
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

// ── pipeline selection + compile target ──────────────────────────────────────

/// Choose the optimization pipeline by model topology (causal LM / ViT / generic).
fn select_pipeline(ai_graph: &AiGraph, patch_budget_ratio: Option<f32>) -> OptPipeline {
    let has_input_ids = ai_graph.input_names.iter().any(|n| n == "input_ids");
    let looks_like_causal_lm = has_input_ids
        && ai_graph
            .output_names
            .iter()
            .any(|n| n == "logits" || n == "output");
    let looks_like_vit = !looks_like_causal_lm
        && ai_graph
            .input_names
            .iter()
            .any(|n| n == "pixel_values" || n == "image" || n == "input_image" || n == "x");
    if looks_like_causal_lm {
        OptPipeline::mvp()
    } else if looks_like_vit {
        OptPipeline::vit(patch_budget_ratio.unwrap_or(1.0))
    } else {
        OptPipeline::generic()
    }
}

/// The Witt level hologram-ai compiles at (W32 = 32-bit residue arithmetic).
fn hologram_witt_level() -> uor_foundation::WittLevel {
    uor_foundation::WittLevel::W32
}
