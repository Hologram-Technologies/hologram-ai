//! Model compilation and inference session.

use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Context;
use hologram_ai_common::{
    AiGraph, OptPipeline, MemoryPlanner, KvCacheLayout,
    lower, LoweringOptions,
};

// ── Model source ──────────────────────────────────────────────────────────────

/// Source for a model to compile.
pub enum ModelSource {
    /// Path to an ONNX model file.
    OnnxPath(PathBuf),
    /// Raw ONNX bytes.
    OnnxBytes(Vec<u8>),
    /// Path to a GGUF model file.
    GgufPath(PathBuf),
    /// Pre-built `AiGraph` (bypass importer).
    AiGraph(AiGraph),
}

// ── Compile options ───────────────────────────────────────────────────────────

/// Options controlling model compilation.
pub struct CompileOptions {
    /// Use memory-mapping for weight loading when possible.
    pub mmap: bool,
}

impl Default for CompileOptions {
    fn default() -> Self { Self { mmap: true } }
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

// ── Compiled model ────────────────────────────────────────────────────────────

/// A fully compiled model ready for repeated inference.
///
/// Thread-safe — wrap in `Arc` to share across sessions.
pub struct CompiledModel {
    /// The compiled `.holo` archive bytes (loaded on each `run()` call).
    archive: Vec<u8>,
    schedule: Arc<hologram::ExecutionSchedule>,
    registry: Arc<hologram::CustomOpRegistry>,
    pub metadata: ModelMetadata,
}

// ── Inference session ─────────────────────────────────────────────────────────

/// Per-session inference state.
pub struct InferenceSession {
    model: Arc<CompiledModel>,
}

impl InferenceSession {
    /// Create a new session for the given compiled model.
    pub fn new(model: Arc<CompiledModel>) -> Self {
        Self { model }
    }

    /// Run a single forward pass and return logits.
    ///
    /// `token_ids` — input token IDs for this batch.
    /// Returns a flat `Vec<f32>` of shape `[seq_len × vocab_size]`.
    pub fn run(&mut self, token_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        let plan = hologram::load_from_bytes(&self.model.archive)
            .context("loading compiled archive")?;

        let mut inputs = hologram::GraphInputs::new();
        inputs.set(0, bytemuck::cast_slice(token_ids).to_vec());

        let outputs = hologram::KvExecutor::execute_with_registry(
            plan.graph(),
            &self.model.schedule,
            &inputs,
            &self.model.registry,
        ).context("hologram execution failed")?;

        let (_, logit_bytes) = outputs.get(0)
            .context("no outputs from execution")?;
        let logits: Vec<f32> = bytemuck::cast_slice(logit_bytes).to_vec();

        Ok(logits)
    }
}

// ── Model compiler ────────────────────────────────────────────────────────────

/// Compiles a `ModelSource` through the full pipeline into a `CompiledModel`.
pub struct ModelCompiler;

impl ModelCompiler {
    /// Compile pipeline:
    ///
    /// 1. Import → `AiGraph`
    /// 2. `OptPipeline::mvp().run()` → optimised `AiGraph`
    /// 3. `MemoryPlanner.plan()` (diagnostic only for now)
    /// 4. `lower()` → `LoweringOutput { graph, registry }`
    /// 5. `hologram::compile(graph)` → `CompilationOutput { archive, schedule }`
    /// 6. Build `CompiledModel`
    pub fn compile(source: ModelSource, opts: CompileOptions) -> anyhow::Result<CompiledModel> {
        // Step 1 — import.
        let ai_graph = import(source, &opts)?;

        // Step 2 — optimize.
        let ai_graph = OptPipeline::mvp()
            .run(ai_graph)
            .context("optimization pass failed")?;

        // Validate before lowering.
        let errs = ai_graph.validate();
        if !errs.is_empty() {
            anyhow::bail!("{} validation error(s): {}", errs.len(), errs[0].message);
        }

        // Step 3 — memory plan (informational for Sprint 001).
        let _plan = MemoryPlanner.plan(&ai_graph)
            .context("memory planning failed")?;

        // Extract metadata before lowering (borrows ai_graph).
        let metadata = extract_metadata(&ai_graph);

        // Step 4 — lower.
        let lower_out = lower(
            &ai_graph,
            &KvCacheLayout::none(),
            LoweringOptions::default(),
        ).context("lowering failed")?;

        // Step 5 — hologram compile → archive + schedule.
        let compilation = hologram::compile(lower_out.graph)
            .context("hologram::compile failed")?;

        Ok(CompiledModel {
            archive:  compilation.archive,
            schedule: Arc::new(compilation.schedule),
            registry: Arc::new(lower_out.registry),
            metadata,
        })
    }
}

fn import(source: ModelSource, opts: &CompileOptions) -> anyhow::Result<AiGraph> {
    match source {
        ModelSource::OnnxPath(path) => {
            hologram_ai_onnx::import_onnx_path(&path)
                .with_context(|| format!("importing ONNX from {path:?}"))
        }
        ModelSource::OnnxBytes(bytes) => {
            hologram_ai_onnx::import_onnx(&bytes)
                .context("importing ONNX from bytes")
        }
        ModelSource::GgufPath(path) => {
            hologram_ai_gguf::import_gguf(
                &path,
                hologram_ai_gguf::GgufImportOptions { mmap: opts.mmap, arch_override: None },
            ).with_context(|| format!("importing GGUF from {path:?}"))
        }
        ModelSource::AiGraph(g) => Ok(g),
    }
}

fn extract_metadata(graph: &AiGraph) -> ModelMetadata {
    use hologram_ai_common::MetaValue;

    let arch = match graph.metadata.get("arch") {
        Some(MetaValue::Str(s)) => s.clone(),
        _ => "unknown".into(),
    };
    let vocab_size  = meta_u32(graph, "vocab_size").unwrap_or(0);
    let context_len = meta_u32(graph, "context_length").unwrap_or(0);
    let n_layers    = meta_u32(graph, "n_layers").unwrap_or(0);
    let n_embd      = meta_u32(graph, "n_embd").unwrap_or(0);

    ModelMetadata { arch, vocab_size, context_len, n_layers, n_embd }
}

fn meta_u32(graph: &AiGraph, key: &str) -> Option<u32> {
    use hologram_ai_common::MetaValue;
    match graph.metadata.get(key) {
        Some(MetaValue::Int(i))   => Some(*i as u32),
        Some(MetaValue::Float(f)) => Some(*f as u32),
        _ => None,
    }
}
