//! Model compilation and inference session.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Context;
use hologram_ai_common::{
    AiGraph, AiParam, OptPipeline, MemoryPlanner, KvCacheLayout,
    lower, LoweringOptions,
};

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

// ── Compiled model ────────────────────────────────────────────────────────────

/// A fully compiled model ready for repeated inference.
///
/// Thread-safe — wrap in `Arc` to share across sessions.
pub struct CompiledModel {
    /// The compiled `.holo` archive bytes.
    archive: Vec<u8>,
    schedule: Arc<hologram::ExecutionSchedule>,
    registry: Arc<hologram::CustomOpRegistry>,
    /// KV-cache layout (None when KV-cache is disabled, e.g. single-pass inference).
    pub kv_layout: Option<KvCacheLayout>,
    pub metadata: ModelMetadata,
}

impl CompiledModel {
    /// Write the compiled `.holo` archive to `path`.
    pub fn save_archive(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating output directory {parent:?}"))?;
            }
        }
        std::fs::write(path, &self.archive)
            .with_context(|| format!("writing .holo archive to {path:?}"))
    }
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

    /// Access the underlying compiled model.
    pub fn model(&self) -> &CompiledModel {
        &self.model
    }

    /// Run a single forward pass and return logits.
    ///
    /// `token_ids` — input token IDs for this batch.
    /// Returns a flat `Vec<f32>` of shape `[seq_len × vocab_size]`.
    pub fn run(&mut self, token_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        let plan = hologram::load_from_bytes(&self.model.archive)
            .context("loading compiled archive")?;

        // Encode token IDs as INT64 little-endian (ONNX convention).
        let input_ids_bytes: Vec<u8> = token_ids
            .iter()
            .flat_map(|&id| (id as i64).to_le_bytes())
            .collect();

        // Attention mask: all ones, same length as token_ids.
        let attention_mask_bytes: Vec<u8> = (0..token_ids.len())
            .flat_map(|_| 1i64.to_le_bytes())
            .collect();

        let mut inputs = hologram::GraphInputs::new();
        inputs.set(0, input_ids_bytes);
        inputs.set(1, attention_mask_bytes);

        let outputs = hologram::KvExecutor::execute_with_weights(
            plan.graph(),
            &self.model.schedule,
            &inputs,
            &self.model.registry,
            plan.weights(),
        ).context("hologram execution failed")?;

        let (_, logit_bytes) = outputs.get(0)
            .context("no outputs from execution")?;
        let logits: Vec<f32> = bytemuck::cast_slice(logit_bytes).to_vec();

        Ok(logits)
    }
}

// ── Model compiler ────────────────────────────────────────────────────────────

/// Compiles a `ModelSource` through the full pipeline into a `CompiledModel`.
///
/// Build a compiler with the default settings via `ModelCompiler::default()`,
/// then call `compile(source)`.
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
    /// Compile pipeline:
    ///
    /// 1. Import → `AiGraph`
    /// 2. `OptPipeline::mvp().run()` → optimised `AiGraph`
    /// 3. `MemoryPlanner.plan()` (KV sizing only for Sprint 001)
    /// 4. `lower()` → `LoweringOutput { graph, registry }`
    /// 5. `hologram::compile(graph)` → `CompilationOutput { archive, schedule }`
    /// 6. Build `CompiledModel`
    pub fn compile(&self, source: ModelSource) -> anyhow::Result<CompiledModel> {
        // Step 1 — import.
        let ai_graph = self.import(source)?;

        // Step 2 — optimize.
        let ai_graph = OptPipeline::mvp()
            .run(ai_graph)
            .context("optimization pass failed")?;

        // Validate before lowering.
        let errs = ai_graph.validate();
        if !errs.is_empty() {
            anyhow::bail!("{} validation error(s): {}", errs.len(), errs[0].message);
        }

        // Step 3 — memory plan (KV sizing).
        let _plan = MemoryPlanner.plan(&ai_graph)
            .context("memory planning failed")?;

        // Extract metadata before lowering (borrows ai_graph).
        let metadata = extract_metadata(&ai_graph);

        // Step 4 — lower.
        let lower_out = lower(
            &ai_graph,
            &KvCacheLayout::none(),
            &LoweringOptions::default(),
        ).context("lowering failed")?;

        // Step 5 — hologram compile → archive + schedule.
        let compilation = hologram::compile(lower_out.graph)
            .context("hologram::compile failed")?;

        // Step 6 — embed weight data in the archive.
        let weight_blob = collect_weight_bytes(&ai_graph)?;
        let archive = if weight_blob.is_empty() {
            compilation.archive
        } else {
            rebuild_archive_with_weights(&compilation.archive, weight_blob)?
        };

        Ok(CompiledModel {
            archive,
            schedule:   Arc::new(compilation.schedule),
            registry:   Arc::new(lower_out.registry),
            kv_layout:  None,
            metadata,
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
            ModelSource::GgufPath(path) => {
                hologram_ai_gguf::import_gguf(
                    &path,
                    hologram_ai_gguf::GgufImportOptions { mmap: self.mmap, arch_override: None },
                ).with_context(|| format!("importing GGUF from {path:?}"))
            }
            ModelSource::GgmlPath(path) => {
                let bytes = std::fs::read(&path)
                    .with_context(|| format!("reading GGML file {path:?}"))?;
                hologram_ai_ggml::import_ggml(&bytes)
                    .with_context(|| format!("importing GGML from {path:?}"))
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

/// Collect weight bytes from all Mmap params in TensorId-sorted order.
///
/// The ordering must match builder.rs which assigns cumulative byte offsets
/// as `source_id` in `ConstantData::Deferred` using the same sorted order.
fn collect_weight_bytes(ai_graph: &AiGraph) -> anyhow::Result<Vec<u8>> {
    let mut sorted: Vec<_> = ai_graph.params.iter()
        .filter(|(_, p)| matches!(p, AiParam::Mmap { .. }))
        .collect();
    if sorted.is_empty() {
        return Ok(Vec::new());
    }
    sorted.sort_by_key(|(&tid, _)| tid);

    let total_size: u64 = sorted.iter()
        .map(|(_, p)| match p { AiParam::Mmap { len, .. } => *len, _ => 0 })
        .sum();
    let mut blob = Vec::with_capacity(total_size as usize);

    for (_, param) in &sorted {
        if let AiParam::Mmap { path, offset, len, .. } = param {
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

/// Rebuild a compiled archive with weight data embedded.
///
/// Extracts the serialized graph from the existing archive and re-assembles
/// using `HoloWriter` with the weight blob added.
fn rebuild_archive_with_weights(archive: &[u8], weights: Vec<u8>) -> anyhow::Result<Vec<u8>> {
    let plan = hologram::load_from_bytes(archive)
        .context("loading compiled archive for weight embedding")?;
    let h = plan.header();
    let graph_bytes = archive[h.graph_offset as usize..(h.graph_offset + h.graph_size) as usize].to_vec();

    hologram::HoloWriter::new()
        .set_graph_bytes(graph_bytes)
        .set_weights(weights)
        .build()
        .map_err(|e| anyhow::anyhow!("rebuilding archive with weights: {e}"))
}
