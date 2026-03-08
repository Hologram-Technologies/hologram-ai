//! Model compilation pipeline.
//!
//! Compiles AI models (ONNX, GGUF) into `.holo` archives via the hologram
//! O(1) LUT runtime. This crate is a **compiler** — it does not own inference
//! sessions or runtime state (see ADR-0016).

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
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

        // Step 2 — optimize.
        let ai_graph = OptPipeline::mvp()
            .run(ai_graph)
            .context("optimization pass failed")?;

        // Validate before lowering.
        let errs = ai_graph.validate();
        if !errs.is_empty() {
            anyhow::bail!("{} validation error(s): {}", errs.len(), errs[0].message);
        }

        // Step 3 — memory plan.
        let _plan = MemoryPlanner.plan(&ai_graph)
            .context("memory planning failed")?;

        // Extract metadata before lowering (borrows ai_graph).
        let metadata = extract_metadata(&ai_graph);
        let import_warnings = ai_graph.warnings.len();
        let node_count = ai_graph.nodes.len();

        // Step 4 — lower.
        let lower_out = lower(
            &ai_graph,
            &KvCacheLayout::none(),
            &LoweringOptions::default(),
        ).context("lowering failed")?;

        // Step 5 — compile.
        let compilation = hologram::compile(lower_out.graph)
            .context("hologram::compile failed")?;

        // Step 6 — embed weights.
        let weight_blob = collect_weight_bytes(&ai_graph)?;
        let total_weight_bytes = weight_blob.len() as u64;

        let archive_bytes = if weight_blob.is_empty() {
            compilation.archive
        } else {
            rebuild_archive_with_weights(&compilation.archive, weight_blob)?
        };

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
