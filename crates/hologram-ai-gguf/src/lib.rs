//! GGUF importer — converts GGUF model files into `AiGraph`.
//!
//! Supports GGUF v2/v3 binary format with LLaMA architecture.

pub mod parser;
pub mod metadata;
pub mod arch;

use std::path::Path;
use hologram_ai_common::AiGraph;
use anyhow::{bail, Context, Result};

/// Options for GGUF import.
pub struct GgufImportOptions {
    pub mmap: bool,
    pub arch_override: Option<String>,
}

impl Default for GgufImportOptions {
    fn default() -> Self {
        Self { mmap: true, arch_override: None }
    }
}

/// Import a GGUF model file from disk.
pub fn import_gguf(path: &Path, opts: GgufImportOptions) -> Result<AiGraph> {
    let data = if opts.mmap {
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening GGUF file: {}", path.display()))?;
        unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("memory-mapping GGUF file: {}", path.display()))?
    } else {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading GGUF file: {}", path.display()))?;
        // Safety: MmapMut from owned bytes is safe — wrap in Mmap-compatible form.
        // For non-mmap mode, we just read all bytes and pass to the bytes API.
        return import_gguf_bytes(&bytes, opts);
    };

    import_from_data(&data, path, opts)
}

/// Import a GGUF model from bytes in memory.
pub fn import_gguf_bytes(bytes: &[u8], opts: GgufImportOptions) -> Result<AiGraph> {
    // Use a temporary path for AiParam::Mmap references — caller must ensure
    // bytes outlive the graph or convert params to Inline before use.
    let dummy_path = Path::new("<memory>");
    import_from_data(bytes, dummy_path, opts)
}

fn import_from_data(data: &[u8], model_path: &Path, opts: GgufImportOptions) -> Result<AiGraph> {
    let gguf = parser::parse_gguf(data).context("parsing GGUF header")?;
    let arch_params = metadata::ArchParams::from_gguf(&gguf, opts.arch_override.as_deref())
        .context("extracting architecture parameters")?;

    match arch_params.arch.as_str() {
        "llama" | "mistral" | "codellama" | "tinyllama" => {
            arch::llama::build_llama_graph(&gguf, &arch_params, model_path)
        }
        other => bail!("unsupported GGUF architecture: {other:?} (supported: llama, mistral, codellama)"),
    }
}

/// Extract tokenizer metadata from a GGUF file (for use by the tokenizer crate).
pub fn extract_tokenizer_meta(path: &Path) -> Result<metadata::TokenizerMeta> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening GGUF file: {}", path.display()))?;
    let data = unsafe { memmap2::Mmap::map(&file) }
        .with_context(|| format!("memory-mapping GGUF file: {}", path.display()))?;
    let gguf = parser::parse_gguf(&data).context("parsing GGUF header")?;
    metadata::TokenizerMeta::from_gguf(&gguf)
}
