//! GGUF metadata key extraction for model architecture and tokenizer.

use crate::parser::{GgufFile, MetaValue};
use anyhow::{Context, Result};

/// Extracted model architecture parameters.
#[derive(Debug, Clone)]
pub struct ArchParams {
    pub arch: String,
    pub context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub feed_forward_length: u32,
    pub vocab_size: u32,
    pub rope_freq_base: f32,
    pub rope_dimension_count: u32,
    pub layer_norm_rms_epsilon: f32,
}

/// Extracted tokenizer metadata from GGUF.
#[derive(Debug, Clone)]
pub struct TokenizerMeta {
    pub model_type: String,
    pub tokens: Vec<String>,
    pub scores: Option<Vec<f32>>,
    pub token_types: Option<Vec<u32>>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub unk_id: Option<u32>,
    pub pad_id: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
}

impl ArchParams {
    /// Extract architecture parameters from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile, arch_override: Option<&str>) -> Result<Self> {
        let meta = &gguf.metadata;

        let arch = match arch_override {
            Some(a) => a.to_string(),
            None => meta
                .get("general.architecture")
                .and_then(|v| v.as_str())
                .context("missing general.architecture")?
                .to_string(),
        };

        let get_u32 = |key: &str| -> Result<u32> {
            meta.get(key)
                .and_then(|v| v.as_u32())
                .with_context(|| format!("missing or invalid metadata key: {key}"))
        };

        let get_f32 = |key: &str, default: f32| -> f32 {
            meta.get(key).and_then(|v| v.as_f32()).unwrap_or(default)
        };

        let prefix = &arch;
        let context_length = get_u32(&format!("{prefix}.context_length"))?;
        let embedding_length = get_u32(&format!("{prefix}.embedding_length"))?;
        let block_count = get_u32(&format!("{prefix}.block_count"))?;
        let head_count = get_u32(&format!("{prefix}.attention.head_count"))?;
        let head_count_kv =
            get_u32(&format!("{prefix}.attention.head_count_kv")).unwrap_or(head_count);
        let feed_forward_length = get_u32(&format!("{prefix}.feed_forward_length"))?;

        // Vocab size can come from metadata or from tensor count.
        let vocab_size = meta
            .get(&format!("{prefix}.vocab_size"))
            .and_then(|v| v.as_u32())
            .or_else(|| {
                meta.get("tokenizer.ggml.tokens").and_then(|v| match v {
                    MetaValue::Array(arr) => Some(arr.len() as u32),
                    _ => None,
                })
            })
            .unwrap_or(32000);

        let rope_freq_base = get_f32(&format!("{prefix}.rope.freq_base"), 10000.0);
        let rope_dimension_count = meta
            .get(&format!("{prefix}.rope.dimension_count"))
            .and_then(|v| v.as_u32())
            .unwrap_or(embedding_length / head_count);
        let layer_norm_rms_epsilon =
            get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"), 1e-5);

        Ok(Self {
            arch,
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            feed_forward_length,
            vocab_size,
            rope_freq_base,
            rope_dimension_count,
            layer_norm_rms_epsilon,
        })
    }
}

impl TokenizerMeta {
    /// Extract tokenizer metadata from GGUF.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let meta = &gguf.metadata;

        let model_type = meta
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();

        let tokens = meta
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .map(|arr| arr.into_iter().map(|s| s.to_string()).collect())
            .context("missing tokenizer.ggml.tokens")?;

        let scores = meta
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_f32_array());

        let token_types = meta.get("tokenizer.ggml.token_type").and_then(|v| match v {
            MetaValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for item in arr {
                    out.push(item.as_u32()?);
                }
                Some(out)
            }
            _ => None,
        });

        let bos_id = meta
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let eos_id = meta
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());
        let unk_id = meta
            .get("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.as_u32());
        let pad_id = meta
            .get("tokenizer.ggml.padding_token_id")
            .and_then(|v| v.as_u32());

        let add_bos = meta
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| match v {
                MetaValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(true);
        let add_eos = meta
            .get("tokenizer.ggml.add_eos_token")
            .and_then(|v| match v {
                MetaValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

        Ok(Self {
            model_type,
            tokens,
            scores,
            token_types,
            bos_id,
            eos_id,
            unk_id,
            pad_id,
            add_bos,
            add_eos,
        })
    }
}
