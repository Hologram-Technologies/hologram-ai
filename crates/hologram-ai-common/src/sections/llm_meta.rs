//! LLM metadata section for `.holo` archives.
//!
//! Custom section embedding KV-cache layout and model identification.
//! Uses `SECTION_CUSTOM_BASE + 0x11` until hologram adds a built-in type.
//! Serialized via rkyv for zero-copy access from memory-mapped archives.

use crate::exec_context::{Section, SECTION_CUSTOM_BASE};

/// Section kind for LLM metadata.
pub const SECTION_LLM_META: u32 = SECTION_CUSTOM_BASE + 0x11;

/// LLM metadata embedded in each sub-archive of a pipeline.
///
/// Zero-copy deserializable via rkyv — can be read directly from a
/// memory-mapped `.holo` archive without allocation.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct LlmMetaSection {
    /// Number of transformer layers.
    pub n_layers: u32,
    /// Number of KV-cache heads per layer.
    pub n_kv_heads: u32,
    /// Dimension per attention head.
    pub head_dim: u32,
    /// Maximum sequence length (context window).
    pub max_seq_len: u32,
    /// Total KV-cache bytes.
    pub kv_cache_bytes: u64,
    /// Model architecture identifier (e.g. "llama", "gpt2").
    pub model_type: String,
    /// Prefill layer name in the pipeline archive.
    pub prefill_layer: String,
    /// Decode layer name in the pipeline archive.
    pub decode_layer: String,
}

impl LlmMetaSection {
    /// Build from KV-cache layout and model metadata.
    pub fn from_layout(layout: &crate::mem::KvCacheLayout, model_type: String) -> Self {
        Self {
            n_layers: layout.n_layers,
            n_kv_heads: layout.n_kv_heads,
            head_dim: layout.head_dim,
            max_seq_len: layout.max_seq_len,
            kv_cache_bytes: layout.byte_size(),
            model_type,
            prefill_layer: "lm.prefill".into(),
            decode_layer: "lm.decode".into(),
        }
    }

    /// Zero-copy access from raw bytes (e.g. memory-mapped archive section).
    pub fn from_bytes(bytes: &[u8]) -> Result<&ArchivedLlmMetaSection, rkyv::rancor::Error> {
        rkyv::access::<ArchivedLlmMetaSection, rkyv::rancor::Error>(bytes)
    }

    /// Deserialize from raw bytes into an owned `LlmMetaSection`.
    pub fn deserialize_from(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
    }
}

impl Section for LlmMetaSection {
    fn section_kind(&self) -> u32 {
        SECTION_LLM_META
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("LlmMetaSection serialization")
            .to_vec()
    }
}

impl LlmMetaSection {
    /// Section kind constant (matches [`Section::section_kind`]).
    pub fn section_id() -> u32 {
        SECTION_LLM_META
    }

    /// Deserialize from custom-section bytes.
    pub fn from_context_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Self::deserialize_from(bytes).map_err(|e| anyhow::anyhow!("deserialize LlmMeta: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;
    use crate::mem::KvCacheLayout;

    #[test]
    fn roundtrip_rkyv() {
        let layout = KvCacheLayout {
            n_layers: 22,
            n_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 2048,
            dtype: DType::F32,
        };
        let section = LlmMetaSection::from_layout(&layout, "llama".into());
        assert_eq!(section.section_kind(), SECTION_LLM_META);

        let bytes = section.to_bytes();

        // Zero-copy access.
        let archived = LlmMetaSection::from_bytes(&bytes).unwrap();
        assert_eq!(archived.n_layers, 22);
        assert_eq!(archived.model_type.as_str(), "llama");
        assert_eq!(archived.kv_cache_bytes, layout.byte_size());

        // Full deserialization.
        let deserialized = LlmMetaSection::deserialize_from(&bytes).unwrap();
        assert_eq!(deserialized.n_layers, 22);
        assert_eq!(deserialized.model_type, "llama");
        assert_eq!(deserialized.kv_cache_bytes, layout.byte_size());
    }
}
