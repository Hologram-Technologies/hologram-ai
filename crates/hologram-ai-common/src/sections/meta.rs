//! Pipeline metadata section for `.holo` archives.
//!
//! Describes the components in a multi-component pipeline archive, their roles,
//! weight sharing groups, and data flow connections. Generalizes the former
//! LLM-specific metadata to support any N-component model (CALM, Whisper,
//! Stable Diffusion, MoE, etc.).
//!
//! Serialized via rkyv for zero-copy access from memory-mapped archives.

use crate::exec_context::{Section, SECTION_CUSTOM_BASE};

/// Section kind for pipeline component metadata.
pub const SECTION_META: u32 = SECTION_CUSTOM_BASE + 0x11;

/// Pipeline metadata describing N components and their relationships.
///
/// Zero-copy deserializable via rkyv — can be read directly from a
/// memory-mapped `.holo` archive without allocation.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct MetaSection {
    /// Components in this pipeline archive.
    pub components: Vec<ComponentDescriptor>,
    /// Data flow between components (output port → input port).
    pub connections: Vec<ComponentConnection>,
}

/// Describes a single component in a pipeline archive.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ComponentDescriptor {
    /// Pipeline key (matches PipelineWriter model name, e.g. "lm.prefill").
    pub name: String,
    /// What role this component plays.
    pub role: ComponentRole,
    /// Components sharing this value share weights. Used by weight
    /// deduplication to avoid storing duplicate blobs.
    pub weight_group: String,
    /// If `Some`, this component reuses weights from the named component
    /// rather than storing its own copy.
    pub weight_source: Option<String>,
}

/// Role of a component in a multi-component pipeline.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum ComponentRole {
    /// Full-sequence forward pass (prompt processing).
    Prefill,
    /// Single-token autoregressive decode step.
    Decode,
    /// Encoder (e.g., VAE encoder, Whisper audio encoder).
    Encoder,
    /// Decoder (e.g., VAE decoder, autoencoder decoder).
    Decoder,
    /// Transformer backbone.
    Backbone,
    /// Generative head (e.g., CALM energy-based head).
    GenerativeHead,
    /// Generic single forward pass.
    Forward,
    /// Application-specific role.
    Custom(String),
}

/// Describes a data flow connection between two components.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ComponentConnection {
    /// Source component name.
    pub from_component: String,
    /// Source output port name.
    pub from_output: String,
    /// Destination component name.
    pub to_component: String,
    /// Destination input port name.
    pub to_input: String,
}

impl MetaSection {
    /// Build an LLM pipeline MetaSection from KV-cache layout and model type.
    ///
    /// Convenience constructor for the common 2-component LLM case
    /// (prefill + decode).
    pub fn llm(layout: &crate::mem::KvCacheLayout, _model_type: &str) -> Self {
        Self {
            components: vec![
                ComponentDescriptor {
                    name: "lm.prefill".into(),
                    role: ComponentRole::Prefill,
                    weight_group: "lm".into(),
                    weight_source: None,
                },
                ComponentDescriptor {
                    name: "lm.decode".into(),
                    role: ComponentRole::Decode,
                    weight_group: "lm".into(),
                    weight_source: Some("lm.prefill".into()),
                },
            ],
            connections: vec![ComponentConnection {
                from_component: "lm.prefill".into(),
                from_output: format!(
                    "kv_cache:{}x{}x{}x{}",
                    layout.n_layers, layout.n_kv_heads, layout.max_seq_len, layout.head_dim
                ),
                to_component: "lm.decode".into(),
                to_input: "kv_cache".into(),
            }],
        }
    }

    /// Build a MetaSection from component descriptors and connections.
    pub fn new(
        components: Vec<ComponentDescriptor>,
        connections: Vec<ComponentConnection>,
    ) -> Self {
        Self {
            components,
            connections,
        }
    }

    /// Zero-copy access from raw bytes (e.g. memory-mapped archive section).
    pub fn from_bytes(bytes: &[u8]) -> Result<&ArchivedMetaSection, rkyv::rancor::Error> {
        rkyv::access::<ArchivedMetaSection, rkyv::rancor::Error>(bytes)
    }

    /// Deserialize from raw bytes into an owned `MetaSection`.
    pub fn deserialize_from(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
    }
}

impl Section for MetaSection {
    fn section_kind(&self) -> u32 {
        SECTION_META
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("MetaSection serialization should not fail")
            .to_vec()
    }
}

impl MetaSection {
    /// Section kind constant (matches [`Section::section_kind`]).
    pub fn section_id() -> u32 {
        SECTION_META
    }

    /// Deserialize from custom-section bytes.
    pub fn from_context_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Self::deserialize_from(bytes).map_err(|e| anyhow::anyhow!("deserialize MetaSection: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;
    use crate::mem::KvCacheLayout;

    #[test]
    fn roundtrip_llm_meta() {
        let layout = KvCacheLayout {
            n_layers: 22,
            n_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 2048,
            dtype: DType::F32,
        };
        let section = MetaSection::llm(&layout, "llama");
        assert_eq!(section.section_kind(), SECTION_META);
        assert_eq!(section.components.len(), 2);
        assert_eq!(section.connections.len(), 1);

        let bytes = section.to_bytes();

        // Zero-copy access.
        let archived = MetaSection::from_bytes(&bytes).expect("zero-copy access should succeed");
        assert_eq!(archived.components.len(), 2);
        assert_eq!(archived.components[0].name.as_str(), "lm.prefill");
        assert_eq!(archived.connections.len(), 1);

        // Full deserialization.
        let deserialized =
            MetaSection::deserialize_from(&bytes).expect("deserialization should succeed");
        assert_eq!(deserialized.components.len(), 2);
        assert_eq!(deserialized.components[1].name, "lm.decode");
        assert_eq!(
            deserialized.components[1].weight_source,
            Some("lm.prefill".into())
        );
    }

    #[test]
    fn roundtrip_multi_component() {
        let section = MetaSection::new(
            vec![
                ComponentDescriptor {
                    name: "ae.encoder".into(),
                    role: ComponentRole::Encoder,
                    weight_group: "autoencoder".into(),
                    weight_source: None,
                },
                ComponentDescriptor {
                    name: "backbone".into(),
                    role: ComponentRole::Backbone,
                    weight_group: "backbone".into(),
                    weight_source: None,
                },
                ComponentDescriptor {
                    name: "gen.head".into(),
                    role: ComponentRole::GenerativeHead,
                    weight_group: "backbone".into(),
                    weight_source: None,
                },
                ComponentDescriptor {
                    name: "ae.decoder".into(),
                    role: ComponentRole::Decoder,
                    weight_group: "autoencoder".into(),
                    weight_source: Some("ae.encoder".into()),
                },
            ],
            vec![
                ComponentConnection {
                    from_component: "ae.encoder".into(),
                    from_output: "latent".into(),
                    to_component: "backbone".into(),
                    to_input: "input".into(),
                },
                ComponentConnection {
                    from_component: "backbone".into(),
                    from_output: "hidden".into(),
                    to_component: "gen.head".into(),
                    to_input: "hidden".into(),
                },
                ComponentConnection {
                    from_component: "gen.head".into(),
                    from_output: "predicted_z".into(),
                    to_component: "ae.decoder".into(),
                    to_input: "latent".into(),
                },
            ],
        );

        let bytes = section.to_bytes();
        let deserialized =
            MetaSection::deserialize_from(&bytes).expect("deserialization should succeed");
        assert_eq!(deserialized.components.len(), 4);
        assert_eq!(deserialized.connections.len(), 3);
        assert_eq!(deserialized.components[0].role, ComponentRole::Encoder);
        assert_eq!(deserialized.components[3].role, ComponentRole::Decoder);
    }
}
