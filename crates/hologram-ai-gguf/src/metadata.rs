//! GGUF metadata extraction and conversion.

use crate::error::{GgufError, Result};
use hologram_ai_common::{Activation, FFNType, NormType, TransformerConfig};

/// Supported model architectures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Architecture {
    /// LLaMA architecture (LLaMA, LLaMA 2, LLaMA 3).
    Llama,
    /// Mistral architecture.
    Mistral,
    /// Qwen architecture.
    Qwen,
    /// Qwen2 architecture.
    Qwen2,
    /// DeepSeek architecture.
    DeepSeek,
    /// Phi architecture.
    Phi,
    /// Gemma architecture.
    Gemma,
    /// Unknown architecture.
    Unknown(String),
}

impl Architecture {
    /// Parse architecture from GGUF metadata string.
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama" => Self::Llama,
            "mistral" => Self::Mistral,
            "qwen" => Self::Qwen,
            "qwen2" => Self::Qwen2,
            "deepseek" | "deepseek2" => Self::DeepSeek,
            "phi" | "phi2" | "phi3" => Self::Phi,
            "gemma" | "gemma2" => Self::Gemma,
            other => Self::Unknown(other.to_string()),
        }
    }
}

/// GGUF model metadata.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    /// Model architecture.
    pub architecture: Architecture,
    /// Number of transformer layers.
    pub block_count: u32,
    /// Hidden size (embedding dimension).
    pub embedding_length: u32,
    /// Number of attention heads.
    pub attention_head_count: u32,
    /// Number of KV heads (for GQA).
    pub attention_head_count_kv: u32,
    /// Feed-forward hidden size.
    pub feed_forward_length: u32,
    /// RoPE base frequency.
    pub rope_freq_base: f32,
    /// Maximum context length.
    pub context_length: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

impl GgufMetadata {
    /// Convert to TransformerConfig for the generic builder.
    pub fn to_transformer_config(&self) -> Result<TransformerConfig> {
        // Determine norm type based on architecture
        let norm_type = match self.architecture {
            Architecture::Llama
            | Architecture::Mistral
            | Architecture::Qwen
            | Architecture::Qwen2
            | Architecture::DeepSeek
            | Architecture::Gemma => NormType::RMSNorm,
            Architecture::Phi => NormType::LayerNorm,
            Architecture::Unknown(_) => NormType::RMSNorm, // Default to RMSNorm
        };

        // Determine activation based on architecture
        let hidden_act = match self.architecture {
            Architecture::Llama
            | Architecture::Mistral
            | Architecture::Qwen
            | Architecture::Qwen2
            | Architecture::DeepSeek
            | Architecture::Gemma => Activation::SiLU,
            Architecture::Phi => Activation::GELU,
            Architecture::Unknown(_) => Activation::SiLU,
        };

        // Determine FFN type based on architecture
        let ffn_type = match self.architecture {
            Architecture::Llama
            | Architecture::Mistral
            | Architecture::Qwen
            | Architecture::Qwen2
            | Architecture::DeepSeek
            | Architecture::Gemma => FFNType::Gated,
            Architecture::Phi => FFNType::Standard,
            Architecture::Unknown(_) => FFNType::Gated,
        };

        // Check for unsupported architectures
        if let Architecture::Unknown(ref name) = self.architecture {
            // Allow unknown architectures but log a warning
            tracing::warn!(
                "Unknown architecture '{}', using default transformer config",
                name
            );
        }

        let config = TransformerConfig {
            num_layers: self.block_count,
            hidden_size: self.embedding_length,
            num_attention_heads: self.attention_head_count,
            num_kv_heads: if self.attention_head_count_kv != self.attention_head_count {
                Some(self.attention_head_count_kv)
            } else {
                None
            },
            intermediate_size: self.feed_forward_length,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.context_length,
            norm_type,
            norm_eps: self.rms_norm_eps,
            hidden_act,
            rope_theta: Some(self.rope_freq_base),
            rope_scaling: None,
            ffn_type,
            tie_word_embeddings: false, // GGUF models typically don't tie embeddings
            head_dim: None,
            attention_bias: false,
            mlp_bias: false,
        };

        // Validate the config
        config.validate().map_err(|e| GgufError::InvalidMetadata {
            key: "config".to_string(),
            message: e,
        })?;

        Ok(config)
    }
}

impl Default for GgufMetadata {
    /// Default metadata matching LLaMA-7B.
    fn default() -> Self {
        Self {
            architecture: Architecture::Llama,
            block_count: 32,
            embedding_length: 4096,
            attention_head_count: 32,
            attention_head_count_kv: 32,
            feed_forward_length: 11008,
            rope_freq_base: 10000.0,
            context_length: 4096,
            vocab_size: 32000,
            rms_norm_eps: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_parsing() {
        assert_eq!(Architecture::parse("llama"), Architecture::Llama);
        assert_eq!(Architecture::parse("LLAMA"), Architecture::Llama);
        assert_eq!(Architecture::parse("mistral"), Architecture::Mistral);
        assert_eq!(Architecture::parse("qwen"), Architecture::Qwen);
        assert_eq!(Architecture::parse("qwen2"), Architecture::Qwen2);
        assert_eq!(Architecture::parse("deepseek"), Architecture::DeepSeek);
        assert_eq!(Architecture::parse("phi"), Architecture::Phi);
        assert_eq!(Architecture::parse("gemma"), Architecture::Gemma);

        match Architecture::parse("custom") {
            Architecture::Unknown(architecture) => assert_eq!(architecture, "custom"),
            _ => panic!("Expected Unknown"),
        }
    }

    #[test]
    fn test_default_metadata() {
        let meta = GgufMetadata::default();
        assert_eq!(meta.architecture, Architecture::Llama);
        assert_eq!(meta.block_count, 32);
        assert_eq!(meta.embedding_length, 4096);
    }

    #[test]
    fn test_to_transformer_config_llama() {
        let meta = GgufMetadata::default();
        let config = meta.to_transformer_config().unwrap();

        assert_eq!(config.num_layers, 32);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.norm_type, NormType::RMSNorm);
        assert_eq!(config.hidden_act, Activation::SiLU);
        assert_eq!(config.ffn_type, FFNType::Gated);
    }

    #[test]
    fn test_to_transformer_config_gqa() {
        let meta = GgufMetadata {
            attention_head_count: 32,
            attention_head_count_kv: 8, // GQA
            ..Default::default()
        };
        let config = meta.to_transformer_config().unwrap();

        assert_eq!(config.num_kv_heads, Some(8));
        assert!(config.is_gqa());
    }

    #[test]
    fn test_to_transformer_config_phi() {
        let meta = GgufMetadata {
            architecture: Architecture::Phi,
            ..Default::default()
        };
        let config = meta.to_transformer_config().unwrap();

        assert_eq!(config.norm_type, NormType::LayerNorm);
        assert_eq!(config.hidden_act, Activation::GELU);
        assert_eq!(config.ffn_type, FFNType::Standard);
    }

    #[test]
    fn test_to_transformer_config_unknown_architecture() {
        let meta = GgufMetadata {
            architecture: Architecture::Unknown("new_model".to_string()),
            ..Default::default()
        };
        // Should still work with defaults
        let config = meta.to_transformer_config().unwrap();
        assert_eq!(config.norm_type, NormType::RMSNorm);
    }
}
