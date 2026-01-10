//! Transformer configuration types.

use serde::{Deserialize, Serialize};

/// Normalization type used in the transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NormType {
    /// Layer normalization (used in GPT-2, BERT, etc.)
    LayerNorm,
    /// RMS normalization (used in LLaMA, Mistral, etc.)
    #[default]
    RMSNorm,
}

/// Activation function used in FFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Activation {
    /// Sigmoid Linear Unit: x * sigmoid(x)
    #[default]
    SiLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Rectified Linear Unit
    ReLU,
    /// GELU with tanh approximation
    GELUTanh,
}

/// Feed-forward network type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FFNType {
    /// Gated FFN: down(gate(x) * up(x)) - used in LLaMA, Mistral
    #[default]
    Gated,
    /// Standard FFN: down(act(up(x))) - used in GPT-2, BERT
    Standard,
}

/// RoPE (Rotary Position Embedding) scaling configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoPEScaling {
    /// Scaling type (e.g., "linear", "dynamic", "yarn")
    pub scaling_type: String,
    /// Scaling factor
    pub factor: f32,
    /// Original max position embeddings (for dynamic scaling)
    pub original_max_position_embeddings: Option<u32>,
}

/// Complete transformer configuration.
///
/// This struct captures all the parameters needed to build a transformer model.
/// It can be populated from GGUF metadata or HuggingFace config.json.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Number of transformer layers.
    pub num_layers: u32,

    /// Hidden size (embedding dimension).
    pub hidden_size: u32,

    /// Number of attention heads.
    pub num_attention_heads: u32,

    /// Number of key-value heads for GQA (None = same as attention heads).
    pub num_kv_heads: Option<u32>,

    /// Intermediate size in FFN (often 4x hidden_size or ~2.67x for gated FFN).
    pub intermediate_size: u32,

    /// Vocabulary size.
    pub vocab_size: u32,

    /// Maximum sequence length / position embeddings.
    pub max_position_embeddings: u32,

    /// Normalization type.
    #[serde(default)]
    pub norm_type: NormType,

    /// Epsilon for normalization layers.
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,

    /// Activation function in FFN.
    #[serde(default)]
    pub hidden_act: Activation,

    /// RoPE base frequency (theta).
    pub rope_theta: Option<f32>,

    /// RoPE scaling configuration.
    pub rope_scaling: Option<RoPEScaling>,

    /// FFN type (gated or standard).
    #[serde(default)]
    pub ffn_type: FFNType,

    /// Whether to tie input/output embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Head dimension (if different from hidden_size / num_attention_heads).
    pub head_dim: Option<u32>,

    /// Attention bias (whether QKV projections have bias).
    #[serde(default)]
    pub attention_bias: bool,

    /// MLP bias (whether FFN layers have bias).
    #[serde(default)]
    pub mlp_bias: bool,
}

fn default_norm_eps() -> f32 {
    1e-6
}

impl TransformerConfig {
    /// Get the head dimension.
    pub fn head_dimension(&self) -> u32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Get the number of KV heads (defaults to num_attention_heads if not set).
    pub fn kv_heads(&self) -> u32 {
        self.num_kv_heads.unwrap_or(self.num_attention_heads)
    }

    /// Check if using Grouped Query Attention (GQA).
    pub fn is_gqa(&self) -> bool {
        self.kv_heads() < self.num_attention_heads
    }

    /// Get the number of query groups (for GQA).
    pub fn num_query_groups(&self) -> u32 {
        self.num_attention_heads / self.kv_heads()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_layers == 0 {
            return Err("num_layers must be > 0".to_string());
        }
        if self.hidden_size == 0 {
            return Err("hidden_size must be > 0".to_string());
        }
        if self.num_attention_heads == 0 {
            return Err("num_attention_heads must be > 0".to_string());
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err("hidden_size must be divisible by num_attention_heads".to_string());
        }
        if matches!(self.num_kv_heads, Some(kv_heads) if !self.num_attention_heads.is_multiple_of(kv_heads))
        {
            return Err("num_attention_heads must be divisible by num_kv_heads".to_string());
        }
        if self.intermediate_size == 0 {
            return Err("intermediate_size must be > 0".to_string());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".to_string());
        }
        Ok(())
    }
}

impl Default for TransformerConfig {
    /// Default configuration matching LLaMA-7B.
    fn default() -> Self {
        Self {
            num_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_kv_heads: None,
            intermediate_size: 11008,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-6,
            hidden_act: Activation::SiLU,
            rope_theta: Some(10000.0),
            rope_scaling: None,
            ffn_type: FFNType::Gated,
            tie_word_embeddings: false,
            head_dim: None,
            attention_bias: false,
            mlp_bias: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TransformerConfig::default();
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.head_dimension(), 128);
        assert!(!config.is_gqa());
    }

    #[test]
    fn test_gqa_config() {
        let config = TransformerConfig {
            num_attention_heads: 32,
            num_kv_heads: Some(8),
            ..Default::default()
        };
        assert!(config.is_gqa());
        assert_eq!(config.kv_heads(), 8);
        assert_eq!(config.num_query_groups(), 4);
    }

    #[test]
    fn test_validation_valid() {
        let config = TransformerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_invalid_layers() {
        let config = TransformerConfig {
            num_layers: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_invalid_hidden_size() {
        let config = TransformerConfig {
            hidden_size: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_misaligned_heads() {
        let config = TransformerConfig {
            hidden_size: 4096,
            num_attention_heads: 30, // 4096 % 30 != 0
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_invalid_gqa() {
        let config = TransformerConfig {
            num_attention_heads: 32,
            num_kv_heads: Some(5), // 32 % 5 != 0
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_custom_head_dim() {
        let config = TransformerConfig {
            hidden_size: 4096,
            num_attention_heads: 32,
            head_dim: Some(64), // Custom head dim
            ..Default::default()
        };
        assert_eq!(config.head_dimension(), 64);
    }

    #[test]
    fn test_norm_type_default() {
        assert_eq!(NormType::default(), NormType::RMSNorm);
    }

    #[test]
    fn test_activation_default() {
        assert_eq!(Activation::default(), Activation::SiLU);
    }

    #[test]
    fn test_ffn_type_default() {
        assert_eq!(FFNType::default(), FFNType::Gated);
    }

    #[test]
    fn test_rope_scaling() {
        let scaling = RoPEScaling {
            scaling_type: "linear".to_string(),
            factor: 2.0,
            original_max_position_embeddings: Some(4096),
        };
        assert_eq!(scaling.scaling_type, "linear");
        assert_eq!(scaling.factor, 2.0);
    }

    #[test]
    fn test_config_serialize_deserialize() {
        let config = TransformerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: TransformerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }
}
