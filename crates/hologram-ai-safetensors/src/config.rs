//! HuggingFace config.json parsing and conversion.

use crate::error::{Result, SafeTensorsError};
use hologram_ai_common::{Activation, FFNType, NormType, TransformerConfig};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// HuggingFace model configuration.
///
/// This struct captures the common fields from HuggingFace config.json files.
/// Different model types may have additional fields which are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    /// Model architectures (e.g., ["LlamaForCausalLM"]).
    pub architectures: Option<Vec<String>>,

    /// Model type string (e.g., "llama", "mistral", "qwen2").
    pub model_type: Option<String>,

    /// Number of hidden layers.
    #[serde(alias = "n_layer", alias = "num_layers")]
    pub num_hidden_layers: Option<u32>,

    /// Hidden size (embedding dimension).
    #[serde(alias = "n_embd", alias = "d_model")]
    pub hidden_size: Option<u32>,

    /// Number of attention heads.
    #[serde(alias = "n_head")]
    pub num_attention_heads: Option<u32>,

    /// Number of KV heads (for GQA).
    #[serde(alias = "n_head_kv")]
    pub num_key_value_heads: Option<u32>,

    /// Intermediate size in MLP.
    #[serde(alias = "n_inner")]
    pub intermediate_size: Option<u32>,

    /// Vocabulary size.
    pub vocab_size: Option<u32>,

    /// Maximum sequence length.
    #[serde(alias = "n_positions", alias = "n_ctx")]
    pub max_position_embeddings: Option<u32>,

    /// RMS norm epsilon.
    pub rms_norm_eps: Option<f32>,

    /// Layer norm epsilon (alternative name).
    pub layer_norm_eps: Option<f32>,

    /// Hidden activation function.
    pub hidden_act: Option<String>,

    /// RoPE theta (base frequency).
    pub rope_theta: Option<f32>,

    /// RoPE scaling configuration.
    pub rope_scaling: Option<RoPEScalingConfig>,

    /// Whether to tie input/output embeddings.
    pub tie_word_embeddings: Option<bool>,

    /// Torch dtype.
    pub torch_dtype: Option<String>,

    /// Head dimension (if specified).
    pub head_dim: Option<u32>,

    /// Whether attention layers have bias.
    pub attention_bias: Option<bool>,

    /// Whether MLP layers have bias.
    pub mlp_bias: Option<bool>,
}

/// RoPE scaling configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct RoPEScalingConfig {
    /// Scaling type.
    #[serde(rename = "type")]
    pub scaling_type: Option<String>,
    /// Scaling factor.
    pub factor: Option<f32>,
    /// Original max position embeddings.
    pub original_max_position_embeddings: Option<u32>,
}

impl HfConfig {
    /// Load config from a path.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config: HfConfig = serde_json::from_reader(reader)?;
        Ok(config)
    }

    /// Get the model type/architecture.
    pub fn get_model_type(&self) -> Option<&str> {
        self.model_type.as_deref().or_else(|| {
            self.architectures
                .as_ref()
                .and_then(|archs| archs.first().map(|s| s.as_str()))
        })
    }

    /// Convert to TransformerConfig for the generic builder.
    pub fn to_transformer_config(&self) -> Result<TransformerConfig> {
        let model_type = self
            .get_model_type()
            .ok_or_else(|| SafeTensorsError::MissingConfigField("model_type".to_string()))?
            .to_lowercase();

        // Determine norm type based on architecture
        let norm_type = match model_type.as_str() {
            "llama" | "mistral" | "qwen" | "qwen2" | "gemma" | "deepseek" => NormType::RMSNorm,
            "gpt2" | "gpt_neox" | "phi" => NormType::LayerNorm,
            _ => NormType::RMSNorm, // Default
        };

        // Determine activation
        let hidden_act = self
            .hidden_act
            .as_deref()
            .map(|act| match act.to_lowercase().as_str() {
                "silu" | "swiglu" => Activation::SiLU,
                "gelu" => Activation::GELU,
                "gelu_new" | "gelu_fast" => Activation::GELUTanh,
                "relu" => Activation::ReLU,
                _ => Activation::SiLU,
            })
            .unwrap_or(Activation::SiLU);

        // Determine FFN type based on architecture
        let ffn_type = match model_type.as_str() {
            "llama" | "mistral" | "qwen" | "qwen2" | "gemma" | "deepseek" => FFNType::Gated,
            "gpt2" | "phi" => FFNType::Standard,
            _ => FFNType::Gated,
        };

        // Get required fields
        let num_layers = self
            .num_hidden_layers
            .ok_or_else(|| SafeTensorsError::MissingConfigField("num_hidden_layers".to_string()))?;
        let hidden_size = self
            .hidden_size
            .ok_or_else(|| SafeTensorsError::MissingConfigField("hidden_size".to_string()))?;
        let num_attention_heads = self.num_attention_heads.ok_or_else(|| {
            SafeTensorsError::MissingConfigField("num_attention_heads".to_string())
        })?;
        let intermediate_size = self
            .intermediate_size
            .ok_or_else(|| SafeTensorsError::MissingConfigField("intermediate_size".to_string()))?;
        let vocab_size = self
            .vocab_size
            .ok_or_else(|| SafeTensorsError::MissingConfigField("vocab_size".to_string()))?;

        // Get optional fields with defaults
        let max_position_embeddings = self.max_position_embeddings.unwrap_or(4096);
        let norm_eps = self.rms_norm_eps.or(self.layer_norm_eps).unwrap_or(1e-6);

        // Handle GQA
        let num_kv_heads = self
            .num_key_value_heads
            .filter(|&kv| kv != num_attention_heads);

        // Handle RoPE scaling
        let rope_scaling = self.rope_scaling.as_ref().and_then(|rs| {
            Some(hologram_ai_common::RoPEScaling {
                scaling_type: rs.scaling_type.clone()?,
                factor: rs.factor?,
                original_max_position_embeddings: rs.original_max_position_embeddings,
            })
        });

        let config = TransformerConfig {
            num_layers,
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            norm_type,
            norm_eps,
            hidden_act,
            rope_theta: self.rope_theta,
            rope_scaling,
            ffn_type,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
            head_dim: self.head_dim,
            attention_bias: self.attention_bias.unwrap_or(false),
            mlp_bias: self.mlp_bias.unwrap_or(false),
        };

        // Validate
        config
            .validate()
            .map_err(SafeTensorsError::MissingConfigField)?;

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_llama_config() -> HfConfig {
        HfConfig {
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            model_type: Some("llama".to_string()),
            num_hidden_layers: Some(32),
            hidden_size: Some(4096),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            intermediate_size: Some(11008),
            vocab_size: Some(32000),
            max_position_embeddings: Some(4096),
            rms_norm_eps: Some(1e-6),
            layer_norm_eps: None,
            hidden_act: Some("silu".to_string()),
            rope_theta: Some(10000.0),
            rope_scaling: None,
            tie_word_embeddings: Some(false),
            torch_dtype: Some("float16".to_string()),
            head_dim: None,
            attention_bias: None,
            mlp_bias: None,
        }
    }

    #[test]
    fn test_get_model_type() {
        let config = sample_llama_config();
        assert_eq!(config.get_model_type(), Some("llama"));

        let config2 = HfConfig {
            model_type: None,
            architectures: Some(vec!["MistralForCausalLM".to_string()]),
            ..sample_llama_config()
        };
        assert_eq!(config2.get_model_type(), Some("MistralForCausalLM"));
    }

    #[test]
    fn test_to_transformer_config_llama() {
        let hf_config = sample_llama_config();
        let config = hf_config.to_transformer_config().unwrap();

        assert_eq!(config.num_layers, 32);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, Some(8));
        assert!(config.is_gqa());
        assert_eq!(config.norm_type, NormType::RMSNorm);
        assert_eq!(config.hidden_act, Activation::SiLU);
        assert_eq!(config.ffn_type, FFNType::Gated);
    }

    #[test]
    fn test_to_transformer_config_missing_field() {
        let config = HfConfig {
            num_hidden_layers: None,
            ..sample_llama_config()
        };
        let result = config.to_transformer_config();
        assert!(result.is_err());
    }

    #[test]
    fn test_activation_parsing() {
        let mut config = sample_llama_config();

        config.hidden_act = Some("silu".to_string());
        let tc = config.to_transformer_config().unwrap();
        assert_eq!(tc.hidden_act, Activation::SiLU);

        config.hidden_act = Some("gelu".to_string());
        let tc = config.to_transformer_config().unwrap();
        assert_eq!(tc.hidden_act, Activation::GELU);

        config.hidden_act = Some("relu".to_string());
        let tc = config.to_transformer_config().unwrap();
        assert_eq!(tc.hidden_act, Activation::ReLU);
    }

    #[test]
    fn test_norm_eps_fallback() {
        let mut config = sample_llama_config();
        config.rms_norm_eps = None;
        config.layer_norm_eps = Some(1e-5);

        let tc = config.to_transformer_config().unwrap();
        assert!((tc.norm_eps - 1e-5).abs() < 1e-10);
    }
}
