//! Generic transformer builder.

use crate::error::{CommonError, Result};
use crate::transformer::attention::AttentionBuilder;
use crate::transformer::config::TransformerConfig;
use crate::transformer::ffn::FFNBuilder;
use crate::transformer::norm::NormBuilder;
use crate::weights::WeightMap;
use hologram::ir::{ConstantData, DType, Dim, GraphBuilder, NodeIndex, OperationGraph, Shape};

/// Generic transformer builder that constructs IR graphs from configuration.
///
/// This builder can create any standard decoder-only transformer architecture
/// by reading the configuration parameters. No architecture-specific code is needed.
///
/// # Supported Models
///
/// - LLaMA / LLaMA 2 / LLaMA 3
/// - Mistral / Mixtral (standard attention)
/// - Qwen / Qwen2
/// - DeepSeek (standard variant)
/// - Phi-2 / Phi-3
/// - Gemma
///
/// # Example
///
/// ```ignore
/// use hologram_ai_common::transformer::{GenericTransformerBuilder, TransformerConfig};
///
/// let config = TransformerConfig::default(); // LLaMA-7B config
/// let builder = GenericTransformerBuilder::new();
/// let graph = builder.build(&config, &weights)?;
/// ```
pub struct GenericTransformerBuilder {
    /// Whether to include RoPE position embeddings.
    pub include_rope: bool,
}

impl GenericTransformerBuilder {
    /// Create a new generic transformer builder.
    pub fn new() -> Self {
        Self { include_rope: true }
    }

    /// Build the complete transformer graph.
    ///
    /// # Arguments
    /// * `config` - Transformer configuration
    /// * `weights` - Weight map containing all model weights
    ///
    /// # Returns
    /// The complete operation graph ready for compilation.
    pub fn build(&self, config: &TransformerConfig, weights: &WeightMap) -> Result<OperationGraph> {
        // Validate configuration
        config.validate().map_err(CommonError::InvalidConfig)?;

        let mut builder = GraphBuilder::new();

        // 1. Input: token IDs [batch, seq_len]
        let input_ids = builder.input(
            "input_ids",
            Shape::new(vec![Dim::Dynamic, Dim::Dynamic]),
            DType::I32,
        );

        // 2. Token embedding
        let hidden_states = self.build_embedding(&mut builder, input_ids, config, weights)?;

        // 3. Transformer layers
        let mut hidden = hidden_states;
        for layer_idx in 0..config.num_layers {
            hidden = self.build_layer(&mut builder, hidden, layer_idx, config, weights)?;
        }

        // 4. Final normalization
        let norm_builder = NormBuilder::new(config);
        let normalized = norm_builder.build_final_norm(&mut builder, hidden, weights)?;

        // 5. Output projection (LM head)
        let logits = self.build_lm_head(&mut builder, normalized, config, weights)?;

        // 6. Register output
        builder.output("logits", logits).map_err(|e| {
            CommonError::GraphBuildError(format!("Output registration failed: {:?}", e))
        })?;

        Ok(builder.build())
    }

    /// Build the token embedding layer.
    fn build_embedding(
        &self,
        builder: &mut GraphBuilder,
        input_ids: NodeIndex,
        config: &TransformerConfig,
        weights: &WeightMap,
    ) -> Result<NodeIndex> {
        let vocab_size = config.vocab_size as usize;
        let hidden_size = config.hidden_size as usize;

        // Get embedding weight
        let embed_weight = weights.get_required("model.embed_tokens.weight")?;

        // Create embedding weight constant
        let embed_const = builder.constant(
            ConstantData::F32(embed_weight.to_f32_vec()),
            Shape::new(vec![Dim::Static(vocab_size), Dim::Static(hidden_size)]),
        );

        // Gather embeddings: [batch, seq] -> [batch, seq, hidden]
        // gather(data, indices, axis) - axis 0 means we're gathering rows from the embedding table
        let embeddings = builder.gather(embed_const, input_ids, 0).map_err(|e| {
            CommonError::GraphBuildError(format!("Embedding gather failed: {:?}", e))
        })?;

        Ok(embeddings)
    }

    /// Build a single transformer layer.
    fn build_layer(
        &self,
        builder: &mut GraphBuilder,
        hidden_states: NodeIndex,
        layer_idx: u32,
        config: &TransformerConfig,
        weights: &WeightMap,
    ) -> Result<NodeIndex> {
        // Pre-attention normalization
        let norm_builder = NormBuilder::new(config);
        let input_norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
        let normed = norm_builder.build_norm(builder, hidden_states, &input_norm_name, weights)?;

        // Self-attention
        let attn_builder = AttentionBuilder::new(config);
        let attn_output = attn_builder.build_attention(builder, normed, layer_idx, weights)?;

        // Residual connection
        let hidden_after_attn = builder.add(hidden_states, attn_output).map_err(|e| {
            CommonError::GraphBuildError(format!("Attention residual add failed: {:?}", e))
        })?;

        // Post-attention normalization
        let post_attn_norm_name =
            format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
        let normed_for_ffn =
            norm_builder.build_norm(builder, hidden_after_attn, &post_attn_norm_name, weights)?;

        // Feed-forward network
        let ffn_builder = FFNBuilder::new(config);
        let ffn_output = ffn_builder.build_ffn(builder, normed_for_ffn, layer_idx, weights)?;

        // Residual connection
        let output = builder.add(hidden_after_attn, ffn_output).map_err(|e| {
            CommonError::GraphBuildError(format!("FFN residual add failed: {:?}", e))
        })?;

        Ok(output)
    }

    /// Build the language model head (output projection).
    fn build_lm_head(
        &self,
        builder: &mut GraphBuilder,
        hidden_states: NodeIndex,
        config: &TransformerConfig,
        weights: &WeightMap,
    ) -> Result<NodeIndex> {
        let vocab_size = config.vocab_size as usize;
        let hidden_size = config.hidden_size as usize;

        // Check if using tied embeddings
        let lm_head_weight = if config.tie_word_embeddings {
            weights.get_required("model.embed_tokens.weight")?
        } else {
            weights.get_required("lm_head.weight")?
        };

        // Create LM head weight constant
        let lm_head_const = builder.constant(
            ConstantData::F32(lm_head_weight.to_f32_vec()),
            Shape::new(vec![Dim::Static(vocab_size), Dim::Static(hidden_size)]),
        );

        // Transpose for matmul: [vocab, hidden] -> [hidden, vocab]
        let lm_head_t = builder.transpose(lm_head_const, vec![1, 0]).map_err(|e| {
            CommonError::GraphBuildError(format!("LM head transpose failed: {:?}", e))
        })?;

        // Project to vocabulary: [batch, seq, hidden] @ [hidden, vocab] -> [batch, seq, vocab]
        let logits = builder
            .matmul(hidden_states, lm_head_t)
            .map_err(|e| CommonError::GraphBuildError(format!("LM head matmul failed: {:?}", e)))?;

        Ok(logits)
    }
}

impl Default for GenericTransformerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = GenericTransformerBuilder::new();
        assert!(builder.include_rope);
    }

    #[test]
    fn test_builder_default() {
        let builder = GenericTransformerBuilder::default();
        assert!(builder.include_rope);
    }

    #[test]
    fn test_config_validation_in_build() {
        let builder = GenericTransformerBuilder::new();
        let invalid_config = TransformerConfig {
            num_layers: 0,
            ..Default::default()
        };
        let weights = WeightMap::new();

        let result = builder.build(&invalid_config, &weights);
        assert!(result.is_err());
    }

    // Note: Full build tests require complete weight maps
    // which would be tested in integration tests with actual models
}
