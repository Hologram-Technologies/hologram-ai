//! Attention block builder.

use crate::error::{CommonError, Result};
use crate::transformer::config::TransformerConfig;
use crate::weights::WeightMap;
use hologram::ir::{ConstantData, Dim, GraphBuilder, NodeIndex, Shape};

/// Attention type variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionType {
    /// Standard multi-head attention.
    #[default]
    Standard,
    /// Sliding window attention (used in Mistral).
    SlidingWindow(u32),
}

/// Builder for attention blocks.
pub struct AttentionBuilder<'a> {
    config: &'a TransformerConfig,
}

impl<'a> AttentionBuilder<'a> {
    /// Create a new attention builder.
    pub fn new(config: &'a TransformerConfig) -> Self {
        Self { config }
    }

    /// Build a single attention block.
    ///
    /// # Arguments
    /// * `builder` - The graph builder
    /// * `hidden_states` - Input hidden states [batch, seq_len, hidden_size]
    /// * `layer_idx` - Layer index for weight naming
    /// * `weights` - Weight map containing Q, K, V, O projections
    ///
    /// # Returns
    /// Output hidden states [batch, seq_len, hidden_size]
    pub fn build_attention(
        &self,
        builder: &mut GraphBuilder,
        hidden_states: NodeIndex,
        layer_idx: u32,
        weights: &WeightMap,
    ) -> Result<NodeIndex> {
        let hidden_size = self.config.hidden_size as i64;
        let num_heads = self.config.num_attention_heads as i64;
        let head_dim = self.config.head_dimension() as i64;
        let num_kv_heads = self.config.kv_heads() as i64;

        // Get weight names for this layer
        let q_weight_name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);
        let k_weight_name = format!("model.layers.{}.self_attn.k_proj.weight", layer_idx);
        let v_weight_name = format!("model.layers.{}.self_attn.v_proj.weight", layer_idx);
        let o_weight_name = format!("model.layers.{}.self_attn.o_proj.weight", layer_idx);

        // Get weights
        let q_weight = weights.get_required(&q_weight_name)?;
        let k_weight = weights.get_required(&k_weight_name)?;
        let v_weight = weights.get_required(&v_weight_name)?;
        let o_weight = weights.get_required(&o_weight_name)?;

        // Create weight constants using ConstantData::F32
        let q_proj_weight = builder.constant(
            ConstantData::F32(q_weight.to_f32_vec()),
            hologram::ir::Shape::new(vec![
                hologram::ir::Dim::Static((num_heads * head_dim) as usize),
                hologram::ir::Dim::Static(hidden_size as usize),
            ]),
        );
        let k_proj_weight = builder.constant(
            ConstantData::F32(k_weight.to_f32_vec()),
            hologram::ir::Shape::new(vec![
                hologram::ir::Dim::Static((num_kv_heads * head_dim) as usize),
                hologram::ir::Dim::Static(hidden_size as usize),
            ]),
        );
        let v_proj_weight = builder.constant(
            ConstantData::F32(v_weight.to_f32_vec()),
            hologram::ir::Shape::new(vec![
                hologram::ir::Dim::Static((num_kv_heads * head_dim) as usize),
                hologram::ir::Dim::Static(hidden_size as usize),
            ]),
        );
        let o_proj_weight = builder.constant(
            ConstantData::F32(o_weight.to_f32_vec()),
            hologram::ir::Shape::new(vec![
                hologram::ir::Dim::Static(hidden_size as usize),
                hologram::ir::Dim::Static((num_heads * head_dim) as usize),
            ]),
        );

        // Q, K, V projections: [batch, seq, hidden] @ [hidden, proj_dim].T -> [batch, seq, proj_dim]
        let query = builder
            .matmul(hidden_states, q_proj_weight)
            .map_err(|e| CommonError::GraphBuildError(format!("Q projection failed: {:?}", e)))?;
        let key = builder
            .matmul(hidden_states, k_proj_weight)
            .map_err(|e| CommonError::GraphBuildError(format!("K projection failed: {:?}", e)))?;
        let value = builder
            .matmul(hidden_states, v_proj_weight)
            .map_err(|e| CommonError::GraphBuildError(format!("V projection failed: {:?}", e)))?;

        // Reshape for multi-head attention
        // Q: [batch, seq, num_heads * head_dim] -> [batch, seq, num_heads, head_dim]
        let query_reshaped = builder
            .reshape(query, vec![-1, -1, num_heads, head_dim])
            .map_err(|e| CommonError::GraphBuildError(format!("Q reshape failed: {:?}", e)))?;

        // K, V: [batch, seq, num_kv_heads * head_dim] -> [batch, seq, num_kv_heads, head_dim]
        let key_reshaped = builder
            .reshape(key, vec![-1, -1, num_kv_heads, head_dim])
            .map_err(|e| CommonError::GraphBuildError(format!("K reshape failed: {:?}", e)))?;
        let value_reshaped = builder
            .reshape(value, vec![-1, -1, num_kv_heads, head_dim])
            .map_err(|e| CommonError::GraphBuildError(format!("V reshape failed: {:?}", e)))?;

        // Transpose to [batch, num_heads, seq, head_dim]
        let query_transposed = builder
            .transpose(query_reshaped, vec![0, 2, 1, 3])
            .map_err(|e| CommonError::GraphBuildError(format!("Q transpose failed: {:?}", e)))?;
        let key_transposed = builder
            .transpose(key_reshaped, vec![0, 2, 1, 3])
            .map_err(|e| CommonError::GraphBuildError(format!("K transpose failed: {:?}", e)))?;
        let value_transposed = builder
            .transpose(value_reshaped, vec![0, 2, 1, 3])
            .map_err(|e| CommonError::GraphBuildError(format!("V transpose failed: {:?}", e)))?;

        // Apply RoPE (Rotary Position Embedding) to Q and K
        let (query_with_rope, key_with_rope) = if self.config.rope_theta.is_some() {
            let q_rope = self.apply_rope(builder, query_transposed, head_dim)?;
            let k_rope = self.apply_rope(builder, key_transposed, head_dim)?;
            (q_rope, k_rope)
        } else {
            (query_transposed, key_transposed)
        };

        // Handle GQA by repeating KV heads if necessary
        let (key_expanded, value_expanded) = if self.config.is_gqa() {
            let num_groups = self.config.num_query_groups() as i64;
            // For GQA, we need to repeat KV heads to match Q heads
            let key_exp = self.expand_kv_heads(builder, key_with_rope, num_kv_heads, num_groups)?;
            let value_exp =
                self.expand_kv_heads(builder, value_transposed, num_kv_heads, num_groups)?;
            (key_exp, value_exp)
        } else {
            (key_with_rope, value_transposed)
        };

        // Scaled dot-product attention
        // scores = Q @ K^T / sqrt(head_dim)
        let key_t = builder
            .transpose(key_expanded, vec![0, 1, 3, 2])
            .map_err(|e| {
                CommonError::GraphBuildError(format!("K transpose for attention failed: {:?}", e))
            })?;
        let scores = builder.matmul(query_with_rope, key_t).map_err(|e| {
            CommonError::GraphBuildError(format!("Attention scores failed: {:?}", e))
        })?;

        // Scale by 1/sqrt(head_dim)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scale_const = builder.constant(
            ConstantData::F32(vec![scale]),
            hologram::ir::Shape::new(vec![hologram::ir::Dim::Static(1)]),
        );
        let scaled_scores = builder
            .mul(scores, scale_const)
            .map_err(|e| CommonError::GraphBuildError(format!("Scaling failed: {:?}", e)))?;

        // Softmax over last dimension
        let attn_weights = builder
            .softmax(scaled_scores, -1)
            .map_err(|e| CommonError::GraphBuildError(format!("Softmax failed: {:?}", e)))?;

        // attn_output = attn_weights @ V
        let attn_output = builder.matmul(attn_weights, value_expanded).map_err(|e| {
            CommonError::GraphBuildError(format!("Attention output failed: {:?}", e))
        })?;

        // Transpose back: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        let attn_transposed = builder
            .transpose(attn_output, vec![0, 2, 1, 3])
            .map_err(|e| {
                CommonError::GraphBuildError(format!("Output transpose failed: {:?}", e))
            })?;

        // Reshape: [batch, seq, heads, head_dim] -> [batch, seq, hidden_size]
        let attn_reshaped = builder
            .reshape(attn_transposed, vec![-1, -1, hidden_size])
            .map_err(|e| CommonError::GraphBuildError(format!("Output reshape failed: {:?}", e)))?;

        // Output projection
        let output = builder.matmul(attn_reshaped, o_proj_weight).map_err(|e| {
            CommonError::GraphBuildError(format!("Output projection failed: {:?}", e))
        })?;

        Ok(output)
    }

    /// Expand KV heads for GQA (Grouped Query Attention).
    ///
    /// For GQA, we need to repeat each KV head `num_groups` times to match
    /// the number of query heads.
    ///
    /// Input: [batch, num_kv_heads, seq, head_dim]
    /// Output: [batch, num_kv_heads * num_groups, seq, head_dim]
    fn expand_kv_heads(
        &self,
        builder: &mut GraphBuilder,
        tensor: NodeIndex,
        num_kv_heads: i64,
        num_groups: i64,
    ) -> Result<NodeIndex> {
        let head_dim = self.config.head_dimension() as i64;

        // Step 1: Reshape to [batch, num_kv_heads, 1, seq, head_dim]
        // This inserts a dimension at position 2 for broadcasting
        let reshape1 = builder
            .reshape(tensor, vec![-1, num_kv_heads, 1, -1, head_dim])
            .map_err(|e| CommonError::GraphBuildError(format!("GQA reshape1 failed: {:?}", e)))?;

        // Step 2: Tile along the new dimension
        // repeats = [1, 1, num_groups, 1, 1] - only repeat along dimension 2
        let repeats = builder.constant(
            ConstantData::I64(vec![1, 1, num_groups, 1, 1]),
            hologram::ir::Shape::new(vec![hologram::ir::Dim::Static(5)]),
        );
        let tiled = builder
            .tile(reshape1, repeats)
            .map_err(|e| CommonError::GraphBuildError(format!("GQA tile failed: {:?}", e)))?;

        // Step 3: Reshape to [batch, num_kv_heads * num_groups, seq, head_dim]
        // This merges dimensions 1 and 2: num_kv_heads * num_groups = num_heads
        let final_shape = builder
            .reshape(tiled, vec![-1, num_kv_heads * num_groups, -1, head_dim])
            .map_err(|e| {
                CommonError::GraphBuildError(format!("GQA final reshape failed: {:?}", e))
            })?;

        Ok(final_shape)
    }

    /// Apply Rotary Position Embedding (RoPE) to a tensor.
    ///
    /// RoPE encodes position information by rotating query and key vectors.
    /// For each position p and dimension pair (2i, 2i+1):
    ///   x_rot[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
    ///   x_rot[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
    /// where θ = p / (theta^(2i/d))
    ///
    /// Input: [batch, num_heads, seq_len, head_dim]
    /// Output: [batch, num_heads, seq_len, head_dim]
    fn apply_rope(
        &self,
        builder: &mut GraphBuilder,
        tensor: NodeIndex,
        head_dim: i64,
    ) -> Result<NodeIndex> {
        let theta = self.config.rope_theta.unwrap_or(10000.0);
        let max_seq_len = self.config.max_position_embeddings as usize;
        let half_dim = (head_dim / 2) as usize;

        // Precompute inverse frequencies: inv_freq[i] = 1 / (theta^(2i/d))
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();

        // Precompute cos and sin tables for all positions
        // Shape: [max_seq_len, half_dim]
        let mut cos_table = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin_table = Vec::with_capacity(max_seq_len * half_dim);

        for pos in 0..max_seq_len {
            for &freq in &inv_freq {
                let angle = pos as f32 * freq;
                cos_table.push(angle.cos());
                sin_table.push(angle.sin());
            }
        }

        // Create cos/sin constants
        // We need to broadcast to [1, 1, seq_len, head_dim]
        // First reshape cos/sin to [max_seq_len, half_dim], then interleave to [max_seq_len, head_dim]
        let mut cos_interleaved = Vec::with_capacity(max_seq_len * head_dim as usize);
        let mut sin_interleaved = Vec::with_capacity(max_seq_len * head_dim as usize);

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let idx = pos * half_dim + i;
                // Duplicate each cos/sin value for the pair
                cos_interleaved.push(cos_table[idx]);
                cos_interleaved.push(cos_table[idx]);
                sin_interleaved.push(sin_table[idx]);
                sin_interleaved.push(sin_table[idx]);
            }
        }

        let cos_const = builder.constant(
            ConstantData::F32(cos_interleaved),
            Shape::new(vec![
                Dim::Static(1),
                Dim::Static(1),
                Dim::Static(max_seq_len),
                Dim::Static(head_dim as usize),
            ]),
        );
        let sin_const = builder.constant(
            ConstantData::F32(sin_interleaved),
            Shape::new(vec![
                Dim::Static(1),
                Dim::Static(1),
                Dim::Static(max_seq_len),
                Dim::Static(head_dim as usize),
            ]),
        );

        // Compute rotate_half(x): for pairs [x0, x1], returns [-x1, x0]
        // We use: x * cos + rotate_half(x) * sin
        // where rotate_half interleaves [-x1, x0, -x3, x2, ...]

        // Create negation pattern: [-1, 1, -1, 1, ...] for rotate_half
        let mut neg_pattern = Vec::with_capacity(head_dim as usize);
        for _ in 0..half_dim {
            neg_pattern.push(-1.0f32);
            neg_pattern.push(1.0f32);
        }
        let neg_const = builder.constant(
            ConstantData::F32(neg_pattern),
            Shape::new(vec![
                Dim::Static(1),
                Dim::Static(1),
                Dim::Static(1),
                Dim::Static(head_dim as usize),
            ]),
        );

        // RoPE rotation: x * cos + rotate_half(x) * sin
        // where rotate_half(x) swaps pairs and negates the first: (-x1, x0, -x3, x2, ...)

        // Step 1: x * cos
        let x_cos = builder
            .mul(tensor, cos_const)
            .map_err(|e| CommonError::GraphBuildError(format!("RoPE x*cos failed: {:?}", e)))?;

        // Step 2: Create rotated version by reshaping and swapping
        // Reshape to [batch, heads, seq, half_dim, 2] then swap last dim
        let reshaped = builder
            .reshape(tensor, vec![-1, -1, -1, half_dim as i64, 2])
            .map_err(|e| {
                CommonError::GraphBuildError(format!("RoPE reshape for rotate failed: {:?}", e))
            })?;

        // Reverse the last dimension: [x0, x1] -> [x1, x0]
        // We can use transpose but that doesn't help here
        // Instead, let's use a gather with reversed indices

        // Create indices for reverse: [1, 0]
        let reverse_indices = builder.constant(
            ConstantData::I64(vec![1, 0]),
            Shape::new(vec![Dim::Static(2)]),
        );

        // Gather along last dimension to reverse
        let reversed = builder.gather(reshaped, reverse_indices, -1).map_err(|e| {
            CommonError::GraphBuildError(format!("RoPE reverse gather failed: {:?}", e))
        })?;

        // Reshape back to [batch, heads, seq, head_dim]
        let reversed_flat = builder
            .reshape(reversed, vec![-1, -1, -1, head_dim])
            .map_err(|e| {
                CommonError::GraphBuildError(format!("RoPE reshape back failed: {:?}", e))
            })?;

        // Apply negation pattern: [-1, 1, -1, 1, ...]
        let rotated = builder
            .mul(reversed_flat, neg_const)
            .map_err(|e| CommonError::GraphBuildError(format!("RoPE negation failed: {:?}", e)))?;

        // Step 3: rotated * sin
        let rotated_sin = builder.mul(rotated, sin_const).map_err(|e| {
            CommonError::GraphBuildError(format!("RoPE rotated*sin failed: {:?}", e))
        })?;

        // Step 4: x_cos + rotated_sin
        let result = builder
            .add(x_cos, rotated_sin)
            .map_err(|e| CommonError::GraphBuildError(format!("RoPE add failed: {:?}", e)))?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_type_default() {
        assert_eq!(AttentionType::default(), AttentionType::Standard);
    }

    #[test]
    fn test_attention_type_sliding_window() {
        let attn = AttentionType::SlidingWindow(4096);
        match attn {
            AttentionType::SlidingWindow(size) => assert_eq!(size, 4096),
            _ => panic!("Expected SlidingWindow"),
        }
    }

    #[test]
    fn test_attention_builder_creation() {
        let config = TransformerConfig::default();
        let _builder = AttentionBuilder::new(&config);
    }

    #[test]
    fn test_rope_frequency_computation() {
        // Test that RoPE inverse frequencies are computed correctly
        let theta = 10000.0f32;
        let head_dim = 128;
        let half_dim = head_dim / 2;

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();

        // First frequency should be 1.0 (theta^0 = 1)
        assert!((inv_freq[0] - 1.0).abs() < 1e-6);

        // Frequencies should decrease
        for i in 1..half_dim {
            assert!(inv_freq[i] < inv_freq[i - 1]);
        }

        // Last frequency should be approximately 1/theta
        let expected_last = 1.0 / theta.powf((2 * (half_dim - 1)) as f32 / head_dim as f32);
        assert!((inv_freq[half_dim - 1] - expected_last).abs() < 1e-6);
    }

    #[test]
    fn test_rope_config_enabled() {
        let config = TransformerConfig {
            rope_theta: Some(10000.0),
            ..Default::default()
        };
        assert!(config.rope_theta.is_some());
        assert_eq!(config.rope_theta.unwrap(), 10000.0);
    }

    #[test]
    fn test_rope_config_disabled() {
        let config = TransformerConfig {
            rope_theta: None,
            ..Default::default()
        };
        assert!(config.rope_theta.is_none());
    }
}
