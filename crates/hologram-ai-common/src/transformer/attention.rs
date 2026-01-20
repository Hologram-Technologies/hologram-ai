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

/// Projection weight constants for attention.
struct ProjectionWeights {
    q_proj: NodeIndex,
    k_proj: NodeIndex,
    v_proj: NodeIndex,
    o_proj: NodeIndex,
}

/// QKV tensors after projection and reshape.
struct QkvTensors {
    query: NodeIndex,
    key: NodeIndex,
    value: NodeIndex,
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
        let head_dim = self.config.head_dimension() as i64;

        // Phase 1: Create projection weight constants
        let proj_weights = self.create_projection_weights(builder, layer_idx, weights)?;

        // Phase 2: Project and reshape QKV
        let qkv = self.project_qkv(builder, hidden_states, &proj_weights)?;

        // Phase 3: Apply RoPE if configured
        let (query_with_rope, key_with_rope) =
            self.apply_rope_if_configured(builder, qkv.query, qkv.key, head_dim)?;

        // Phase 4: Expand KV heads for GQA if needed
        let (key_expanded, value_expanded) =
            self.expand_kv_if_gqa(builder, key_with_rope, qkv.value)?;

        // Phase 5: Compute scaled dot-product attention
        let attn_output = self.compute_attention(
            builder,
            query_with_rope,
            key_expanded,
            value_expanded,
            head_dim,
        )?;

        // Phase 6: Project output
        let output = self.project_output(builder, attn_output, proj_weights.o_proj)?;

        Ok(output)
    }

    /// Create projection weight constants for Q, K, V, O.
    fn create_projection_weights(
        &self,
        builder: &mut GraphBuilder,
        layer_idx: u32,
        weights: &WeightMap,
    ) -> Result<ProjectionWeights> {
        let hidden_size = self.config.hidden_size as i64;
        let num_heads = self.config.num_attention_heads as i64;
        let head_dim = self.config.head_dimension() as i64;
        let num_kv_heads = self.config.kv_heads() as i64;

        // Get weight names for this layer
        let q_weight_name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);
        let k_weight_name = format!("model.layers.{}.self_attn.k_proj.weight", layer_idx);
        let v_weight_name = format!("model.layers.{}.self_attn.v_proj.weight", layer_idx);
        let o_weight_name = format!("model.layers.{}.self_attn.o_proj.weight", layer_idx);

        // Get weights from weight map
        let q_weight = weights.get_required(&q_weight_name)?;
        let k_weight = weights.get_required(&k_weight_name)?;
        let v_weight = weights.get_required(&v_weight_name)?;
        let o_weight = weights.get_required(&o_weight_name)?;

        // Create weight constants
        let q_proj = builder.constant(
            ConstantData::F32(q_weight.to_f32_vec()),
            Shape::new(vec![
                Dim::Static((num_heads * head_dim) as usize),
                Dim::Static(hidden_size as usize),
            ]),
        );
        let k_proj = builder.constant(
            ConstantData::F32(k_weight.to_f32_vec()),
            Shape::new(vec![
                Dim::Static((num_kv_heads * head_dim) as usize),
                Dim::Static(hidden_size as usize),
            ]),
        );
        let v_proj = builder.constant(
            ConstantData::F32(v_weight.to_f32_vec()),
            Shape::new(vec![
                Dim::Static((num_kv_heads * head_dim) as usize),
                Dim::Static(hidden_size as usize),
            ]),
        );
        let o_proj = builder.constant(
            ConstantData::F32(o_weight.to_f32_vec()),
            Shape::new(vec![
                Dim::Static(hidden_size as usize),
                Dim::Static((num_heads * head_dim) as usize),
            ]),
        );

        Ok(ProjectionWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    /// Project input to Q, K, V and reshape for multi-head attention.
    ///
    /// Input: [batch, seq, hidden_size]
    /// Output: Q, K, V each as [batch, num_heads, seq, head_dim]
    fn project_qkv(
        &self,
        builder: &mut GraphBuilder,
        hidden_states: NodeIndex,
        proj_weights: &ProjectionWeights,
    ) -> Result<QkvTensors> {
        let num_heads = self.config.num_attention_heads as i64;
        let head_dim = self.config.head_dimension() as i64;
        let num_kv_heads = self.config.kv_heads() as i64;

        // Q, K, V projections: [batch, seq, hidden] @ [hidden, proj_dim].T -> [batch, seq, proj_dim]
        let query = builder
            .matmul(hidden_states, proj_weights.q_proj)
            .map_err(|e| CommonError::GraphBuildError(format!("Q projection failed: {e:?}")))?;
        let key = builder
            .matmul(hidden_states, proj_weights.k_proj)
            .map_err(|e| CommonError::GraphBuildError(format!("K projection failed: {e:?}")))?;
        let value = builder
            .matmul(hidden_states, proj_weights.v_proj)
            .map_err(|e| CommonError::GraphBuildError(format!("V projection failed: {e:?}")))?;

        // Reshape for multi-head attention
        // Q: [batch, seq, num_heads * head_dim] -> [batch, seq, num_heads, head_dim]
        let query_reshaped = builder
            .reshape(query, vec![-1, -1, num_heads, head_dim])
            .map_err(|e| CommonError::GraphBuildError(format!("Q reshape failed: {e:?}")))?;

        // K, V: [batch, seq, num_kv_heads * head_dim] -> [batch, seq, num_kv_heads, head_dim]
        let key_reshaped = builder
            .reshape(key, vec![-1, -1, num_kv_heads, head_dim])
            .map_err(|e| CommonError::GraphBuildError(format!("K reshape failed: {e:?}")))?;
        let value_reshaped = builder
            .reshape(value, vec![-1, -1, num_kv_heads, head_dim])
            .map_err(|e| CommonError::GraphBuildError(format!("V reshape failed: {e:?}")))?;

        // Transpose to [batch, num_heads, seq, head_dim]
        let query_transposed = builder
            .transpose(query_reshaped, vec![0, 2, 1, 3])
            .map_err(|e| CommonError::GraphBuildError(format!("Q transpose failed: {e:?}")))?;
        let key_transposed = builder
            .transpose(key_reshaped, vec![0, 2, 1, 3])
            .map_err(|e| CommonError::GraphBuildError(format!("K transpose failed: {e:?}")))?;
        let value_transposed = builder
            .transpose(value_reshaped, vec![0, 2, 1, 3])
            .map_err(|e| CommonError::GraphBuildError(format!("V transpose failed: {e:?}")))?;

        Ok(QkvTensors {
            query: query_transposed,
            key: key_transposed,
            value: value_transposed,
        })
    }

    /// Apply RoPE to Q and K if configured, otherwise return unchanged.
    fn apply_rope_if_configured(
        &self,
        builder: &mut GraphBuilder,
        query: NodeIndex,
        key: NodeIndex,
        head_dim: i64,
    ) -> Result<(NodeIndex, NodeIndex)> {
        if self.config.rope_theta.is_some() {
            let q_rope = self.apply_rope(builder, query, head_dim)?;
            let k_rope = self.apply_rope(builder, key, head_dim)?;
            Ok((q_rope, k_rope))
        } else {
            Ok((query, key))
        }
    }

    /// Expand KV heads for GQA if configured.
    fn expand_kv_if_gqa(
        &self,
        builder: &mut GraphBuilder,
        key: NodeIndex,
        value: NodeIndex,
    ) -> Result<(NodeIndex, NodeIndex)> {
        if self.config.is_gqa() {
            let num_kv_heads = self.config.kv_heads() as i64;
            let num_groups = self.config.num_query_groups() as i64;
            let key_exp = self.expand_kv_heads(builder, key, num_kv_heads, num_groups)?;
            let value_exp = self.expand_kv_heads(builder, value, num_kv_heads, num_groups)?;
            Ok((key_exp, value_exp))
        } else {
            Ok((key, value))
        }
    }

    /// Compute scaled dot-product attention.
    ///
    /// Input: Q, K, V each as [batch, num_heads, seq, head_dim]
    /// Output: [batch, num_heads, seq, head_dim]
    fn compute_attention(
        &self,
        builder: &mut GraphBuilder,
        query: NodeIndex,
        key: NodeIndex,
        value: NodeIndex,
        head_dim: i64,
    ) -> Result<NodeIndex> {
        // scores = Q @ K^T / sqrt(head_dim)
        let key_t = builder.transpose(key, vec![0, 1, 3, 2]).map_err(|e| {
            CommonError::GraphBuildError(format!("K transpose for attention failed: {e:?}"))
        })?;
        let scores = builder
            .matmul(query, key_t)
            .map_err(|e| CommonError::GraphBuildError(format!("Attention scores failed: {e:?}")))?;

        // Scale by 1/sqrt(head_dim)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scale_const = builder.constant(
            ConstantData::F32(vec![scale]),
            Shape::new(vec![Dim::Static(1)]),
        );
        let scaled_scores = builder
            .mul(scores, scale_const)
            .map_err(|e| CommonError::GraphBuildError(format!("Scaling failed: {e:?}")))?;

        // Softmax over last dimension
        let attn_weights = builder
            .softmax(scaled_scores, -1)
            .map_err(|e| CommonError::GraphBuildError(format!("Softmax failed: {e:?}")))?;

        // attn_output = attn_weights @ V
        let attn_output = builder
            .matmul(attn_weights, value)
            .map_err(|e| CommonError::GraphBuildError(format!("Attention output failed: {e:?}")))?;

        Ok(attn_output)
    }

    /// Project attention output back to hidden size.
    ///
    /// Input: [batch, num_heads, seq, head_dim]
    /// Output: [batch, seq, hidden_size]
    fn project_output(
        &self,
        builder: &mut GraphBuilder,
        attn_output: NodeIndex,
        o_proj_weight: NodeIndex,
    ) -> Result<NodeIndex> {
        let hidden_size = self.config.hidden_size as i64;

        // Transpose back: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        let attn_transposed = builder
            .transpose(attn_output, vec![0, 2, 1, 3])
            .map_err(|e| CommonError::GraphBuildError(format!("Output transpose failed: {e:?}")))?;

        // Reshape: [batch, seq, heads, head_dim] -> [batch, seq, hidden_size]
        let attn_reshaped = builder
            .reshape(attn_transposed, vec![-1, -1, hidden_size])
            .map_err(|e| CommonError::GraphBuildError(format!("Output reshape failed: {e:?}")))?;

        // Output projection
        let output = builder.matmul(attn_reshaped, o_proj_weight).map_err(|e| {
            CommonError::GraphBuildError(format!("Output projection failed: {e:?}"))
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

        // Phase 1: Precompute and create cos/sin constants
        let (cos_const, sin_const) =
            self.create_rope_cos_sin_constants(builder, theta, max_seq_len, head_dim)?;

        // Phase 2: Create negation pattern constant
        let neg_const = self.create_rope_negation_pattern(builder, head_dim);

        // Phase 3: Apply the rotation formula: x * cos + rotate_half(x) * sin
        let result =
            self.apply_rope_rotation(builder, tensor, cos_const, sin_const, neg_const, head_dim)?;

        Ok(result)
    }

    /// Precompute and create cos/sin constants for RoPE.
    ///
    /// Returns interleaved cos and sin tables as graph constants with shape
    /// [1, 1, max_seq_len, head_dim] for broadcasting.
    fn create_rope_cos_sin_constants(
        &self,
        builder: &mut GraphBuilder,
        theta: f32,
        max_seq_len: usize,
        head_dim: i64,
    ) -> Result<(NodeIndex, NodeIndex)> {
        // Compute interleaved cos/sin tables
        let (cos_interleaved, sin_interleaved) =
            compute_rope_tables(theta, max_seq_len, head_dim as usize);

        // Create graph constants
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

        Ok((cos_const, sin_const))
    }

    /// Create negation pattern [-1, 1, -1, 1, ...] for rotate_half operation.
    fn create_rope_negation_pattern(&self, builder: &mut GraphBuilder, head_dim: i64) -> NodeIndex {
        let half_dim = (head_dim / 2) as usize;
        let mut neg_pattern = Vec::with_capacity(head_dim as usize);
        for _ in 0..half_dim {
            neg_pattern.push(-1.0f32);
            neg_pattern.push(1.0f32);
        }

        builder.constant(
            ConstantData::F32(neg_pattern),
            Shape::new(vec![
                Dim::Static(1),
                Dim::Static(1),
                Dim::Static(1),
                Dim::Static(head_dim as usize),
            ]),
        )
    }

    /// Apply the RoPE rotation formula: x * cos + rotate_half(x) * sin
    ///
    /// rotate_half(x) swaps pairs and negates the first: (-x1, x0, -x3, x2, ...)
    fn apply_rope_rotation(
        &self,
        builder: &mut GraphBuilder,
        tensor: NodeIndex,
        cos_const: NodeIndex,
        sin_const: NodeIndex,
        neg_const: NodeIndex,
        head_dim: i64,
    ) -> Result<NodeIndex> {
        let half_dim = (head_dim / 2) as usize;

        // x * cos
        let x_cos = builder
            .mul(tensor, cos_const)
            .map_err(|e| CommonError::GraphBuildError(format!("RoPE x*cos failed: {e:?}")))?;

        // Create rotated version: rotate_half(x)
        let rotated = self.compute_rotate_half(builder, tensor, neg_const, half_dim, head_dim)?;

        // rotated * sin
        let rotated_sin = builder
            .mul(rotated, sin_const)
            .map_err(|e| CommonError::GraphBuildError(format!("RoPE rotated*sin failed: {e:?}")))?;

        // x_cos + rotated_sin
        let result = builder
            .add(x_cos, rotated_sin)
            .map_err(|e| CommonError::GraphBuildError(format!("RoPE add failed: {e:?}")))?;

        Ok(result)
    }

    /// Compute rotate_half(x): for pairs [x0, x1], returns [-x1, x0].
    ///
    /// This is done by:
    /// 1. Reshape to [..., half_dim, 2]
    /// 2. Reverse the last dimension: [x0, x1] -> [x1, x0]
    /// 3. Reshape back to [..., head_dim]
    /// 4. Apply negation pattern: [-1, 1, -1, 1, ...]
    fn compute_rotate_half(
        &self,
        builder: &mut GraphBuilder,
        tensor: NodeIndex,
        neg_const: NodeIndex,
        half_dim: usize,
        head_dim: i64,
    ) -> Result<NodeIndex> {
        // Reshape to [batch, heads, seq, half_dim, 2]
        let reshaped = builder
            .reshape(tensor, vec![-1, -1, -1, half_dim as i64, 2])
            .map_err(|e| {
                CommonError::GraphBuildError(format!("RoPE reshape for rotate failed: {e:?}"))
            })?;

        // Reverse indices [1, 0] to swap pairs
        let reverse_indices = builder.constant(
            ConstantData::I64(vec![1, 0]),
            Shape::new(vec![Dim::Static(2)]),
        );

        // Gather along last dimension to reverse: [x0, x1] -> [x1, x0]
        let reversed = builder.gather(reshaped, reverse_indices, -1).map_err(|e| {
            CommonError::GraphBuildError(format!("RoPE reverse gather failed: {e:?}"))
        })?;

        // Reshape back to [batch, heads, seq, head_dim]
        let reversed_flat = builder
            .reshape(reversed, vec![-1, -1, -1, head_dim])
            .map_err(|e| {
                CommonError::GraphBuildError(format!("RoPE reshape back failed: {e:?}"))
            })?;

        // Apply negation pattern: [-1, 1, -1, 1, ...]
        let rotated = builder
            .mul(reversed_flat, neg_const)
            .map_err(|e| CommonError::GraphBuildError(format!("RoPE negation failed: {e:?}")))?;

        Ok(rotated)
    }
}

/// Precompute interleaved cos/sin tables for RoPE.
///
/// Returns (cos_interleaved, sin_interleaved) where each has shape
/// [max_seq_len * head_dim] with values duplicated for each dimension pair.
fn compute_rope_tables(theta: f32, max_seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;

    // Compute inverse frequencies: inv_freq[i] = 1 / (theta^(2i/d))
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
        .collect();

    // Precompute cos and sin for all positions
    let mut cos_table = Vec::with_capacity(max_seq_len * half_dim);
    let mut sin_table = Vec::with_capacity(max_seq_len * half_dim);

    for pos in 0..max_seq_len {
        for &freq in &inv_freq {
            let angle = pos as f32 * freq;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }

    // Interleave: duplicate each value for the dimension pair
    let mut cos_interleaved = Vec::with_capacity(max_seq_len * head_dim);
    let mut sin_interleaved = Vec::with_capacity(max_seq_len * head_dim);

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let idx = pos * half_dim + i;
            cos_interleaved.push(cos_table[idx]);
            cos_interleaved.push(cos_table[idx]);
            sin_interleaved.push(sin_table[idx]);
            sin_interleaved.push(sin_table[idx]);
        }
    }

    (cos_interleaved, sin_interleaved)
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
