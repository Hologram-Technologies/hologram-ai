//! Normalization layer builders.

use crate::error::{CommonError, Result};
use crate::transformer::config::{NormType, TransformerConfig};
use crate::weights::WeightMap;
use hologram::ir::{ConstantData, Dim, GraphBuilder, NodeIndex, NodeOp, Shape};

/// Builder for normalization layers.
pub struct NormBuilder<'a> {
    config: &'a TransformerConfig,
}

impl<'a> NormBuilder<'a> {
    /// Create a new normalization builder.
    pub fn new(config: &'a TransformerConfig) -> Self {
        Self { config }
    }

    /// Build a normalization layer.
    ///
    /// # Arguments
    /// * `builder` - The graph builder
    /// * `input` - Input tensor [batch, seq_len, hidden_size]
    /// * `weight_name` - Name of the weight tensor in the weight map
    /// * `weights` - Weight map containing normalization weights
    ///
    /// # Returns
    /// Normalized output [batch, seq_len, hidden_size]
    pub fn build_norm(
        &self,
        builder: &mut GraphBuilder,
        input: NodeIndex,
        weight_name: &str,
        weights: &WeightMap,
    ) -> Result<NodeIndex> {
        match self.config.norm_type {
            NormType::RMSNorm => self.build_rms_norm(builder, input, weight_name, weights),
            NormType::LayerNorm => self.build_layer_norm(builder, input, weight_name, weights),
        }
    }

    /// Build RMSNorm layer.
    ///
    /// RMSNorm: y = x / sqrt(mean(x^2) + eps) * scale
    fn build_rms_norm(
        &self,
        builder: &mut GraphBuilder,
        input: NodeIndex,
        weight_name: &str,
        weights: &WeightMap,
    ) -> Result<NodeIndex> {
        let hidden_size = self.config.hidden_size as usize;
        let eps = self.config.norm_eps;

        // Get weight
        let weight = weights.get_required(weight_name)?;

        // Create weight constant
        let weight_const = builder.constant(
            ConstantData::F32(weight.to_f32_vec()),
            Shape::new(vec![Dim::Static(hidden_size)]),
        );

        // RMSNorm operation: takes input and scale, normalizes over last axis
        builder
            .rms_norm(input, weight_const, eps, vec![-1])
            .map_err(|e| CommonError::GraphBuildError(format!("RMSNorm failed: {:?}", e)))
    }

    /// Build LayerNorm layer.
    ///
    /// LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * scale + bias
    fn build_layer_norm(
        &self,
        builder: &mut GraphBuilder,
        input: NodeIndex,
        weight_name: &str,
        weights: &WeightMap,
    ) -> Result<NodeIndex> {
        let hidden_size = self.config.hidden_size as usize;
        let eps = self.config.norm_eps;

        // Get weight and bias
        let weight = weights.get_required(weight_name)?;
        let bias_name = weight_name.replace(".weight", ".bias");
        let bias = weights.get(&bias_name);

        // Create weight constant
        let weight_const = builder.constant(
            ConstantData::F32(weight.to_f32_vec()),
            Shape::new(vec![Dim::Static(hidden_size)]),
        );

        // Get input shape and dtype for LayerNorm operation
        let input_node = builder
            .graph()
            .node(input)
            .ok_or_else(|| CommonError::GraphBuildError("Invalid input node".to_string()))?;
        let shape = input_node.shape.clone();
        let dtype = input_node.dtype;

        // Add LayerNorm operation using direct graph access
        let graph = builder.graph_mut();
        let norm_idx = graph.add_op(
            NodeOp::LayerNorm {
                epsilon: eps,
                axes: vec![-1],
            },
            shape,
            dtype,
        );
        graph.connect(input, norm_idx);
        graph.connect(weight_const, norm_idx);

        // Add bias if present
        if let Some(bias_weight) = bias {
            let bias_const = builder.constant(
                ConstantData::F32(bias_weight.to_f32_vec()),
                Shape::new(vec![Dim::Static(hidden_size)]),
            );
            builder.add(norm_idx, bias_const).map_err(|e| {
                CommonError::GraphBuildError(format!("LayerNorm bias add failed: {:?}", e))
            })
        } else {
            Ok(norm_idx)
        }
    }

    /// Build input embedding layer norm (if model uses it).
    /// Some models apply layer normalization to embeddings before the transformer layers.
    #[allow(dead_code)]
    pub fn build_input_norm(
        &self,
        builder: &mut GraphBuilder,
        input: NodeIndex,
        weights: &WeightMap,
    ) -> Result<NodeIndex> {
        let weight_name = "model.embed_tokens.weight"; // Some models normalize embeddings
        if weights.contains(weight_name) {
            self.build_norm(builder, input, weight_name, weights)
        } else {
            Ok(input) // No input norm
        }
    }

    /// Build final layer norm before output.
    pub fn build_final_norm(
        &self,
        builder: &mut GraphBuilder,
        input: NodeIndex,
        weights: &WeightMap,
    ) -> Result<NodeIndex> {
        let weight_name = "model.norm.weight";
        self.build_norm(builder, input, weight_name, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_builder_creation() {
        let config = TransformerConfig::default();
        let builder = NormBuilder::new(&config);
        assert_eq!(builder.config.norm_type, NormType::RMSNorm);
        assert!((builder.config.norm_eps - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_norm_type_rms() {
        let config = TransformerConfig {
            norm_type: NormType::RMSNorm,
            ..Default::default()
        };
        assert_eq!(config.norm_type, NormType::RMSNorm);
    }

    #[test]
    fn test_norm_type_layer() {
        let config = TransformerConfig {
            norm_type: NormType::LayerNorm,
            ..Default::default()
        };
        assert_eq!(config.norm_type, NormType::LayerNorm);
    }

    #[test]
    fn test_default_norm_eps() {
        let config = TransformerConfig::default();
        assert!((config.norm_eps - 1e-6).abs() < 1e-10);
    }
}
