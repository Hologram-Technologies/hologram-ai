//! Feed-forward network (FFN) block builder.

use crate::error::{CommonError, Result};
use crate::weights::WeightMap;
use crate::transformer::config::{Activation, FFNType, TransformerConfig};
use hologram::ir::{GraphBuilder, NodeIndex, ConstantData};

/// Builder for FFN blocks.
pub struct FFNBuilder<'a> {
    config: &'a TransformerConfig,
}

impl<'a> FFNBuilder<'a> {
    /// Create a new FFN builder.
    pub fn new(config: &'a TransformerConfig) -> Self {
        Self { config }
    }

    /// Build a feed-forward network block.
    ///
    /// # Arguments
    /// * `builder` - The graph builder
    /// * `hidden_states` - Input hidden states [batch, seq_len, hidden_size]
    /// * `layer_idx` - Layer index for weight naming
    /// * `weights` - Weight map containing FFN weights
    ///
    /// # Returns
    /// Output hidden states [batch, seq_len, hidden_size]
    pub fn build_ffn(
        &self,
        builder: &mut GraphBuilder,
        hidden_states: NodeIndex,
        layer_idx: u32,
        weights: &WeightMap,
    ) -> Result<NodeIndex> {
        match self.config.ffn_type {
            FFNType::Gated => self.build_gated_ffn(builder, hidden_states, layer_idx, weights),
            FFNType::Standard => self.build_standard_ffn(builder, hidden_states, layer_idx, weights),
        }
    }

    /// Build a gated FFN (LLaMA-style): down(act(gate(x)) * up(x))
    fn build_gated_ffn(
        &self,
        builder: &mut GraphBuilder,
        hidden_states: NodeIndex,
        layer_idx: u32,
        weights: &WeightMap,
    ) -> Result<NodeIndex> {
        let hidden_size = self.config.hidden_size as usize;
        let intermediate_size = self.config.intermediate_size as usize;

        // Get weight names
        let gate_weight_name = format!("model.layers.{}.mlp.gate_proj.weight", layer_idx);
        let up_weight_name = format!("model.layers.{}.mlp.up_proj.weight", layer_idx);
        let down_weight_name = format!("model.layers.{}.mlp.down_proj.weight", layer_idx);

        // Get weights
        let gate_weight = weights.get_required(&gate_weight_name)?;
        let up_weight = weights.get_required(&up_weight_name)?;
        let down_weight = weights.get_required(&down_weight_name)?;

        // Create weight constants
        let gate_proj = builder.constant(
            ConstantData::F32(gate_weight.to_f32_vec()),
            hologram::ir::Shape::new(vec![
                hologram::ir::Dim::Static(intermediate_size),
                hologram::ir::Dim::Static(hidden_size),
            ]),
        );
        let up_proj = builder.constant(
            ConstantData::F32(up_weight.to_f32_vec()),
            hologram::ir::Shape::new(vec![
                hologram::ir::Dim::Static(intermediate_size),
                hologram::ir::Dim::Static(hidden_size),
            ]),
        );
        let down_proj = builder.constant(
            ConstantData::F32(down_weight.to_f32_vec()),
            hologram::ir::Shape::new(vec![
                hologram::ir::Dim::Static(hidden_size),
                hologram::ir::Dim::Static(intermediate_size),
            ]),
        );

        // gate = hidden @ gate_proj.T
        let gate = builder.matmul(hidden_states, gate_proj)
            .map_err(|e| CommonError::GraphBuildError(format!("Gate projection failed: {:?}", e)))?;

        // Apply activation to gate
        let gate_activated = self.apply_activation(builder, gate)?;

        // up = hidden @ up_proj.T
        let up = builder.matmul(hidden_states, up_proj)
            .map_err(|e| CommonError::GraphBuildError(format!("Up projection failed: {:?}", e)))?;

        // gate_activated * up
        let gated = builder.mul(gate_activated, up)
            .map_err(|e| CommonError::GraphBuildError(format!("Gated multiply failed: {:?}", e)))?;

        // output = gated @ down_proj.T
        let output = builder.matmul(gated, down_proj)
            .map_err(|e| CommonError::GraphBuildError(format!("Down projection failed: {:?}", e)))?;

        Ok(output)
    }

    /// Build a standard FFN (GPT-style): down(act(up(x)))
    fn build_standard_ffn(
        &self,
        builder: &mut GraphBuilder,
        hidden_states: NodeIndex,
        layer_idx: u32,
        weights: &WeightMap,
    ) -> Result<NodeIndex> {
        let hidden_size = self.config.hidden_size as usize;
        let intermediate_size = self.config.intermediate_size as usize;

        // Get weight names (different naming for standard FFN)
        let up_weight_name = format!("model.layers.{}.mlp.fc1.weight", layer_idx);
        let down_weight_name = format!("model.layers.{}.mlp.fc2.weight", layer_idx);

        // Get weights
        let up_weight = weights.get_required(&up_weight_name)?;
        let down_weight = weights.get_required(&down_weight_name)?;

        // Create weight constants
        let up_proj = builder.constant(
            ConstantData::F32(up_weight.to_f32_vec()),
            hologram::ir::Shape::new(vec![
                hologram::ir::Dim::Static(intermediate_size),
                hologram::ir::Dim::Static(hidden_size),
            ]),
        );
        let down_proj = builder.constant(
            ConstantData::F32(down_weight.to_f32_vec()),
            hologram::ir::Shape::new(vec![
                hologram::ir::Dim::Static(hidden_size),
                hologram::ir::Dim::Static(intermediate_size),
            ]),
        );

        // up = hidden @ up_proj.T
        let up = builder.matmul(hidden_states, up_proj)
            .map_err(|e| CommonError::GraphBuildError(format!("Up projection failed: {:?}", e)))?;

        // Apply activation
        let up_activated = self.apply_activation(builder, up)?;

        // output = up_activated @ down_proj.T
        let output = builder.matmul(up_activated, down_proj)
            .map_err(|e| CommonError::GraphBuildError(format!("Down projection failed: {:?}", e)))?;

        Ok(output)
    }

    /// Apply the configured activation function.
    fn apply_activation(&self, builder: &mut GraphBuilder, input: NodeIndex) -> Result<NodeIndex> {
        match self.config.hidden_act {
            Activation::SiLU => {
                // SiLU(x) = x * sigmoid(x)
                let sigmoid = builder.sigmoid(input)
                    .map_err(|e| CommonError::GraphBuildError(format!("Sigmoid failed: {:?}", e)))?;
                builder.mul(input, sigmoid)
                    .map_err(|e| CommonError::GraphBuildError(format!("SiLU multiply failed: {:?}", e)))
            }
            Activation::GELU | Activation::GELUTanh => {
                builder.gelu(input)
                    .map_err(|e| CommonError::GraphBuildError(format!("GELU failed: {:?}", e)))
            }
            Activation::ReLU => {
                builder.relu(input)
                    .map_err(|e| CommonError::GraphBuildError(format!("ReLU failed: {:?}", e)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_builder_creation() {
        let config = TransformerConfig::default();
        let builder = FFNBuilder::new(&config);
        assert_eq!(builder.config.ffn_type, FFNType::Gated);
        assert_eq!(builder.config.hidden_act, Activation::SiLU);
    }

    #[test]
    fn test_ffn_type_gated() {
        let config = TransformerConfig {
            ffn_type: FFNType::Gated,
            ..Default::default()
        };
        assert_eq!(config.ffn_type, FFNType::Gated);
    }

    #[test]
    fn test_ffn_type_standard() {
        let config = TransformerConfig {
            ffn_type: FFNType::Standard,
            ..Default::default()
        };
        assert_eq!(config.ffn_type, FFNType::Standard);
    }

    #[test]
    fn test_activation_types() {
        assert_eq!(Activation::default(), Activation::SiLU);
    }
}
