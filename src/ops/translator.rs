//! Main ONNX operation dispatcher.
//!
//! This module provides the central translation function that dispatches
//! ONNX operations to their specific translators based on the operation type.

use hologram_ir::{GraphBuilder, NodeIndex};
use crate::core::{OnnxError, Result};
use crate::proto::NodeProto;
use std::collections::HashMap;
use tracing::{trace, warn};

/// Translate a single ONNX node to IR operations.
///
/// This is the main dispatcher that routes operations to their specific translators
/// based on the `op_type` field of the ONNX node.
///
/// # Arguments
///
/// * `node` - ONNX node to translate
/// * `builder` - IR graph builder
/// * `value_map` - Map from ONNX value names to IR node indices
///
/// # Returns
///
/// Vector of output node indices (one per output of the ONNX node)
///
/// # Errors
///
/// Returns error if:
/// - Operation is not supported
/// - Required inputs are missing
/// - Attribute parsing fails
/// - Shape inference fails
pub fn translate_onnx_node(
    node: &NodeProto,
    builder: &mut GraphBuilder,
    value_map: &mut HashMap<String, NodeIndex>,
) -> Result<Vec<NodeIndex>> {
    // Get input node indices
    let inputs: Result<Vec<NodeIndex>> = node
        .input
        .iter()
        .map(|input_name| {
            if input_name.is_empty() {
                // Optional input not provided
                Err(OnnxError::MissingInput(format!("Empty input name in node '{}'", node.name)))
            } else {
                value_map.get(input_name).copied().ok_or_else(|| {
                    OnnxError::MissingInput(format!(
                        "Input '{}' not found for node '{}' ({})",
                        input_name, node.name, node.op_type
                    ))
                })
            }
        })
        .collect();

    let inputs = inputs?;
    trace!("Translating {} with {} inputs", node.op_type, inputs.len());

    // Dispatch to operation-specific translator
    let output = match node.op_type.as_str() {
        // ===== CORE ARITHMETIC =====
        "Add" => {
            let result = builder.add(inputs[0], inputs[1])?;
            vec![result]
        }
        "Sub" => {
            let result = builder.sub(inputs[0], inputs[1])?;
            vec![result]
        }
        "Mul" => {
            let result = builder.mul(inputs[0], inputs[1])?;
            vec![result]
        }
        "Div" => {
            let result = builder.div(inputs[0], inputs[1])?;
            vec![result]
        }
        "MatMul" => {
            let result = builder.matmul(inputs[0], inputs[1])?;
            vec![result]
        }
        "Gemm" => crate::ops::core::translate_gemm(&inputs, &node.attribute, builder)?,
        "Pow" => crate::ops::core::translate_pow(&inputs, &node.attribute, builder)?,

        // ===== SHAPE OPERATIONS =====
        "Reshape" => crate::ops::shape::translate_reshape(&inputs, &node.attribute, builder)?,
        "Transpose" => crate::ops::shape::translate_transpose(&inputs, &node.attribute, builder)?,
        "Concat" => crate::ops::shape::translate_concat(&inputs, &node.attribute, builder)?,
        "Squeeze" => crate::ops::shape::translate_squeeze(&inputs, &node.attribute, builder)?,
        "Unsqueeze" => crate::ops::shape::translate_unsqueeze(&inputs, &node.attribute, builder)?,
        "Flatten" => crate::ops::shape::translate_flatten(&inputs, &node.attribute, builder)?,
        "Expand" => crate::ops::shape::translate_expand(&inputs, &node.attribute, builder)?,
        "Split" => crate::ops::shape::translate_split(&inputs, &node.attribute, builder)?,

        // ===== ACTIVATION FUNCTIONS =====
        "Relu" => {
            let result = builder.relu(inputs[0])?;
            vec![result]
        }
        "Sigmoid" => {
            let result = builder.sigmoid(inputs[0])?;
            vec![result]
        }
        "Tanh" => {
            let result = builder.tanh(inputs[0])?;
            vec![result]
        }
        "Gelu" => {
            let result = builder.gelu(inputs[0])?;
            vec![result]
        }
        "Softmax" => crate::ops::activation::translate_softmax(&inputs, &node.attribute, builder)?,
        "Clip" => crate::ops::activation::translate_clip(&inputs, &node.attribute, builder)?,
        "LeakyRelu" => crate::ops::activation::translate_leaky_relu(&inputs, &node.attribute, builder)?,
        "Elu" => crate::ops::activation::translate_elu(&inputs, &node.attribute, builder)?,
        "Selu" => crate::ops::activation::translate_selu(&inputs, &node.attribute, builder)?,
        "PRelu" => crate::ops::activation::translate_prelu(&inputs, &node.attribute, builder)?,
        "Swish" => crate::ops::activation::translate_swish(&inputs, &node.attribute, builder)?,
        "Erf" => {
            let result = builder.erf(inputs[0])?;
            vec![result]
        }

        // ===== NORMALIZATION =====
        "LayerNormalization" | "LayerNorm" => crate::ops::norm::translate_layer_norm(&inputs, &node.attribute, builder)?,
        "BatchNormalization" => crate::ops::norm::translate_batch_norm(&inputs, &node.attribute, builder)?,
        "GroupNormalization" => crate::ops::norm::translate_group_norm(&inputs, &node.attribute, builder)?,
        "InstanceNormalization" => crate::ops::norm::translate_instance_norm(&inputs, &node.attribute, builder)?,

        // ===== REDUCTION =====
        "ReduceMean" => crate::ops::reduction::translate_reduce_mean(&inputs, &node.attribute, builder)?,
        "ReduceSum" => crate::ops::reduction::translate_reduce_sum(&inputs, &node.attribute, builder)?,
        "ReduceMax" => crate::ops::reduction::translate_reduce_max(&inputs, &node.attribute, builder)?,
        "ReduceMin" => crate::ops::reduction::translate_reduce_min(&inputs, &node.attribute, builder)?,
        "ReduceProd" => crate::ops::reduction::translate_reduce_prod(&inputs, &node.attribute, builder)?,


        // ===== UNARY OPERATIONS =====
        "Sqrt" => crate::ops::unary::translate_sqrt(&inputs, builder)?,
        "Exp" => crate::ops::unary::translate_exp(&inputs, builder)?,
        "Log" => crate::ops::unary::translate_log(&inputs, builder)?,
        "Abs" => crate::ops::unary::translate_abs(&inputs, builder)?,
        "Neg" => {
            let result = builder.neg(inputs[0])?;
            vec![result]
        }
        "Reciprocal" => crate::ops::unary::translate_reciprocal(&inputs, builder)?,
        "Sin" => crate::ops::unary::translate_sin(&inputs, builder)?,
        "Cos" => crate::ops::unary::translate_cos(&inputs, builder)?,
        "Tan" => crate::ops::unary::translate_tan(&inputs, builder)?,

        // ===== LOGICAL =====
        "Equal" => crate::ops::logical::translate_equal(&inputs, builder)?,
        "Greater" => crate::ops::logical::translate_greater(&inputs, builder)?,
        "Less" => crate::ops::logical::translate_less(&inputs, builder)?,
        "And" => crate::ops::logical::translate_and(&inputs, builder)?,
        "Or" => crate::ops::logical::translate_or(&inputs, builder)?,
        "Not" => crate::ops::logical::translate_not(&inputs, builder)?,
        "Where" => crate::ops::logical::translate_where(&inputs, builder)?,

        // ===== ADVANCED =====
        "Cast" => crate::ops::advanced::translate_cast(&inputs, &node.attribute, builder)?,
        "Identity" => {
            // Identity is a no-op, just return input
            vec![inputs[0]]
        }
        "Dropout" => {
            // During inference, Dropout is identity
            // Return input and mask (mask is all ones, but we skip it)
            vec![inputs[0]]
        }

        // ===== UNSUPPORTED =====
        _ => {
            warn!("Unsupported ONNX operation: {}", node.op_type);
            return Err(OnnxError::unsupported_op(&node.op_type, 13)); // Default opset 13
        }
    };

    trace!("Translated {} -> {} outputs", node.op_type, output.len());
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            name: format!("{}_node", op_type),
            op_type: op_type.to_string(),
            input: inputs.into_iter().map(|s| s.to_string()).collect(),
            output: outputs.into_iter().map(|s| s.to_string()).collect(),
            attribute: vec![],
            ..Default::default()
        }
    }

    #[test]
    fn test_translate_add() {
        let mut builder = GraphBuilder::new();
        let mut value_map = HashMap::new();

        // Create inputs
        let a = builder.input("a", hologram_ir::Shape::static_shape(&[2, 3]), hologram_ir::DType::F32);
        let b = builder.input("b", hologram_ir::Shape::static_shape(&[2, 3]), hologram_ir::DType::F32);
        value_map.insert("a".to_string(), a);
        value_map.insert("b".to_string(), b);

        // Translate Add
        let node = make_node("Add", vec!["a", "b"], vec!["c"]);
        let result = translate_onnx_node(&node, &mut builder, &mut value_map);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_relu() {
        let mut builder = GraphBuilder::new();
        let mut value_map = HashMap::new();

        let x = builder.input("x", hologram_ir::Shape::static_shape(&[1, 10]), hologram_ir::DType::F32);
        value_map.insert("x".to_string(), x);

        let node = make_node("Relu", vec!["x"], vec!["y"]);
        let result = translate_onnx_node(&node, &mut builder, &mut value_map);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_identity() {
        let mut builder = GraphBuilder::new();
        let mut value_map = HashMap::new();

        let x = builder.input("x", hologram_ir::Shape::static_shape(&[1, 10]), hologram_ir::DType::F32);
        value_map.insert("x".to_string(), x);

        let node = make_node("Identity", vec!["x"], vec!["y"]);
        let result = translate_onnx_node(&node, &mut builder, &mut value_map);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], x); // Identity returns same node
    }

    #[test]
    fn test_translate_unsupported() {
        let mut builder = GraphBuilder::new();
        let mut value_map = HashMap::new();

        let x = builder.input("x", hologram_ir::Shape::static_shape(&[1, 10]), hologram_ir::DType::F32);
        value_map.insert("x".to_string(), x);

        let node = make_node("UnsupportedOp", vec!["x"], vec!["y"]);
        let result = translate_onnx_node(&node, &mut builder, &mut value_map);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_unsupported_op());
    }

    #[test]
    fn test_translate_missing_input() {
        let mut builder = GraphBuilder::new();
        let mut value_map = HashMap::new();

        let node = make_node("Add", vec!["missing_a", "missing_b"], vec!["c"]);
        let result = translate_onnx_node(&node, &mut builder, &mut value_map);
        assert!(result.is_err());
    }
}
