//! Main ONNX operation translator and dispatcher.
//!
//! This module provides the central translation function that dispatches
//! ONNX operations to their specific translators.

use hologram_compiler::ir::{IRBuilder, NodeId};
use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use std::collections::HashMap;
use tracing::{debug, trace};

use crate::ops::{
    activation::*, advanced::*, conv::*, core::*, norm::*, pool::*, reduction::*, shape::*,
    unary::*,
};

/// Trait for operation translators.
///
/// Implement this trait to add support for new ONNX operations.
pub trait OpTranslator {
    /// Translate ONNX operation to hologram IR.
    fn translate(
        &self,
        inputs: &[NodeId],
        attrs: &[AttributeProto],
        input_shapes: &HashMap<String, SymbolicShape>,
        builder: &mut IRBuilder,
    ) -> Result<NodeId>;

    /// Infer output shape for this operation.
    fn infer_shape(
        &self,
        input_shapes: &[&SymbolicShape],
        attrs: &[AttributeProto],
    ) -> Result<SymbolicShape>;
}

/// Translate ONNX operation to hologram IR.
///
/// This is the main entry point for operation translation. It dispatches
/// to specific operation translators based on the operation type.
///
/// # Performance
///
/// - **O(1) dispatch** via match statement (compiler optimizes to jump table)
/// - All translators use **zero-copy** operations where possible
/// - Symbolic shapes enable **compile-time** shape inference
///
/// # Arguments
///
/// * `op_type` - ONNX operation type (e.g., "MatMul", "Add", "ReLU")
/// * `inputs` - Input node IDs from the IR builder
/// * `attrs` - ONNX operation attributes
/// * `shapes` - Symbolic shapes for all named tensors
/// * `builder` - IR builder for creating new nodes
///
/// # Returns
///
/// Returns the IR node ID for the operation output.
///
/// # Errors
///
/// Returns `OnnxError::UnsupportedOp` if the operation is not supported.
pub fn translate_onnx_op(
    op_type: &str,
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    debug!(
        "Translating ONNX op: {} with {} inputs",
        op_type,
        inputs.len()
    );
    trace!("Operation attributes: {} attrs", attrs.len());

    // Dispatch to specific translator based on operation type
    // NOTE: This match is compiled to a jump table for O(1) dispatch
    match op_type {
        // Core operations
        "MatMul" => translate_matmul(inputs, attrs, shapes, builder),
        "Gemm" => translate_gemm(inputs, attrs, shapes, builder),
        "Add" => translate_add(inputs, attrs, shapes, builder),
        "Sub" => translate_sub(inputs, attrs, shapes, builder),
        "Mul" => translate_mul(inputs, attrs, shapes, builder),
        "Div" => translate_div(inputs, attrs, shapes, builder),
        "Pow" => translate_pow(inputs, attrs, shapes, builder),
        "Cast" => translate_cast(inputs, attrs, shapes, builder),

        // Activation functions
        "Relu" => translate_relu(inputs, attrs, shapes, builder),
        "Sigmoid" => translate_sigmoid(inputs, attrs, shapes, builder),
        "Tanh" => translate_tanh(inputs, attrs, shapes, builder),
        "Softmax" => translate_softmax(inputs, attrs, shapes, builder),

        // Advanced activation functions
        "Gelu" => translate_gelu(inputs, attrs, shapes, builder),
        "Swish" => translate_swish(inputs, attrs, shapes, builder),
        "Elu" => translate_elu(inputs, attrs, shapes, builder),
        "Selu" => translate_selu(inputs, attrs, shapes, builder),

        // Shape operations
        "Reshape" => translate_reshape(inputs, attrs, shapes, builder),
        "Transpose" => translate_transpose(inputs, attrs, shapes, builder),
        "Squeeze" => translate_squeeze(inputs, attrs, shapes, builder),
        "Unsqueeze" => translate_unsqueeze(inputs, attrs, shapes, builder),
        "Concat" => translate_concat(inputs, attrs, shapes, builder),
        "Split" => translate_split(inputs, attrs, shapes, builder),
        "Flatten" => translate_flatten(inputs, attrs, shapes, builder),

        // Convolution operations
        "Conv" => translate_conv(inputs, attrs, shapes, builder),
        "ConvTranspose" => translate_conv_transpose(inputs, attrs, shapes, builder),

        // Normalization operations
        "BatchNormalization" => translate_batch_normalization(inputs, attrs, shapes, builder),
        "LayerNormalization" => translate_layer_normalization(inputs, attrs, shapes, builder),
        "InstanceNormalization" => translate_instance_normalization(inputs, attrs, shapes, builder),
        "GroupNormalization" => translate_group_normalization(inputs, attrs, shapes, builder),

        // Pooling operations
        "MaxPool" => translate_max_pool(inputs, attrs, shapes, builder),
        "AveragePool" => translate_average_pool(inputs, attrs, shapes, builder),
        "GlobalAveragePool" => translate_global_average_pool(inputs, attrs, shapes, builder),

        // Reduction operations
        "ReduceSum" => translate_reduce_sum(inputs, attrs, shapes, builder),
        "ReduceMean" => translate_reduce_mean(inputs, attrs, shapes, builder),
        "ReduceMax" => translate_reduce_max(inputs, attrs, shapes, builder),
        "ReduceMin" => translate_reduce_min(inputs, attrs, shapes, builder),
        "ReduceProd" => translate_reduce_prod(inputs, attrs, shapes, builder),

        // Advanced operations
        "Attention" => translate_attention(inputs, attrs, shapes, builder),
        "MultiHeadAttention" => translate_multi_head_attention(inputs, attrs, shapes, builder),
        "LSTM" => translate_lstm(inputs, attrs, shapes, builder),
        "GRU" => translate_gru(inputs, attrs, shapes, builder),
        "RNN" => translate_rnn(inputs, attrs, shapes, builder),

        // Unary operations
        "Sqrt" => translate_sqrt(inputs, attrs, shapes, builder),
        "Exp" => translate_exp(inputs, attrs, shapes, builder),
        "Log" => translate_log(inputs, attrs, shapes, builder),
        "Neg" => translate_neg(inputs, attrs, shapes, builder),
        "Abs" => translate_abs(inputs, attrs, shapes, builder),
        "Reciprocal" => translate_reciprocal(inputs, attrs, shapes, builder),

        // Unsupported operation
        _ => {
            debug!("Unsupported ONNX operation: {}", op_type);
            Err(OnnxError::unsupported_op(op_type, 13)) // Opset 13 is common
        }
    }
}

/// Infer output shape for ONNX operation.
///
/// This function performs symbolic shape inference for ONNX operations.
/// It's called during graph translation to propagate shape information.
///
/// # Performance
///
/// - **O(1) dispatch** via match statement
/// - All shape inference is **compile-time** (no runtime overhead)
/// - Supports **symbolic dimensions** (Dim::Var, Dim::Expr)
///
/// # Arguments
///
/// * `op_type` - ONNX operation type
/// * `input_shapes` - Input shapes (may contain symbolic dimensions)
/// * `attrs` - ONNX operation attributes
///
/// # Returns
///
/// Returns the inferred output shape (may contain symbolic dimensions).
///
/// # Errors
///
/// Returns `OnnxError::ShapeInferenceError` if shape inference fails.
pub fn infer_op_output_shape(
    op_type: &str,
    input_shapes: &[&SymbolicShape],
    attrs: &[AttributeProto],
) -> Result<SymbolicShape> {
    trace!("Inferring output shape for op: {}", op_type);

    // Dispatch to operation-specific shape inference
    match op_type {
        // Core operations
        "MatMul" => {
            if input_shapes.len() != 2 {
                return Err(OnnxError::ShapeInferenceError(format!(
                    "MatMul expects 2 inputs, got {}",
                    input_shapes.len()
                )));
            }
            input_shapes[0].infer_matmul(input_shapes[1])
        }

        "Gemm" => {
            // Gemm: Y = alpha * A @ B + beta * C
            // Output shape is same as MatMul(A, B)
            if input_shapes.len() < 2 {
                return Err(OnnxError::ShapeInferenceError(format!(
                    "Gemm expects at least 2 inputs, got {}",
                    input_shapes.len()
                )));
            }
            input_shapes[0].infer_matmul(input_shapes[1])
        }

        // Binary operations (broadcasting)
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
            if input_shapes.len() != 2 {
                return Err(OnnxError::ShapeInferenceError(format!(
                    "{} expects 2 inputs, got {}",
                    op_type,
                    input_shapes.len()
                )));
            }
            input_shapes[0].infer_binary_op(input_shapes[1])
        }

        // Unary operations (shape unchanged)
        "Relu" | "Sigmoid" | "Tanh" => {
            if input_shapes.is_empty() {
                return Err(OnnxError::ShapeInferenceError(format!(
                    "{} expects 1 input, got 0",
                    op_type
                )));
            }
            Ok(input_shapes[0].clone())
        }

        "Softmax" => {
            // Softmax output shape same as input
            if input_shapes.is_empty() {
                return Err(OnnxError::ShapeInferenceError(
                    "Softmax expects 1 input, got 0".to_string(),
                ));
            }
            Ok(input_shapes[0].clone())
        }

        "Transpose" => {
            if input_shapes.is_empty() {
                return Err(OnnxError::ShapeInferenceError(
                    "Transpose expects 1 input, got 0".to_string(),
                ));
            }
            // Parse perm attribute
            use crate::utils::parse_attr_ints;
            let perm = parse_attr_ints(attrs, "perm", vec![])?;
            let perm_opt = if perm.is_empty() {
                None
            } else {
                Some(perm.as_slice())
            };
            input_shapes[0].infer_transpose(perm_opt)
        }

        "Reshape" => {
            // Reshape output shape is specified in attributes or second input
            // For now, return error as we need more context
            Err(OnnxError::ShapeInferenceError(
                "Reshape shape inference requires target shape".to_string(),
            ))
        }

        "Squeeze" | "Unsqueeze" => {
            // These modify dimensions, but we need attribute context
            Err(OnnxError::ShapeInferenceError(format!(
                "{} shape inference requires axes attribute",
                op_type
            )))
        }

        "Concat" => {
            // Concat needs axis and all input shapes
            Err(OnnxError::ShapeInferenceError(
                "Concat shape inference requires axis and all inputs".to_string(),
            ))
        }

        "Split" => {
            // Split produces multiple outputs
            Err(OnnxError::ShapeInferenceError(
                "Split has multiple outputs, use operation-specific inference".to_string(),
            ))
        }

        _ => Err(OnnxError::unsupported_op(op_type, 13)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_compiler::shapes::Dim;

    #[test]
    fn test_infer_binary_op_concrete() {
        let shape1 = SymbolicShape::concrete(vec![2, 3, 4]);
        let shape2 = SymbolicShape::concrete(vec![2, 3, 4]);

        let result = infer_op_output_shape("Add", &[&shape1, &shape2], &[]).unwrap();

        assert_eq!(
            result.dims(),
            &[Dim::Concrete(2), Dim::Concrete(3), Dim::Concrete(4)]
        );
    }

    #[test]
    fn test_infer_matmul_concrete() {
        let shape1 = SymbolicShape::concrete(vec![2, 3]);
        let shape2 = SymbolicShape::concrete(vec![3, 4]);

        let result = infer_op_output_shape("MatMul", &[&shape1, &shape2], &[]).unwrap();

        assert_eq!(result.dims(), &[Dim::Concrete(2), Dim::Concrete(4)]);
    }

    #[test]
    fn test_infer_unary_op() {
        let shape = SymbolicShape::concrete(vec![1, 2, 3, 4]);

        for op in &["Relu", "Sigmoid", "Tanh"] {
            let result = infer_op_output_shape(op, &[&shape], &[]).unwrap();

            assert_eq!(result.dims(), shape.dims());
        }
    }

    #[test]
    fn test_infer_symbolic_shapes() {
        let shape1 = SymbolicShape::symbolic(vec!["batch", "seq_len", "hidden"]);
        let shape2 = SymbolicShape::concrete(vec![1, 1, 512]);

        let result = infer_op_output_shape("Add", &[&shape1, &shape2], &[]).unwrap();

        // Result should preserve symbolic dimensions
        assert!(matches!(result.dims()[0], Dim::Var(_)));
    }

    #[test]
    fn test_unsupported_op() {
        let shape = SymbolicShape::concrete(vec![1, 2, 3]);

        let result = infer_op_output_shape("UnsupportedOp", &[&shape], &[]);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OnnxError::UnsupportedOp { .. }
        ));
    }

    #[test]
    fn test_wrong_input_count() {
        let shape = SymbolicShape::concrete(vec![2, 3]);

        // Add expects 2 inputs
        let result = infer_op_output_shape("Add", &[&shape], &[]);
        assert!(result.is_err());

        // MatMul expects 2 inputs
        let result = infer_op_output_shape("MatMul", &[], &[]);
        assert!(result.is_err());
    }
}
