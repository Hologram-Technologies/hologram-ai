//! Shape operation - returns tensor dimensions as a 1D constant.

use anyhow::{Context, Result};
use hologram::compiler::{ConstantData, DType};

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX Shape operation.
///
/// Returns the shape of the input tensor as a 1D int64 tensor.
/// This operation is always constant-foldable since tensor shapes
/// are known at compile time.
pub struct ShapeOp;

impl OpTranslator for ShapeOp {
    fn op_type(&self) -> &'static str {
        "Shape"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let input_name = node.input.first()?;
        let input_node = ctx.get_node(input_name)?;

        // Shape returns the input's dimensions as a 1D int64 tensor
        let shape_values: Vec<i64> = input_node.shape.iter().map(|&d| d as i64).collect();
        let output_shape = vec![shape_values.len()];

        tracing::debug!(
            "Shape '{}': input shape {:?} -> constant {:?}",
            node.output.first().unwrap_or(&String::new()),
            input_node.shape,
            shape_values
        );

        Some(TranslateResult::constant(
            output_shape,
            DType::I64,
            ConstantData::I64(shape_values),
        ))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        // Shape should always be folded - compute the constant directly
        let input_name = node.input.first().context("Shape has no input")?;
        let input_node = ctx.get_node(input_name).context("Shape input not found")?;

        // Shape returns the input's dimensions as a 1D int64 tensor
        let shape_values: Vec<i64> = input_node.shape.iter().map(|&d| d as i64).collect();
        let output_shape = vec![shape_values.len()];

        Ok(TranslateResult::constant(
            output_shape,
            DType::I64,
            ConstantData::I64(shape_values),
        ))
    }
}

/// ONNX ConstantOfShape operation.
///
/// Creates a tensor with a given shape filled with a constant value.
/// The shape is provided as input (1D tensor of dimensions).
/// The value is provided as an attribute.
pub struct ConstantOfShapeOp;

impl OpTranslator for ConstantOfShapeOp {
    fn op_type(&self) -> &'static str {
        "ConstantOfShape"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        // Get the shape from the constant input
        let shape_name = node.input.first()?;
        let shape_values = ctx.get_constant_i64(shape_name)?;
        let output_shape: Vec<usize> = shape_values.iter().map(|&d| d as usize).collect();

        // Get the fill value from attribute (defaults to 0.0f32)
        let (fill_value, dtype) = get_fill_value(node);

        // Create the constant data
        let total_size: usize = output_shape.iter().product();
        let const_data = create_constant_data(fill_value, total_size, dtype);

        tracing::debug!(
            "ConstantOfShape '{}': shape {:?}, value {}",
            node.output.first().unwrap_or(&String::new()),
            output_shape,
            fill_value
        );

        Some(TranslateResult::constant(output_shape, dtype, const_data))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        // ConstantOfShape should always be foldable since its shape input comes from Shape op
        // which is always constant
        let shape_name = node
            .input
            .first()
            .context("ConstantOfShape has no shape input")?;

        // Get shape from constant input (required)
        let shape_values = ctx
            .get_constant_i64(shape_name)
            .context("ConstantOfShape: shape input must be a constant")?;

        let output_shape: Vec<usize> = shape_values.iter().map(|&d| d as usize).collect();
        let (fill_value, dtype) = get_fill_value(node);
        let total_size: usize = output_shape.iter().product();
        let const_data = create_constant_data(fill_value, total_size, dtype);

        Ok(TranslateResult::constant(output_shape, dtype, const_data))
    }
}

/// Get fill value and dtype from ConstantOfShape node attributes.
fn get_fill_value(node: &proto::NodeProto) -> (f64, DType) {
    // Find the 'value' attribute (TensorProto)
    for attr in &node.attribute {
        if attr.name == "value"
            && let Some(tensor) = &attr.t
        {
            match tensor.data_type {
                1 => {
                    // F32
                    let val = if !tensor.float_data.is_empty() {
                        tensor.float_data[0] as f64
                    } else if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 4 {
                        let bytes: [u8; 4] = tensor.raw_data[..4].try_into().unwrap_or([0; 4]);
                        f32::from_le_bytes(bytes) as f64
                    } else {
                        0.0
                    };
                    return (val, DType::F32);
                }
                7 => {
                    // I64
                    let val = if !tensor.int64_data.is_empty() {
                        tensor.int64_data[0] as f64
                    } else if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 8 {
                        let bytes: [u8; 8] = tensor.raw_data[..8].try_into().unwrap_or([0; 8]);
                        i64::from_le_bytes(bytes) as f64
                    } else {
                        0.0
                    };
                    return (val, DType::I64);
                }
                6 => {
                    // I32
                    let val = if !tensor.int32_data.is_empty() {
                        tensor.int32_data[0] as f64
                    } else if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 4 {
                        let bytes: [u8; 4] = tensor.raw_data[..4].try_into().unwrap_or([0; 4]);
                        i32::from_le_bytes(bytes) as f64
                    } else {
                        0.0
                    };
                    return (val, DType::I32);
                }
                _ => {}
            }
        }
    }
    // Default: 0.0f32
    (0.0, DType::F32)
}

/// Create constant data filled with a value.
fn create_constant_data(value: f64, size: usize, dtype: DType) -> ConstantData {
    match dtype {
        DType::F32 => ConstantData::F32(vec![value as f32; size]),
        DType::F64 => ConstantData::F64(vec![value; size]),
        DType::I32 => ConstantData::I32(vec![value as i32; size]),
        DType::I64 => ConstantData::I64(vec![value as i64; size]),
        _ => ConstantData::F32(vec![value as f32; size]), // Fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::compiler::{OpKind, OperationGraph};
    use std::collections::HashMap;

    #[test]
    fn test_shape_constant_fold() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Add input with shape [1, 5, 512]
        let input = hologram::compiler::OpNode::new(0, OpKind::Input, vec![1, 5, 512], DType::F32)
            .with_name("input".to_string());
        graph.nodes.push(input);
        value_to_node.insert("input".to_string(), 0);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["shape_out".to_string()],
            op_type: "Shape".to_string(),
            ..Default::default()
        };

        let op = ShapeOp;
        let result = op.try_fold(&node, &ctx).expect("Should fold");

        assert_eq!(result.shape, vec![3]); // 3D input -> shape tensor of length 3
        assert_eq!(result.dtype, DType::I64);

        if let Some(ConstantData::I64(data)) = result.constant_data {
            assert_eq!(data, vec![1, 5, 512]);
        } else {
            panic!("Expected I64 constant data");
        }
    }
}
