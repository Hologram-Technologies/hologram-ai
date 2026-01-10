//! ConstantOfShape operation translator.

use hologram::ir::{GraphBuilder, NodeIndex, ConstantData, Shape, DType, NodeOp};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, OnnxAttributes, InputRequirement, TranslationError};

/// Translator for ONNX ConstantOfShape operation.
///
/// Creates a constant tensor filled with a specified value (default: 0.0f32)
/// using the shape from the input tensor.
///
/// # Inputs
/// - shape: 1D int64 tensor containing the output shape dimensions
///
/// # Attributes
/// - value: Optional 1-element tensor specifying the fill value (default: 0.0f32)
///
/// # Outputs
/// - output: Tensor of the specified shape filled with the specified value
#[derive(Debug, Default)]
pub struct ConstantOfShapeTranslator;

impl OnnxTranslator for ConstantOfShapeTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "ConstantOfShape"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(1)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let shape_input = inputs[0];

        // Get the shape input node
        let shape_node = builder.graph().node(shape_input).ok_or_else(|| {
            TranslationError::IrBuilder("ConstantOfShape: shape input not found".to_string())
        })?;

        // Parse fill value from attribute (default is 0.0 as float32)
        let (fill_value_data, _fill_dtype) = parse_fill_value(node);

        // Check if shape input is constant - if so, use constant folding (optimization)
        if let NodeOp::Constant { data } = &shape_node.op {
            let shape_dims = match data {
                ConstantData::I64(values) => {
                    values.iter().map(|&v| v as usize).collect::<Vec<_>>()
                }
                ConstantData::I32(values) => {
                    values.iter().map(|&v| v as usize).collect::<Vec<_>>()
                }
                _ => {
                    return Err(TranslationError::invalid_attribute(
                        "shape",
                        "shape must be int32 or int64 tensor",
                    ));
                }
            };

            // Calculate total number of elements
            let total_elements = shape_dims.iter().product::<usize>();

            // Create filled tensor
            let filled_data = match fill_value_data {
                ConstantData::F32(vals) if !vals.is_empty() => {
                    ConstantData::F32(vec![vals[0]; total_elements])
                }
                ConstantData::F64(vals) if !vals.is_empty() => {
                    ConstantData::F64(vec![vals[0]; total_elements])
                }
                ConstantData::I64(vals) if !vals.is_empty() => {
                    ConstantData::I64(vec![vals[0]; total_elements])
                }
                ConstantData::I32(vals) if !vals.is_empty() => {
                    ConstantData::I32(vec![vals[0]; total_elements])
                }
                ConstantData::Bool(vals) if !vals.is_empty() => {
                    ConstantData::Bool(vec![vals[0]; total_elements])
                }
                ConstantData::U8(vals) if !vals.is_empty() => {
                    ConstantData::U8(vec![vals[0]; total_elements])
                }
                _ => ConstantData::F32(vec![0.0; total_elements]),
            };

            let output_shape = Shape::static_shape(&shape_dims);
            let result = builder.constant(filled_data, output_shape);

            return Ok(vec![result]);
        }

        // Dynamic path - shape is computed at runtime
        // Create a constant node for the fill value (scalar)
        let fill_value_node = builder.constant(fill_value_data, Shape::static_shape(&[]));

        // Use dynamic ConstantOfShape operation
        let result = builder
            .constant_of_shape(shape_input, fill_value_node)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }
}

/// Parse the fill value from the node's "value" attribute.
///
/// Returns the fill value data and dtype. Defaults to 0.0f32 if no attribute.
fn parse_fill_value(node: &NodeProto) -> (ConstantData, DType) {
    use crate::proto::tensor_proto::DataType;

    if let Some(tensor) = node.get_tensor("value") {
        let data_type = DataType::try_from(tensor.data_type).ok();

        match data_type {
            Some(DataType::Float) => {
                let val = if !tensor.float_data.is_empty() {
                    tensor.float_data[0]
                } else if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 4 {
                    let bytes: [u8; 4] = tensor.raw_data[0..4].try_into().unwrap_or([0; 4]);
                    f32::from_le_bytes(bytes)
                } else {
                    0.0f32
                };
                (ConstantData::F32(vec![val]), DType::F32)
            }
            Some(DataType::Double) => {
                let val = if !tensor.double_data.is_empty() {
                    tensor.double_data[0]
                } else if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 8 {
                    let bytes: [u8; 8] = tensor.raw_data[0..8].try_into().unwrap_or([0; 8]);
                    f64::from_le_bytes(bytes)
                } else {
                    0.0f64
                };
                (ConstantData::F64(vec![val]), DType::F64)
            }
            Some(DataType::Int64) => {
                let val = if !tensor.int64_data.is_empty() {
                    tensor.int64_data[0]
                } else if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 8 {
                    let bytes: [u8; 8] = tensor.raw_data[0..8].try_into().unwrap_or([0; 8]);
                    i64::from_le_bytes(bytes)
                } else {
                    0i64
                };
                (ConstantData::I64(vec![val]), DType::I64)
            }
            Some(DataType::Int32) => {
                let val = if !tensor.int32_data.is_empty() {
                    tensor.int32_data[0]
                } else if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 4 {
                    let bytes: [u8; 4] = tensor.raw_data[0..4].try_into().unwrap_or([0; 4]);
                    i32::from_le_bytes(bytes)
                } else {
                    0i32
                };
                (ConstantData::I32(vec![val]), DType::I32)
            }
            Some(DataType::Bool) => {
                let val = if !tensor.int32_data.is_empty() {
                    tensor.int32_data[0] != 0
                } else if !tensor.raw_data.is_empty() {
                    tensor.raw_data[0] != 0
                } else {
                    false
                };
                (ConstantData::Bool(vec![val]), DType::Bool)
            }
            _ => (ConstantData::F32(vec![0.0]), DType::F32),
        }
    } else {
        // Default: 0.0 as float32
        (ConstantData::F32(vec![0.0]), DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{AttributeProto, TensorProto, attribute_proto::AttributeType};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "constant_of_shape_test".to_string(),
            op_type: "ConstantOfShape".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_fill_value(tensor: TensorProto) -> NodeProto {
        NodeProto {
            name: "constant_of_shape_test".to_string(),
            op_type: "ConstantOfShape".to_string(),
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                t: Some(tensor),
                r#type: AttributeType::Tensor as i32,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn make_f32_fill_tensor(value: f32) -> TensorProto {
        use crate::proto::tensor_proto::DataType;
        TensorProto {
            dims: vec![1],
            data_type: DataType::Float as i32,
            float_data: vec![value],
            ..Default::default()
        }
    }

    fn make_i64_fill_tensor(value: i64) -> TensorProto {
        use crate::proto::tensor_proto::DataType;
        TensorProto {
            dims: vec![1],
            data_type: DataType::Int64 as i32,
            int64_data: vec![value],
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_constant_of_shape_default_value() {
        let translator = ConstantOfShapeTranslator;
        let mut builder = GraphBuilder::new();

        // Create a constant shape tensor [2, 3]
        let shape = builder.constant(ConstantData::I64(vec![2, 3]), Shape::static_shape(&[2]));
        let node = make_node();

        let result = translator.translate(&node, &[shape], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);

        // Verify output is a 2x3 tensor filled with 0.0 (default)
        let output_node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(output_node.shape.rank(), 2);

        if let NodeOp::Constant { data } = &output_node.op {
            if let ConstantData::F32(values) = data {
                assert_eq!(values.len(), 6); // 2 * 3
                assert!(values.iter().all(|&v| v == 0.0));
            } else {
                panic!("Expected F32 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_constant_of_shape_custom_f32_value() {
        let translator = ConstantOfShapeTranslator;
        let mut builder = GraphBuilder::new();

        let shape = builder.constant(ConstantData::I64(vec![3]), Shape::static_shape(&[1]));
        let node = make_node_with_fill_value(make_f32_fill_tensor(5.0));

        let result = translator.translate(&node, &[shape], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();

        if let NodeOp::Constant { data } = &output_node.op {
            if let ConstantData::F32(values) = data {
                assert_eq!(values.len(), 3);
                assert!(values.iter().all(|&v| v == 5.0));
            } else {
                panic!("Expected F32 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_constant_of_shape_i64_value() {
        let translator = ConstantOfShapeTranslator;
        let mut builder = GraphBuilder::new();

        let shape = builder.constant(ConstantData::I64(vec![2, 2]), Shape::static_shape(&[2]));
        let node = make_node_with_fill_value(make_i64_fill_tensor(42));

        let result = translator.translate(&node, &[shape], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();

        if let NodeOp::Constant { data } = &output_node.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values.len(), 4);
                assert!(values.iter().all(|&v| v == 42));
            } else {
                panic!("Expected I64 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_constant_of_shape_scalar() {
        let translator = ConstantOfShapeTranslator;
        let mut builder = GraphBuilder::new();

        // Empty shape means scalar output
        let shape = builder.constant(ConstantData::I64(vec![]), Shape::static_shape(&[0]));
        let node = make_node_with_fill_value(make_f32_fill_tensor(1.0));

        let result = translator.translate(&node, &[shape], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();

        // Scalar has 1 element
        if let NodeOp::Constant { data } = &output_node.op {
            if let ConstantData::F32(values) = data {
                assert_eq!(values.len(), 1);
                assert_eq!(values[0], 1.0);
            } else {
                panic!("Expected F32 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_constant_of_shape_3d() {
        let translator = ConstantOfShapeTranslator;
        let mut builder = GraphBuilder::new();

        let shape = builder.constant(ConstantData::I64(vec![2, 3, 4]), Shape::static_shape(&[3]));
        let node = make_node();

        let result = translator.translate(&node, &[shape], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();

        assert_eq!(output_node.shape.rank(), 3);
        if let NodeOp::Constant { data } = &output_node.op {
            if let ConstantData::F32(values) = data {
                assert_eq!(values.len(), 24); // 2 * 3 * 4
            } else {
                panic!("Expected F32 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_constant_of_shape_no_inputs() {
        let translator = ConstantOfShapeTranslator;
        let err = translator.input_requirement().validate(0, "ConstantOfShape");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount { expected: 1, got: 0, .. }
        ));
    }

    #[test]
    fn test_constant_of_shape_too_many_inputs() {
        let translator = ConstantOfShapeTranslator;
        let err = translator.input_requirement().validate(2, "ConstantOfShape");
        assert!(err.is_err());
    }

    #[test]
    fn test_constant_of_shape_invalid_shape_dtype() {
        let translator = ConstantOfShapeTranslator;
        let mut builder = GraphBuilder::new();

        // Use F32 instead of I64 for shape (invalid)
        let shape = builder.constant(ConstantData::F32(vec![2.0, 3.0]), Shape::static_shape(&[2]));
        let node = make_node();

        let result = translator.translate(&node, &[shape], &mut builder);
        assert!(result.is_err());
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_constant_of_shape_op_type() {
        let translator = ConstantOfShapeTranslator;
        assert_eq!(translator.onnx_op_type(), "ConstantOfShape");
    }

    #[test]
    fn test_constant_of_shape_supports_folding() {
        let translator = ConstantOfShapeTranslator;
        assert!(translator.supports_constant_folding());
    }

    // ===== parse_fill_value Tests =====

    #[test]
    fn test_parse_fill_value_default() {
        let node = make_node();
        let (data, dtype) = parse_fill_value(&node);
        assert_eq!(dtype, DType::F32);
        if let ConstantData::F32(vals) = data {
            assert_eq!(vals, vec![0.0]);
        } else {
            panic!("Expected F32 data");
        }
    }

    #[test]
    fn test_parse_fill_value_custom_f32() {
        let node = make_node_with_fill_value(make_f32_fill_tensor(2.5));
        let (data, dtype) = parse_fill_value(&node);
        assert_eq!(dtype, DType::F32);
        if let ConstantData::F32(vals) = data {
            assert!((vals[0] - 2.5).abs() < 1e-6);
        } else {
            panic!("Expected F32 data");
        }
    }

    #[test]
    fn test_parse_fill_value_i64() {
        let node = make_node_with_fill_value(make_i64_fill_tensor(999));
        let (data, dtype) = parse_fill_value(&node);
        assert_eq!(dtype, DType::I64);
        if let ConstantData::I64(vals) = data {
            assert_eq!(vals, vec![999]);
        } else {
            panic!("Expected I64 data");
        }
    }
}
