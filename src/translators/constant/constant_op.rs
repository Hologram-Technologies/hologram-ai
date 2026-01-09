//! Constant operation translator.

use hologram::ir::{GraphBuilder, NodeIndex, ConstantData, Shape, DType};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, OnnxAttributes, InputRequirement, TranslationError};

/// Translator for ONNX Constant operation.
///
/// Creates a constant tensor from the value attribute stored in the node.
/// Supports multiple data types including F32, F64, I32, I64, Bool, and U8.
#[derive(Debug, Default)]
pub struct ConstantTranslator;

impl OnnxTranslator for ConstantTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Constant"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(0)
    }

    fn translate(
        &self,
        node: &NodeProto,
        _inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        // Get value tensor from attributes
        let tensor = node.get_tensor("value").ok_or_else(|| {
            TranslationError::missing_attribute("Constant", "value")
        })?;

        // Extract constant data from tensor
        let (constant_data, _dtype, shape) = extract_constant_from_tensor(tensor)?;

        // Create constant node (dtype is inferred from ConstantData)
        let result = builder.constant(constant_data, shape);

        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        // Constant is already a constant, no folding needed
        false
    }
}

/// Extract constant data from ONNX TensorProto.
///
/// Converts ONNX tensor data to hologram-ir ConstantData format.
///
/// # Arguments
///
/// * `tensor` - ONNX TensorProto
///
/// # Returns
///
/// Tuple of (ConstantData, DType, Shape)
///
/// # Errors
///
/// Returns error if data type is not supported or extraction fails
fn extract_constant_from_tensor(
    tensor: &crate::proto::TensorProto,
) -> Result<(ConstantData, DType, Shape), TranslationError> {
    use crate::proto::tensor_proto::DataType;

    // Extract shape
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    let shape = Shape::static_shape(&dims);

    // Get data type
    let data_type = DataType::try_from(tensor.data_type)
        .map_err(|_| TranslationError::invalid_attribute("value", "unknown data type"))?;

    let (constant_data, dtype) = match data_type {
        DataType::Float => {
            let data = if !tensor.raw_data.is_empty() {
                bytemuck::cast_slice(&tensor.raw_data).to_vec()
            } else {
                tensor.float_data.clone()
            };
            (ConstantData::F32(data), DType::F32)
        }

        DataType::Float16 => {
            // Convert f16 to f32
            let data = if !tensor.raw_data.is_empty() {
                let u16_data: &[u16] = bytemuck::cast_slice(&tensor.raw_data);
                u16_data
                    .iter()
                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                    .collect()
            } else {
                tensor
                    .int32_data
                    .iter()
                    .map(|&bits| half::f16::from_bits(bits as u16).to_f32())
                    .collect()
            };
            (ConstantData::F32(data), DType::F32)
        }

        DataType::Double => {
            let data = if !tensor.raw_data.is_empty() {
                bytemuck::cast_slice(&tensor.raw_data).to_vec()
            } else {
                tensor.double_data.clone()
            };
            (ConstantData::F64(data), DType::F64)
        }

        DataType::Int32 => {
            let data = if !tensor.raw_data.is_empty() {
                bytemuck::cast_slice(&tensor.raw_data).to_vec()
            } else {
                tensor.int32_data.clone()
            };
            (ConstantData::I32(data), DType::I32)
        }

        DataType::Int64 => {
            let data = if !tensor.raw_data.is_empty() {
                bytemuck::cast_slice(&tensor.raw_data).to_vec()
            } else {
                tensor.int64_data.clone()
            };
            (ConstantData::I64(data), DType::I64)
        }

        DataType::Uint8 => {
            let data = if !tensor.raw_data.is_empty() {
                tensor.raw_data.clone()
            } else {
                tensor.int32_data.iter().map(|&x| x as u8).collect()
            };
            (ConstantData::U8(data), DType::U8)
        }

        DataType::Int8 => {
            // INT8 stored as U8 in hologram-ir
            let data = if !tensor.raw_data.is_empty() {
                tensor.raw_data.clone()
            } else {
                tensor.int32_data.iter().map(|&x| x as u8).collect()
            };
            (ConstantData::U8(data), DType::I8)
        }

        DataType::Bool => {
            let data = if !tensor.raw_data.is_empty() {
                tensor.raw_data.iter().map(|&b| b != 0).collect()
            } else {
                tensor.int32_data.iter().map(|&x| x != 0).collect()
            };
            (ConstantData::Bool(data), DType::Bool)
        }

        _ => {
            return Err(TranslationError::invalid_attribute(
                "value",
                format!("unsupported data type: {:?}", data_type),
            ));
        }
    };

    Ok((constant_data, dtype, shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{AttributeProto, TensorProto, attribute_proto::AttributeType};

    fn make_node_with_tensor(tensor: TensorProto) -> NodeProto {
        NodeProto {
            name: "constant_test".to_string(),
            op_type: "Constant".to_string(),
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                t: Some(tensor),
                r#type: AttributeType::Tensor as i32,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn make_f32_tensor(dims: Vec<i64>, data: Vec<f32>) -> TensorProto {
        use crate::proto::tensor_proto::DataType;
        TensorProto {
            dims,
            data_type: DataType::Float as i32,
            float_data: data,
            ..Default::default()
        }
    }

    fn make_i64_tensor(dims: Vec<i64>, data: Vec<i64>) -> TensorProto {
        use crate::proto::tensor_proto::DataType;
        TensorProto {
            dims,
            data_type: DataType::Int64 as i32,
            int64_data: data,
            ..Default::default()
        }
    }

    fn make_bool_tensor(dims: Vec<i64>, data: Vec<i32>) -> TensorProto {
        use crate::proto::tensor_proto::DataType;
        TensorProto {
            dims,
            data_type: DataType::Bool as i32,
            int32_data: data,
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_constant_f32_2d() {
        let translator = ConstantTranslator;
        let mut builder = GraphBuilder::new();

        let tensor = make_f32_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let node = make_node_with_tensor(tensor);

        let result = translator.translate(&node, &[], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);

        // Verify output shape
        let output_node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(output_node.shape.rank(), 2);
        assert_eq!(output_node.dtype, DType::F32);
    }

    #[test]
    fn test_constant_i64_1d() {
        let translator = ConstantTranslator;
        let mut builder = GraphBuilder::new();

        let tensor = make_i64_tensor(vec![3], vec![100, 200, 300]);
        let node = make_node_with_tensor(tensor);

        let result = translator.translate(&node, &[], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(output_node.shape.rank(), 1);
        assert_eq!(output_node.dtype, DType::I64);
    }

    #[test]
    fn test_constant_scalar() {
        let translator = ConstantTranslator;
        let mut builder = GraphBuilder::new();

        let tensor = make_f32_tensor(vec![], vec![42.0]);
        let node = make_node_with_tensor(tensor);

        let result = translator.translate(&node, &[], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(output_node.shape.rank(), 0);
    }

    #[test]
    fn test_constant_bool() {
        let translator = ConstantTranslator;
        let mut builder = GraphBuilder::new();

        let tensor = make_bool_tensor(vec![2], vec![1, 0]);
        let node = make_node_with_tensor(tensor);

        let result = translator.translate(&node, &[], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(output_node.dtype, DType::Bool);
    }

    #[test]
    fn test_constant_empty_tensor() {
        let translator = ConstantTranslator;
        let mut builder = GraphBuilder::new();

        let tensor = make_f32_tensor(vec![0], vec![]);
        let node = make_node_with_tensor(tensor);

        let result = translator.translate(&node, &[], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(output_node.shape.dims.len(), 1);
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_constant_with_inputs_fails_validation() {
        let translator = ConstantTranslator;
        let err = translator.input_requirement().validate(1, "Constant");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount { expected: 0, got: 1, .. }
        ));
    }

    #[test]
    fn test_constant_missing_value_attribute() {
        let translator = ConstantTranslator;
        let mut builder = GraphBuilder::new();

        let node = NodeProto {
            name: "constant_test".to_string(),
            op_type: "Constant".to_string(),
            attribute: vec![],
            ..Default::default()
        };

        let result = translator.translate(&node, &[], &mut builder);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TranslationError::MissingAttribute { op, name } if op == "Constant" && name == "value"
        ));
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_constant_op_type() {
        let translator = ConstantTranslator;
        assert_eq!(translator.onnx_op_type(), "Constant");
    }

    #[test]
    fn test_constant_no_folding_support() {
        let translator = ConstantTranslator;
        assert!(!translator.supports_constant_folding());
    }

    #[test]
    fn test_input_requirement_exact_zero() {
        let translator = ConstantTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::Exact(0)));
        assert!(req.accepts_zero());
    }

    // ===== Extract Constant Tests =====

    #[test]
    fn test_extract_f32_from_raw_data() {
        use crate::proto::tensor_proto::DataType;

        let raw_bytes: Vec<u8> = vec![1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let tensor = TensorProto {
            dims: vec![3],
            data_type: DataType::Float as i32,
            raw_data: raw_bytes,
            ..Default::default()
        };

        let result = extract_constant_from_tensor(&tensor);
        assert!(result.is_ok());

        let (data, dtype, shape) = result.unwrap();
        assert_eq!(dtype, DType::F32);
        assert_eq!(shape.rank(), 1);
        if let ConstantData::F32(values) = data {
            assert_eq!(values, vec![1.0, 2.0, 3.0]);
        } else {
            panic!("Expected F32 data");
        }
    }

    #[test]
    fn test_extract_i64_from_int64_data() {
        use crate::proto::tensor_proto::DataType;

        let tensor = TensorProto {
            dims: vec![2, 2],
            data_type: DataType::Int64 as i32,
            int64_data: vec![1, 2, 3, 4],
            ..Default::default()
        };

        let result = extract_constant_from_tensor(&tensor);
        assert!(result.is_ok());

        let (data, dtype, _) = result.unwrap();
        assert_eq!(dtype, DType::I64);
        if let ConstantData::I64(values) = data {
            assert_eq!(values, vec![1, 2, 3, 4]);
        } else {
            panic!("Expected I64 data");
        }
    }
}
