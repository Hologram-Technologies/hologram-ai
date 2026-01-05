//! ONNX constant operations.
//!
//! This module provides translators for constant operations including:
//! - Constant: Create a constant tensor from attribute
//! - ConstantOfShape: Create a constant tensor of given shape
//! - Identity: Pass-through operation
//! - Shape: Get shape of tensor (not yet fully supported)

use hologram_ir::{GraphBuilder, NodeIndex, ConstantData, DType, Shape};
use crate::core::{OnnxError, Result};
use crate::proto::{AttributeProto, TensorProto};
use crate::ops::utils::parse_attr_tensor;
use bytemuck;
use half;

/// Translate ONNX Constant operation to IR.
///
/// ONNX Constant creates a constant tensor from the value attribute.
///
/// # Arguments
///
/// * `inputs` - (none)
/// * `attrs` - Attributes including value (TensorProto)
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
///
/// # Errors
///
/// Returns error if:
/// - value attribute is missing
/// - Tensor data type is not supported
/// - Tensor data extraction fails
pub fn translate_constant(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if !inputs.is_empty() {
        return Err(OnnxError::InvalidModel(format!(
            "Constant requires 0 inputs, got {}",
            inputs.len()
        )));
    }

    // Get value tensor from attributes
    let tensor = parse_attr_tensor(attrs, "value")?;

    // Extract constant data from tensor
    let (constant_data, _dtype, shape) = extract_constant_from_tensor(tensor)?;

    // Create constant node (dtype is inferred from ConstantData)
    let result = builder.constant(constant_data, shape);

    Ok(vec![result])
}

/// Translate ONNX ConstantOfShape operation to IR.
///
/// ONNX ConstantOfShape creates a constant tensor of the given shape
/// filled with a specified value.
///
/// # Arguments
///
/// * `inputs` - [shape] (must be a constant)
/// * `attrs` - Attributes including value (TensorProto with single element)
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
///
/// # Errors
///
/// Returns error if:
/// - Input count is not 1
/// - Shape input is not constant (not yet supported)
/// - Value tensor format is invalid
pub fn translate_constant_of_shape(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "ConstantOfShape requires 1 input (shape), got {}",
            inputs.len()
        )));
    }

    // Get the shape input
    let shape_node = builder.graph().node(inputs[0])
        .ok_or_else(|| OnnxError::InvalidModel("ConstantOfShape: shape input not found".to_string()))?;

    use hologram_ir::{NodeOp, ConstantData, Shape, DType};

    // Get the fill value from attributes (default is 0.0 as float32)
    let (fill_value_data, _fill_dtype) = if let Some(value_attr) = attrs.iter().find(|a| a.name == "value") {
        // Parse the tensor value from attribute
        if let Some(ref tensor) = value_attr.t {
            use crate::proto::tensor_proto::DataType;
            let data_type = DataType::try_from(tensor.data_type)
                .map_err(|_| OnnxError::InvalidModel("ConstantOfShape: invalid data type".to_string()))?;

            match data_type {
                DataType::Float => {
                    let val = if !tensor.float_data.is_empty() {
                        tensor.float_data[0]
                    } else {
                        0.0f32
                    };
                    (ConstantData::F32(vec![val]), DType::F32)
                }
                DataType::Int64 => {
                    let val = if !tensor.int64_data.is_empty() {
                        tensor.int64_data[0]
                    } else {
                        0i64
                    };
                    (ConstantData::I64(vec![val]), DType::I64)
                }
                DataType::Int32 => {
                    let val = if !tensor.int32_data.is_empty() {
                        tensor.int32_data[0]
                    } else {
                        0i32
                    };
                    (ConstantData::I32(vec![val]), DType::I32)
                }
                _ => (ConstantData::F32(vec![0.0]), DType::F32), // Default
            }
        } else {
            (ConstantData::F32(vec![0.0]), DType::F32) // Default
        }
    } else {
        // No value attribute, use default 0.0 as float32
        (ConstantData::F32(vec![0.0]), DType::F32)
    };

    // Check if shape input is constant - if so, use constant folding (optimization)
    if let NodeOp::Constant { data } = &shape_node.op {
        let shape_dims = match data {
            ConstantData::I64(values) => {
                values.iter().map(|&v| v as usize).collect::<Vec<_>>()
            }
            ConstantData::I32(values) => {
                values.iter().map(|&v| v as usize).collect::<Vec<_>>()
            }
            _ => return Err(OnnxError::InvalidModel(
                "ConstantOfShape: shape must be int32 or int64 tensor".to_string()
            )),
        };

        // Calculate total number of elements
        let total_elements = shape_dims.iter().product::<usize>();

        // Create filled tensor
        let filled_data = match fill_value_data {
            ConstantData::F32(vals) if !vals.is_empty() => {
                ConstantData::F32(vec![vals[0]; total_elements])
            }
            ConstantData::I64(vals) if !vals.is_empty() => {
                ConstantData::I64(vec![vals[0]; total_elements])
            }
            ConstantData::I32(vals) if !vals.is_empty() => {
                ConstantData::I32(vec![vals[0]; total_elements])
            }
            _ => ConstantData::F32(vec![0.0; total_elements]),
        };

        let output_shape = Shape::static_shape(&shape_dims);
        let result = builder.constant(filled_data, output_shape);

        tracing::debug!("ConstantOfShape: static path, shape = {:?}, elements = {}", shape_dims, total_elements);

        return Ok(vec![result]);
    }

    // Dynamic path - shape is computed at runtime
    tracing::debug!("ConstantOfShape: dynamic path (runtime-computed shape)");

    // Create a constant node for the fill value (scalar)
    let fill_value_node = builder.constant(fill_value_data, Shape::static_shape(&[]));

    // Use dynamic ConstantOfShape operation
    let result = builder.constant_of_shape(inputs[0], fill_value_node)?;

    Ok(vec![result])
}

/// Translate ONNX Identity operation to IR.
///
/// Identity is a no-op that passes the input through unchanged.
///
/// # Arguments
///
/// * `inputs` - [input]
/// * `attrs` - (none)
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node (same as input)
///
/// # Errors
///
/// Returns error if input count is not 1
pub fn translate_identity(
    inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    _builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Identity requires 1 input, got {}",
            inputs.len()
        )));
    }

    // Identity is a no-op, just return the input
    Ok(vec![inputs[0]])
}

/// Translate ONNX Shape operation to IR.
///
/// Shape operation extracts the shape of a tensor as a 1D int64 tensor.
/// Uses hologram-ir's runtime Shape operation to support symbolic dimensions.
///
/// # Arguments
///
/// * `inputs` - [data]
/// * `attrs` - Optional attributes (start, end for partial shape extraction)
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node (1D int64 tensor containing shape dimensions)
///
/// # Errors
///
/// Returns error if:
/// - Input count is not 1
/// - Start/end indices are invalid
pub fn translate_shape_op(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Shape requires 1 input, got {}",
            inputs.len()
        )));
    }

    // Get input node to determine rank
    let input_node = builder.graph().node(inputs[0])
        .ok_or_else(|| OnnxError::InvalidModel("Shape: input node not found".to_string()))?;
    let rank = input_node.shape.rank() as i64;

    // Extract start/end indices if specified (ONNX opset 15+)
    let start = attrs
        .iter()
        .find(|a| a.name == "start")
        .map(|a| a.i)
        .unwrap_or(0);

    let end = attrs
        .iter()
        .find(|a| a.name == "end")
        .map(|a| a.i)
        .unwrap_or(rank);

    // Normalize negative indices
    let start_idx = if start < 0 { rank + start } else { start };
    let end_idx = if end < 0 { rank + end } else { end };

    // Validate range
    if start_idx < 0 || end_idx > rank || start_idx > end_idx {
        return Err(OnnxError::InvalidModel(format!(
            "Shape: invalid range [{}:{}] for rank {}",
            start, end, rank
        )));
    }

    // Optimization: If all dimensions in the range are static, do constant folding
    use hologram_ir::Dim;
    let dims = &input_node.shape.dims;
    let mut all_static = true;
    let mut shape_values = Vec::new();

    for i in start_idx..end_idx {
        match &dims[i as usize] {
            Dim::Static(size) => shape_values.push(*size as i64),
            Dim::Symbolic(_) | Dim::Dynamic => {
                all_static = false;
                break;
            }
        }
    }

    if all_static {
        // All dimensions are static, use constant folding
        let output_shape = Shape::static_shape(&[shape_values.len()]);
        let constant_data = ConstantData::I64(shape_values);
        let result = builder.constant(constant_data, output_shape);
        Ok(vec![result])
    } else {
        // Has symbolic/dynamic dimensions, use runtime Shape operation
        let result = builder.shape(inputs[0], start_idx, end_idx)?;
        Ok(vec![result])
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
fn extract_constant_from_tensor(tensor: &TensorProto) -> Result<(ConstantData, DType, Shape)> {
    // ONNX data type constants
    const FLOAT: i32 = 1;
    const UINT8: i32 = 2;
    const INT8: i32 = 3;
    const INT32: i32 = 6;
    const INT64: i32 = 7;
    const BOOL: i32 = 9;
    const FLOAT16: i32 = 10;
    const DOUBLE: i32 = 11;

    // Extract shape
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    let shape = Shape::static_shape(&dims);

    let (constant_data, dtype) = match tensor.data_type {
        FLOAT => {
            let data = if !tensor.raw_data.is_empty() {
                bytemuck::cast_slice(&tensor.raw_data).to_vec()
            } else {
                tensor.float_data.clone()
            };
            (ConstantData::F32(data), DType::F32)
        }

        FLOAT16 => {
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

        DOUBLE => {
            let data = if !tensor.raw_data.is_empty() {
                bytemuck::cast_slice(&tensor.raw_data).to_vec()
            } else {
                tensor.double_data.clone()
            };
            (ConstantData::F64(data), DType::F64)
        }

        INT32 => {
            let data = if !tensor.raw_data.is_empty() {
                bytemuck::cast_slice(&tensor.raw_data).to_vec()
            } else {
                tensor.int32_data.clone()
            };
            (ConstantData::I32(data), DType::I32)
        }

        INT64 => {
            let data = if !tensor.raw_data.is_empty() {
                bytemuck::cast_slice(&tensor.raw_data).to_vec()
            } else {
                tensor.int64_data.clone()
            };
            (ConstantData::I64(data), DType::I64)
        }

        UINT8 => {
            let data = if !tensor.raw_data.is_empty() {
                tensor.raw_data.clone()
            } else {
                tensor.int32_data.iter().map(|&x| x as u8).collect()
            };
            (ConstantData::U8(data), DType::U8)
        }

        INT8 => {
            // INT8 stored as U8 in hologram-ir
            let data = if !tensor.raw_data.is_empty() {
                tensor.raw_data.clone()
            } else {
                tensor.int32_data.iter().map(|&x| x as u8).collect()
            };
            (ConstantData::U8(data), DType::I8)
        }

        BOOL => {
            let data = if !tensor.raw_data.is_empty() {
                tensor.raw_data.iter().map(|&b| b != 0).collect()
            } else {
                tensor.int32_data.iter().map(|&x| x != 0).collect()
            };
            (ConstantData::Bool(data), DType::Bool)
        }

        _ => {
            return Err(OnnxError::unsupported_op("Constant", 13));
        }
    };

    Ok((constant_data, dtype, shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::attribute_proto::AttributeType;
    use hologram_ir::NodeOp;

    fn make_tensor_attr(name: &str, tensor: TensorProto) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            t: Some(tensor),
            r#type: AttributeType::Tensor as i32,
            ..Default::default()
        }
    }

    fn make_f32_tensor(dims: Vec<i64>, data: Vec<f32>) -> TensorProto {
        TensorProto {
            dims,
            data_type: 1, // FLOAT
            float_data: data,
            ..Default::default()
        }
    }

    fn make_i64_tensor(dims: Vec<i64>, data: Vec<i64>) -> TensorProto {
        TensorProto {
            dims,
            data_type: 7, // INT64
            int64_data: data,
            ..Default::default()
        }
    }

    #[test]
    fn test_translate_constant_f32() {
        let mut builder = GraphBuilder::new();

        let tensor = make_f32_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let attrs = vec![make_tensor_attr("value", tensor)];

        let result = translate_constant(&[], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_constant_i64() {
        let mut builder = GraphBuilder::new();

        let tensor = make_i64_tensor(vec![3], vec![1, 2, 3]);
        let attrs = vec![make_tensor_attr("value", tensor)];

        let result = translate_constant(&[], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_constant_scalar() {
        let mut builder = GraphBuilder::new();

        let tensor = make_f32_tensor(vec![], vec![42.0]);
        let attrs = vec![make_tensor_attr("value", tensor)];

        let result = translate_constant(&[], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_constant_invalid_inputs() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1]), DType::F32);
        let tensor = make_f32_tensor(vec![1], vec![1.0]);
        let attrs = vec![make_tensor_attr("value", tensor)];

        let result = translate_constant(&[input], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_constant_missing_value() {
        let mut builder = GraphBuilder::new();

        let attrs = vec![];
        let result = translate_constant(&[], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_constant_of_shape_default() {
        let mut builder = GraphBuilder::new();

        // Create a constant shape tensor [2, 3]
        let shape = builder.constant(ConstantData::I64(vec![2, 3]), Shape::static_shape(&[2]));
        let attrs = vec![];

        let result = translate_constant_of_shape(&[shape], &attrs, &mut builder);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1);

        // Verify output is a 2x3 tensor filled with 0.0 (default)
        let node = builder.graph().node(output[0]).unwrap();
        assert_eq!(node.shape.dims.len(), 2);
        assert_eq!(node.shape.dims[0], hologram_ir::Dim::Static(2));
        assert_eq!(node.shape.dims[1], hologram_ir::Dim::Static(3));

        if let NodeOp::Constant { data } = &node.op {
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
    fn test_translate_constant_of_shape_with_value() {
        use crate::proto::attribute_proto::AttributeType;
        use crate::proto::{TensorProto, tensor_proto::DataType};

        let mut builder = GraphBuilder::new();

        // Create a constant shape tensor [3]
        let shape = builder.constant(ConstantData::I64(vec![3]), Shape::static_shape(&[1]));

        // Create attribute with fill value of 1.0
        let value_tensor = TensorProto {
            data_type: DataType::Float as i32,
            float_data: vec![1.0],
            ..Default::default()
        };

        let attrs = vec![AttributeProto {
            name: "value".to_string(),
            t: Some(value_tensor),
            r#type: AttributeType::Tensor as i32,
            ..Default::default()
        }];

        let result = translate_constant_of_shape(&[shape], &attrs, &mut builder);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1);

        // Verify output is a [3] tensor filled with 1.0
        let node = builder.graph().node(output[0]).unwrap();
        if let NodeOp::Constant { data } = &node.op {
            if let ConstantData::F32(values) = data {
                assert_eq!(values.len(), 3);
                assert!(values.iter().all(|&v| v == 1.0));
            } else {
                panic!("Expected F32 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_translate_identity() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let result = translate_identity(&[input], &[], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], input); // Identity returns same node
    }

    #[test]
    fn test_translate_identity_invalid_inputs() {
        let mut builder = GraphBuilder::new();

        let input1 = builder.input("input1", Shape::static_shape(&[1]), DType::F32);
        let input2 = builder.input("input2", Shape::static_shape(&[1]), DType::F32);

        let result = translate_identity(&[input1, input2], &[], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_shape_op_static() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let result = translate_shape_op(&[input], &[], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);

        // Output should be a constant int64 tensor [1, 3, 32, 32]
        let output_node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(output_node.shape.rank(), 1);
        assert_eq!(output_node.dtype, DType::I64);

        // Verify constant data
        if let NodeOp::Constant { data } = &output_node.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values.as_slice(), &[1i64, 3, 32, 32]);
            } else {
                panic!("Expected I64 constant data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_translate_shape_op_with_start_end() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[2, 3, 4, 5]), DType::F32);

        // Extract shape[1:3] = [3, 4]
        let attrs = vec![
            AttributeProto {
                name: "start".to_string(),
                i: 1,
                ..Default::default()
            },
            AttributeProto {
                name: "end".to_string(),
                i: 3,
                ..Default::default()
            },
        ];

        let result = translate_shape_op(&[input], &attrs, &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();

        if let NodeOp::Constant { data } = &output_node.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values.as_slice(), &[3i64, 4]);
            } else {
                panic!("Expected I64 constant data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_translate_shape_op_symbolic() {
        use hologram_ir::Dim;

        let mut builder = GraphBuilder::new();

        // Create input with symbolic dimension
        let shape = Shape::new(vec![
            Dim::Static(2),
            Dim::Symbolic("batch".to_string()),
            Dim::Static(768),
        ]);
        let input = builder.input("input", shape, DType::F32);

        // Should now succeed with runtime Shape operation
        let result = translate_shape_op(&[input], &[], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);

        // Output should be a Shape operation node (not constant for symbolic dims)
        let output_node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(output_node.shape.rank(), 1);
        assert_eq!(output_node.dtype, DType::I64);
        // Output shape is [3] (the rank of the input)
        assert_eq!(output_node.shape.dims[0], Dim::Static(3));
    }

    #[test]
    fn test_translate_shape_op_invalid_inputs() {
        let mut builder = GraphBuilder::new();

        let input1 = builder.input("input1", Shape::static_shape(&[1]), DType::F32);
        let input2 = builder.input("input2", Shape::static_shape(&[1]), DType::F32);

        let result = translate_shape_op(&[input1, input2], &[], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_constant_f32() {
        let tensor = make_f32_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let result = extract_constant_from_tensor(&tensor);
        assert!(result.is_ok());

        let (data, dtype, shape) = result.unwrap();
        assert!(matches!(data, ConstantData::F32(_)));
        assert_eq!(dtype, DType::F32);
        assert_eq!(shape.rank(), 2);
    }

    #[test]
    fn test_extract_constant_i64() {
        let tensor = make_i64_tensor(vec![3], vec![1, 2, 3]);
        let result = extract_constant_from_tensor(&tensor);
        assert!(result.is_ok());

        let (data, dtype, shape) = result.unwrap();
        assert!(matches!(data, ConstantData::I64(_)));
        assert_eq!(dtype, DType::I64);
        assert_eq!(shape.rank(), 1);
    }

    #[test]
    fn test_extract_constant_bool() {
        let tensor = TensorProto {
            dims: vec![2],
            data_type: 9, // BOOL
            int32_data: vec![1, 0],
            ..Default::default()
        };
        let result = extract_constant_from_tensor(&tensor);
        assert!(result.is_ok());

        let (data, dtype, _shape) = result.unwrap();
        assert!(matches!(data, ConstantData::Bool(_)));
        assert_eq!(dtype, DType::Bool);
    }
}
