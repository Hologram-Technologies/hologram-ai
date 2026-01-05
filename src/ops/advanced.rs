//! ONNX advanced operations.

use hologram_ir::{GraphBuilder, NodeIndex, DType};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::parse_attr_int;

/// Translate ONNX Cast to IR.
pub fn translate_cast(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Cast requires 1 input".into()));
    }

    let to_type = parse_attr_int(attrs, "to", 1)?;

    // Convert ONNX type to DType
    let dtype = match to_type {
        1 => DType::F32,
        2 => DType::U8,
        3 => DType::I8,
        6 => DType::I32,
        7 => DType::I64,
        10 => DType::F16,
        11 => DType::F64,
        _ => DType::F32,
    };

    // Constant folding: if input is a constant, cast it at compile time
    use hologram_ir::{NodeOp, ConstantData};

    let input_node = builder.graph().node(inputs[0])
        .ok_or_else(|| OnnxError::InvalidModel("Cast: input not found".to_string()))?;

    if let NodeOp::Constant { data } = &input_node.op {
        // Perform constant folding for common type casts
        let folded_data = match (data, dtype) {
            // I64 → I64 (no-op)
            (ConstantData::I64(values), DType::I64) => Some(ConstantData::I64(values.clone())),

            // I32 → I64
            (ConstantData::I32(values), DType::I64) => {
                Some(ConstantData::I64(values.iter().map(|&v| v as i64).collect()))
            }

            // I64 → I32
            (ConstantData::I64(values), DType::I32) => {
                Some(ConstantData::I32(values.iter().map(|&v| v as i32).collect()))
            }

            // F32 → F32 (no-op)
            (ConstantData::F32(values), DType::F32) => Some(ConstantData::F32(values.clone())),

            // I64 → F32
            (ConstantData::I64(values), DType::F32) => {
                Some(ConstantData::F32(values.iter().map(|&v| v as f32).collect()))
            }

            // I32 → F32
            (ConstantData::I32(values), DType::F32) => {
                Some(ConstantData::F32(values.iter().map(|&v| v as f32).collect()))
            }

            // F32 → I64
            (ConstantData::F32(values), DType::I64) => {
                Some(ConstantData::I64(values.iter().map(|&v| v as i64).collect()))
            }

            // F32 → I32
            (ConstantData::F32(values), DType::I32) => {
                Some(ConstantData::I32(values.iter().map(|&v| v as i32).collect()))
            }

            _ => None,  // Cast not supported for constant folding, fall through
        };

        if let Some(const_data) = folded_data {
            let output_shape = input_node.shape.clone();
            let result = builder.constant(const_data, output_shape);
            return Ok(vec![result]);
        }
    }

    // No constant folding, create regular cast node
    let result = builder.cast(inputs[0], dtype)?;

    Ok(vec![result])
}

/// Translate ONNX Range to IR.
///
/// Range generates a sequence of numbers from start to limit (exclusive) with step delta.
/// Supports both constant folding (when all inputs are constants) and runtime range generation.
///
/// Formula: output[i] = start + (i * delta)
/// Number of elements: max(ceil((limit - start) / delta), 0)
pub fn translate_range(
    inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() != 3 {
        return Err(OnnxError::InvalidModel(format!(
            "Range requires 3 inputs (start, limit, delta), got {}",
            inputs.len()
        )));
    }

    // Get the three input nodes
    let start_node = builder.graph().node(inputs[0])
        .ok_or_else(|| OnnxError::InvalidModel("Range: start input not found".to_string()))?;
    let limit_node = builder.graph().node(inputs[1])
        .ok_or_else(|| OnnxError::InvalidModel("Range: limit input not found".to_string()))?;
    let delta_node = builder.graph().node(inputs[2])
        .ok_or_else(|| OnnxError::InvalidModel("Range: delta input not found".to_string()))?;

    // Check if all inputs are constants - if so, use constant folding (optimization)
    use hologram_ir::{NodeOp, ConstantData, Shape};

    let all_constants = matches!(start_node.op, NodeOp::Constant { .. })
        && matches!(limit_node.op, NodeOp::Constant { .. })
        && matches!(delta_node.op, NodeOp::Constant { .. });

    if !all_constants {
        // Dynamic range - use runtime Range operation
        tracing::debug!("Range: dynamic path (non-constant inputs)");
        let result = builder.range(inputs[0], inputs[1], inputs[2])?;
        return Ok(vec![result]);
    }

    // Static range - constant folding path
    let start_data = match &start_node.op {
        NodeOp::Constant { data } => data,
        _ => unreachable!(),
    };

    let limit_data = match &limit_node.op {
        NodeOp::Constant { data } => data,
        _ => unreachable!(),
    };

    let delta_data = match &delta_node.op {
        NodeOp::Constant { data } => data,
        _ => unreachable!(),
    };

    // Extract scalar values and compute range based on data type
    match (start_data, limit_data, delta_data) {
        (ConstantData::I64(start_vec), ConstantData::I64(limit_vec), ConstantData::I64(delta_vec)) => {
            if start_vec.is_empty() || limit_vec.is_empty() || delta_vec.is_empty() {
                return Err(OnnxError::InvalidModel("Range: inputs must be non-empty".to_string()));
            }

            let start = start_vec[0];
            let limit = limit_vec[0];
            let delta = delta_vec[0];

            if delta == 0 {
                return Err(OnnxError::InvalidModel("Range: delta cannot be zero".to_string()));
            }

            // Compute number of elements: max(ceil((limit - start) / delta), 0)
            let num_elements = if (delta > 0 && start >= limit) || (delta < 0 && start <= limit) {
                0
            } else {
                ((limit - start + delta - delta.signum()) / delta).max(0) as usize
            };

            // Generate range
            let mut values = Vec::with_capacity(num_elements);
            for i in 0..num_elements {
                values.push(start + (i as i64) * delta);
            }

            let output_shape = Shape::static_shape(&[values.len()]);
            let constant_data = ConstantData::I64(values);
            let result = builder.constant(constant_data, output_shape);
            Ok(vec![result])
        }
        (ConstantData::I32(start_vec), ConstantData::I32(limit_vec), ConstantData::I32(delta_vec)) => {
            if start_vec.is_empty() || limit_vec.is_empty() || delta_vec.is_empty() {
                return Err(OnnxError::InvalidModel("Range: inputs must be non-empty".to_string()));
            }

            let start = start_vec[0];
            let limit = limit_vec[0];
            let delta = delta_vec[0];

            if delta == 0 {
                return Err(OnnxError::InvalidModel("Range: delta cannot be zero".to_string()));
            }

            let num_elements = if (delta > 0 && start >= limit) || (delta < 0 && start <= limit) {
                0
            } else {
                ((limit - start + delta - delta.signum()) / delta).max(0) as usize
            };

            let mut values = Vec::with_capacity(num_elements);
            for i in 0..num_elements {
                values.push(start + (i as i32) * delta);
            }

            let output_shape = Shape::static_shape(&[values.len()]);
            let constant_data = ConstantData::I32(values);
            let result = builder.constant(constant_data, output_shape);
            Ok(vec![result])
        }
        (ConstantData::F32(start_vec), ConstantData::F32(limit_vec), ConstantData::F32(delta_vec)) => {
            if start_vec.is_empty() || limit_vec.is_empty() || delta_vec.is_empty() {
                return Err(OnnxError::InvalidModel("Range: inputs must be non-empty".to_string()));
            }

            let start = start_vec[0];
            let limit = limit_vec[0];
            let delta = delta_vec[0];

            if delta == 0.0 {
                return Err(OnnxError::InvalidModel("Range: delta cannot be zero".to_string()));
            }

            let num_elements = if (delta > 0.0 && start >= limit) || (delta < 0.0 && start <= limit) {
                0
            } else {
                ((limit - start) / delta).ceil().max(0.0) as usize
            };

            let mut values = Vec::with_capacity(num_elements);
            for i in 0..num_elements {
                values.push(start + (i as f32) * delta);
            }

            let output_shape = Shape::static_shape(&[values.len()]);
            let constant_data = ConstantData::F32(values);
            let result = builder.constant(constant_data, output_shape);
            Ok(vec![result])
        }
        (ConstantData::F64(start_vec), ConstantData::F64(limit_vec), ConstantData::F64(delta_vec)) => {
            if start_vec.is_empty() || limit_vec.is_empty() || delta_vec.is_empty() {
                return Err(OnnxError::InvalidModel("Range: inputs must be non-empty".to_string()));
            }

            let start = start_vec[0];
            let limit = limit_vec[0];
            let delta = delta_vec[0];

            if delta == 0.0 {
                return Err(OnnxError::InvalidModel("Range: delta cannot be zero".to_string()));
            }

            let num_elements = if (delta > 0.0 && start >= limit) || (delta < 0.0 && start <= limit) {
                0
            } else {
                ((limit - start) / delta).ceil().max(0.0) as usize
            };

            let mut values = Vec::with_capacity(num_elements);
            for i in 0..num_elements {
                values.push(start + (i as f64) * delta);
            }

            let output_shape = Shape::static_shape(&[values.len()]);
            let constant_data = ConstantData::F64(values);
            let result = builder.constant(constant_data, output_shape);
            Ok(vec![result])
        }
        _ => Err(OnnxError::InvalidModel(
            "Range: type mismatch or unsupported data types (all inputs must have same type: i32, i64, f32, or f64)".to_string()
        )),
    }
}

/// Translate ONNX Trilu (triangular lower/upper) operation to IR.
///
/// Trilu returns the upper or lower triangular part of 2-D matrices or batches of 2-D matrices.
/// Used for attention masking in transformers.
///
/// # Arguments
///
/// * `inputs` - [data, k (optional)] where:
///   - data: Input tensor of shape [*, N, M]
///   - k: Optional scalar diagonal offset (default: 0)
/// * `attrs` - Attributes including:
///   - upper: bool (default: true) - if true, return upper triangle; else lower triangle
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node (same shape as data)
///
/// # Errors
///
/// Returns error if:
/// - Input count is not 1 or 2
/// - Data tensor is less than 2-D
pub fn translate_trilu(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() || inputs.len() > 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Trilu requires 1 or 2 inputs (data, k optional), got {}",
            inputs.len()
        )));
    }

    // Parse upper attribute (default: true)
    let upper = crate::ops::utils::parse_attr_int(attrs, "upper", 1)? != 0;

    // Get k parameter if provided
    let k = if inputs.len() > 1 {
        Some(inputs[1])
    } else {
        None
    };

    // Create Trilu node using builder
    let result = builder.trilu(inputs[0], k, upper)?;

    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::attribute_proto::AttributeType;
    use hologram_ir::Shape;

    fn make_int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }
    }

    #[test]
    fn test_translate_cast_f32_to_i32() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3]), DType::F32);

        let attrs = vec![make_int_attr("to", 6)]; // 6 = INT32
        let result = translate_cast(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_cast_f32_to_f64() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3]), DType::F32);

        let attrs = vec![make_int_attr("to", 11)]; // 11 = DOUBLE
        let result = translate_cast(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_cast_i64_to_f32() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[10]), DType::I64);

        let attrs = vec![make_int_attr("to", 1)]; // 1 = FLOAT
        let result = translate_cast(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_cast_u8_to_f16() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[256]), DType::U8);

        let attrs = vec![make_int_attr("to", 10)]; // 10 = FLOAT16
        let result = translate_cast(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_cast_no_inputs() {
        let mut builder = GraphBuilder::new();

        let attrs = vec![make_int_attr("to", 1)];
        let result = translate_cast(&[], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_cast_unknown_type_defaults_to_f32() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[5]), DType::I32);

        let attrs = vec![make_int_attr("to", 999)]; // Unknown type
        let result = translate_cast(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_range_i64_ascending() {
        use hologram_ir::ConstantData;

        let mut builder = GraphBuilder::new();
        let start = builder.constant(ConstantData::I64(vec![3]), Shape::static_shape(&[]));
        let limit = builder.constant(ConstantData::I64(vec![9]), Shape::static_shape(&[]));
        let delta = builder.constant(ConstantData::I64(vec![3]), Shape::static_shape(&[]));

        let result = translate_range(&[start, limit, delta], &[], &mut builder);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1);

        // Verify the generated constant: [3, 6]
        let node = builder.graph().node(output[0]).unwrap();
        if let hologram_ir::NodeOp::Constant { data } = &node.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values, &vec![3, 6]);
            } else {
                panic!("Expected I64 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_translate_range_i64_descending() {
        use hologram_ir::ConstantData;

        let mut builder = GraphBuilder::new();
        let start = builder.constant(ConstantData::I64(vec![10]), Shape::static_shape(&[]));
        let limit = builder.constant(ConstantData::I64(vec![4]), Shape::static_shape(&[]));
        let delta = builder.constant(ConstantData::I64(vec![-2]), Shape::static_shape(&[]));

        let result = translate_range(&[start, limit, delta], &[], &mut builder);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1);

        // Verify the generated constant: [10, 8, 6]
        let node = builder.graph().node(output[0]).unwrap();
        if let hologram_ir::NodeOp::Constant { data } = &node.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values, &vec![10, 8, 6]);
            } else {
                panic!("Expected I64 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_translate_range_f32() {
        use hologram_ir::ConstantData;

        let mut builder = GraphBuilder::new();
        let start = builder.constant(ConstantData::F32(vec![0.0]), Shape::static_shape(&[]));
        let limit = builder.constant(ConstantData::F32(vec![1.0]), Shape::static_shape(&[]));
        let delta = builder.constant(ConstantData::F32(vec![0.25]), Shape::static_shape(&[]));

        let result = translate_range(&[start, limit, delta], &[], &mut builder);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1);

        // Verify the generated constant: [0.0, 0.25, 0.5, 0.75]
        let node = builder.graph().node(output[0]).unwrap();
        if let hologram_ir::NodeOp::Constant { data } = &node.op {
            if let ConstantData::F32(values) = data {
                assert_eq!(values.len(), 4);
                assert!((values[0] - 0.0).abs() < 1e-6);
                assert!((values[1] - 0.25).abs() < 1e-6);
                assert!((values[2] - 0.5).abs() < 1e-6);
                assert!((values[3] - 0.75).abs() < 1e-6);
            } else {
                panic!("Expected F32 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_translate_range_empty_sequence() {
        use hologram_ir::ConstantData;

        let mut builder = GraphBuilder::new();
        let start = builder.constant(ConstantData::I64(vec![10]), Shape::static_shape(&[]));
        let limit = builder.constant(ConstantData::I64(vec![5]), Shape::static_shape(&[]));
        let delta = builder.constant(ConstantData::I64(vec![1]), Shape::static_shape(&[]));

        let result = translate_range(&[start, limit, delta], &[], &mut builder);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1);

        // Verify empty sequence
        let node = builder.graph().node(output[0]).unwrap();
        if let hologram_ir::NodeOp::Constant { data } = &node.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values.len(), 0);
            } else {
                panic!("Expected I64 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_translate_range_zero_delta_error() {
        use hologram_ir::ConstantData;

        let mut builder = GraphBuilder::new();
        let start = builder.constant(ConstantData::I64(vec![0]), Shape::static_shape(&[]));
        let limit = builder.constant(ConstantData::I64(vec![10]), Shape::static_shape(&[]));
        let delta = builder.constant(ConstantData::I64(vec![0]), Shape::static_shape(&[]));

        let result = translate_range(&[start, limit, delta], &[], &mut builder);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("delta cannot be zero"));
    }

    #[test]
    fn test_translate_range_wrong_input_count() {
        use hologram_ir::ConstantData;

        let mut builder = GraphBuilder::new();
        let start = builder.constant(ConstantData::I64(vec![0]), Shape::static_shape(&[]));
        let limit = builder.constant(ConstantData::I64(vec![10]), Shape::static_shape(&[]));

        let result = translate_range(&[start, limit], &[], &mut builder);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Range requires 3 inputs"));
    }

    #[test]
    fn test_translate_range_type_mismatch() {
        use hologram_ir::ConstantData;

        let mut builder = GraphBuilder::new();
        let start = builder.constant(ConstantData::I64(vec![0]), Shape::static_shape(&[]));
        let limit = builder.constant(ConstantData::F32(vec![10.0]), Shape::static_shape(&[]));
        let delta = builder.constant(ConstantData::I64(vec![1]), Shape::static_shape(&[]));

        let result = translate_range(&[start, limit, delta], &[], &mut builder);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("type mismatch"));
    }
}
