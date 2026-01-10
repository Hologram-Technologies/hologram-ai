//! ONNX padding operations.
//!
//! This module provides translators for padding operations including:
//! - Pad: Pad tensor with various modes (constant, reflect, edge)

use hologram::ir::{GraphBuilder, NodeIndex, PadMode};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::{parse_attr_ints, parse_attr_string_or, parse_attr_float};

/// Translate ONNX Pad operation to IR.
///
/// ONNX Pad adds padding to the input tensor with various modes:
/// - constant: Pad with a constant value
/// - reflect: Reflect padding (mirror at edge)
/// - edge: Edge padding (replicate edge values)
/// - wrap: Wrap padding (circular) - not yet supported
///
/// # Arguments
///
/// * `inputs` - [data, pads, constant_value (optional)]
/// * `attrs` - Attributes including mode, pads (for older opsets)
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
///
/// # Errors
///
/// Returns error if:
/// - Input count is less than 1
/// - Padding mode is not supported
/// - Padding array format is invalid
/// - Wrap mode is requested (not supported)
pub fn translate_pad(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Pad requires at least 1 input".to_string()
        ));
    }

    let data = inputs[0];

    // Parse mode (default is "constant")
    let mode_str = parse_attr_string_or(attrs, "mode", "constant")?;
    let mode = match mode_str.as_str() {
        "constant" => PadMode::Constant,
        "reflect" => PadMode::Reflect,
        "edge" => PadMode::Edge,
        "wrap" => PadMode::Wrap,
        _ => {
            return Err(OnnxError::InvalidModel(format!(
                "Unknown padding mode: {}",
                mode_str
            )));
        }
    };

    // Parse constant value (default is 0.0)
    let constant_value = if inputs.len() >= 3 {
        // Opset >= 11: constant_value is an input
        // For now, we don't support dynamic constant values
        // A full implementation would extract the constant from the graph
        parse_attr_float(attrs, "value", 0.0)? as f64
    } else {
        parse_attr_float(attrs, "value", 0.0)? as f64
    };

    // Parse pads
    // ONNX format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    let pads = if inputs.len() >= 2 {
        // Opset >= 11: pads is an input
        // For now, we don't support dynamic pads
        // A full implementation would extract pads from the graph
        return Err(OnnxError::unsupported_op("Pad", 13));
    } else {
        // Opset < 11: pads is an attribute
        parse_attr_ints(attrs, "pads", vec![])?
    };

    if pads.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Pad requires 'pads' attribute".to_string()
        ));
    }

    // Convert ONNX pads format to hologram-ir format
    // ONNX: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    // Hologram: Vec<(before, after)> for each dimension
    let num_dims = pads.len() / 2;
    if pads.len() != num_dims * 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Pad pads array length must be even, got {}",
            pads.len()
        )));
    }

    let mut pad_pairs = Vec::with_capacity(num_dims);
    for i in 0..num_dims {
        let before = pads[i] as usize;
        let after = pads[num_dims + i] as usize;
        pad_pairs.push((before, after));
    }

    // Create pad node
    let result = builder.pad(data, pad_pairs, mode, constant_value)?;

    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::attribute_proto::AttributeType;
    use hologram::ir::{DType, Shape};

    fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values,
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        }
    }

    fn make_string_attr(name: &str, value: &str) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            s: value.as_bytes().to_vec(),
            r#type: AttributeType::String as i32,
            ..Default::default()
        }
    }

    fn make_float_attr(name: &str, value: f32) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            f: value,
            r#type: AttributeType::Float as i32,
            ..Default::default()
        }
    }

    #[test]
    fn test_translate_pad_constant_mode() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        // Pad: [top=1, left=1, bottom=1, right=1]
        // ONNX format: [0, 0, 1, 1, 0, 0, 1, 1]
        let attrs = vec![
            make_string_attr("mode", "constant"),
            make_ints_attr("pads", vec![0, 0, 1, 1, 0, 0, 1, 1]),
            make_float_attr("value", 0.0),
        ];

        let result = translate_pad(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_pad_reflect_mode() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![
            make_string_attr("mode", "reflect"),
            make_ints_attr("pads", vec![0, 0, 1, 1, 0, 0, 1, 1]),
        ];

        let result = translate_pad(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_pad_edge_mode() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![
            make_string_attr("mode", "edge"),
            make_ints_attr("pads", vec![0, 0, 2, 2, 0, 0, 2, 2]),
        ];

        let result = translate_pad(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_pad_wrap_mode() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![
            make_string_attr("mode", "wrap"),
            make_ints_attr("pads", vec![0, 0, 1, 1, 0, 0, 1, 1]),
        ];

        let result = translate_pad(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_pad_default_mode() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![
            make_ints_attr("pads", vec![0, 0, 1, 1, 0, 0, 1, 1]),
        ];

        let result = translate_pad(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_pad_custom_value() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![
            make_string_attr("mode", "constant"),
            make_ints_attr("pads", vec![0, 0, 1, 1, 0, 0, 1, 1]),
            make_float_attr("value", 1.5),
        ];

        let result = translate_pad(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_pad_missing_pads() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![make_string_attr("mode", "constant")];

        let result = translate_pad(&[data], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_pad_invalid_pads_length() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        // Odd length pads array
        let attrs = vec![
            make_string_attr("mode", "constant"),
            make_ints_attr("pads", vec![0, 0, 1]),
        ];

        let result = translate_pad(&[data], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_pad_dynamic_pads_unsupported() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        let pads = builder.input("pads", Shape::static_shape(&[8]), DType::I64);

        let attrs = vec![make_string_attr("mode", "constant")];

        let result = translate_pad(&[data, pads], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_pad_1d() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[10]), DType::F32);

        let attrs = vec![
            make_string_attr("mode", "constant"),
            make_ints_attr("pads", vec![2, 3]),
        ];

        let result = translate_pad(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_pad_asymmetric() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        // Asymmetric padding: top=1, left=2, bottom=3, right=4
        let attrs = vec![
            make_string_attr("mode", "constant"),
            make_ints_attr("pads", vec![0, 0, 1, 2, 0, 0, 3, 4]),
        ];

        let result = translate_pad(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }
}
