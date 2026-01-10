//! ONNX normalization operations.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::{parse_attr_float, parse_attr_int};

/// Translate ONNX LayerNormalization to IR.
pub fn translate_layer_norm(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("LayerNorm requires at least 1 input".into()));
    }

    let epsilon = parse_attr_float(attrs, "epsilon", 1e-5)?;
    let axis = parse_attr_int(attrs, "axis", -1)? as i32;

    // Get input node to determine rank
    let input_node = builder.graph().node(inputs[0])
        .ok_or_else(|| OnnxError::InvalidModel("Invalid input node".into()))?;
    let rank = input_node.shape.rank() as i32;

    // Normalize over last dimensions from axis onwards
    let axes: Vec<i32> = if axis < 0 {
        (axis..0).map(|i| rank + i).collect()
    } else {
        (axis..rank).collect()
    };

    let result = builder.unary(
        hologram::ir::NodeOp::LayerNorm { epsilon, axes },
        inputs[0]
    )?;

    Ok(vec![result])
}

/// Translate ONNX BatchNormalization to IR.
pub fn translate_batch_norm(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 5 {
        return Err(OnnxError::InvalidModel("BatchNorm requires 5 inputs".into()));
    }

    let epsilon = parse_attr_float(attrs, "epsilon", 1e-5)?;
    let momentum = parse_attr_float(attrs, "momentum", 0.9)?;

    // BatchNorm: (x - mean) / sqrt(var + eps) * scale + bias
    let result = builder.unary(
        hologram::ir::NodeOp::BatchNorm { epsilon, momentum },
        inputs[0]
    )?;

    Ok(vec![result])
}

/// Translate ONNX GroupNormalization to IR.
pub fn translate_group_norm(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 3 {
        return Err(OnnxError::InvalidModel("GroupNorm requires 3 inputs".into()));
    }

    let epsilon = parse_attr_float(attrs, "epsilon", 1e-5)?;
    let _num_groups = parse_attr_int(attrs, "num_groups", 1)?;

    // GroupNorm: normalize within groups
    // Approximate with LayerNorm over spatial dimensions
    let axes = vec![-2, -1]; // Normalize over last 2 dims

    let result = builder.unary(
        hologram::ir::NodeOp::LayerNorm { epsilon, axes },
        inputs[0]
    )?;

    Ok(vec![result])
}

/// Translate ONNX InstanceNormalization to IR.
pub fn translate_instance_norm(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 3 {
        return Err(OnnxError::InvalidModel("InstanceNorm requires 3 inputs".into()));
    }

    let epsilon = parse_attr_float(attrs, "epsilon", 1e-5)?;

    // InstanceNorm: normalize per instance (spatial dimensions)
    // Normalize over last 2 dimensions (H, W)
    let axes = vec![-2, -1];

    let result = builder.unary(
        hologram::ir::NodeOp::LayerNorm { epsilon, axes },
        inputs[0]
    )?;

    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::attribute_proto::AttributeType;
    use hologram::ir::{DType, Shape};

    fn make_float_attr(name: &str, value: f32) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            f: value,
            r#type: AttributeType::Float as i32,
            ..Default::default()
        }
    }

    fn make_int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }
    }

    #[test]
    fn test_translate_layer_norm_basic() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let attrs = vec![
            make_float_attr("epsilon", 1e-5),
            make_int_attr("axis", -1),
        ];

        let result = translate_layer_norm(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_layer_norm_custom_axis() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[4, 5, 6]), DType::F32);

        let attrs = vec![
            make_float_attr("epsilon", 1e-6),
            make_int_attr("axis", 1),
        ];

        let result = translate_layer_norm(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_layer_norm_default_epsilon() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 2, 2]), DType::F32);

        let attrs = vec![make_int_attr("axis", -1)];

        let result = translate_layer_norm(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_layer_norm_no_inputs() {
        let mut builder = GraphBuilder::new();
        let attrs = vec![make_float_attr("epsilon", 1e-5)];

        let result = translate_layer_norm(&[], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_batch_norm_basic() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        let scale = builder.input("scale", Shape::static_shape(&[3]), DType::F32);
        let bias = builder.input("bias", Shape::static_shape(&[3]), DType::F32);
        let mean = builder.input("mean", Shape::static_shape(&[3]), DType::F32);
        let var = builder.input("var", Shape::static_shape(&[3]), DType::F32);

        let attrs = vec![
            make_float_attr("epsilon", 1e-5),
            make_float_attr("momentum", 0.9),
        ];

        let result = translate_batch_norm(&[input, scale, bias, mean, var], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_batch_norm_custom_params() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 64, 16, 16]), DType::F32);
        let scale = builder.input("scale", Shape::static_shape(&[64]), DType::F32);
        let bias = builder.input("bias", Shape::static_shape(&[64]), DType::F32);
        let mean = builder.input("mean", Shape::static_shape(&[64]), DType::F32);
        let var = builder.input("var", Shape::static_shape(&[64]), DType::F32);

        let attrs = vec![
            make_float_attr("epsilon", 1e-3),
            make_float_attr("momentum", 0.99),
        ];

        let result = translate_batch_norm(&[input, scale, bias, mean, var], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_batch_norm_insufficient_inputs() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        let scale = builder.input("scale", Shape::static_shape(&[3]), DType::F32);

        let attrs = vec![make_float_attr("epsilon", 1e-5)];

        let result = translate_batch_norm(&[input, scale], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_group_norm_basic() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 6, 8, 8]), DType::F32);
        let scale = builder.input("scale", Shape::static_shape(&[6]), DType::F32);
        let bias = builder.input("bias", Shape::static_shape(&[6]), DType::F32);

        let attrs = vec![
            make_float_attr("epsilon", 1e-5),
            make_int_attr("num_groups", 2),
        ];

        let result = translate_group_norm(&[input, scale, bias], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_group_norm_insufficient_inputs() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 6, 8, 8]), DType::F32);

        let attrs = vec![make_int_attr("num_groups", 2)];

        let result = translate_group_norm(&[input], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_instance_norm_basic() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);
        let scale = builder.input("scale", Shape::static_shape(&[64]), DType::F32);
        let bias = builder.input("bias", Shape::static_shape(&[64]), DType::F32);

        let attrs = vec![make_float_attr("epsilon", 1e-5)];

        let result = translate_instance_norm(&[input, scale, bias], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_instance_norm_custom_epsilon() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 32, 16, 16]), DType::F32);
        let scale = builder.input("scale", Shape::static_shape(&[32]), DType::F32);
        let bias = builder.input("bias", Shape::static_shape(&[32]), DType::F32);

        let attrs = vec![make_float_attr("epsilon", 1e-3)];

        let result = translate_instance_norm(&[input, scale, bias], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_instance_norm_insufficient_inputs() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let attrs = vec![make_float_attr("epsilon", 1e-5)];

        let result = translate_instance_norm(&[input], &attrs, &mut builder);
        assert!(result.is_err());
    }
}
