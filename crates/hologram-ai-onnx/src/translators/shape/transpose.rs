//! Transpose operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{ConstantData, Dim, GraphBuilder, NodeIndex, NodeOp, Shape};

/// Translator for ONNX Transpose operation.
///
/// Transpose permutes the axes of a tensor according to the `perm` attribute.
///
/// # Inputs
/// - data: Input tensor to transpose
///
/// # Attributes
/// - perm (optional): Permutation of axes. If not specified, reverses all axes.
///
/// # Constant Folding
/// If the input is a constant tensor, the transpose is performed at compile time.
#[derive(Debug, Default)]
pub struct TransposeTranslator;

impl OnnxTranslator for TransposeTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Transpose"
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
        let data = inputs[0];

        // Get the input node
        let input_node = builder
            .graph()
            .node(data)
            .ok_or_else(|| TranslationError::IrBuilder("Transpose: input not found".to_string()))?;

        // Determine permutation
        let perm: Vec<usize> = if let Some(perm_attr) = node.get_ints("perm") {
            perm_attr.iter().map(|&x| x as usize).collect()
        } else {
            // Default: reverse all axes
            (0..input_node.op.shape.rank()).rev().collect()
        };

        // Validate permutation
        let rank = input_node.op.shape.rank();
        if perm.len() != rank {
            return Err(TranslationError::invalid_attribute(
                "perm",
                format!(
                    "perm length ({}) must match input rank ({})",
                    perm.len(),
                    rank
                ),
            ));
        }

        // Check for duplicate or out-of-range values
        let mut seen = vec![false; rank];
        for &p in &perm {
            if p >= rank {
                return Err(TranslationError::invalid_attribute(
                    "perm",
                    format!("perm value {} is out of range for rank {}", p, rank),
                ));
            }
            if seen[p] {
                return Err(TranslationError::invalid_attribute(
                    "perm",
                    format!("duplicate axis {} in perm", p),
                ));
            }
            seen[p] = true;
        }

        // Check if input is a Constant for constant folding
        if let NodeOp::Constant { data: const_data } = &input_node.op.op {
            // Get input shape dimensions as static values
            let in_dims: Vec<usize> = input_node
                .op
                .shape
                .dims
                .iter()
                .map(|d| d.static_value().unwrap_or(1))
                .collect();

            // Calculate output shape
            let out_dims: Vec<Dim> = perm.iter().map(|&p| Dim::Static(in_dims[p])).collect();
            let out_shape = Shape::new(out_dims);

            // Perform the transpose
            let transposed_data = transpose_constant_data(const_data, &in_dims, &perm);

            tracing::debug!(
                "Transpose: constant folding {:?} with perm {:?} -> shape {:?}",
                in_dims,
                perm,
                out_shape
            );

            let result = builder.constant(transposed_data, out_shape);
            return Ok(vec![result]);
        }

        // Non-constant path: emit transpose op
        let perm_i32: Vec<i32> = perm.iter().map(|&x| x as i32).collect();
        let result = builder
            .transpose(data, perm_i32)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }
}

/// Transpose constant data according to permutation.
fn transpose_constant_data(data: &ConstantData, in_dims: &[usize], perm: &[usize]) -> ConstantData {
    match data {
        ConstantData::F32(values) => ConstantData::F32(transpose_nd(values, in_dims, perm)),
        ConstantData::F64(values) => ConstantData::F64(transpose_nd(values, in_dims, perm)),
        ConstantData::I32(values) => ConstantData::I32(transpose_nd(values, in_dims, perm)),
        ConstantData::I64(values) => ConstantData::I64(transpose_nd(values, in_dims, perm)),
        ConstantData::Bool(values) => ConstantData::Bool(transpose_nd(values, in_dims, perm)),
        ConstantData::U8(values) => ConstantData::U8(transpose_nd(values, in_dims, perm)),
    }
}

/// Generic N-dimensional transpose.
fn transpose_nd<T: Clone>(data: &[T], in_dims: &[usize], perm: &[usize]) -> Vec<T> {
    let ndim = in_dims.len();
    if ndim == 0 || data.is_empty() {
        return data.to_vec();
    }

    // Calculate output dimensions
    let out_dims: Vec<usize> = perm.iter().map(|&p| in_dims[p]).collect();

    // Calculate strides for input tensor
    let mut in_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
    }

    // Calculate strides for output tensor
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }

    let total_elements: usize = out_dims.iter().product();
    let mut result = Vec::with_capacity(total_elements);

    // For each output position, compute corresponding input position
    for out_idx in 0..total_elements {
        // Convert flat index to multi-dimensional index in output space
        let mut out_coords = vec![0usize; ndim];
        let mut remaining = out_idx;
        for i in 0..ndim {
            out_coords[i] = remaining / out_strides[i];
            remaining %= out_strides[i];
        }

        // Map output coordinates to input coordinates using inverse permutation
        let mut in_coords = vec![0usize; ndim];
        for i in 0..ndim {
            in_coords[perm[i]] = out_coords[i];
        }

        // Convert multi-dimensional index to flat index in input space
        let in_idx: usize = in_coords
            .iter()
            .zip(in_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();

        result.push(data[in_idx].clone());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::DType;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "transpose_test".to_string(),
            op_type: "Transpose".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_perm(perm: Vec<i64>) -> NodeProto {
        NodeProto {
            name: "transpose_test".to_string(),
            op_type: "Transpose".to_string(),
            attribute: vec![AttributeProto {
                name: "perm".to_string(),
                ints: perm,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_transpose_default_perm() {
        let translator = TransposeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);

        // Default perm reverses axes: [2,3,4] -> [4,3,2]
        let node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(node.op.shape.dims, Shape::static_shape(&[4, 3, 2]).dims);
    }

    #[test]
    fn test_transpose_explicit_perm() {
        let translator = TransposeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let result = translator.translate(&make_node_with_perm(vec![0, 2, 1]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        // perm [0,2,1]: [2,3,4] -> [2,4,3]
        let node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(node.op.shape.dims, Shape::static_shape(&[2, 4, 3]).dims);
    }

    #[test]
    fn test_transpose_2d() {
        let translator = TransposeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[3, 4]), DType::F32);

        let result = translator.translate(&make_node_with_perm(vec![1, 0]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        // Transpose: [3,4] -> [4,3]
        let node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(node.op.shape.dims, Shape::static_shape(&[4, 3]).dims);
    }

    #[test]
    fn test_transpose_constant_folding() {
        let translator = TransposeTranslator;
        let mut builder = GraphBuilder::new();

        // 2x3 matrix: [[1,2,3], [4,5,6]]
        let data = builder.constant(
            ConstantData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Shape::static_shape(&[2, 3]),
        );

        let result = translator.translate(&make_node_with_perm(vec![1, 0]), &[data], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // Should be constant folded
        if let NodeOp::Constant { data } = &node.op.op {
            if let ConstantData::F32(values) = data {
                // Transposed: [[1,4], [2,5], [3,6]]
                assert_eq!(values.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            } else {
                panic!("Expected F32 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_transpose_4d_batch_channel_swap() {
        let translator = TransposeTranslator;
        let mut builder = GraphBuilder::new();

        // NCHW -> NHWC
        let x = builder.input("x", Shape::static_shape(&[1, 3, 224, 224]), DType::F32);

        let result =
            translator.translate(&make_node_with_perm(vec![0, 2, 3, 1]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(
            node.op.shape.dims,
            Shape::static_shape(&[1, 224, 224, 3]).dims
        );
    }

    #[test]
    fn test_transpose_identity_perm() {
        let translator = TransposeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // Identity permutation
        let result = translator.translate(&make_node_with_perm(vec![0, 1]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(node.op.shape.dims, Shape::static_shape(&[2, 3]).dims);
    }

    #[test]
    fn test_transpose_1d() {
        let translator = TransposeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[5]), DType::F32);

        let result = translator.translate(&make_node_with_perm(vec![0]), &[x], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_transpose_no_inputs() {
        let translator = TransposeTranslator;
        let err = translator.input_requirement().validate(0, "Transpose");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount {
                expected: 1,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_transpose_too_many_inputs() {
        let translator = TransposeTranslator;
        let err = translator.input_requirement().validate(2, "Transpose");
        assert!(err.is_err());
    }

    #[test]
    fn test_transpose_invalid_perm_length() {
        let translator = TransposeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // Wrong perm length
        let result = translator.translate(&make_node_with_perm(vec![0, 1, 2]), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("perm length"));
    }

    #[test]
    fn test_transpose_perm_out_of_range() {
        let translator = TransposeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // Out of range axis
        let result = translator.translate(&make_node_with_perm(vec![0, 5]), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn test_transpose_duplicate_axis() {
        let translator = TransposeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // Duplicate axis
        let result = translator.translate(&make_node_with_perm(vec![1, 1]), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_op_type() {
        let translator = TransposeTranslator;
        assert_eq!(translator.onnx_op_type(), "Transpose");
    }

    #[test]
    fn test_input_requirement() {
        let translator = TransposeTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::Exact(1)));
    }

    #[test]
    fn test_supports_constant_folding() {
        let translator = TransposeTranslator;
        assert!(translator.supports_constant_folding());
    }

    // ===== Helper Function Tests =====

    #[test]
    fn test_transpose_nd_2x3() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose_nd(&data, &[2, 3], &[1, 0]);
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_nd_empty() {
        let data: Vec<f32> = vec![];
        let result = transpose_nd(&data, &[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_transpose_nd_scalar() {
        let data = vec![42.0f32];
        let result = transpose_nd(&data, &[], &[]);
        assert_eq!(result, vec![42.0]);
    }
}
