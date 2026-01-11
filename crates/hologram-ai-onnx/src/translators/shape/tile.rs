//! Tile operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{ConstantData, Dim, GraphBuilder, NodeIndex, NodeOp, Shape};

/// Translator for ONNX Tile operation.
///
/// Tile replicates the input tensor `repeats` times along each dimension.
///
/// # Inputs
/// - data: Input tensor to tile
/// - repeats: 1D tensor specifying the number of repeats per dimension
///
/// # Output Shape
/// output_shape[i] = input_shape[i] * repeats[i]
///
/// # Constant Folding
/// If both inputs are constants, the tile is performed at compile time.
#[derive(Debug, Default)]
pub struct TileTranslator;

impl OnnxTranslator for TileTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Tile"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(2)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let data = inputs[0];
        let repeats_input = inputs[1];

        // Get data node
        let data_node = builder
            .graph()
            .node(data)
            .ok_or_else(|| TranslationError::IrBuilder("Tile: data input not found".to_string()))?;
        let data_shape = data_node.op.shape.clone();

        // Get repeats node
        let repeats_node = builder.graph().node(repeats_input).ok_or_else(|| {
            TranslationError::IrBuilder("Tile: repeats input not found".to_string())
        })?;

        // Try to get repeats as constant for shape inference
        let repeats: Option<Vec<i64>> = match &repeats_node.op.op {
            NodeOp::Constant { data } => match data {
                ConstantData::I64(vals) => Some(vals.clone()),
                ConstantData::I32(vals) => Some(vals.iter().map(|&v| v as i64).collect()),
                _ => None,
            },
            _ => None,
        };

        // If repeats is not constant, delegate to runtime tile
        let Some(repeats) = repeats else {
            let result = builder
                .tile(data, repeats_input)
                .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
            return Ok(vec![result]);
        };

        // Validate repeats length matches input rank
        let input_dims = &data_shape.dims;
        if input_dims.len() != repeats.len() {
            return Err(TranslationError::ShapeInference(format!(
                "Tile: repeats length ({}) must match input rank ({})",
                repeats.len(),
                input_dims.len()
            )));
        }

        // Compute output shape
        let output_dims: Vec<Dim> = input_dims
            .iter()
            .zip(repeats.iter())
            .map(|(dim, &rep)| match dim {
                Dim::Static(n) => Dim::Static((*n as i64 * rep) as usize),
                Dim::Symbolic(name) => {
                    // For symbolic dims, keep as symbolic (could create expression)
                    Dim::Symbolic(name.clone())
                }
                Dim::Dynamic => Dim::Dynamic,
            })
            .collect();
        let output_shape = Shape::new(output_dims);

        tracing::debug!(
            "Tile: repeats = {:?}, input shape = {:?}, output shape = {:?}",
            repeats,
            data_shape,
            output_shape
        );

        // Try constant folding if data is also constant
        if let NodeOp::Constant { data: const_data } = &data_node.op.op {
            // Get concrete dimensions for constant folding
            let input_dims_concrete: Vec<usize> = input_dims
                .iter()
                .filter_map(|d| match d {
                    Dim::Static(n) => Some(*n),
                    _ => None,
                })
                .collect();

            if input_dims_concrete.len() == input_dims.len() {
                // All dims are static, we can tile at compile time
                let output_dims_concrete: Vec<usize> = output_shape
                    .dims
                    .iter()
                    .filter_map(|d| match d {
                        Dim::Static(n) => Some(*n),
                        _ => None,
                    })
                    .collect();

                if output_dims_concrete.len() == output_shape.dims.len()
                    && let Some(tiled_data) =
                        tile_constant_data(const_data, &input_dims_concrete, &output_dims_concrete)
                {
                    tracing::debug!("Tile: constant folding succeeded");
                    let result = builder.constant(tiled_data, output_shape);
                    return Ok(vec![result]);
                }
            }
        }

        // Runtime tile
        let result = builder
            .tile(data, repeats_input)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }
}

/// Tile constant data at compile time.
fn tile_constant_data(
    data: &ConstantData,
    input_dims: &[usize],
    output_dims: &[usize],
) -> Option<ConstantData> {
    match data {
        ConstantData::F32(values) => {
            Some(ConstantData::F32(tile_nd(values, input_dims, output_dims)))
        }
        ConstantData::F64(values) => {
            Some(ConstantData::F64(tile_nd(values, input_dims, output_dims)))
        }
        ConstantData::I32(values) => {
            Some(ConstantData::I32(tile_nd(values, input_dims, output_dims)))
        }
        ConstantData::I64(values) => {
            Some(ConstantData::I64(tile_nd(values, input_dims, output_dims)))
        }
        ConstantData::Bool(values) => {
            Some(ConstantData::Bool(tile_nd(values, input_dims, output_dims)))
        }
        ConstantData::U8(values) => {
            Some(ConstantData::U8(tile_nd(values, input_dims, output_dims)))
        }
    }
}

/// Generic N-dimensional tile operation.
fn tile_nd<T: Clone>(data: &[T], input_dims: &[usize], output_dims: &[usize]) -> Vec<T> {
    let output_size: usize = output_dims.iter().product();
    if output_size == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(output_size);

    // For each position in output, find corresponding input position
    for out_idx in 0..output_size {
        // Convert linear index to multi-dimensional indices
        let mut remaining = out_idx;
        let mut out_coords = vec![0usize; output_dims.len()];
        for i in (0..output_dims.len()).rev() {
            out_coords[i] = remaining % output_dims[i];
            remaining /= output_dims[i];
        }

        // Map to input coordinates using modulo
        let in_coords: Vec<usize> = out_coords
            .iter()
            .zip(input_dims.iter())
            .map(|(&out_c, &in_dim)| out_c % in_dim)
            .collect();

        // Convert back to linear index
        let mut in_idx = 0;
        let mut stride = 1;
        for i in (0..input_dims.len()).rev() {
            in_idx += in_coords[i] * stride;
            stride *= input_dims[i];
        }

        result.push(data[in_idx].clone());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::DType;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "tile_test".to_string(),
            op_type: "Tile".to_string(),
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_tile_2d() {
        let translator = TileTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3]), DType::F32);
        let repeats = builder.constant(ConstantData::I64(vec![2, 3]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, repeats], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_tile_1d() {
        let translator = TileTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[4]), DType::F32);
        let repeats = builder.constant(ConstantData::I64(vec![3]), Shape::static_shape(&[1]));

        let result = translator.translate(&make_node(), &[data, repeats], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tile_3d() {
        let translator = TileTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3, 4]), DType::F32);
        let repeats = builder.constant(ConstantData::I64(vec![1, 2, 1]), Shape::static_shape(&[3]));

        let result = translator.translate(&make_node(), &[data, repeats], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tile_identity() {
        let translator = TileTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3]), DType::F32);
        let repeats = builder.constant(ConstantData::I64(vec![1, 1]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, repeats], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tile_constant_folding() {
        let translator = TileTranslator;
        let mut builder = GraphBuilder::new();

        // 2x2 matrix: [[1, 2], [3, 4]]
        let data = builder.constant(
            ConstantData::F32(vec![1.0, 2.0, 3.0, 4.0]),
            Shape::static_shape(&[2, 2]),
        );
        let repeats = builder.constant(ConstantData::I64(vec![2, 2]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, repeats], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // Should be constant folded
        if let NodeOp::Constant { data } = &node.op.op {
            if let ConstantData::F32(values) = data {
                // Tiled to 4x4
                assert_eq!(values.len(), 16);
                // First row should be [1, 2, 1, 2]
                assert_eq!(values[0], 1.0);
                assert_eq!(values[1], 2.0);
                assert_eq!(values[2], 1.0);
                assert_eq!(values[3], 2.0);
            } else {
                panic!("Expected F32 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_tile_dynamic_repeats() {
        let translator = TileTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3]), DType::F32);
        // Non-constant repeats
        let repeats = builder.input("repeats", Shape::static_shape(&[2]), DType::I64);

        let result = translator.translate(&make_node(), &[data, repeats], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tile_large_repeat() {
        let translator = TileTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 1]), DType::F32);
        let repeats =
            builder.constant(ConstantData::I64(vec![100, 100]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, repeats], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tile_scalar() {
        let translator = TileTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[]), DType::F32);
        let repeats = builder.constant(ConstantData::I64(vec![]), Shape::static_shape(&[0]));

        let result = translator.translate(&make_node(), &[data, repeats], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_tile_no_inputs() {
        let translator = TileTranslator;
        let err = translator.input_requirement().validate(0, "Tile");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount {
                expected: 2,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_tile_one_input() {
        let translator = TileTranslator;
        let err = translator.input_requirement().validate(1, "Tile");
        assert!(err.is_err());
    }

    #[test]
    fn test_tile_too_many_inputs() {
        let translator = TileTranslator;
        let err = translator.input_requirement().validate(3, "Tile");
        assert!(err.is_err());
    }

    #[test]
    fn test_tile_rank_mismatch() {
        let translator = TileTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3]), DType::F32);
        // Wrong number of repeats
        let repeats = builder.constant(ConstantData::I64(vec![2, 3, 4]), Shape::static_shape(&[3]));

        let result = translator.translate(&make_node(), &[data, repeats], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("must match"));
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_op_type() {
        let translator = TileTranslator;
        assert_eq!(translator.onnx_op_type(), "Tile");
    }

    #[test]
    fn test_input_requirement() {
        let translator = TileTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::Exact(2)));
        assert!(!req.accepts_zero());
    }

    #[test]
    fn test_supports_constant_folding() {
        let translator = TileTranslator;
        assert!(translator.supports_constant_folding());
    }

    // ===== Helper Function Tests =====

    #[test]
    fn test_tile_nd_2x2_to_4x4() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = tile_nd(&data, &[2, 2], &[4, 4]);
        assert_eq!(result.len(), 16);
        // First row: [1, 2, 1, 2]
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 2.0);
    }

    #[test]
    fn test_tile_nd_1d() {
        let data = vec![1, 2, 3];
        let result = tile_nd(&data, &[3], &[9]);
        assert_eq!(result, vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn test_tile_nd_empty() {
        let data: Vec<f32> = vec![];
        let result = tile_nd(&data, &[0], &[0]);
        assert!(result.is_empty());
    }
}
