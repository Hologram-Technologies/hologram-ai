//! Tile operation - repeat tensor along each dimension.

use anyhow::{Context, Result};
use hologram::compiler::{ConstantData, OpKind};

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX Tile operation.
///
/// Constructs a tensor by tiling a given tensor.
/// Repeats the input tensor along each dimension N times where N is
/// specified in the repeats input.
pub struct TileOp;

impl OpTranslator for TileOp {
    fn op_type(&self) -> &'static str {
        "Tile"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let input_name = node.input.first()?;
        let repeats_name = node.input.get(1)?;

        // Only fold if both inputs are constant
        if !ctx.is_constant(input_name) || !ctx.is_constant(repeats_name) {
            return None;
        }

        let input_data = ctx.get_constant_data(input_name)?;
        let input_node = ctx.get_node(input_name)?;
        let repeats = ctx.get_constant_i64(repeats_name)?;

        // Compute output shape: input_shape[i] * repeats[i]
        let output_shape: Vec<usize> = input_node
            .shape
            .iter()
            .zip(repeats.iter())
            .map(|(&dim, &rep)| dim * rep as usize)
            .collect();

        // Tile the constant data
        let tiled_data = tile_constant(input_data, &input_node.shape, &repeats)?;

        tracing::debug!(
            "Tile '{}': {:?} x {:?} -> {:?}",
            node.output.first().unwrap_or(&String::new()),
            input_node.shape,
            repeats,
            output_shape
        );

        Some(TranslateResult::constant(
            output_shape,
            input_node.dtype,
            tiled_data,
        ))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("Tile has no input")?;
        let repeats_name = node.input.get(1).context("Tile has no repeats input")?;

        let input_node = ctx.get_node(input_name).context("Tile input not found")?;

        // Get repeats from constant (required for Tile)
        let repeats = ctx
            .get_constant_i64(repeats_name)
            .context("Tile requires constant repeats input")?;

        let multiples: Vec<usize> = repeats.iter().map(|&r| r as usize).collect();

        let output_shape: Vec<usize> = input_node
            .shape
            .iter()
            .zip(multiples.iter())
            .map(|(&dim, &rep)| dim * rep)
            .collect();

        // Use native hologram Tile operation
        Ok(TranslateResult::runtime_with_inputs(
            OpKind::Tile { multiples },
            output_shape,
            input_node.dtype,
            1, // Only the data input creates an edge, not the repeats
        ))
    }
}

/// Tile constant data by repeating along each dimension.
fn tile_constant(
    data: &ConstantData,
    src_shape: &[usize],
    repeats: &[i64],
) -> Option<ConstantData> {
    let output_shape: Vec<usize> = src_shape
        .iter()
        .zip(repeats.iter())
        .map(|(&dim, &rep)| dim * rep as usize)
        .collect();

    let total_size: usize = output_shape.iter().product();

    macro_rules! tile_impl {
        ($variant:ident, $values:expr) => {{
            let mut result = Vec::with_capacity(total_size);
            for i in 0..total_size {
                let src_idx = tile_index(i, &output_shape, src_shape);
                result.push($values[src_idx].clone());
            }
            Some(ConstantData::$variant(result))
        }};
    }

    match data {
        ConstantData::F32(v) => tile_impl!(F32, v),
        ConstantData::F64(v) => tile_impl!(F64, v),
        ConstantData::I32(v) => tile_impl!(I32, v),
        ConstantData::I64(v) => tile_impl!(I64, v),
        ConstantData::U8(v) => tile_impl!(U8, v),
        ConstantData::U16(v) => tile_impl!(U16, v),
        ConstantData::U32(v) => tile_impl!(U32, v),
        ConstantData::Bool(v) => tile_impl!(Bool, v),
    }
}

/// Compute the source index for a tile operation.
/// For tiling, we wrap around the source dimensions.
fn tile_index(flat_idx: usize, output_shape: &[usize], input_shape: &[usize]) -> usize {
    if input_shape.is_empty() || input_shape.iter().product::<usize>() == 1 {
        return 0; // Scalar
    }

    let mut result = 0;
    let mut stride = 1;
    let mut out_stride = 1;

    for i in (0..input_shape.len()).rev() {
        let out_coord = (flat_idx / out_stride) % output_shape[i];
        // Wrap coordinate into source shape
        let src_coord = out_coord % input_shape[i];
        result += src_coord * stride;
        stride *= input_shape[i];
        out_stride *= output_shape[i];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::compiler::{DType, OpNode, OperationGraph};
    use std::collections::HashMap;

    #[test]
    fn test_tile_1d() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Input [1, 2, 3]
        let input =
            OpNode::new(0, OpKind::Constant, vec![3], DType::I64).with_name("input".to_string());
        graph.nodes.push(input);
        value_to_node.insert("input".to_string(), 0);
        graph.constants.push(ConstantData::I64(vec![1, 2, 3]));

        // Repeats [2]
        let repeats =
            OpNode::new(1, OpKind::Constant, vec![1], DType::I64).with_name("repeats".to_string());
        graph.nodes.push(repeats);
        value_to_node.insert("repeats".to_string(), 1);
        graph.constants.push(ConstantData::I64(vec![2]));

        let ctx = TranslateContext::new(&graph, &value_to_node, &graph.constants);

        let node = proto::NodeProto {
            input: vec!["input".to_string(), "repeats".to_string()],
            output: vec!["out".to_string()],
            op_type: "Tile".to_string(),
            ..Default::default()
        };

        let result = TileOp.try_fold(&node, &ctx).expect("Should fold");
        assert_eq!(result.shape, vec![6]);

        if let Some(ConstantData::I64(data)) = result.constant_data {
            assert_eq!(data, vec![1, 2, 3, 1, 2, 3]);
        } else {
            panic!("Expected I64 constant data");
        }
    }

    #[test]
    fn test_tile_2d() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Input [[1, 2], [3, 4]] shape [2, 2]
        let input =
            OpNode::new(0, OpKind::Constant, vec![2, 2], DType::I64).with_name("input".to_string());
        graph.nodes.push(input);
        value_to_node.insert("input".to_string(), 0);
        graph.constants.push(ConstantData::I64(vec![1, 2, 3, 4]));

        // Repeats [2, 3]
        let repeats =
            OpNode::new(1, OpKind::Constant, vec![2], DType::I64).with_name("repeats".to_string());
        graph.nodes.push(repeats);
        value_to_node.insert("repeats".to_string(), 1);
        graph.constants.push(ConstantData::I64(vec![2, 3]));

        let ctx = TranslateContext::new(&graph, &value_to_node, &graph.constants);

        let node = proto::NodeProto {
            input: vec!["input".to_string(), "repeats".to_string()],
            output: vec!["out".to_string()],
            op_type: "Tile".to_string(),
            ..Default::default()
        };

        let result = TileOp.try_fold(&node, &ctx).expect("Should fold");
        assert_eq!(result.shape, vec![4, 6]);

        if let Some(ConstantData::I64(data)) = result.constant_data {
            // Expected: first row [1,2,1,2,1,2] repeated 4 times (2 original rows * 2 repeats)
            // [[1,2,1,2,1,2], [3,4,3,4,3,4], [1,2,1,2,1,2], [3,4,3,4,3,4]]
            assert_eq!(
                data,
                vec![
                    1, 2, 1, 2, 1, 2, // row 0
                    3, 4, 3, 4, 3, 4, // row 1
                    1, 2, 1, 2, 1, 2, // row 2 (repeat of row 0)
                    3, 4, 3, 4, 3, 4, // row 3 (repeat of row 1)
                ]
            );
        } else {
            panic!("Expected I64 constant data");
        }
    }

    #[test]
    fn test_tile_scalar() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Input scalar [5]
        let input =
            OpNode::new(0, OpKind::Constant, vec![1], DType::F32).with_name("input".to_string());
        graph.nodes.push(input);
        value_to_node.insert("input".to_string(), 0);
        graph.constants.push(ConstantData::F32(vec![5.0]));

        // Repeats [4]
        let repeats =
            OpNode::new(1, OpKind::Constant, vec![1], DType::I64).with_name("repeats".to_string());
        graph.nodes.push(repeats);
        value_to_node.insert("repeats".to_string(), 1);
        graph.constants.push(ConstantData::I64(vec![4]));

        let ctx = TranslateContext::new(&graph, &value_to_node, &graph.constants);

        let node = proto::NodeProto {
            input: vec!["input".to_string(), "repeats".to_string()],
            output: vec!["out".to_string()],
            op_type: "Tile".to_string(),
            ..Default::default()
        };

        let result = TileOp.try_fold(&node, &ctx).expect("Should fold");
        assert_eq!(result.shape, vec![4]);

        if let Some(ConstantData::F32(data)) = result.constant_data {
            assert_eq!(data, vec![5.0, 5.0, 5.0, 5.0]);
        } else {
            panic!("Expected F32 constant data");
        }
    }
}
