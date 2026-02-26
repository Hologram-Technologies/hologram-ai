//! Concat operation - concatenate tensors along an axis.

use anyhow::{Context, Result, bail};
use hologram::compiler::{ConstantData, OpKind};

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX Concat operation.
pub struct ConcatOp;

impl OpTranslator for ConcatOp {
    fn op_type(&self) -> &'static str {
        "Concat"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        // All inputs must be constants
        let all_constant = node.input.iter().all(|name| ctx.is_constant(name));

        if !all_constant || node.input.is_empty() {
            return None;
        }

        let axis = get_axis(node)?;

        // Collect input data and shapes
        let mut input_data = Vec::new();
        let mut input_shapes = Vec::new();
        let mut dtype = None;

        for name in &node.input {
            let node_ref = ctx.get_node(name)?;
            let data = ctx.get_constant_data(name)?.clone();
            input_shapes.push(node_ref.shape.clone());
            input_data.push(data);
            dtype = Some(node_ref.dtype);
        }

        let dtype = dtype?;
        let first_shape = &input_shapes[0];
        let rank = first_shape.len();
        let axis = normalize_axis(axis, rank);

        // Compute output shape
        let concat_dim: usize = input_shapes.iter().map(|s| s[axis]).sum();
        let mut output_shape = first_shape.clone();
        output_shape[axis] = concat_dim;

        // Concatenate data
        let result = concat_constant_data(&input_data, &input_shapes, axis)?;

        tracing::debug!(
            "Concat '{}': {} inputs -> {:?}",
            node.output.first().unwrap_or(&String::new()),
            node.input.len(),
            output_shape
        );

        Some(TranslateResult::constant(output_shape, dtype, result))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        if node.input.is_empty() {
            bail!("Concat has no inputs");
        }

        let axis = get_axis(node).context("Concat missing axis attribute")?;

        let first_name = node.input.first().context("Concat has no inputs")?;
        let first_node = ctx.get_node(first_name).context("First input not found")?;

        let rank = first_node.shape.len();
        let axis = normalize_axis(axis, rank);

        // Compute output shape
        let mut output_shape = first_node.shape.clone();
        let concat_dim: usize = node
            .input
            .iter()
            .map(|name| {
                ctx.get_node(name)
                    .map(|n| n.shape.get(axis).copied().unwrap_or(0))
                    .unwrap_or(0)
            })
            .sum();
        output_shape[axis] = concat_dim;

        Ok(TranslateResult::runtime(
            OpKind::Concat {
                axis,
                num_inputs: node.input.len(),
            },
            output_shape,
            first_node.dtype,
        ))
    }
}

fn get_axis(node: &proto::NodeProto) -> Option<i64> {
    node.attribute
        .iter()
        .find(|a| a.name == "axis")
        .map(|a| a.i)
}

fn normalize_axis(axis: i64, rank: usize) -> usize {
    if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    }
}

fn concat_constant_data(
    inputs: &[ConstantData],
    shapes: &[Vec<usize>],
    axis: usize,
) -> Option<ConstantData> {
    if inputs.is_empty() {
        return None;
    }

    // For axis 0 concatenation (most common), just append data
    if axis == 0 {
        return concat_axis_0(inputs);
    }

    // For other axes, we need to interleave data
    concat_general(inputs, shapes, axis)
}

fn concat_axis_0(inputs: &[ConstantData]) -> Option<ConstantData> {
    macro_rules! concat_impl {
        ($variant:ident) => {{
            let mut result = Vec::new();
            for input in inputs {
                if let ConstantData::$variant(data) = input {
                    result.extend_from_slice(data);
                } else {
                    return None;
                }
            }
            Some(ConstantData::$variant(result))
        }};
    }

    match &inputs[0] {
        ConstantData::F32(_) => concat_impl!(F32),
        ConstantData::F64(_) => concat_impl!(F64),
        ConstantData::I32(_) => concat_impl!(I32),
        ConstantData::I64(_) => concat_impl!(I64),
        ConstantData::U8(_) => concat_impl!(U8),
        ConstantData::U16(_) => concat_impl!(U16),
        ConstantData::U32(_) => concat_impl!(U32),
        ConstantData::Bool(_) => concat_impl!(Bool),
    }
}

fn concat_general(
    inputs: &[ConstantData],
    shapes: &[Vec<usize>],
    _axis: usize,
) -> Option<ConstantData> {
    // This is more complex - for now, return None and let runtime handle it
    // Full implementation would require computing proper offsets and interleaving

    // Simple case: 1D tensors
    if shapes[0].len() == 1 {
        return concat_axis_0(inputs);
    }

    None // Fall back to runtime for complex cases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_axis_0() {
        let a = ConstantData::I64(vec![1, 2, 3]);
        let b = ConstantData::I64(vec![4, 5, 6]);

        let result = concat_axis_0(&[a, b]).unwrap();

        if let ConstantData::I64(v) = result {
            assert_eq!(v, vec![1, 2, 3, 4, 5, 6]);
        } else {
            panic!("Expected I64");
        }
    }
}
