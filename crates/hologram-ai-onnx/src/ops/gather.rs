//! Gather operation - index into a tensor using indices.

use anyhow::{Context, Result, bail};
use hologram::compiler::{ConstantData, OpKind};

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX Gather operation.
///
/// Gathers elements from a tensor along an axis using indices.
/// When both inputs are constants, this operation is folded at compile time.
pub struct GatherOp;

impl OpTranslator for GatherOp {
    fn op_type(&self) -> &'static str {
        "Gather"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let data_name = node.input.first()?;
        let indices_name = node.input.get(1)?;

        // Both inputs must be constants
        if !ctx.is_constant(data_name) || !ctx.is_constant(indices_name) {
            return None;
        }

        let data_node = ctx.get_node(data_name)?;
        let indices_node = ctx.get_node(indices_name)?;
        let data_const = ctx.get_constant_data(data_name)?.clone();
        let indices = ctx.get_constant_i64(indices_name)?;
        let axis = get_axis(node).unwrap_or(0);

        let (result_data, result_shape) = gather_constant(
            &data_const,
            &data_node.shape,
            &indices,
            &indices_node.shape,
            axis,
        )
        .ok()?;

        tracing::debug!(
            "Gather '{}': constant-folded to shape {:?}",
            node.output.first().unwrap_or(&String::new()),
            result_shape
        );

        Some(TranslateResult::constant(
            result_shape,
            data_node.dtype,
            result_data,
        ))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let data_name = node.input.first().context("Gather has no data input")?;
        let indices_name = node.input.get(1).context("Gather has no indices input")?;

        let data_node = ctx.get_node(data_name).context("Gather data not found")?;
        let indices_node = ctx
            .get_node(indices_name)
            .context("Gather indices not found")?;

        let axis = get_axis(node).unwrap_or(0);
        let axis = normalize_axis(axis, data_node.shape.len());

        // Compute output shape
        let mut output_shape = Vec::new();
        output_shape.extend_from_slice(&data_node.shape[..axis]);
        output_shape.extend_from_slice(&indices_node.shape);
        output_shape.extend_from_slice(&data_node.shape[axis + 1..]);

        Ok(TranslateResult::runtime(
            OpKind::Gather { axis },
            output_shape,
            data_node.dtype,
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

fn gather_constant(
    data: &ConstantData,
    data_shape: &[usize],
    indices: &[i64],
    indices_shape: &[usize],
    axis: i64,
) -> Result<(ConstantData, Vec<usize>)> {
    let rank = data_shape.len();
    let axis = normalize_axis(axis, rank);

    if axis >= rank {
        bail!("Gather axis {} out of bounds for rank {}", axis, rank);
    }

    // Compute output shape
    let mut output_shape = Vec::new();
    output_shape.extend_from_slice(&data_shape[..axis]);
    output_shape.extend_from_slice(indices_shape);
    output_shape.extend_from_slice(&data_shape[axis + 1..]);

    // Scalar index case (most common for shape extraction)
    if indices_shape.is_empty() && indices.len() == 1 {
        let idx = indices[0];
        let normalized_idx = if idx < 0 {
            (data_shape[axis] as i64 + idx) as usize
        } else {
            idx as usize
        };

        let inner_size: usize = data_shape[axis + 1..].iter().product();
        let stride: usize = data_shape[axis..].iter().product();
        let outer_count: usize = data_shape[..axis].iter().product();

        macro_rules! gather_slice {
            ($data_vec:expr, $variant:ident) => {{
                let start = normalized_idx * inner_size;
                let mut result = Vec::new();
                for outer in 0..outer_count.max(1) {
                    let base = outer * stride + start;
                    result.extend_from_slice(&$data_vec[base..base + inner_size]);
                }
                Ok((ConstantData::$variant(result), output_shape))
            }};
        }

        match data {
            ConstantData::I64(v) => gather_slice!(v, I64),
            ConstantData::I32(v) => gather_slice!(v, I32),
            ConstantData::F32(v) => gather_slice!(v, F32),
            ConstantData::F64(v) => gather_slice!(v, F64),
            ConstantData::U8(v) => gather_slice!(v, U8),
            ConstantData::U16(v) => gather_slice!(v, U16),
            ConstantData::U32(v) => gather_slice!(v, U32),
            ConstantData::Bool(v) => gather_slice!(v, Bool),
        }
    } else {
        // Complex Gather: indices is not a scalar
        // General algorithm: for each index position, gather the corresponding slice from data
        //
        // output_shape = data_shape[:axis] + indices_shape + data_shape[axis+1:]
        // For position bias: data=[32,8], indices=[512,512], axis=0 -> output=[512,512,8]

        let indices_count: usize = indices_shape.iter().product();
        let inner_size: usize = data_shape[axis + 1..].iter().product();
        let outer_count: usize = data_shape[..axis].iter().product();
        let axis_size = data_shape[axis];

        // For axis=0 case (most common)
        if axis == 0 && outer_count <= 1 {
            macro_rules! gather_multi {
                ($data_vec:expr, $variant:ident) => {{
                    let mut result = Vec::with_capacity(indices_count * inner_size);
                    for &idx in indices.iter() {
                        let normalized_idx = if idx < 0 {
                            (axis_size as i64 + idx) as usize
                        } else {
                            idx as usize
                        };

                        // Bounds check
                        if normalized_idx >= axis_size {
                            bail!(
                                "Gather index {} out of bounds for axis size {}",
                                idx,
                                axis_size
                            );
                        }

                        let start = normalized_idx * inner_size;
                        result.extend_from_slice(&$data_vec[start..start + inner_size]);
                    }
                    Ok((ConstantData::$variant(result), output_shape))
                }};
            }

            match data {
                ConstantData::I64(v) => gather_multi!(v, I64),
                ConstantData::I32(v) => gather_multi!(v, I32),
                ConstantData::F32(v) => gather_multi!(v, F32),
                ConstantData::F64(v) => gather_multi!(v, F64),
                ConstantData::U8(v) => gather_multi!(v, U8),
                ConstantData::U16(v) => gather_multi!(v, U16),
                ConstantData::U32(v) => gather_multi!(v, U32),
                ConstantData::Bool(v) => gather_multi!(v, Bool),
            }
        } else {
            // More complex case with outer dimensions - not yet implemented
            bail!(
                "Complex Gather with outer dimensions not yet implemented (axis={}, outer_count={})",
                axis,
                outer_count
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_scalar_index() {
        let data = ConstantData::I64(vec![10, 20, 30]);
        let indices = vec![1i64];

        let (result, shape) = gather_constant(&data, &[3], &indices, &[], 0).unwrap();

        assert_eq!(shape, Vec::<usize>::new()); // Scalar output
        if let ConstantData::I64(v) = result {
            assert_eq!(v, vec![20]);
        } else {
            panic!("Expected I64");
        }
    }

    #[test]
    fn test_gather_from_shape() {
        // Simulating Shape([1, 5, 512]) -> Gather(index=1) -> 5
        let data = ConstantData::I64(vec![1, 5, 512]);
        let indices = vec![1i64];

        let (result, shape) = gather_constant(&data, &[3], &indices, &[], 0).unwrap();

        assert_eq!(shape, Vec::<usize>::new());
        if let ConstantData::I64(v) = result {
            assert_eq!(v, vec![5]);
        } else {
            panic!("Expected I64");
        }
    }
}
