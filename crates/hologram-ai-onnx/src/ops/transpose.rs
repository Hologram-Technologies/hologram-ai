//! Transpose operation - permute tensor dimensions.

use anyhow::{Context, Result};
use hologram::compiler::{ConstantData, OpKind};

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX Transpose operation.
pub struct TransposeOp;

impl OpTranslator for TransposeOp {
    fn op_type(&self) -> &'static str {
        "Transpose"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let input_name = node.input.first()?;

        if !ctx.is_constant(input_name) {
            return None;
        }

        let input_node = ctx.get_node(input_name)?;
        let input_const = ctx.get_constant_data(input_name)?;

        let perm = get_perm(node, input_node.shape.len());
        let output_shape = permute_shape(&input_node.shape, &perm);

        // For constants, we need to actually permute the data
        let result_data = transpose_constant(input_const, &input_node.shape, &perm)?;

        Some(TranslateResult::constant(
            output_shape,
            input_node.dtype,
            result_data,
        ))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("Transpose has no input")?;
        let input_node = ctx
            .get_node(input_name)
            .context("Transpose input not found")?;

        let perm = get_perm(node, input_node.shape.len());
        let output_shape = permute_shape(&input_node.shape, &perm);

        Ok(TranslateResult::runtime(
            OpKind::Transpose { perm },
            output_shape,
            input_node.dtype,
        ))
    }
}

fn get_perm(node: &proto::NodeProto, rank: usize) -> Vec<usize> {
    node.attribute
        .iter()
        .find(|a| a.name == "perm")
        .map(|a| a.ints.iter().map(|&v| v as usize).collect())
        .unwrap_or_else(|| {
            // Default: reverse dimensions
            (0..rank).rev().collect()
        })
}

fn permute_shape(shape: &[usize], perm: &[usize]) -> Vec<usize> {
    perm.iter().map(|&i| shape[i]).collect()
}

fn transpose_constant(
    data: &ConstantData,
    shape: &[usize],
    perm: &[usize],
) -> Option<ConstantData> {
    // For simple cases like scalars or 1D, data doesn't change
    if shape.len() <= 1 {
        return Some(data.clone());
    }

    // For 2D transpose, we can implement efficiently
    if shape.len() == 2 && perm == [1, 0] {
        return transpose_2d(data, shape);
    }

    // For more complex cases, implement general transpose
    // This is expensive but correct
    general_transpose(data, shape, perm)
}

fn transpose_2d(data: &ConstantData, shape: &[usize]) -> Option<ConstantData> {
    let rows = shape[0];
    let cols = shape[1];

    macro_rules! transpose_impl {
        ($data_vec:expr, $variant:ident) => {{
            let mut result = vec![Default::default(); rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    result[j * rows + i] = $data_vec[i * cols + j].clone();
                }
            }
            Some(ConstantData::$variant(result))
        }};
    }

    match data {
        ConstantData::F32(v) => transpose_impl!(v, F32),
        ConstantData::F64(v) => transpose_impl!(v, F64),
        ConstantData::I32(v) => transpose_impl!(v, I32),
        ConstantData::I64(v) => transpose_impl!(v, I64),
        _ => None,
    }
}

fn general_transpose(data: &ConstantData, shape: &[usize], perm: &[usize]) -> Option<ConstantData> {
    let total: usize = shape.iter().product();
    let output_shape = permute_shape(shape, perm);

    // Compute strides for input and output
    let input_strides = compute_strides(shape);
    let output_strides = compute_strides(&output_shape);

    macro_rules! transpose_impl {
        ($data_vec:expr, $variant:ident) => {{
            let mut result = vec![Default::default(); total];

            for i in 0..total {
                // Convert flat index to coords in output space
                let mut output_coords = vec![0; shape.len()];
                let mut remaining = i;
                for (d, &stride) in output_strides.iter().enumerate() {
                    output_coords[d] = remaining / stride;
                    remaining %= stride;
                }

                // Map output coords to input coords via inverse perm
                let mut input_coords = vec![0; shape.len()];
                for (output_dim, &input_dim) in perm.iter().enumerate() {
                    input_coords[input_dim] = output_coords[output_dim];
                }

                // Convert input coords to flat index
                let input_idx: usize = input_coords
                    .iter()
                    .zip(input_strides.iter())
                    .map(|(&c, &s)| c * s)
                    .sum();

                result[i] = $data_vec[input_idx].clone();
            }

            Some(ConstantData::$variant(result))
        }};
    }

    match data {
        ConstantData::F32(v) => transpose_impl!(v, F32),
        ConstantData::F64(v) => transpose_impl!(v, F64),
        ConstantData::I32(v) => transpose_impl!(v, I32),
        ConstantData::I64(v) => transpose_impl!(v, I64),
        _ => None,
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permute_shape() {
        assert_eq!(permute_shape(&[2, 3, 4], &[2, 0, 1]), vec![4, 2, 3]);
        assert_eq!(permute_shape(&[2, 3], &[1, 0]), vec![3, 2]);
    }

    #[test]
    fn test_transpose_2d() {
        let data = ConstantData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = transpose_2d(&data, &[2, 3]).unwrap();

        if let ConstantData::F32(v) = result {
            // Original: [[1,2,3], [4,5,6]]
            // Transposed: [[1,4], [2,5], [3,6]]
            assert_eq!(v, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        } else {
            panic!("Expected F32");
        }
    }
}
