//! Reshape operation - change tensor dimensions.

use anyhow::{Context, Result};
use hologram::compiler::OpKind;

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX Reshape operation.
pub struct ReshapeOp;

impl OpTranslator for ReshapeOp {
    fn op_type(&self) -> &'static str {
        "Reshape"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let data_name = node.input.first()?;

        if !ctx.is_constant(data_name) {
            return None;
        }

        let data_node = ctx.get_node(data_name)?;
        let data_const = ctx.get_constant_data(data_name)?.clone();

        // Get target shape from constant
        let target_shape = get_target_shape(node, ctx, &data_node.shape)?;

        tracing::debug!(
            "Reshape '{}': constant-folded {:?} -> {:?}",
            node.output.first().unwrap_or(&String::new()),
            data_node.shape,
            target_shape
        );

        Some(TranslateResult::constant(
            target_shape,
            data_node.dtype,
            data_const,
        ))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let data_name = node.input.first().context("Reshape has no data input")?;
        let data_node = ctx.get_node(data_name).context("Reshape data not found")?;

        let target_shape = get_target_shape(node, ctx, &data_node.shape)
            .context("Could not determine target shape")?;

        // Only the first input (data) creates an edge, not the shape input
        Ok(TranslateResult::runtime_with_inputs(
            OpKind::Reshape {
                shape: target_shape.clone(),
            },
            target_shape,
            data_node.dtype,
            1, // Only 1 data input (the data tensor), shape is metadata
        ))
    }
}

fn get_target_shape(
    node: &proto::NodeProto,
    ctx: &TranslateContext,
    input_shape: &[usize],
) -> Option<Vec<usize>> {
    let shape_name = node.input.get(1)?;

    // Try to get shape from constant
    if let Some(shape_values) = ctx.get_constant_i64(shape_name) {
        return Some(resolve_shape(&shape_values, input_shape));
    }

    // If shape is not constant, try to infer from context
    let shape_node = ctx.get_node(shape_name)?;

    // For transformer patterns, infer from shape node dimensions
    if shape_node.shape.len() == 1 {
        // Shape tensor tells us the output rank
        let output_rank = shape_node.shape[0];
        let total_elems: usize = input_shape.iter().product();

        // Common transformer patterns
        match output_rank {
            2 => {
                // [batch*seq, hidden] pattern
                return Some(infer_2d_shape(input_shape, total_elems));
            }
            4 => {
                // [batch, heads, seq, head_dim] pattern
                return Some(infer_4d_attention_shape(input_shape, total_elems));
            }
            _ => {}
        }
    }

    // Fallback: preserve input shape (might be wrong but avoids compile failure)
    tracing::warn!(
        "Reshape '{}': inferring shape {:?} from input {:?}. Shape node is not constant.",
        node.name,
        input_shape,
        input_shape
    );
    Some(input_shape.to_vec())
}

fn resolve_shape(shape_values: &[i64], input_shape: &[usize]) -> Vec<usize> {
    let total_elems: usize = input_shape.iter().product();

    // Find -1 (inferred dimension) and compute it
    let mut result: Vec<usize> = shape_values
        .iter()
        .map(|&v| if v == -1 { 0 } else { v as usize })
        .collect();

    // Handle 0 values (copy from input)
    for (i, &v) in shape_values.iter().enumerate() {
        if v == 0 && i < input_shape.len() {
            result[i] = input_shape[i];
        }
    }

    // Compute inferred dimension
    let known_product: usize = result.iter().filter(|&&v| v != 0).product();
    if let Some(inferred) = total_elems.checked_div(known_product) {
        for dim in &mut result {
            if *dim == 0 {
                *dim = inferred;
                break;
            }
        }
    }

    result
}

fn infer_2d_shape(input_shape: &[usize], total: usize) -> Vec<usize> {
    if input_shape.len() >= 2 {
        // [batch, seq, hidden] -> [batch*seq, hidden]
        let last = *input_shape.last().unwrap_or(&1);
        let first = total / last;
        vec![first, last]
    } else {
        vec![total]
    }
}

fn infer_4d_attention_shape(input_shape: &[usize], _total: usize) -> Vec<usize> {
    // Common pattern: [seq, hidden] -> [batch=1, heads, seq, head_dim]
    if input_shape.len() == 2 {
        let seq = input_shape[0];
        let hidden = input_shape[1];

        // Try common head dimensions
        for num_heads in [8, 12, 16, 6, 4] {
            if hidden.is_multiple_of(num_heads) {
                let head_dim = hidden / num_heads;
                return vec![1, num_heads, seq, head_dim];
            }
        }
    }

    // Fallback
    input_shape.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_shape_basic() {
        let result = resolve_shape(&[2, 3, 4], &[24]);
        assert_eq!(result, vec![2, 3, 4]);
    }

    #[test]
    fn test_resolve_shape_infer() {
        let result = resolve_shape(&[2, -1, 4], &[24]);
        assert_eq!(result, vec![2, 3, 4]);
    }

    #[test]
    fn test_resolve_shape_copy() {
        let result = resolve_shape(&[0, 3, -1], &[2, 3, 4]);
        assert_eq!(result, vec![2, 3, 4]);
    }
}
