//! Comparison operations - Greater, Less, Equal, etc.

use anyhow::{Context, Result, bail};
use hologram::compiler::{ConstantData, DType, OpKind};

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX Greater operation.
///
/// Element-wise comparison returning boolean tensor.
pub struct GreaterOp;

impl OpTranslator for GreaterOp {
    fn op_type(&self) -> &'static str {
        "Greater"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let a_name = node.input.first()?;
        let b_name = node.input.get(1)?;

        if !ctx.is_constant(a_name) || !ctx.is_constant(b_name) {
            return None;
        }

        let a_const = ctx.get_constant_data(a_name)?;
        let b_const = ctx.get_constant_data(b_name)?;
        let a_node = ctx.get_node(a_name)?;
        let b_node = ctx.get_node(b_name)?;

        let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape)?;
        let result = compare_constants(
            a_const,
            b_const,
            &a_node.shape,
            &b_node.shape,
            &output_shape,
            |a, b| a > b,
        )?;

        tracing::debug!(
            "Greater '{}': constant-folded to shape {:?}",
            node.output.first().unwrap_or(&String::new()),
            output_shape
        );

        Some(TranslateResult::constant(output_shape, DType::F32, result))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let a_name = node.input.first().context("Greater has no first input")?;
        let b_name = node.input.get(1).context("Greater has no second input")?;

        let a_node = ctx
            .get_node(a_name)
            .context("Greater first input not found")?;
        let b_node = ctx
            .get_node(b_name)
            .context("Greater second input not found")?;

        let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape).ok_or_else(|| {
            anyhow::anyhow!(
                "Cannot broadcast shapes {:?} and {:?}",
                a_node.shape,
                b_node.shape
            )
        })?;

        Ok(TranslateResult::runtime(
            OpKind::Greater,
            output_shape,
            DType::F32,
        ))
    }
}

/// ONNX Less operation.
pub struct LessOp;

impl OpTranslator for LessOp {
    fn op_type(&self) -> &'static str {
        "Less"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let a_name = node.input.first()?;
        let b_name = node.input.get(1)?;

        if !ctx.is_constant(a_name) || !ctx.is_constant(b_name) {
            return None;
        }

        let a_const = ctx.get_constant_data(a_name)?;
        let b_const = ctx.get_constant_data(b_name)?;
        let a_node = ctx.get_node(a_name)?;
        let b_node = ctx.get_node(b_name)?;

        let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape)?;
        let result = compare_constants(
            a_const,
            b_const,
            &a_node.shape,
            &b_node.shape,
            &output_shape,
            |a, b| a < b,
        )?;

        Some(TranslateResult::constant(output_shape, DType::F32, result))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let a_name = node.input.first().context("Less has no first input")?;
        let b_name = node.input.get(1).context("Less has no second input")?;

        let a_node = ctx.get_node(a_name).context("Less first input not found")?;
        let b_node = ctx
            .get_node(b_name)
            .context("Less second input not found")?;

        let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape).ok_or_else(|| {
            anyhow::anyhow!(
                "Cannot broadcast shapes {:?} and {:?}",
                a_node.shape,
                b_node.shape
            )
        })?;

        Ok(TranslateResult::runtime(
            OpKind::Less,
            output_shape,
            DType::F32,
        ))
    }
}

/// ONNX LessOrEqual operation.
pub struct LessOrEqualOp;

impl OpTranslator for LessOrEqualOp {
    fn op_type(&self) -> &'static str {
        "LessOrEqual"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let a_name = node.input.first()?;
        let b_name = node.input.get(1)?;

        if !ctx.is_constant(a_name) || !ctx.is_constant(b_name) {
            return None;
        }

        let a_const = ctx.get_constant_data(a_name)?;
        let b_const = ctx.get_constant_data(b_name)?;
        let a_node = ctx.get_node(a_name)?;
        let b_node = ctx.get_node(b_name)?;

        let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape)?;
        let result = compare_constants(
            a_const,
            b_const,
            &a_node.shape,
            &b_node.shape,
            &output_shape,
            |a, b| a <= b,
        )?;

        Some(TranslateResult::constant(output_shape, DType::F32, result))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let a_name = node
            .input
            .first()
            .context("LessOrEqual has no first input")?;
        let b_name = node
            .input
            .get(1)
            .context("LessOrEqual has no second input")?;

        let a_node = ctx
            .get_node(a_name)
            .context("LessOrEqual first input not found")?;
        let b_node = ctx
            .get_node(b_name)
            .context("LessOrEqual second input not found")?;

        let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape).ok_or_else(|| {
            anyhow::anyhow!(
                "Cannot broadcast shapes {:?} and {:?}",
                a_node.shape,
                b_node.shape
            )
        })?;

        Ok(TranslateResult::runtime(
            OpKind::LessOrEqual,
            output_shape,
            DType::F32,
        ))
    }
}

/// ONNX GreaterOrEqual operation.
pub struct GreaterOrEqualOp;

impl OpTranslator for GreaterOrEqualOp {
    fn op_type(&self) -> &'static str {
        "GreaterOrEqual"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let a_name = node.input.first()?;
        let b_name = node.input.get(1)?;

        if !ctx.is_constant(a_name) || !ctx.is_constant(b_name) {
            return None;
        }

        let a_const = ctx.get_constant_data(a_name)?;
        let b_const = ctx.get_constant_data(b_name)?;
        let a_node = ctx.get_node(a_name)?;
        let b_node = ctx.get_node(b_name)?;

        let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape)?;
        let result = compare_constants(
            a_const,
            b_const,
            &a_node.shape,
            &b_node.shape,
            &output_shape,
            |a, b| a >= b,
        )?;

        Some(TranslateResult::constant(output_shape, DType::F32, result))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let a_name = node
            .input
            .first()
            .context("GreaterOrEqual has no first input")?;
        let b_name = node
            .input
            .get(1)
            .context("GreaterOrEqual has no second input")?;

        let a_node = ctx
            .get_node(a_name)
            .context("GreaterOrEqual first input not found")?;
        let b_node = ctx
            .get_node(b_name)
            .context("GreaterOrEqual second input not found")?;

        let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape).ok_or_else(|| {
            anyhow::anyhow!(
                "Cannot broadcast shapes {:?} and {:?}",
                a_node.shape,
                b_node.shape
            )
        })?;

        Ok(TranslateResult::runtime(
            OpKind::GreaterOrEqual,
            output_shape,
            DType::F32,
        ))
    }
}

/// ONNX Equal operation.
///
/// Note: Equal is constant-folded only since hologram doesn't have Equal OpKind.
/// For runtime Equal, the model should be redesigned to avoid it.
pub struct EqualOp;

impl OpTranslator for EqualOp {
    fn op_type(&self) -> &'static str {
        "Equal"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let a_name = node.input.first()?;
        let b_name = node.input.get(1)?;

        if !ctx.is_constant(a_name) || !ctx.is_constant(b_name) {
            return None;
        }

        let a_const = ctx.get_constant_data(a_name)?;
        let b_const = ctx.get_constant_data(b_name)?;
        let a_node = ctx.get_node(a_name)?;
        let b_node = ctx.get_node(b_name)?;

        let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape)?;
        let result = compare_constants_eq(
            a_const,
            b_const,
            &a_node.shape,
            &b_node.shape,
            &output_shape,
        )?;

        Some(TranslateResult::constant(output_shape, DType::F32, result))
    }

    fn translate(
        &self,
        _node: &proto::NodeProto,
        _ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        // Equal at runtime is not directly supported by hologram.
        // This should only happen if both inputs are not constant.
        bail!(
            "Equal operation requires constant inputs for folding. Runtime Equal is not supported."
        )
    }
}

/// ONNX Where operation - conditional select.
pub struct WhereOp;

impl OpTranslator for WhereOp {
    fn op_type(&self) -> &'static str {
        "Where"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        // Where(condition, x, y) -> x where condition else y
        let cond_name = node.input.first()?;
        let x_name = node.input.get(1)?;
        let y_name = node.input.get(2)?;

        // All three must be constants to fold
        if !ctx.is_constant(cond_name) || !ctx.is_constant(x_name) || !ctx.is_constant(y_name) {
            return None;
        }

        let cond_const = ctx.get_constant_data(cond_name)?;
        let x_const = ctx.get_constant_data(x_name)?;
        let y_const = ctx.get_constant_data(y_name)?;

        let cond_node = ctx.get_node(cond_name)?;
        let x_node = ctx.get_node(x_name)?;
        let y_node = ctx.get_node(y_name)?;

        // Compute broadcast output shape (condition, x, y all broadcast together)
        let temp_shape = broadcast_shapes(&cond_node.shape, &x_node.shape)?;
        let output_shape = broadcast_shapes(&temp_shape, &y_node.shape)?;

        // Perform constant folding
        let result = where_constants(
            cond_const,
            x_const,
            y_const,
            &cond_node.shape,
            &x_node.shape,
            &y_node.shape,
            &output_shape,
        )?;

        tracing::debug!(
            "Where '{}': constant-folded to shape {:?}",
            node.output.first().unwrap_or(&String::new()),
            output_shape
        );

        Some(TranslateResult::constant(
            output_shape,
            x_node.dtype,
            result,
        ))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        // Where(condition, x, y) -> x where condition else y
        let cond_name = node.input.first().context("Where has no condition input")?;
        let x_name = node.input.get(1).context("Where has no X input")?;
        let y_name = node.input.get(2).context("Where has no Y input")?;

        let cond_node = ctx
            .get_node(cond_name)
            .context("Where condition not found")?;
        let x_node = ctx.get_node(x_name).context("Where X not found")?;
        let y_node = ctx.get_node(y_name).context("Where Y not found")?;

        // Output shape is broadcast of all three
        let temp_shape = broadcast_shapes(&cond_node.shape, &x_node.shape)
            .ok_or_else(|| anyhow::anyhow!("Cannot broadcast condition and X shapes"))?;
        let output_shape = broadcast_shapes(&temp_shape, &y_node.shape)
            .ok_or_else(|| anyhow::anyhow!("Cannot broadcast with Y shape"))?;

        Ok(TranslateResult::runtime(
            OpKind::Where,
            output_shape,
            x_node.dtype,
        ))
    }
}

/// Broadcast two shapes according to NumPy rules.
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_rank = a.len().max(b.len());
    let mut result = vec![0usize; max_rank];

    for i in 0..max_rank {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if a_dim == b_dim {
            result[max_rank - 1 - i] = a_dim;
        } else if a_dim == 1 {
            result[max_rank - 1 - i] = b_dim;
        } else if b_dim == 1 {
            result[max_rank - 1 - i] = a_dim;
        } else {
            return None; // Incompatible dimensions
        }
    }

    Some(result)
}

/// Compare constant tensors element-wise with broadcasting.
///
/// Returns F32 (1.0 for true, 0.0 for false) instead of Bool because
/// hologram's Where instruction expects F32 for conditions.
fn compare_constants<F>(
    a: &ConstantData,
    b: &ConstantData,
    a_shape: &[usize],
    b_shape: &[usize],
    output_shape: &[usize],
    compare: F,
) -> Option<ConstantData>
where
    F: Fn(f64, f64) -> bool,
{
    let total_size: usize = output_shape.iter().product();
    let mut result = Vec::with_capacity(total_size);

    // Convert to f64 for comparison
    let a_vals: Vec<f64> = match a {
        ConstantData::F32(v) => v.iter().map(|&x| x as f64).collect(),
        ConstantData::F64(v) => v.clone(),
        ConstantData::I32(v) => v.iter().map(|&x| x as f64).collect(),
        ConstantData::I64(v) => v.iter().map(|&x| x as f64).collect(),
        _ => return None,
    };

    let b_vals: Vec<f64> = match b {
        ConstantData::F32(v) => v.iter().map(|&x| x as f64).collect(),
        ConstantData::F64(v) => v.clone(),
        ConstantData::I32(v) => v.iter().map(|&x| x as f64).collect(),
        ConstantData::I64(v) => v.iter().map(|&x| x as f64).collect(),
        _ => return None,
    };

    // Return F32 (1.0/0.0) instead of Bool because hologram's Where expects F32
    for i in 0..total_size {
        let a_idx = broadcast_index(i, output_shape, a_shape);
        let b_idx = broadcast_index(i, output_shape, b_shape);
        result.push(if compare(a_vals[a_idx], b_vals[b_idx]) {
            1.0f32
        } else {
            0.0f32
        });
    }

    Some(ConstantData::F32(result))
}

/// Compare constant tensors for equality.
///
/// Returns F32 (1.0 for true, 0.0 for false) instead of Bool because
/// hologram's Where instruction expects F32 for conditions.
fn compare_constants_eq(
    a: &ConstantData,
    b: &ConstantData,
    a_shape: &[usize],
    b_shape: &[usize],
    output_shape: &[usize],
) -> Option<ConstantData> {
    let total_size: usize = output_shape.iter().product();
    let mut result = Vec::with_capacity(total_size);

    macro_rules! compare_eq {
        ($av:expr, $bv:expr) => {{
            for i in 0..total_size {
                let a_idx = broadcast_index(i, output_shape, a_shape);
                let b_idx = broadcast_index(i, output_shape, b_shape);
                result.push(if $av[a_idx] == $bv[b_idx] {
                    1.0f32
                } else {
                    0.0f32
                });
            }
        }};
    }

    match (a, b) {
        (ConstantData::I64(av), ConstantData::I64(bv)) => compare_eq!(av, bv),
        (ConstantData::I32(av), ConstantData::I32(bv)) => compare_eq!(av, bv),
        (ConstantData::F32(av), ConstantData::F32(bv)) => compare_eq!(av, bv),
        (ConstantData::F64(av), ConstantData::F64(bv)) => compare_eq!(av, bv),
        _ => return None,
    }

    Some(ConstantData::F32(result))
}

/// Compute broadcast index.
fn broadcast_index(flat_idx: usize, output_shape: &[usize], input_shape: &[usize]) -> usize {
    if input_shape.is_empty() || input_shape.iter().product::<usize>() == 1 {
        return 0; // Scalar
    }

    let rank_diff = output_shape.len() - input_shape.len();
    let mut result = 0;
    let mut stride = 1;

    for i in (0..input_shape.len()).rev() {
        let out_idx = i + rank_diff;
        let coord = (flat_idx / stride_at(output_shape, out_idx)) % output_shape[out_idx];
        let input_coord = if input_shape[i] == 1 { 0 } else { coord };
        result += input_coord * stride;
        stride *= input_shape[i];
    }

    result
}

fn stride_at(shape: &[usize], idx: usize) -> usize {
    shape[idx + 1..].iter().product()
}

/// Perform Where selection on constant tensors with broadcasting.
///
/// condition is treated as boolean: non-zero = true, zero = false.
/// For each element: result[i] = x[i] if condition[i] else y[i]
fn where_constants(
    cond: &ConstantData,
    x: &ConstantData,
    y: &ConstantData,
    cond_shape: &[usize],
    x_shape: &[usize],
    y_shape: &[usize],
    output_shape: &[usize],
) -> Option<ConstantData> {
    let total_size: usize = output_shape.iter().product();

    // Convert condition to boolean (non-zero = true)
    let cond_bool: Vec<bool> = match cond {
        ConstantData::F32(v) => v.iter().map(|&c| c != 0.0).collect(),
        ConstantData::F64(v) => v.iter().map(|&c| c != 0.0).collect(),
        ConstantData::I32(v) => v.iter().map(|&c| c != 0).collect(),
        ConstantData::I64(v) => v.iter().map(|&c| c != 0).collect(),
        ConstantData::U8(v) => v.iter().map(|&c| c != 0).collect(),
        _ => return None,
    };

    // Handle different data types for x and y
    match (x, y) {
        (ConstantData::F32(x_vals), ConstantData::F32(y_vals)) => {
            let mut result = Vec::with_capacity(total_size);
            for i in 0..total_size {
                let cond_idx = broadcast_index(i, output_shape, cond_shape);
                let x_idx = broadcast_index(i, output_shape, x_shape);
                let y_idx = broadcast_index(i, output_shape, y_shape);
                result.push(if cond_bool[cond_idx] {
                    x_vals[x_idx]
                } else {
                    y_vals[y_idx]
                });
            }
            Some(ConstantData::F32(result))
        }
        (ConstantData::F64(x_vals), ConstantData::F64(y_vals)) => {
            let mut result = Vec::with_capacity(total_size);
            for i in 0..total_size {
                let cond_idx = broadcast_index(i, output_shape, cond_shape);
                let x_idx = broadcast_index(i, output_shape, x_shape);
                let y_idx = broadcast_index(i, output_shape, y_shape);
                result.push(if cond_bool[cond_idx] {
                    x_vals[x_idx]
                } else {
                    y_vals[y_idx]
                });
            }
            Some(ConstantData::F64(result))
        }
        (ConstantData::I32(x_vals), ConstantData::I32(y_vals)) => {
            let mut result = Vec::with_capacity(total_size);
            for i in 0..total_size {
                let cond_idx = broadcast_index(i, output_shape, cond_shape);
                let x_idx = broadcast_index(i, output_shape, x_shape);
                let y_idx = broadcast_index(i, output_shape, y_shape);
                result.push(if cond_bool[cond_idx] {
                    x_vals[x_idx]
                } else {
                    y_vals[y_idx]
                });
            }
            Some(ConstantData::I32(result))
        }
        (ConstantData::I64(x_vals), ConstantData::I64(y_vals)) => {
            let mut result = Vec::with_capacity(total_size);
            for i in 0..total_size {
                let cond_idx = broadcast_index(i, output_shape, cond_shape);
                let x_idx = broadcast_index(i, output_shape, x_shape);
                let y_idx = broadcast_index(i, output_shape, y_shape);
                result.push(if cond_bool[cond_idx] {
                    x_vals[x_idx]
                } else {
                    y_vals[y_idx]
                });
            }
            Some(ConstantData::I64(result))
        }
        // Mixed types: convert to F32
        _ => {
            let x_f32 = constant_to_f32(x)?;
            let y_f32 = constant_to_f32(y)?;
            let mut result = Vec::with_capacity(total_size);
            for i in 0..total_size {
                let cond_idx = broadcast_index(i, output_shape, cond_shape);
                let x_idx = broadcast_index(i, output_shape, x_shape);
                let y_idx = broadcast_index(i, output_shape, y_shape);
                result.push(if cond_bool[cond_idx] {
                    x_f32[x_idx]
                } else {
                    y_f32[y_idx]
                });
            }
            Some(ConstantData::F32(result))
        }
    }
}

/// Convert constant data to f32 values.
fn constant_to_f32(data: &ConstantData) -> Option<Vec<f32>> {
    match data {
        ConstantData::F32(v) => Some(v.clone()),
        ConstantData::F64(v) => Some(v.iter().map(|&x| x as f32).collect()),
        ConstantData::I32(v) => Some(v.iter().map(|&x| x as f32).collect()),
        ConstantData::I64(v) => Some(v.iter().map(|&x| x as f32).collect()),
        ConstantData::U8(v) => Some(v.iter().map(|&x| x as f32).collect()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(broadcast_shapes(&[3, 4], &[4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[1, 4], &[3, 1]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[5], &[5]), Some(vec![5]));
        assert_eq!(broadcast_shapes(&[3, 4], &[5, 4]), None);
    }

    #[test]
    fn test_where_constants_simple() {
        // condition: [1, 0, 1, 0] (bool mask)
        // x: [10, 20, 30, 40]
        // y: [1, 2, 3, 4]
        // expected: [10, 2, 30, 4]
        let cond = ConstantData::F32(vec![1.0, 0.0, 1.0, 0.0]);
        let x = ConstantData::F32(vec![10.0, 20.0, 30.0, 40.0]);
        let y = ConstantData::F32(vec![1.0, 2.0, 3.0, 4.0]);

        let result = where_constants(&cond, &x, &y, &[4], &[4], &[4], &[4]).unwrap();

        if let ConstantData::F32(vals) = result {
            assert_eq!(vals, vec![10.0, 2.0, 30.0, 4.0]);
        } else {
            panic!("Expected F32 result");
        }
    }

    #[test]
    fn test_where_constants_with_broadcast() {
        // condition: [1, 0] shape [2]
        // x: [10, 20] shape [2]
        // y: [99] shape [1] (broadcast to [2])
        // expected: [10, 99]
        let cond = ConstantData::F32(vec![1.0, 0.0]);
        let x = ConstantData::F32(vec![10.0, 20.0]);
        let y = ConstantData::F32(vec![99.0]);

        let result = where_constants(&cond, &x, &y, &[2], &[2], &[1], &[2]).unwrap();

        if let ConstantData::F32(vals) = result {
            assert_eq!(vals, vec![10.0, 99.0]);
        } else {
            panic!("Expected F32 result");
        }
    }

    #[test]
    fn test_where_constants_position_bias_pattern() {
        // Simulates T5 position bias bucket selection:
        // condition: is_small (abs_distance < threshold)
        // x: small_bucket (abs_distance itself)
        // y: large_bucket (log-space bucket index)
        //
        // For distances [0, 1, 2, 10, 20]:
        // - is_small (<8): [1, 1, 1, 0, 0]
        // - small_bucket: [0, 1, 2, 10, 20]
        // - large_bucket: [8, 9, 10, 11, 12] (fake log values)
        // expected: [0, 1, 2, 11, 12]
        let cond = ConstantData::F32(vec![1.0, 1.0, 1.0, 0.0, 0.0]);
        let x = ConstantData::F32(vec![0.0, 1.0, 2.0, 10.0, 20.0]);
        let y = ConstantData::F32(vec![8.0, 9.0, 10.0, 11.0, 12.0]);

        let result = where_constants(&cond, &x, &y, &[5], &[5], &[5], &[5]).unwrap();

        if let ConstantData::F32(vals) = result {
            // When condition is true (1.0), select from x (small_bucket)
            // When condition is false (0.0), select from y (large_bucket)
            assert_eq!(vals, vec![0.0, 1.0, 2.0, 11.0, 12.0]);
        } else {
            panic!("Expected F32 result");
        }
    }

    #[test]
    fn test_where_constants_2d_broadcast() {
        // Test 2D broadcasting similar to position bias [seq, seq]
        // condition: [[1, 0], [1, 1]] shape [2, 2]
        // x: [[10, 20], [30, 40]] shape [2, 2]
        // y: [99, 88] shape [2] (broadcast along axis 0)
        // expected: [[10, 88], [30, 40]]
        let cond = ConstantData::F32(vec![1.0, 0.0, 1.0, 1.0]);
        let x = ConstantData::F32(vec![10.0, 20.0, 30.0, 40.0]);
        let y = ConstantData::F32(vec![99.0, 88.0]);

        let result = where_constants(&cond, &x, &y, &[2, 2], &[2, 2], &[2], &[2, 2]).unwrap();

        if let ConstantData::F32(vals) = result {
            assert_eq!(vals, vec![10.0, 88.0, 30.0, 40.0]);
        } else {
            panic!("Expected F32 result");
        }
    }
}
