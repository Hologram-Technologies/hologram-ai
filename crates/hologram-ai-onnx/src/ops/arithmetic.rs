//! Arithmetic operations - Add, Sub, Mul, Div, etc.

use anyhow::{Context, Result, bail};
use hologram::compiler::{ConstantData, OpKind};

use super::{BroadcastInfo, OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

// ============================================================================
// Binary Operations
// ============================================================================

/// ONNX Add operation.
pub struct AddOp;

impl OpTranslator for AddOp {
    fn op_type(&self) -> &'static str {
        "Add"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        binary_constant_fold(node, ctx, |a, b| a + b, |a, b| a + b)
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        binary_translate(node, ctx, OpKind::Add)
    }
}

/// ONNX Sub operation.
pub struct SubOp;

impl OpTranslator for SubOp {
    fn op_type(&self) -> &'static str {
        "Sub"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        binary_constant_fold(node, ctx, |a, b| a - b, |a, b| a - b)
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        binary_translate(node, ctx, OpKind::Sub)
    }
}

/// ONNX Mul operation.
pub struct MulOp;

impl OpTranslator for MulOp {
    fn op_type(&self) -> &'static str {
        "Mul"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        binary_constant_fold(node, ctx, |a, b| a * b, |a, b| a * b)
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        binary_translate(node, ctx, OpKind::Mul)
    }
}

/// ONNX Div operation.
pub struct DivOp;

impl OpTranslator for DivOp {
    fn op_type(&self) -> &'static str {
        "Div"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        binary_constant_fold(node, ctx, |a, b| a / b, |a, b| a / b)
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        binary_translate(node, ctx, OpKind::Div)
    }
}

/// ONNX Pow operation.
pub struct PowOp;

impl OpTranslator for PowOp {
    fn op_type(&self) -> &'static str {
        "Pow"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        binary_constant_fold(
            node,
            ctx,
            |a, b| (a as f64).powi(b as i32) as i64,
            |a, b| a.powf(b),
        )
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        binary_translate(node, ctx, OpKind::Pow)
    }
}

/// ONNX Min operation - element-wise minimum.
pub struct MinOp;

impl OpTranslator for MinOp {
    fn op_type(&self) -> &'static str {
        "Min"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        binary_constant_fold(node, ctx, |a, b| a.min(b), |a, b| a.min(b))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        binary_translate(node, ctx, OpKind::ElemMin)
    }
}

/// ONNX Max operation - element-wise maximum.
pub struct MaxOp;

impl OpTranslator for MaxOp {
    fn op_type(&self) -> &'static str {
        "Max"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        binary_constant_fold(node, ctx, |a, b| a.max(b), |a, b| a.max(b))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        binary_translate(node, ctx, OpKind::ElemMax)
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

/// ONNX Sqrt operation.
pub struct SqrtOp;

impl OpTranslator for SqrtOp {
    fn op_type(&self) -> &'static str {
        "Sqrt"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        unary_constant_fold_f32(node, ctx, |x| x.sqrt())
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        unary_translate(node, ctx, OpKind::Sqrt)
    }
}

/// ONNX Log operation.
pub struct LogOp;

impl OpTranslator for LogOp {
    fn op_type(&self) -> &'static str {
        "Log"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        unary_constant_fold_f32(node, ctx, |x| x.ln())
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        unary_translate(node, ctx, OpKind::Log)
    }
}

/// ONNX Exp operation.
pub struct ExpOp;

impl OpTranslator for ExpOp {
    fn op_type(&self) -> &'static str {
        "Exp"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        unary_constant_fold_f32(node, ctx, |x| x.exp())
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        unary_translate(node, ctx, OpKind::Exp)
    }
}

/// ONNX Abs operation.
pub struct AbsOp;

impl OpTranslator for AbsOp {
    fn op_type(&self) -> &'static str {
        "Abs"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let input_name = node.input.first()?;
        if !ctx.is_constant(input_name) {
            return None;
        }

        let input_node = ctx.get_node(input_name)?;
        let input_const = ctx.get_constant_data(input_name)?;

        let result = match input_const {
            ConstantData::I64(v) => ConstantData::I64(v.iter().map(|x| x.abs()).collect()),
            ConstantData::I32(v) => ConstantData::I32(v.iter().map(|x| x.abs()).collect()),
            ConstantData::F32(v) => ConstantData::F32(v.iter().map(|x| x.abs()).collect()),
            ConstantData::F64(v) => ConstantData::F64(v.iter().map(|x| x.abs()).collect()),
            _ => return None,
        };

        Some(TranslateResult::constant(
            input_node.shape.clone(),
            input_node.dtype,
            result,
        ))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        unary_translate(node, ctx, OpKind::Abs)
    }
}

/// ONNX Neg operation.
pub struct NegOp;

impl OpTranslator for NegOp {
    fn op_type(&self) -> &'static str {
        "Neg"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let input_name = node.input.first()?;
        if !ctx.is_constant(input_name) {
            return None;
        }

        let input_node = ctx.get_node(input_name)?;
        let input_const = ctx.get_constant_data(input_name)?;

        let result = match input_const {
            ConstantData::I64(v) => ConstantData::I64(v.iter().map(|x| -x).collect()),
            ConstantData::I32(v) => ConstantData::I32(v.iter().map(|x| -x).collect()),
            ConstantData::F32(v) => ConstantData::F32(v.iter().map(|x| -x).collect()),
            ConstantData::F64(v) => ConstantData::F64(v.iter().map(|x| -x).collect()),
            _ => return None,
        };

        Some(TranslateResult::constant(
            input_node.shape.clone(),
            input_node.dtype,
            result,
        ))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        // Neg is implemented as Mul by -1
        let input_name = node.input.first().context("Neg has no input")?;
        let input_node = ctx.get_node(input_name).context("Neg input not found")?;

        Ok(TranslateResult::runtime(
            OpKind::Mul, // Will need scalar broadcast of -1
            input_node.shape.clone(),
            input_node.dtype,
        ))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn binary_translate(
    node: &proto::NodeProto,
    ctx: &TranslateContext,
    op_kind: OpKind,
) -> Result<TranslateResult> {
    let a_name = node.input.first().context("Binary op has no first input")?;
    let b_name = node.input.get(1).context("Binary op has no second input")?;

    let a_node = ctx.get_node(a_name).context("First input not found")?;
    let b_node = ctx.get_node(b_name).context("Second input not found")?;

    let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape).with_context(|| {
        format!(
            "Incompatible shapes for {}: {:?} and {:?}",
            node.op_type, a_node.shape, b_node.shape
        )
    })?;

    // Check if broadcasting is needed
    let a_needs_broadcast = a_node.shape != output_shape;
    let b_needs_broadcast = b_node.shape != output_shape;

    if !a_needs_broadcast && !b_needs_broadcast {
        // No broadcasting needed - simple case
        return Ok(TranslateResult::runtime(
            op_kind,
            output_shape,
            a_node.dtype,
        ));
    }

    tracing::debug!(
        "{} '{}': broadcasting {:?} and {:?} -> {:?}",
        node.op_type,
        node.output.first().unwrap_or(&String::new()),
        a_node.shape,
        b_node.shape,
        output_shape
    );

    // Broadcasting required - signal to builder to insert Expand nodes
    Ok(TranslateResult::broadcast_binary(
        op_kind,
        output_shape.clone(),
        a_node.dtype,
        BroadcastInfo {
            a_shape: a_node.shape.clone(),
            b_shape: b_node.shape.clone(),
            output_shape,
            a_needs_broadcast,
            b_needs_broadcast,
        },
    ))
}

fn unary_translate(
    node: &proto::NodeProto,
    ctx: &TranslateContext,
    op_kind: OpKind,
) -> Result<TranslateResult> {
    let input_name = node.input.first().context("Unary op has no input")?;
    let input_node = ctx.get_node(input_name).context("Input not found")?;

    Ok(TranslateResult::runtime(
        op_kind,
        input_node.shape.clone(),
        input_node.dtype,
    ))
}

fn binary_constant_fold<Fi, Ff>(
    node: &proto::NodeProto,
    ctx: &TranslateContext,
    int_op: Fi,
    float_op: Ff,
) -> Option<TranslateResult>
where
    Fi: Fn(i64, i64) -> i64,
    Ff: Fn(f32, f32) -> f32,
{
    let a_name = node.input.first()?;
    let b_name = node.input.get(1)?;

    if !ctx.is_constant(a_name) || !ctx.is_constant(b_name) {
        return None;
    }

    let a_node = ctx.get_node(a_name)?;
    let b_node = ctx.get_node(b_name)?;
    let a_const = ctx.get_constant_data(a_name)?;
    let b_const = ctx.get_constant_data(b_name)?;

    let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape).ok()?;

    let result = match (a_const, b_const) {
        (ConstantData::I64(a), ConstantData::I64(b)) => {
            let data =
                broadcast_binary_op(a, &a_node.shape, b, &b_node.shape, &output_shape, &int_op)?;
            ConstantData::I64(data)
        }
        (ConstantData::I32(a), ConstantData::I32(b)) => {
            let a64: Vec<i64> = a.iter().map(|&x| x as i64).collect();
            let b64: Vec<i64> = b.iter().map(|&x| x as i64).collect();
            let data = broadcast_binary_op(
                &a64,
                &a_node.shape,
                &b64,
                &b_node.shape,
                &output_shape,
                &int_op,
            )?;
            ConstantData::I32(data.iter().map(|&x| x as i32).collect())
        }
        (ConstantData::F32(a), ConstantData::F32(b)) => {
            let data = broadcast_binary_op_f32(
                a,
                &a_node.shape,
                b,
                &b_node.shape,
                &output_shape,
                &float_op,
            )?;
            ConstantData::F32(data)
        }
        _ => return None,
    };

    Some(TranslateResult::constant(
        output_shape,
        a_node.dtype,
        result,
    ))
}

fn unary_constant_fold_f32<F>(
    node: &proto::NodeProto,
    ctx: &TranslateContext,
    op: F,
) -> Option<TranslateResult>
where
    F: Fn(f32) -> f32,
{
    let input_name = node.input.first()?;
    if !ctx.is_constant(input_name) {
        return None;
    }

    let input_node = ctx.get_node(input_name)?;
    let input_const = ctx.get_constant_data(input_name)?;

    let result = match input_const {
        ConstantData::F32(v) => ConstantData::F32(v.iter().map(|&x| op(x)).collect()),
        ConstantData::F64(v) => ConstantData::F64(v.iter().map(|&x| op(x as f32) as f64).collect()),
        _ => return None,
    };

    Some(TranslateResult::constant(
        input_node.shape.clone(),
        input_node.dtype,
        result,
    ))
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_rank = a.len().max(b.len());
    let mut result = vec![0; max_rank];

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
            bail!("Incompatible shapes: {:?} and {:?}", a, b);
        }
    }

    Ok(result)
}

fn broadcast_binary_op<F>(
    a: &[i64],
    a_shape: &[usize],
    b: &[i64],
    b_shape: &[usize],
    output_shape: &[usize],
    op: &F,
) -> Option<Vec<i64>>
where
    F: Fn(i64, i64) -> i64,
{
    let output_size: usize = output_shape.iter().product();
    let mut result = Vec::with_capacity(output_size);

    for i in 0..output_size {
        let a_idx = compute_broadcast_index(i, output_shape, a_shape);
        let b_idx = compute_broadcast_index(i, output_shape, b_shape);
        result.push(op(a[a_idx], b[b_idx]));
    }

    Some(result)
}

fn broadcast_binary_op_f32<F>(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    output_shape: &[usize],
    op: &F,
) -> Option<Vec<f32>>
where
    F: Fn(f32, f32) -> f32,
{
    let output_size: usize = output_shape.iter().product();
    let mut result = Vec::with_capacity(output_size);

    for i in 0..output_size {
        let a_idx = compute_broadcast_index(i, output_shape, a_shape);
        let b_idx = compute_broadcast_index(i, output_shape, b_shape);
        result.push(op(a[a_idx], b[b_idx]));
    }

    Some(result)
}

fn compute_broadcast_index(
    flat_idx: usize,
    output_shape: &[usize],
    input_shape: &[usize],
) -> usize {
    if input_shape.is_empty() {
        return 0; // Scalar
    }

    let rank_diff = output_shape.len() - input_shape.len();
    let mut idx = 0;
    let mut remaining = flat_idx;

    // Compute strides for input
    let mut input_strides = vec![1; input_shape.len()];
    for i in (0..input_shape.len().saturating_sub(1)).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    for (i, &_dim) in output_shape.iter().enumerate() {
        let stride: usize = output_shape[i + 1..].iter().product::<usize>().max(1);
        let coord = remaining / stride;
        remaining %= stride;

        if i >= rank_diff {
            let input_i = i - rank_diff;
            let input_coord = if input_shape[input_i] == 1 { 0 } else { coord };
            idx += input_coord * input_strides[input_i];
        }
    }

    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::compiler::{DType, OperationGraph};
    use std::collections::HashMap;

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(broadcast_shapes(&[5], &[1]).unwrap(), vec![5]);
        assert_eq!(broadcast_shapes(&[], &[5]).unwrap(), vec![5]);
        assert_eq!(broadcast_shapes(&[1, 5], &[5, 1]).unwrap(), vec![5, 5]);
        assert!(broadcast_shapes(&[3], &[5]).is_err());
    }

    #[test]
    fn test_scalar_broadcast() {
        let a = vec![10i64];
        let b = vec![1i64, 2, 3];
        let output_shape = vec![3];

        let result = broadcast_binary_op(&a, &[], &b, &[3], &output_shape, &|x, y| x + y).unwrap();
        assert_eq!(result, vec![11, 12, 13]);
    }

    /// Helper to create a test context with given inputs.
    fn make_test_context(inputs: &[(&str, Vec<usize>)]) -> (OperationGraph, HashMap<String, u32>) {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        for (idx, (name, shape)) in inputs.iter().enumerate() {
            let node = hologram::compiler::OpNode::new(
                idx as u32,
                OpKind::Input,
                shape.clone(),
                DType::F32,
            )
            .with_name(name.to_string());
            graph.nodes.push(node);
            value_to_node.insert(name.to_string(), idx as u32);
        }

        (graph, value_to_node)
    }

    #[test]
    fn test_binary_translate_no_broadcast() {
        // Same shapes - no broadcasting needed
        let (graph, value_to_node) = make_test_context(&[("a", vec![1, 512]), ("b", vec![1, 512])]);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let node = proto::NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["out".to_string()],
            op_type: "Sub".to_string(),
            ..Default::default()
        };

        let result = SubOp.translate(&node, &ctx).expect("Should translate");

        // No broadcast info when shapes match
        assert!(
            result.broadcast_info.is_none(),
            "No broadcasting needed for same shapes"
        );
        assert_eq!(result.shape, vec![1, 512]);
        assert!(matches!(result.op_kind, OpKind::Sub));
    }

    #[test]
    fn test_binary_translate_broadcast_scalar() {
        // Scalar broadcast: [1, 512] - [1] -> [1, 512]
        let (graph, value_to_node) = make_test_context(&[("a", vec![1, 512]), ("b", vec![1])]);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let node = proto::NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["out".to_string()],
            op_type: "Sub".to_string(),
            ..Default::default()
        };

        let result = SubOp.translate(&node, &ctx).expect("Should translate");

        // Should have broadcast info
        assert!(
            result.broadcast_info.is_some(),
            "Should need broadcasting for scalar"
        );

        let info = result.broadcast_info.unwrap();
        assert_eq!(info.a_shape, vec![1, 512]);
        assert_eq!(info.b_shape, vec![1]);
        assert_eq!(info.output_shape, vec![1, 512]);
        assert!(!info.a_needs_broadcast, "A already has output shape");
        assert!(info.b_needs_broadcast, "B needs to be expanded");
    }

    #[test]
    fn test_binary_translate_broadcast_different_ranks() {
        // Different ranks: [2, 3, 4] + [4] -> [2, 3, 4]
        let (graph, value_to_node) = make_test_context(&[("a", vec![2, 3, 4]), ("b", vec![4])]);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let node = proto::NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["out".to_string()],
            op_type: "Add".to_string(),
            ..Default::default()
        };

        let result = AddOp.translate(&node, &ctx).expect("Should translate");

        let info = result.broadcast_info.expect("Should have broadcast info");
        assert_eq!(info.output_shape, vec![2, 3, 4]);
        assert!(!info.a_needs_broadcast);
        assert!(info.b_needs_broadcast);
    }

    #[test]
    fn test_binary_translate_broadcast_both_inputs() {
        // Both need broadcast: [1, 5] * [5, 1] -> [5, 5]
        let (graph, value_to_node) = make_test_context(&[("a", vec![1, 5]), ("b", vec![5, 1])]);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let node = proto::NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["out".to_string()],
            op_type: "Mul".to_string(),
            ..Default::default()
        };

        let result = MulOp.translate(&node, &ctx).expect("Should translate");

        let info = result.broadcast_info.expect("Should have broadcast info");
        assert_eq!(info.output_shape, vec![5, 5]);
        assert!(info.a_needs_broadcast, "A [1,5] needs expand to [5,5]");
        assert!(info.b_needs_broadcast, "B [5,1] needs expand to [5,5]");
    }

    #[test]
    fn test_binary_translate_t5_layernorm_case() {
        // T5 LayerNorm case: [1, 512] - [1] -> [1, 512]
        // This is the exact case causing the SizeMismatch bug
        let (graph, value_to_node) =
            make_test_context(&[("hidden_states", vec![1, 512]), ("mean", vec![1])]);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let node = proto::NodeProto {
            input: vec!["hidden_states".to_string(), "mean".to_string()],
            output: vec!["centered".to_string()],
            op_type: "Sub".to_string(),
            ..Default::default()
        };

        let result = SubOp.translate(&node, &ctx).expect("Should translate");

        // This MUST have broadcast_info to fix the T5 bug
        assert!(
            result.broadcast_info.is_some(),
            "T5 LayerNorm Sub MUST have broadcast_info"
        );

        let info = result.broadcast_info.unwrap();
        assert_eq!(info.output_shape, vec![1, 512]);
        assert!(!info.a_needs_broadcast);
        assert!(info.b_needs_broadcast);
    }
}
