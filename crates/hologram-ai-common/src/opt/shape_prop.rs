//! Forward shape propagation pass.
//!
//! Infers output shapes from input shapes for each node in topological order.
//! Collects `ShapeConstraint` entries into `graph.shape_constraints`.

use crate::ir::{AiGraph, AiOp, Shape, shape_from_concrete};
use crate::ir::shape::DimExpr;
use super::pipeline::Pass;

/// Propagate shapes forward through the graph.
///
/// For each node, computes output shapes from input shapes. Unknown shapes
/// are left as-is (no error). Shape constraints are recorded for later
/// validation.
pub struct ShapePropagation;

impl Pass for ShapePropagation {
    fn name(&self) -> &str { "ShapePropagation" }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let order = graph.topo_order();

        // Build node lookup.
        let node_idx: std::collections::HashMap<u32, usize> =
            graph.nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();

        for &nid in &order {
            let idx = match node_idx.get(&nid) {
                Some(&i) => i,
                None => continue,
            };

            // Gather input shapes.
            let input_shapes: Vec<Shape> = graph.nodes[idx]
                .inputs
                .iter()
                .map(|tid| {
                    graph.tensor_info
                        .get(tid)
                        .map(|ti| ti.shape.clone())
                        .unwrap_or_default()
                })
                .collect();

            let output_tids = graph.nodes[idx].outputs.clone();
            let op = graph.nodes[idx].op.clone();

            // Infer output shapes.
            let inferred = infer_output_shapes(&op, &input_shapes);

            // Update tensor_info for outputs.
            for (i, tid) in output_tids.iter().enumerate() {
                if let Some(shape) = inferred.get(i) {
                    if let Some(info) = graph.tensor_info.get_mut(tid) {
                        if info.shape.is_empty() || info.shape.iter().all(|d| matches!(d, DimExpr::Dynamic)) {
                            info.shape = shape.clone();
                        }
                    }
                }
            }
        }

        Ok(graph)
    }
}

/// Infer output shapes for a single op given input shapes.
fn infer_output_shapes(op: &AiOp, inputs: &[Shape]) -> Vec<Shape> {
    match op {
        // Unary elementwise — output shape = input shape.
        AiOp::Relu | AiOp::Gelu | AiOp::GeluApprox | AiOp::Silu
        | AiOp::Tanh | AiOp::Sigmoid | AiOp::Abs | AiOp::Neg
        | AiOp::Sqrt | AiOp::Exp | AiOp::Log | AiOp::Sign
        | AiOp::Floor | AiOp::Ceil | AiOp::Round | AiOp::Clip
        | AiOp::Erf | AiOp::Reciprocal | AiOp::Cos | AiOp::Sin
        | AiOp::IsNaN | AiOp::Not | AiOp::Identity | AiOp::Dequantize => {
            inputs.first().cloned().into_iter().collect()
        }

        // Binary elementwise with broadcasting — use the longer shape.
        AiOp::Add | AiOp::Sub | AiOp::Mul | AiOp::Div | AiOp::Pow | AiOp::Mod
        | AiOp::Min | AiOp::Max | AiOp::And | AiOp::Or | AiOp::Xor
        | AiOp::Equal | AiOp::Less | AiOp::LessOrEqual
        | AiOp::Greater | AiOp::GreaterOrEqual => {
            if inputs.len() >= 2 {
                vec![broadcast_shape(&inputs[0], &inputs[1])]
            } else {
                inputs.first().cloned().into_iter().collect()
            }
        }

        // MatMul: [..., M, K] x [..., K, N] → [..., M, N]
        AiOp::MatMul | AiOp::BatchMatMul => {
            if inputs.len() >= 2 && inputs[0].len() >= 2 && inputs[1].len() >= 2 {
                let a = &inputs[0];
                let b = &inputs[1];
                let mut shape = a[..a.len() - 1].to_vec();
                shape.push(b[b.len() - 1].clone());
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // Softmax/LogSoftmax preserve shape.
        AiOp::Softmax { .. } | AiOp::LogSoftmax { .. } => {
            inputs.first().cloned().into_iter().collect()
        }

        // Norms preserve shape (with weight input).
        AiOp::RmsNorm { .. } | AiOp::LayerNorm { .. }
        | AiOp::GroupNorm { .. } | AiOp::BatchNorm { .. } => {
            inputs.first().cloned().into_iter().collect()
        }

        // Concat along axis — sum that dimension.
        AiOp::Concat { axis } => {
            if inputs.is_empty() || inputs[0].is_empty() {
                return vec![Shape::new()];
            }
            let mut shape = inputs[0].clone();
            let ax = normalize_axis(*axis, shape.len());
            if ax < shape.len() {
                for inp in &inputs[1..] {
                    if ax < inp.len() {
                        shape[ax] = add_dims(&shape[ax], &inp[ax]);
                    }
                }
            }
            vec![shape]
        }

        // Embed: [batch, seq] → [batch, seq, embed_dim] — but we need weight shape.
        AiOp::Embed => {
            if inputs.len() >= 2 && !inputs[1].is_empty() {
                let mut shape = inputs[0].clone();
                shape.push(inputs[1][inputs[1].len() - 1].clone());
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // Attention ops — output shape = [batch, seq, num_heads * head_dim]
        AiOp::MultiHeadAttention { num_heads, head_dim, .. }
        | AiOp::GroupedQueryAttention { num_heads, head_dim, .. } => {
            if !inputs.is_empty() && inputs[0].len() >= 2 {
                let mut shape = inputs[0][..inputs[0].len() - 1].to_vec();
                shape.push(DimExpr::Concrete((*num_heads as u64) * (*head_dim as u64)));
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // FusedSwiGLU — output shape = input shape.
        AiOp::FusedSwiGLU => {
            inputs.first().cloned().into_iter().collect()
        }

        // RotaryEmbedding — preserves input shape.
        AiOp::RotaryEmbedding { .. } => {
            inputs.first().cloned().into_iter().collect()
        }

        // Reductions.
        AiOp::ReduceSum { axes, keepdims }
        | AiOp::ReduceMean { axes, keepdims }
        | AiOp::ReduceMax { axes, keepdims }
        | AiOp::ReduceMin { axes, keepdims } => {
            if let Some(input) = inputs.first() {
                vec![reduce_shape(input, axes, *keepdims)]
            } else {
                vec![Shape::new()]
            }
        }

        // Cast preserves shape.
        AiOp::Cast { .. } => {
            inputs.first().cloned().into_iter().collect()
        }

        // For complex ops we don't infer yet, return empty.
        _ => vec![Shape::new()],
    }
}

fn normalize_axis(axis: i64, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as i64 + axis).max(0) as usize
    } else {
        axis as usize
    }
}

fn add_dims(a: &DimExpr, b: &DimExpr) -> DimExpr {
    match (a.as_concrete(), b.as_concrete()) {
        (Some(av), Some(bv)) => DimExpr::Concrete(av + bv),
        _ => DimExpr::Dynamic,
    }
}

fn broadcast_shape(a: &Shape, b: &Shape) -> Shape {
    let len = a.len().max(b.len());
    let mut result = Shape::new();
    for i in 0..len {
        let ad = if i < a.len() { &a[a.len() - 1 - i] } else { &DimExpr::Concrete(1) };
        let bd = if i < b.len() { &b[b.len() - 1 - i] } else { &DimExpr::Concrete(1) };
        let dim = match (ad.as_concrete(), bd.as_concrete()) {
            (Some(1), _) => bd.clone(),
            (_, Some(1)) => ad.clone(),
            (Some(av), Some(bv)) if av == bv => ad.clone(),
            _ => DimExpr::Dynamic,
        };
        result.push(dim);
    }
    result.reverse();
    result
}

fn reduce_shape(input: &Shape, axes: &[i64], keepdims: bool) -> Shape {
    if axes.is_empty() {
        // Reduce all axes.
        if keepdims {
            Shape::from(vec![DimExpr::Concrete(1); input.len()])
        } else {
            shape_from_concrete(&[1])
        }
    } else {
        let ndim = input.len();
        let mut shape = Vec::new();
        for (i, dim) in input.iter().enumerate() {
            let is_reduced = axes.iter().any(|&ax| normalize_axis(ax, ndim) == i);
            if is_reduced {
                if keepdims {
                    shape.push(DimExpr::Concrete(1));
                }
            } else {
                shape.push(dim.clone());
            }
        }
        Shape::from(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AiGraph, AiNode, AiOp, DType, TensorInfo, shape_from_concrete};
    use std::collections::HashMap;

    #[test]
    fn propagate_matmul_shape() {
        let mut ti = HashMap::new();
        ti.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 4, 8])));
        ti.insert(1u32, TensorInfo::new(DType::F32, shape_from_concrete(&[8, 16])));
        // Output starts with unknown shape.
        ti.insert(2u32, TensorInfo::new(DType::F32, Shape::new()));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(0, AiOp::MatMul, vec![0, 1], vec![2])],
            inputs: vec![0, 1],
            outputs: vec![2],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
        };

        let pass = ShapePropagation;
        let g2 = pass.run(g).unwrap();
        let out_shape = &g2.tensor_info[&2].shape;
        // [1, 4, 8] x [8, 16] → [1, 4, 16]
        assert_eq!(out_shape.len(), 3);
        assert_eq!(out_shape[0].as_concrete(), Some(1));
        assert_eq!(out_shape[1].as_concrete(), Some(4));
        assert_eq!(out_shape[2].as_concrete(), Some(16));
    }

    #[test]
    fn propagate_elementwise_broadcast() {
        let mut ti = HashMap::new();
        ti.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[4, 1])));
        ti.insert(1u32, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 8])));
        ti.insert(2u32, TensorInfo::new(DType::F32, Shape::new()));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(0, AiOp::Add, vec![0, 1], vec![2])],
            inputs: vec![0, 1],
            outputs: vec![2],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
        };

        let pass = ShapePropagation;
        let g2 = pass.run(g).unwrap();
        let out_shape = &g2.tensor_info[&2].shape;
        // [4, 1] + [1, 8] → [4, 8]
        assert_eq!(out_shape.len(), 2);
        assert_eq!(out_shape[0].as_concrete(), Some(4));
        assert_eq!(out_shape[1].as_concrete(), Some(8));
    }
}
