//! Shape healing pass — fills in empty shapes after concretization.
//!
//! Runs after `concretize_all_dims` when all `DimExpr` values are `Concrete`.
//! For any tensor that still has an empty shape, tries to infer it from:
//!
//! 1. The producing op's semantics + already-resolved input shapes
//! 2. Element count conservation (for Reshape)
//! 3. Broadcasting rules (for elementwise ops)
//!
//! This is a safety net — it catches whatever `ShapePropagation` + `DataPropagation`
//! missed. It's not a full shape inference pass; it only fills *empty* shapes.

use super::pipeline::Pass;
use crate::ir::shape::DimExpr;
use crate::ir::{AiGraph, AiOp, Shape};

pub struct ShapeHealing;

impl Pass for ShapeHealing {
    fn name(&self) -> &str {
        "ShapeHealing"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let order = graph.topo_order();
        let node_idx: std::collections::HashMap<u32, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id, i))
            .collect();

        // Multiple passes: some shapes depend on others.
        for _round in 0..3 {
            let mut changed = false;

            for &nid in order.iter() {
                let idx = match node_idx.get(&nid) {
                    Some(&i) => i,
                    None => continue,
                };

                let output_tids = graph.nodes[idx].outputs.clone();
                // Only heal empty shapes.
                let needs_heal = output_tids.iter().any(|tid| {
                    graph
                        .tensor_info
                        .get(tid)
                        .map(|ti| ti.shape.is_empty())
                        .unwrap_or(false)
                });
                if !needs_heal {
                    continue;
                }

                let input_shapes: Vec<Shape> = graph.nodes[idx]
                    .inputs
                    .iter()
                    .map(|tid| {
                        graph
                            .tensor_info
                            .get(tid)
                            .map(|ti| ti.shape.clone())
                            .unwrap_or_default()
                    })
                    .collect();

                let op = graph.nodes[idx].op.clone();

                if let Some(inferred) = heal_shape(&op, &input_shapes, &graph, idx) {
                    for (i, tid) in output_tids.iter().enumerate() {
                        if let Some(shape) = inferred.get(i) {
                            if !shape.is_empty() {
                                if let Some(info) = graph.tensor_info.get_mut(tid) {
                                    if info.shape.is_empty() {
                                        info.shape = shape.clone();
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }

        Ok(graph)
    }
}

/// Infer output shape for an op using concrete input shapes.
fn heal_shape(
    op: &AiOp,
    input_shapes: &[Shape],
    graph: &AiGraph,
    node_idx: usize,
) -> Option<Vec<Shape>> {
    match op {
        // Reshape: use known_i64_values or infer from element count.
        AiOp::Reshape { .. } => {
            let node = &graph.nodes[node_idx];
            // Try known_i64_values from the shape input tensor.
            let shape_vals = node.inputs.get(1).and_then(|tid| {
                graph
                    .tensor_info
                    .get(tid)
                    .and_then(|ti| ti.known_i64_values.as_ref())
            });

            if let Some(vals) = shape_vals {
                let data_shape = input_shapes.first();
                let data_elems: u64 = data_shape
                    .map(|s| {
                        s.iter()
                            .filter_map(|d| d.as_concrete())
                            .product::<u64>()
                            .max(1)
                    })
                    .unwrap_or(1);

                // Resolve all values, computing -1 from element conservation.
                let neg1_count = vals.iter().filter(|v| **v == Some(-1)).count();
                let known_product: i64 = vals
                    .iter()
                    .filter_map(|v| *v)
                    .filter(|&v| v > 0)
                    .product::<i64>()
                    .max(1);

                let shape: Vec<DimExpr> = vals
                    .iter()
                    .enumerate()
                    .map(|(i, v)| match v {
                        Some(0) => {
                            // Copy from data input at same position.
                            data_shape
                                .and_then(|ds| ds.get(i))
                                .and_then(|d| d.as_concrete())
                                .map(DimExpr::Concrete)
                                .unwrap_or(DimExpr::Concrete(1))
                        }
                        Some(-1) if neg1_count == 1 && known_product > 0 => {
                            let resolved = data_elems as i64 / known_product;
                            DimExpr::Concrete(resolved.max(1) as u64)
                        }
                        Some(n) if *n > 0 => DimExpr::Concrete(*n as u64),
                        None => {
                            // Unknown: try to inherit from data input.
                            data_shape
                                .and_then(|ds| ds.get(i))
                                .and_then(|d| d.as_concrete())
                                .map(DimExpr::Concrete)
                                .unwrap_or(DimExpr::Concrete(1))
                        }
                        _ => DimExpr::Concrete(1),
                    })
                    .collect();

                Some(vec![Shape::from(shape)])
            } else {
                None
            }
        }

        // Squeeze with empty axes: remove all size-1 dims.
        AiOp::Squeeze { axes } if axes.is_empty() => {
            if let Some(input) = input_shapes.first() {
                if !input.is_empty() {
                    let shape: Vec<DimExpr> = input
                        .iter()
                        .filter(|d| d.as_concrete() != Some(1))
                        .cloned()
                        .collect();
                    Some(vec![Shape::from(shape)])
                } else {
                    None
                }
            } else {
                None
            }
        }

        // Elementwise unary: copy input shape.
        _ if matches!(
            op.category(),
            crate::ir::op::OpCategory::UnaryElementwise
                | crate::ir::op::OpCategory::ShapePreserving
        ) =>
        {
            input_shapes.first().cloned().map(|s| vec![s])
        }

        // Elementwise binary: broadcast.
        _ if matches!(
            op.category(),
            crate::ir::op::OpCategory::BinaryElementwise
                | crate::ir::op::OpCategory::BinaryComparison
        ) =>
        {
            if input_shapes.len() >= 2 && !input_shapes[0].is_empty() && !input_shapes[1].is_empty()
            {
                Some(vec![broadcast_shape(&input_shapes[0], &input_shapes[1])])
            } else {
                input_shapes
                    .first()
                    .filter(|s| !s.is_empty())
                    .cloned()
                    .map(|s| vec![s])
            }
        }

        // Identity-like: copy input shape.
        AiOp::Identity | AiOp::Cast { .. } => input_shapes.first().cloned().map(|s| vec![s]),

        // MatMul: [batch..., M, K] x [batch..., K, N] → [batch..., M, N]
        AiOp::MatMul | AiOp::BatchMatMul => {
            if input_shapes.len() >= 2 && input_shapes[0].len() >= 2 && input_shapes[1].len() >= 2 {
                let a = &input_shapes[0];
                let b = &input_shapes[1];
                // Output = a's batch + second-to-last dims, b's last dim.
                let mut shape: Vec<DimExpr> = a[..a.len() - 1].to_vec();
                shape.push(b[b.len() - 1].clone());
                Some(vec![Shape::from(shape)])
            } else {
                None
            }
        }

        // Gemm: [M, K] x [K, N] → [M, N] (with optional transposes)
        AiOp::Gemm {
            trans_a, trans_b, ..
        } => {
            if input_shapes.len() >= 2 && input_shapes[0].len() >= 2 && input_shapes[1].len() >= 2 {
                let a = &input_shapes[0];
                let b = &input_shapes[1];
                let m = if *trans_a {
                    a.last().cloned()
                } else {
                    a.first().cloned()
                };
                let n = if *trans_b {
                    b.first().cloned()
                } else {
                    b.last().cloned()
                };
                if let (Some(m_dim), Some(n_dim)) = (m, n) {
                    Some(vec![Shape::from(vec![m_dim, n_dim])])
                } else {
                    None
                }
            } else {
                None
            }
        }

        // Transpose: permute input dims.
        AiOp::Transpose { perm } => {
            if let Some(input) = input_shapes.first() {
                if !input.is_empty() && perm.len() == input.len() {
                    let shape: Vec<DimExpr> =
                        perm.iter().map(|&p| input[p as usize].clone()).collect();
                    Some(vec![Shape::from(shape)])
                } else {
                    None
                }
            } else {
                None
            }
        }

        // Concat: sum along axis, keep other dims.
        AiOp::Concat { axis } => {
            if input_shapes.is_empty() || input_shapes.iter().any(|s| s.is_empty()) {
                return None;
            }
            let first = &input_shapes[0];
            let norm_axis = if *axis < 0 {
                (first.len() as i64 + *axis) as usize
            } else {
                *axis as usize
            };
            if norm_axis >= first.len() {
                return None;
            }
            // Sum the axis dimension across all inputs.
            let mut axis_sum: u64 = 0;
            for s in input_shapes {
                axis_sum += s.get(norm_axis).and_then(|d| d.as_concrete())?;
            }
            let mut shape = first.clone();
            shape[norm_axis] = DimExpr::Concrete(axis_sum);
            Some(vec![shape])
        }

        // Conv2d: output spatial dims from convolution arithmetic.
        AiOp::Conv {
            kernel_shape,
            strides,
            pads,
            dilations,
            ..
        } => {
            if let Some(input) = input_shapes.first() {
                if input.len() >= 4 {
                    // input: [N, C_in, H, W], weight: [C_out, C_in/groups, kH, kW]
                    let n = input[0].clone();
                    let c_out = input_shapes
                        .get(1)
                        .and_then(|w| w.first())
                        .cloned()
                        .unwrap_or(input[1].clone());
                    let h_in = input[2].as_concrete().unwrap_or(1);
                    let w_in = input[3].as_concrete().unwrap_or(1);
                    let kh = kernel_shape.first().copied().unwrap_or(1);
                    let kw = kernel_shape.get(1).copied().unwrap_or(1);
                    let sh = strides.first().copied().unwrap_or(1);
                    let sw = strides.get(1).copied().unwrap_or(1);
                    let ph = pads.first().copied().unwrap_or(0);
                    let pw = pads.get(1).copied().unwrap_or(0);
                    let dh = dilations.first().copied().unwrap_or(1);
                    let dw = dilations.get(1).copied().unwrap_or(1);
                    let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
                    let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
                    Some(vec![Shape::from(vec![
                        n,
                        c_out,
                        DimExpr::Concrete(h_out),
                        DimExpr::Concrete(w_out),
                    ])])
                } else {
                    None
                }
            } else {
                None
            }
        }

        // Softmax, norms: shape-preserving.
        AiOp::Softmax { .. }
        | AiOp::LogSoftmax { .. }
        | AiOp::RmsNorm { .. }
        | AiOp::LayerNorm { .. }
        | AiOp::GroupNorm { .. }
        | AiOp::InstanceNorm { .. } => input_shapes.first().cloned().map(|s| vec![s]),

        // Gather: replace indexed axis with indices shape.
        AiOp::Gather { axis } | AiOp::GatherElements { axis } => {
            if input_shapes.len() >= 2 && !input_shapes[0].is_empty() && !input_shapes[1].is_empty()
            {
                let data = &input_shapes[0];
                let indices = &input_shapes[1];
                let norm_axis = if *axis < 0 {
                    (data.len() as i64 + *axis) as usize
                } else {
                    *axis as usize
                };
                let mut shape = Vec::new();
                for (i, d) in data.iter().enumerate() {
                    if i == norm_axis {
                        shape.extend(indices.iter().cloned());
                    } else {
                        shape.push(d.clone());
                    }
                }
                Some(vec![Shape::from(shape)])
            } else {
                None
            }
        }

        _ => None,
    }
}

fn broadcast_shape(a: &Shape, b: &Shape) -> Shape {
    let len = a.len().max(b.len());
    let mut result = Shape::new();
    for i in 0..len {
        let ad = if i < a.len() {
            &a[a.len() - 1 - i]
        } else {
            &DimExpr::Concrete(1)
        };
        let bd = if i < b.len() {
            &b[b.len() - 1 - i]
        } else {
            &DimExpr::Concrete(1)
        };
        let dim = match (ad.as_concrete(), bd.as_concrete()) {
            (Some(1), _) => bd.clone(),
            (_, Some(1)) => ad.clone(),
            (Some(av), Some(bv)) if av == bv => ad.clone(),
            _ => ad.clone(), // best guess
        };
        result.push(dim);
    }
    result.reverse();
    result
}
