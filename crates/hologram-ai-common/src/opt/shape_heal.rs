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

            for &nid in &order {
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
                input_shapes.first().filter(|s| !s.is_empty()).cloned().map(|s| vec![s])
            }
        }

        // Identity-like: copy input shape.
        AiOp::Identity | AiOp::Cast { .. } => input_shapes.first().cloned().map(|s| vec![s]),

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
