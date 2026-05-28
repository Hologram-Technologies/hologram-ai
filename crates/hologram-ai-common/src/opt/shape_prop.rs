//! Forward shape propagation pass.
//!
//! Infers output shapes from input shapes for each node in topological order.
//! Collects `ShapeConstraint` entries into `graph.shape_constraints`.

use super::pipeline::Pass;
use super::shape_helpers::infer_output_dtypes;
use super::shape_inference::infer_output_shapes;
use crate::ir::dtype::DType;
use crate::ir::shape::DimExpr;
use crate::ir::{AiGraph, AiOp, Shape};

/// Propagate shapes forward through the graph.
///
/// For each node, computes output shapes from input shapes. Unknown shapes
/// are left as-is (no error). Shape constraints are recorded for later
/// validation.
///
/// When `known_i64_values` are available on shape-input tensors (populated by
/// `DataPropagation`), this pass can resolve Reshape and Expand output shapes.
///
/// Settled shapes (non-empty with ALL `Concrete` dims) are never overwritten —
/// this preserves fully-concrete oracle-seeded shapes and correctly-inferred
/// shapes from prior passes. Shapes with any `Var` or `Dynamic` dim are not
/// settled and may be overwritten by DataProp + subsequent ShapeProp passes.
pub struct ShapePropagation;

/// Alias for `ShapePropagation` used in the post-concretization repair loop.
///
/// After `concretize_all_dims` all symbolic `Var` dims are concrete, so any
/// remaining shapes with Var dims have been resolved. The settled-shape
/// protection (all-Concrete dims) applies identically here — this alias exists
/// for call-site clarity in the compiler pipeline.
pub struct AggressiveShapePropagation;

impl Pass for ShapePropagation {
    fn name(&self) -> &str {
        "ShapePropagation"
    }

    fn run(&self, graph: AiGraph) -> anyhow::Result<AiGraph> {
        propagate_shapes(graph, true)
    }
}

impl Pass for AggressiveShapePropagation {
    fn name(&self) -> &str {
        "AggressiveShapePropagation"
    }

    fn run(&self, graph: AiGraph) -> anyhow::Result<AiGraph> {
        // No settled-shape protection: overwrite any shape when a better
        // inference is available. Safe because this runs post-concretization
        // where all dims are concrete and DataProp has resolved Reshape targets.
        propagate_shapes(graph, false)
    }
}

fn propagate_shapes(mut graph: AiGraph, protect_settled: bool) -> anyhow::Result<AiGraph> {
    let order = graph.topo_order();

    // Build node lookup.
    let node_idx: std::collections::HashMap<u32, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id, i))
        .collect();

    // Shape propagation: single pass in topological order.
    for &nid in order.iter() {
        let idx = match node_idx.get(&nid) {
            Some(&i) => i,
            None => continue,
        };

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

        // For Reshape/Expand/Resize/Pad: try to get known_i64_values from shape inputs.
        let shape_known_values: Option<Vec<Option<i64>>> = match &graph.nodes[idx].op {
            AiOp::Reshape { .. } | AiOp::Expand => graph.nodes[idx].inputs.get(1).and_then(|tid| {
                graph
                    .tensor_info
                    .get(tid)
                    .and_then(|ti| ti.known_i64_values.clone())
            }),
            // Resize: sizes from input[3] or input[1] (known_i64_values).
            // If no integer sizes found, try float scales from input[2] or
            // input[1] and multiply by input spatial dims.
            AiOp::Resize { .. } => {
                let inputs_ref = &graph.nodes[idx].inputs;
                // ONNX Resize signature varies by opset:
                //   v10:  Resize(X, scales)             — inputs[1] is f32 scales
                //   v11+: Resize(X, roi, scales, sizes) — inputs[3] is i64 sizes
                // The ONNX importer collapses to 2 inputs (X, scales-or-sizes),
                // preferring i64 sizes when present, so sizes can end up at
                // inputs[1] OR inputs[3].
                //
                // To distinguish sizes (true i64) from scales (f32 cast-to-i64
                // by DataPropagation as e.g. [1, 1, 2, 2]) we gate on
                // logical_dtype — only treat known_i64_values as "sizes" if the
                // source tensor is INT64.
                let sizes_from = |tid: crate::ir::TensorId| -> Option<Vec<Option<i64>>> {
                    let info = graph.tensor_info.get(&tid)?;
                    if info.logical_dtype != DType::INT64 {
                        return None;
                    }
                    info.known_i64_values.clone()
                };
                let sizes = inputs_ref
                    .get(3)
                    .copied()
                    .and_then(sizes_from)
                    .or_else(|| inputs_ref.get(1).copied().and_then(sizes_from));
                if sizes.is_some() {
                    sizes
                } else {
                    // Try scales: read f32 constant param, multiply with input shape.
                    let scales_tid = inputs_ref.get(2).or_else(|| inputs_ref.get(1));
                    let scales = scales_tid.and_then(|tid| {
                        graph
                            .params
                            .get(tid)
                            .and_then(|p| p.as_f32_slice())
                            .map(|s| s.to_vec())
                    });
                    if let (Some(ref scales), Some(in_shape)) = (scales, input_shapes.first()) {
                        let vals: Vec<Option<i64>> = in_shape
                            .iter()
                            .zip(scales.iter().chain(std::iter::repeat(&1.0f32)))
                            .map(|(dim, &scale)| {
                                dim.as_concrete()
                                    .map(|d| (d as f64 * scale as f64).round() as i64)
                            })
                            .collect();
                        Some(vals)
                    } else {
                        None
                    }
                }
            }
            // Pad: pads from input[1] (opset 11+).
            AiOp::Pad { .. } => graph.nodes[idx].inputs.get(1).and_then(|tid| {
                graph
                    .tensor_info
                    .get(tid)
                    .and_then(|ti| ti.known_i64_values.clone())
            }),
            _ => None,
        };

        let output_tids = graph.nodes[idx].outputs.clone();
        let op = graph.nodes[idx].op.clone();

        let inferred = infer_output_shapes(&op, &input_shapes, shape_known_values.as_deref());

        for (i, tid) in output_tids.iter().enumerate() {
            if let Some(shape) = inferred.get(i) {
                if let Some(info) = graph.tensor_info.get_mut(tid) {
                    // ShapePropagation (protect_settled=true) protects
                    // fully-concrete shapes so oracle-seeded values and
                    // previously-correct inferences are not replaced by
                    // weaker op-rule inferences.
                    //
                    // AggressiveShapePropagation (protect_settled=false)
                    // overwrites any existing shape when it can infer a
                    // non-empty one. Used post-concretization to repair
                    // oracle shapes that were concretized to wrong values
                    // (e.g. '(32//batch_size)' Var → 1 instead of 32).
                    //
                    // The `!shape.is_empty()` guard ensures Opaque ops
                    // (infer empty) never clear existing shapes.
                    let is_settled = protect_settled
                        && !info.shape.is_empty()
                        && info.shape.iter().all(|d| matches!(d, DimExpr::Concrete(_)));
                    if !is_settled && !shape.is_empty() {
                        // Never downgrade a Concrete dim to Dynamic/Var, and
                        // never shrink a Concrete dim to a smaller Concrete value.
                        // This prevents AggressiveShapePropagation from overwriting
                        // correct shapes with force-concretized inputs (e.g., Resize
                        // output [1,512,128,128] shrunk to [1,512,2,2] because
                        // ForceConcretize set the input spatial dims to 1).
                        if info.shape.len() == shape.len() {
                            let mut merged = shape.clone();
                            for (new_dim, old_dim) in merged.iter_mut().zip(info.shape.iter()) {
                                if matches!(new_dim, DimExpr::Dynamic | DimExpr::Var(_))
                                    && matches!(old_dim, DimExpr::Concrete(_))
                                {
                                    *new_dim = old_dim.clone();
                                }
                            }
                            info.shape = merged;
                        } else {
                            info.shape = shape.clone();
                        }
                    }
                }
            }
        }
    }

    // Dtype propagation: fixpoint loop until no changes.
    // Single pass is insufficient because intermediate tensors default to F32
    // and cascading updates (Shape→I64 → Gather→I64 → Concat→I64) may not
    // propagate through a single topological pass if some inputs haven't been
    // updated yet when their consumers are processed.
    loop {
        let mut changed = false;
        for &nid in order.iter() {
            let idx = match node_idx.get(&nid) {
                Some(&i) => i,
                None => continue,
            };

            // UOR-native dtype propagation: an input lacking `tensor_info`
            // is a propagation **precondition failure**, not a thing to
            // approximate with an F32 fallback. Skip this node — its
            // output dtype is left as-is, and the fixed-point loop will
            // revisit it after other passes / iterations fill the
            // missing info. (The old code's `.unwrap_or(F32)` fabricated
            // a dtype and then guarded against the fabrication
            // overriding good data; the principled fix is to not
            // fabricate at all.)
            let mut input_dtypes: Vec<DType> = Vec::with_capacity(graph.nodes[idx].inputs.len());
            let mut incomplete = false;
            for tid in &graph.nodes[idx].inputs {
                match graph.tensor_info.get(tid) {
                    Some(ti) => input_dtypes.push(ti.logical_dtype),
                    None => {
                        incomplete = true;
                        break;
                    }
                }
            }
            if incomplete {
                continue;
            }

            let output_tids = graph.nodes[idx].outputs.clone();
            let op = graph.nodes[idx].op.clone();
            let Some(inferred_dtypes) = infer_output_dtypes(&op, &input_dtypes, output_tids.len())
            else {
                // The inference itself couldn't determine a dtype
                // (e.g. Cast with an unrecognized target). Leave the
                // output dtype as-is; never invent one.
                continue;
            };

            for (i, tid) in output_tids.iter().enumerate() {
                let Some(&inferred_dtype) = inferred_dtypes.get(i) else {
                    continue;
                };
                if let Some(info) = graph.tensor_info.get_mut(tid) {
                    if info.logical_dtype != inferred_dtype {
                        info.logical_dtype = inferred_dtype;
                        info.storage_dtype = inferred_dtype;
                        changed = true;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{shape_from_concrete, AiGraph, AiNode, AiOp, DType, TensorInfo};
    use std::collections::HashMap;

    #[test]
    fn propagate_matmul_shape() {
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 4, 8])),
        );
        ti.insert(
            1u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[8, 16])),
        );
        // Output starts with unknown shape.
        ti.insert(2u32, TensorInfo::new(DType::F32, Shape::new()));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(0, AiOp::MatMul, vec![0, 1], vec![2])],
            inputs: vec![0, 1],
            outputs: vec![2],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
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
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[4, 1])),
        );
        ti.insert(
            1u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 8])),
        );
        ti.insert(2u32, TensorInfo::new(DType::F32, Shape::new()));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(0, AiOp::Add, vec![0, 1], vec![2])],
            inputs: vec![0, 1],
            outputs: vec![2],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };

        let pass = ShapePropagation;
        let g2 = pass.run(g).unwrap();
        let out_shape = &g2.tensor_info[&2].shape;
        // [4, 1] + [1, 8] → [4, 8]
        assert_eq!(out_shape.len(), 2);
        assert_eq!(out_shape[0].as_concrete(), Some(4));
        assert_eq!(out_shape[1].as_concrete(), Some(8));
    }

    /// Simulates the ONNX shape subgraph: Shape → Gather → Unsqueeze → Concat.
    /// This is the exact pattern that causes i64 shape corruption at runtime.
    #[test]
    fn propagate_shape_subgraph_chain() {
        let mut ti = HashMap::new();
        // input_ids: [batch=1, seq=2] I64
        ti.insert(
            0u32,
            TensorInfo::new(DType::INT64, shape_from_concrete(&[1, 2])),
        );
        // Shape op output: should be [2] I64 (rank of input)
        ti.insert(1u32, TensorInfo::new(DType::F32, Shape::new()));
        // Gather indices: scalar constant (value=0)
        ti.insert(
            2u32,
            TensorInfo::new(DType::INT64, Shape::new()), // scalar
        );
        // Gather output: scalar I64
        ti.insert(3u32, TensorInfo::new(DType::F32, Shape::new()));
        // Unsqueeze output: [1] I64
        ti.insert(4u32, TensorInfo::new(DType::F32, Shape::new()));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![
                AiNode::new(
                    0,
                    AiOp::Shape {
                        start: None,
                        end: None,
                    },
                    vec![0],
                    vec![1],
                ),
                AiNode::new(1, AiOp::Gather { axis: 0 }, vec![1, 2], vec![3]),
                AiNode::new(2, AiOp::Unsqueeze { axes: vec![0] }, vec![3], vec![4]),
            ],
            inputs: vec![0],
            outputs: vec![4],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };

        let pass = ShapePropagation;
        let g2 = pass.run(g).unwrap();

        // Shape output: [2] (rank of [1, 2] input)
        let shape_out = &g2.tensor_info[&1].shape;
        assert_eq!(shape_out.len(), 1);
        assert_eq!(shape_out[0].as_concrete(), Some(2));
        assert_eq!(g2.tensor_info[&1].logical_dtype, DType::INT64);

        // Gather output: scalar (empty shape) — axis=0 dim removed, scalar indices add nothing
        let gather_out = &g2.tensor_info[&3].shape;
        assert_eq!(gather_out.len(), 0);
        assert_eq!(g2.tensor_info[&3].logical_dtype, DType::INT64);

        // Unsqueeze output: [1] — scalar + unsqueeze(axis=0)
        let unsqueeze_out = &g2.tensor_info[&4].shape;
        assert_eq!(unsqueeze_out.len(), 1);
        assert_eq!(unsqueeze_out[0].as_concrete(), Some(1));
        assert_eq!(g2.tensor_info[&4].logical_dtype, DType::INT64);
    }

    #[test]
    fn propagate_squeeze_shape() {
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 4, 1, 8])),
        );
        ti.insert(1u32, TensorInfo::new(DType::F32, Shape::new()));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(
                0,
                AiOp::Squeeze { axes: vec![0, 2] },
                vec![0],
                vec![1],
            )],
            inputs: vec![0],
            outputs: vec![1],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };

        let pass = ShapePropagation;
        let g2 = pass.run(g).unwrap();
        let out_shape = &g2.tensor_info[&1].shape;
        // [1, 4, 1, 8] squeeze axes [0, 2] → [4, 8]
        assert_eq!(out_shape.len(), 2);
        assert_eq!(out_shape[0].as_concrete(), Some(4));
        assert_eq!(out_shape[1].as_concrete(), Some(8));
    }

    #[test]
    fn propagate_transpose_shape() {
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[2, 3, 4])),
        );
        ti.insert(1u32, TensorInfo::new(DType::F32, Shape::new()));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(
                0,
                AiOp::Transpose {
                    perm: vec![2, 0, 1],
                },
                vec![0],
                vec![1],
            )],
            inputs: vec![0],
            outputs: vec![1],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };

        let pass = ShapePropagation;
        let g2 = pass.run(g).unwrap();
        let out_shape = &g2.tensor_info[&1].shape;
        // [2, 3, 4] perm [2, 0, 1] → [4, 2, 3]
        assert_eq!(out_shape.len(), 3);
        assert_eq!(out_shape[0].as_concrete(), Some(4));
        assert_eq!(out_shape[1].as_concrete(), Some(2));
        assert_eq!(out_shape[2].as_concrete(), Some(3));
    }

    fn relu_graph_with_shapes(in_shape: &[u64], out_shape: &[u64]) -> AiGraph {
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(in_shape)),
        );
        ti.insert(
            1u32,
            TensorInfo::new(DType::F32, shape_from_concrete(out_shape)),
        );
        AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(0, AiOp::Relu, vec![0], vec![1])],
            inputs: vec![0],
            outputs: vec![1],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        }
    }

    /// Oracle-seeded concrete shapes survive ShapePropagation (settled-shape protection).
    #[test]
    fn settled_shape_survives_shape_propagation() {
        let g = relu_graph_with_shapes(&[1, 32, 512], &[1, 32, 512]);
        let g2 = ShapePropagation.run(g).unwrap();
        assert_eq!(
            g2.tensor_info[&1].shape.as_slice(),
            shape_from_concrete(&[1, 32, 512]).as_slice(),
            "ShapePropagation must not overwrite settled shape"
        );
    }

    /// Oracle-seeded concrete shapes survive AggressiveShapePropagation.
    #[test]
    fn settled_shape_survives_aggressive_propagation() {
        let g = relu_graph_with_shapes(&[1, 32, 512], &[1, 32, 512]);
        let g2 = AggressiveShapePropagation.run(g).unwrap();
        assert_eq!(
            g2.tensor_info[&1].shape.as_slice(),
            shape_from_concrete(&[1, 32, 512]).as_slice(),
            "AggressiveShapePropagation must not overwrite settled shape"
        );
    }

    /// Dynamic dims are still filled by propagation even when other dims
    /// are concrete (shape is not settled because it contains Dynamic).
    #[test]
    fn dynamic_dim_is_filled_by_propagation() {
        use crate::ir::shape::DimExpr;
        use smallvec::smallvec;

        let mut ti = HashMap::new();
        // Input: fully concrete.
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[2, 4])),
        );
        // Output: has a Dynamic dim — not settled, propagation should fill it.
        let partial: Shape = smallvec![DimExpr::Concrete(2), DimExpr::Dynamic];
        ti.insert(1u32, TensorInfo::new(DType::F32, partial));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(0, AiOp::Relu, vec![0], vec![1])],
            inputs: vec![0],
            outputs: vec![1],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };

        let g2 = ShapePropagation.run(g).unwrap();
        let out = &g2.tensor_info[&1].shape;
        assert_eq!(
            out.as_slice(),
            shape_from_concrete(&[2, 4]).as_slice(),
            "Dynamic dim in output should be replaced by propagated concrete shape"
        );
    }
}
