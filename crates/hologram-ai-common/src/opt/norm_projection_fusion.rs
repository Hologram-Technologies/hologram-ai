//! Norm + Projection fusion (Plan 054).
//!
//! Detects `[Add +] RmsNorm → 2+ MatMul` (multi-way projection) and fuses
//! into `FusedNormProjection`.
//!
//! # Patterns
//!
//! ## Pattern A: Norm → multi-way projection
//! ```text
//! normed = RmsNorm(x, weight, eps)
//!   → MatMul(normed, W_a) → a
//!   → MatMul(normed, W_b) → b
//!  [→ MatMul(normed, W_c) → c]
//! ```
//!
//! ## Pattern B: Add + Norm → multi-way projection
//! ```text
//! normed = FusedLayerNormResidual(x, residual, weight, eps)
//!   → MatMul(normed, W_a) → a
//!   → MatMul(normed, W_b) → b
//!  [→ MatMul(normed, W_c) → c]
//! ```
//!
//! Fused into:
//! ```text
//! concat_out = FusedNormProjection(x, [residual,] weight, W_concat)
//!   → Slice(0..n_a) → a
//!   → Slice(n_a..n_a+n_b) → b
//!  [→ Slice(n_a+n_b..n_total) → c]
//! ```
//!
//! The fused kernel normalizes in a stack buffer (no arena allocation),
//! then projects via a single GEMV into the concatenated output.
//! Downstream Slice nodes split the output (zero-copy at M=1).
//!
//! Constraint: the norm output must have no consumers besides the projections.
//! Only f32 weights are concatenated (Q4 weights skipped to avoid rkyv overflow).

use super::graph_utils::{build_consumer_map, next_node_id};
use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, DType, TensorId};
use std::collections::HashSet;

pub struct NormProjectionFusion;

impl Pass for NormProjectionFusion {
    fn name(&self) -> &str {
        "NormProjectionFusion"
    }

    fn should_run(&self, graph: &AiGraph) -> bool {
        graph.nodes.iter().any(|n| {
            matches!(
                n.op,
                AiOp::RmsNorm { .. } | AiOp::FusedLayerNormResidual { .. }
            )
        })
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        // Map: tensor_id → list of (consuming_node_idx, input_position).
        let consumers = build_consumer_map(&graph);

        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut new_node_groups: Vec<(usize, Vec<AiNode>)> = Vec::new();
        let mut fused_count: u32 = 0;

        let mut next_nid = next_node_id(&graph);

        for (norm_idx, norm_node) in graph.nodes.iter().enumerate() {
            let (epsilon, has_residual_add) = match &norm_node.op {
                AiOp::RmsNorm { epsilon } => (*epsilon as f64, false),
                AiOp::FusedLayerNormResidual { epsilon } => (*epsilon as f64, true),
                _ => continue,
            };

            if to_remove.contains(&norm_idx) {
                continue;
            }

            let norm_out = match norm_node.outputs.first() {
                Some(&tid) => tid,
                None => continue,
            };

            let norm_consumers = match consumers.get(&norm_out) {
                Some(c) => c,
                None => continue,
            };

            // Collect MatMul consumers where norm output is input[0].
            let matmul_consumers: Vec<(usize, TensorId, TensorId)> = norm_consumers
                .iter()
                .filter_map(|&(node_idx, input_pos)| {
                    if input_pos != 0 {
                        return None;
                    }
                    let node = &graph.nodes[node_idx];
                    if !matches!(node.op, AiOp::MatMul) || node.inputs.len() < 2 {
                        return None;
                    }
                    if to_remove.contains(&node_idx) {
                        return None;
                    }
                    let weight_tid = node.inputs[1];
                    let out_tid = *node.outputs.first()?;
                    Some((node_idx, weight_tid, out_tid))
                })
                .collect();

            if matmul_consumers.len() < 2 {
                continue;
            }

            // Norm output must ONLY be consumed by these MatMuls.
            if norm_consumers.len() != matmul_consumers.len() {
                continue;
            }

            // All weights must be f32 parameters with concrete 2D shapes.
            let mut weight_infos: Vec<(TensorId, usize, usize)> = Vec::new(); // (tid, k, n)
            let mut all_valid = true;
            for &(_, weight_tid, _) in &matmul_consumers {
                if !graph.params.contains_key(&weight_tid) {
                    all_valid = false;
                    break;
                }
                match get_2d_shape(&graph, weight_tid) {
                    Some((k, n)) if k >= 256 && n >= 64 => {
                        weight_infos.push((weight_tid, k, n));
                    }
                    _ => {
                        all_valid = false;
                        break;
                    }
                }
            }
            if !all_valid {
                continue;
            }

            // All K dims must match.
            let k = weight_infos[0].1;
            if !weight_infos.iter().all(|w| w.1 == k) {
                continue;
            }

            // Only fuse f32 weights (skip Q4/Q8 to avoid rkyv overflow).
            let all_f32 = matmul_consumers.iter().all(|&(_, weight_tid, _)| {
                graph
                    .tensor_info
                    .get(&weight_tid)
                    .is_some_and(|info| info.storage_dtype == DType::F32)
            });
            if !all_f32 {
                tracing::trace!(
                    norm_idx,
                    "NormProjectionFusion: skipping — non-f32 weights (Q4/Q8)"
                );
                continue;
            }

            let split_sizes: Vec<usize> = weight_infos.iter().map(|w| w.2).collect();

            // Build fused node inputs: [x, (residual,) norm_weight, W_a, W_b, ...]
            // No weight concatenation — pass original weights directly.
            // The fused kernel normalizes once, then does N separate projections
            // from the same normed buffer. This avoids rkyv serialization overflow
            // from inline concatenated weights.
            let mut fused_inputs = if has_residual_add {
                vec![
                    norm_node.inputs[0], // x
                    norm_node.inputs[1], // residual
                    norm_node.inputs[2], // norm_weight
                ]
            } else {
                vec![
                    norm_node.inputs[0], // x
                    norm_node.inputs[1], // norm_weight
                ]
            };
            // Append each projection weight.
            for &(_, weight_tid, _) in &matmul_consumers {
                fused_inputs.push(weight_tid);
            }

            // Create one output per projection (multi-output node).
            let fused_outputs: Vec<TensorId> = matmul_consumers
                .iter()
                .map(|&(_, _, orig_out_tid)| orig_out_tid)
                .collect();

            let fused_node_id = next_nid;
            next_nid += 1;
            let fused_node = AiNode::new(
                fused_node_id,
                AiOp::FusedNormProjection {
                    epsilon,
                    split_sizes: split_sizes.clone(),
                    has_residual_add,
                },
                fused_inputs,
                fused_outputs,
            );

            let new_nodes = vec![fused_node];

            // Mark norm + matmul nodes for removal.
            to_remove.insert(norm_idx);
            for &(mm_idx, _, _) in &matmul_consumers {
                to_remove.insert(mm_idx);
            }
            let insert_idx = norm_idx;
            new_node_groups.push((insert_idx, new_nodes));
            fused_count += 1;

            tracing::debug!(
                norm_idx,
                n_projections = matmul_consumers.len(),
                ?split_sizes,
                has_residual_add,
                "NormProjectionFusion: fused Norm + {}-way projection",
                matmul_consumers.len(),
            );
        }

        if fused_count > 0 {
            tracing::info!(
                fused_count,
                "NormProjectionFusion: fused Norm+Projection groups"
            );
        }

        if !to_remove.is_empty() {
            new_node_groups.sort_by_key(|(idx, _)| *idx);
            let mut result_nodes = Vec::with_capacity(graph.nodes.len());
            let mut insert_iter = new_node_groups.into_iter().peekable();

            for (i, node) in graph.nodes.into_iter().enumerate() {
                while let Some((insert_idx, _)) = insert_iter.peek() {
                    if *insert_idx <= i {
                        let (_, nodes) = insert_iter.next().expect("peeked");
                        result_nodes.extend(nodes);
                    } else {
                        break;
                    }
                }
                if to_remove.contains(&i) {
                    continue;
                }
                result_nodes.push(node);
            }
            for (_, nodes) in insert_iter {
                result_nodes.extend(nodes);
            }

            graph.nodes = result_nodes;
            graph.invalidate_topo_cache();
        }

        Ok(graph)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn get_2d_shape(graph: &AiGraph, tid: TensorId) -> Option<(usize, usize)> {
    use crate::ir::Dim;
    let info = graph.tensor_info.get(&tid)?;
    let dims: Vec<usize> = info
        .shape
        .iter()
        .filter_map(|d| match d {
            Dim::Concrete(n) => Some(*n as usize),
            _ => None,
        })
        .collect();
    if dims.len() == 2 {
        Some((dims[0], dims[1]))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AiGraph, AiNode, AiOp, AiParam, SemanticHint, TensorInfo};

    fn empty_graph() -> AiGraph {
        AiGraph {
            name: "test".to_string(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_names: Vec::new(),
            output_names: Vec::new(),
            params: Default::default(),
            tensor_info: Default::default(),
            metadata: Default::default(),
            warnings: Vec::new(),
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: Default::default(),
            tensor_names: Default::default(),
            topo_cache: Default::default(),
        }
    }

    fn f32_weight(k: usize, n: usize) -> (AiParam, TensorInfo) {
        let data = vec![0u8; k * n * 4];
        let info = TensorInfo {
            shape: crate::ir::shape_from_concrete(&[k as u64, n as u64]),
            logical_dtype: DType::F32,
            storage_dtype: DType::F32,
            quant: hologram_ai_quant::QuantDescriptor::none(),
            known_i64_values: None,
            semantic: SemanticHint::Unknown,
        };
        (
            AiParam::Inline {
                data: data.into(),
                info: info.clone(),
            },
            info,
        )
    }

    #[test]
    fn fuses_rmsnorm_with_two_matmul_consumers() {
        let mut g = empty_graph();
        // x=10, norm_weight=11, W_a=12, W_b=13
        // RmsNorm(x, norm_weight) → normed=20
        // MatMul(normed, W_a) → a=30
        // MatMul(normed, W_b) → b=31
        g.inputs = vec![10];
        g.outputs = vec![30, 31];

        let (param_a, info_a) = f32_weight(512, 256);
        let (param_b, info_b) = f32_weight(512, 256);
        g.params.insert(12, param_a);
        g.params.insert(13, param_b);
        g.tensor_info.insert(12, info_a);
        g.tensor_info.insert(13, info_b);

        g.tensor_info.insert(
            10,
            TensorInfo {
                shape: crate::ir::shape_from_concrete(&[1, 512]),
                logical_dtype: DType::F32,
                storage_dtype: DType::F32,
                quant: hologram_ai_quant::QuantDescriptor::none(),
                known_i64_values: None,
                semantic: SemanticHint::Unknown,
            },
        );

        g.nodes = vec![
            AiNode::new(0, AiOp::RmsNorm { epsilon: 1e-5 }, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::MatMul, vec![20, 12], vec![30]),
            AiNode::new(2, AiOp::MatMul, vec![20, 13], vec![31]),
        ];

        let result = NormProjectionFusion.run(g).expect("pass should succeed");
        // Multi-output: 1 FusedNormProjection with 2 outputs (no Slices needed).
        assert_eq!(result.nodes.len(), 1, "should have 1 fused node");
        assert!(
            matches!(result.nodes[0].op, AiOp::FusedNormProjection { .. }),
            "should be FusedNormProjection, got {:?}",
            result.nodes[0].op
        );

        if let AiOp::FusedNormProjection {
            epsilon,
            split_sizes,
            has_residual_add,
        } = &result.nodes[0].op
        {
            assert!((*epsilon - 1e-5_f64).abs() < 1e-10);
            assert_eq!(split_sizes, &[256, 256]);
            assert!(!has_residual_add);
        }

        // Outputs should be the original MatMul output tids.
        assert_eq!(result.nodes[0].outputs, vec![30, 31]);
        // Inputs: [x, norm_weight, W_a, W_b]
        assert_eq!(result.nodes[0].inputs, vec![10, 11, 12, 13]);
    }

    #[test]
    fn skips_single_matmul_consumer() {
        let mut g = empty_graph();
        g.inputs = vec![10];
        g.outputs = vec![30];

        let (param_a, info_a) = f32_weight(512, 256);
        g.params.insert(12, param_a);
        g.tensor_info.insert(12, info_a);

        g.nodes = vec![
            AiNode::new(0, AiOp::RmsNorm { epsilon: 1e-5 }, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::MatMul, vec![20, 12], vec![30]),
        ];

        let result = NormProjectionFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 2, "should not fuse single consumer");
    }

    #[test]
    fn skips_when_norm_has_non_matmul_consumer() {
        let mut g = empty_graph();
        g.inputs = vec![10];
        g.outputs = vec![30, 31, 40];

        let (param_a, info_a) = f32_weight(512, 256);
        let (param_b, info_b) = f32_weight(512, 256);
        g.params.insert(12, param_a);
        g.params.insert(13, param_b);
        g.tensor_info.insert(12, info_a);
        g.tensor_info.insert(13, info_b);

        g.nodes = vec![
            AiNode::new(0, AiOp::RmsNorm { epsilon: 1e-5 }, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::MatMul, vec![20, 12], vec![30]),
            AiNode::new(2, AiOp::MatMul, vec![20, 13], vec![31]),
            AiNode::new(3, AiOp::Add, vec![20, 10], vec![40]),
        ];

        let result = NormProjectionFusion.run(g).expect("pass should succeed");
        assert_eq!(
            result.nodes.len(),
            4,
            "should not fuse with non-MatMul consumer"
        );
    }
}
