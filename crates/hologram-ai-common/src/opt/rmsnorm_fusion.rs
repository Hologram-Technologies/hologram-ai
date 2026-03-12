//! RMSNorm pattern fusion.
//!
//! Detects the explicit ONNX RMSNorm subgraph and fuses it into a single
//! `AiOp::RmsNorm { epsilon }` node.
//!
//! # Why this is needed
//!
//! Many models (LLaMA 2/3, Mistral, TinyLlama, Qwen, …) implement RMSNorm
//! as a chain of standard ONNX ops rather than a contrib op:
//!
//! ```text
//! x → Pow(x, 2) → ReduceMean(last-dim) → Add(ε) → Sqrt
//!                                                       ↓
//! x ──────────────────────────────────── Div or Reciprocal → Mul
//!                                                               ↓
//!                                           weight (param) → Mul → output
//! ```
//!
//! Without fusion, the `Mul(x, 1/rms)` step reaches the executor as a plain
//! `FloatOp::Mul`.  The executor's `binary_elementwise` uses cycling broadcast
//! (`a[i % len_a] * b[i % len_b]`), which is **incorrect** when `x` has shape
//! `[batch, seq, hidden]` and `1/rms` has shape `[batch, seq, 1]`.  Cycling
//! iterates through all `seq` norm values per hidden element instead of
//! applying each position's own norm.  The result is completely wrong
//! normalization, causing residual-stream values to grow without bound across
//! layers and attention scores to diverge to ±∞/NaN.
//!
//! Fusing the chain into `AiOp::RmsNorm { epsilon }` routes execution through
//! `dispatch_rms_norm`, which correctly normalises each row independently.
//!
//! # Pattern matched
//!
//! ```text
//! pow2    = Pow(x, 2.0)
//! mean    = ReduceMean(pow2)
//! biased  = Add(mean, ε)   or   Add(ε, mean)     [ε is a small scalar param]
//! rms     = Sqrt(biased)
//! recip   = Reciprocal(rms)                       [optional: Div variant skips this]
//! normed  = Mul(x, recip)  or   Div(x, rms)
//! output  = Mul(normed, weight)  or  Mul(weight, normed)   [weight is a param]
//! ```
//!
//! Fused to: `RmsNorm { epsilon: ε }` with inputs `[x, weight]`.
//!
//! The pattern is commutative on `Mul` and `Add`, so argument order doesn't matter.

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, AiParam, TensorId};
use std::collections::{HashMap, HashSet};

/// Fuse explicit ONNX RMSNorm chains into `AiOp::RmsNorm`.
pub struct RmsNormFusion;

impl Pass for RmsNormFusion {
    fn name(&self) -> &str {
        "RmsNormFusion"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        // tid → index in graph.nodes (for the node that produces that tensor).
        let tid_to_node: HashMap<TensorId, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.outputs.iter().map(move |&tid| (tid, i)))
            .collect();

        let mut to_remove: HashSet<usize> = HashSet::new();
        // Maps node_idx → replacement AiNode (replaces the outer Mul).
        let mut replacements: HashMap<usize, AiNode> = HashMap::new();

        let mut next_id = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            // Outer Mul(normed, weight) — one input is a weight param, one is computed.
            if !matches!(node.op, AiOp::Mul) || node.inputs.len() < 2 || node.outputs.is_empty() {
                continue;
            }

            let out_tid = node.outputs[0];

            // Try both orderings of (normed_tid, weight_tid).
            let candidates = [
                (node.inputs[0], node.inputs[1]),
                (node.inputs[1], node.inputs[0]),
            ];
            for (normed_tid, weight_tid) in candidates {
                if !graph.params.contains_key(&weight_tid) {
                    continue;
                }
                if graph.params.contains_key(&normed_tid) {
                    continue; // normed must be a computed tensor
                }

                if let Some((x_tid, epsilon, inner_idxs)) =
                    match_rmsnorm_inner(normed_tid, &tid_to_node, &graph)
                {
                    for idx in &inner_idxs {
                        to_remove.insert(*idx);
                    }
                    to_remove.insert(node_idx);

                    let fused = AiNode::new(
                        next_id,
                        AiOp::RmsNorm { epsilon },
                        vec![x_tid, weight_tid],
                        vec![out_tid],
                    );
                    next_id += 1;
                    replacements.insert(node_idx, fused);

                    tracing::debug!(
                        x = x_tid,
                        weight = weight_tid,
                        epsilon,
                        out = out_tid,
                        "RmsNormFusion: fused RmsNorm chain"
                    );
                    break;
                }
            }
        }

        if replacements.is_empty() {
            return Ok(graph);
        }

        let fused_count = replacements.len();

        let mut new_nodes: Vec<AiNode> = Vec::with_capacity(graph.nodes.len());
        for (idx, node) in graph.nodes.into_iter().enumerate() {
            if let Some(replacement) = replacements.remove(&idx) {
                new_nodes.push(replacement);
            } else if !to_remove.contains(&idx) {
                new_nodes.push(node);
            }
        }
        graph.nodes = new_nodes;

        tracing::info!("RmsNormFusion: fused {fused_count} RmsNorm chain(s)");
        Ok(graph)
    }
}

// ── Pattern matching ─────────────────────────────────────────────────────────

/// Try to match:
/// ```text
/// normed_tid ← Mul(x, Reciprocal(rms))  |  Div(x, rms)
/// rms        ← Sqrt(Add(ReduceMean(Pow(x, 2.0)), ε))
/// ```
///
/// Returns `(x_tid, epsilon, consumed_node_indices)` on success.
fn match_rmsnorm_inner(
    normed_tid: TensorId,
    tid_to_node: &HashMap<TensorId, usize>,
    graph: &AiGraph,
) -> Option<(TensorId, f32, Vec<usize>)> {
    // normed_tid must be computed, not a param.
    if graph.params.contains_key(&normed_tid) {
        return None;
    }
    let inner_idx = *tid_to_node.get(&normed_tid)?;
    let inner_node = &graph.nodes[inner_idx];
    if inner_node.inputs.len() < 2 {
        return None;
    }

    let mut consumed = vec![inner_idx];

    // Match either Mul(x, recip) or Div(x, rms).
    let (x_tid_from_inner, rms_tid) = match &inner_node.op {
        AiOp::Mul => {
            let a = inner_node.inputs[0];
            let b = inner_node.inputs[1];
            // Try a as Reciprocal output (b is x), then b as Reciprocal (a is x).
            if let Some((rms, recip_idx)) = match_reciprocal_node(a, tid_to_node, graph) {
                consumed.push(recip_idx);
                (b, rms)
            } else if let Some((rms, recip_idx)) = match_reciprocal_node(b, tid_to_node, graph) {
                consumed.push(recip_idx);
                (a, rms)
            } else {
                return None;
            }
        }
        AiOp::Div => (inner_node.inputs[0], inner_node.inputs[1]),
        _ => return None,
    };

    // rms_tid ← Sqrt(biased).
    let sqrt_idx = *tid_to_node.get(&rms_tid)?;
    let sqrt_node = &graph.nodes[sqrt_idx];
    if !matches!(sqrt_node.op, AiOp::Sqrt) || sqrt_node.inputs.is_empty() {
        return None;
    }
    consumed.push(sqrt_idx);
    let biased_tid = sqrt_node.inputs[0];

    // biased_tid ← Add(mean, ε) or Add(ε, mean).
    let add_idx = *tid_to_node.get(&biased_tid)?;
    let add_node = &graph.nodes[add_idx];
    if !matches!(add_node.op, AiOp::Add) || add_node.inputs.len() < 2 {
        return None;
    }
    consumed.push(add_idx);

    let (mean_tid, epsilon) =
        if let Some(eps) = scalar_f32_param(add_node.inputs[0], graph) {
            (add_node.inputs[1], eps)
        } else if let Some(eps) = scalar_f32_param(add_node.inputs[1], graph) {
            (add_node.inputs[0], eps)
        } else {
            return None;
        };

    // mean_tid ← ReduceMean(pow2).
    let reduce_idx = *tid_to_node.get(&mean_tid)?;
    let reduce_node = &graph.nodes[reduce_idx];
    if !matches!(reduce_node.op, AiOp::ReduceMean { .. }) || reduce_node.inputs.is_empty() {
        return None;
    }
    consumed.push(reduce_idx);
    let pow2_tid = reduce_node.inputs[0];

    // pow2_tid ← Pow(x, 2.0).
    let pow_idx = *tid_to_node.get(&pow2_tid)?;
    let pow_node = &graph.nodes[pow_idx];
    let x_tid_from_pow = match &pow_node.op {
        AiOp::Pow if pow_node.inputs.len() >= 2 => {
            let a = pow_node.inputs[0];
            let b = pow_node.inputs[1];
            if scalar_f32_param(b, graph).is_some_and(|v| (v - 2.0).abs() < 0.01) {
                a
            } else if scalar_f32_param(a, graph).is_some_and(|v| (v - 2.0).abs() < 0.01) {
                b
            } else {
                return None;
            }
        }
        _ => return None,
    };
    consumed.push(pow_idx);

    // The x from Mul/Div must be the same as the x fed to Pow.
    if x_tid_from_inner != x_tid_from_pow {
        return None;
    }

    Some((x_tid_from_pow, epsilon, consumed))
}

/// If `tid` is produced by a `Reciprocal` node, return `(input_tid, node_idx)`.
fn match_reciprocal_node(
    tid: TensorId,
    tid_to_node: &HashMap<TensorId, usize>,
    graph: &AiGraph,
) -> Option<(TensorId, usize)> {
    if graph.params.contains_key(&tid) {
        return None;
    }
    let idx = *tid_to_node.get(&tid)?;
    let node = &graph.nodes[idx];
    if matches!(node.op, AiOp::Reciprocal) && !node.inputs.is_empty() {
        Some((node.inputs[0], idx))
    } else {
        None
    }
}

/// Read a scalar f32 from an inline param (must be exactly 4 bytes).
fn scalar_f32_param(tid: TensorId, graph: &AiGraph) -> Option<f32> {
    match graph.params.get(&tid)? {
        AiParam::Inline { data, .. } if data.len() == 4 => {
            let arr: [u8; 4] = data.as_slice().try_into().ok()?;
            Some(f32::from_le_bytes(arr))
        }
        _ => None,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        shape::shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorId, TensorInfo,
    };
    use std::collections::HashMap;

    fn f32_param(v: f32) -> AiParam {
        AiParam::Inline {
            data: v.to_le_bytes().to_vec(),
            info: TensorInfo::new(DType::F32, shape_from_concrete(&[])),
        }
    }

    fn vec_param(values: &[f32]) -> AiParam {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        AiParam::Inline {
            data,
            info: TensorInfo::new(DType::F32, shape_from_concrete(&[values.len() as u64])),
        }
    }

    fn build_rmsnorm_graph(hidden: u64) -> AiGraph {
        // Build:
        //   x(0) → Pow(x, TWO(1)) → pow2(2)
        //        → ReduceMean(pow2) → mean(3)
        //        → Add(mean, EPS(4)) → biased(5)
        //        → Sqrt(biased) → rms(6)
        //        → Reciprocal(rms) → recip(7)
        //   x(0) → Mul(x, recip) → normed(8)
        //        → Mul(normed, W(9)) → out(10)
        let x_tid: TensorId = 0;
        let two_tid: TensorId = 1;
        let pow2_tid: TensorId = 2;
        let mean_tid: TensorId = 3;
        let eps_tid: TensorId = 4;
        let biased_tid: TensorId = 5;
        let rms_tid: TensorId = 6;
        let recip_tid: TensorId = 7;
        let normed_tid: TensorId = 8;
        let w_tid: TensorId = 9;
        let out_tid: TensorId = 10;

        let x_shape = shape_from_concrete(&[1, 16, hidden]);
        let scalar_shape = shape_from_concrete(&[]);
        let hidden_shape = shape_from_concrete(&[hidden]);

        let mut tensor_info: HashMap<TensorId, TensorInfo> = HashMap::new();
        tensor_info.insert(x_tid, TensorInfo::new(DType::F32, x_shape.clone()));
        tensor_info.insert(two_tid, TensorInfo::new(DType::F32, scalar_shape.clone()));
        tensor_info.insert(pow2_tid, TensorInfo::new(DType::F32, x_shape.clone()));
        tensor_info.insert(mean_tid, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 16, 1])));
        tensor_info.insert(eps_tid, TensorInfo::new(DType::F32, scalar_shape.clone()));
        tensor_info.insert(biased_tid, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 16, 1])));
        tensor_info.insert(rms_tid, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 16, 1])));
        tensor_info.insert(recip_tid, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 16, 1])));
        tensor_info.insert(normed_tid, TensorInfo::new(DType::F32, x_shape.clone()));
        tensor_info.insert(w_tid, TensorInfo::new(DType::F32, hidden_shape));
        tensor_info.insert(out_tid, TensorInfo::new(DType::F32, x_shape));

        let mut params: HashMap<TensorId, AiParam> = HashMap::new();
        params.insert(two_tid, f32_param(2.0));
        params.insert(eps_tid, f32_param(1e-5));
        params.insert(
            w_tid,
            vec_param(&vec![1.0f32; hidden as usize]),
        );

        let nodes = vec![
            AiNode::new(0, AiOp::Pow, vec![x_tid, two_tid], vec![pow2_tid]),
            AiNode::new(1, AiOp::ReduceMean { axes: vec![-1], keepdims: true }, vec![pow2_tid], vec![mean_tid]),
            AiNode::new(2, AiOp::Add, vec![mean_tid, eps_tid], vec![biased_tid]),
            AiNode::new(3, AiOp::Sqrt, vec![biased_tid], vec![rms_tid]),
            AiNode::new(4, AiOp::Reciprocal, vec![rms_tid], vec![recip_tid]),
            AiNode::new(5, AiOp::Mul, vec![x_tid, recip_tid], vec![normed_tid]),
            AiNode::new(6, AiOp::Mul, vec![normed_tid, w_tid], vec![out_tid]),
        ];

        AiGraph {
            name: "test".to_string(),
            nodes,
            inputs: vec![x_tid],
            outputs: vec![out_tid],
            input_names: vec!["x".to_string()],
            output_names: vec!["out".to_string()],
            params,
            tensor_info,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
        }
    }

    #[test]
    fn fuses_rmsnorm_chain() {
        let graph = build_rmsnorm_graph(64);
        let pass = RmsNormFusion;
        let graph = pass.run(graph).unwrap();

        // Should have exactly 1 node: RmsNorm.
        assert_eq!(graph.nodes.len(), 1, "expected 1 node after fusion, got {}", graph.nodes.len());
        let node = &graph.nodes[0];
        assert!(matches!(node.op, AiOp::RmsNorm { epsilon } if (epsilon - 1e-5).abs() < 1e-8));
        // inputs: [x=0, weight=9], output: [10]
        assert_eq!(node.inputs[0], 0); // x
        assert_eq!(node.inputs[1], 9); // weight
        assert_eq!(node.outputs[0], 10); // out
    }

    #[test]
    fn no_fusion_when_no_pow() {
        // If the Pow is missing, the pattern should not match.
        let mut graph = build_rmsnorm_graph(64);
        // Replace the Pow node with a different op.
        graph.nodes[0].op = AiOp::Relu;
        let pass = RmsNormFusion;
        let graph = pass.run(graph).unwrap();
        // Should still have 7 nodes (no fusion).
        assert_eq!(graph.nodes.len(), 7);
    }

    #[test]
    fn no_fusion_when_x_mismatch() {
        // If the x fed to Mul differs from the x fed to Pow, no fusion.
        let mut graph = build_rmsnorm_graph(64);
        // Give the inner Mul a different first input (e.g., tid=99).
        graph.nodes[5].inputs[0] = 99;
        // Add a dummy tensor_info entry for tid 99.
        graph.tensor_info.insert(
            99,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 16, 64])),
        );
        let pass = RmsNormFusion;
        let graph = pass.run(graph).unwrap();
        assert_eq!(graph.nodes.len(), 7);
    }
}
