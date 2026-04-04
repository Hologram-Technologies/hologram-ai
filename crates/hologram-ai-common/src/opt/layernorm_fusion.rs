//! LayerNorm pattern fusion.
//!
//! Detects the explicit ONNX LayerNorm subgraph and fuses it into a single
//! `AiOp::LayerNorm { axis, epsilon }` node.
//!
//! # Pattern matched
//!
//! ```text
//! mean     = ReduceMean(X, axis=-1)
//! centered = Sub(X, mean)
//! pow2     = Pow(centered, 2.0)
//! var      = ReduceMean(pow2, axis=-1)
//! biased   = Add(var, ε)        // ε is a scalar param
//! std      = Sqrt(biased)
//! normed   = Div(centered, std)
//! scaled   = Mul(normed, weight) // weight is a param or graph input
//! output   = Add(scaled, bias)   // bias is a param or graph input
//! ```
//!
//! Fused to: `LayerNorm { axis: -1, epsilon: ε }` with inputs `[X, weight, bias]`.

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, AiParam, TensorId};
use std::collections::{HashMap, HashSet};

/// Fuse explicit ONNX LayerNorm chains into `AiOp::LayerNorm`.
pub struct LayerNormFusion;

impl Pass for LayerNormFusion {
    fn name(&self) -> &str {
        "LayerNormFusion"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        // tid -> index in graph.nodes (for the node that produces that tensor).
        let tid_to_node: HashMap<TensorId, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.outputs.iter().map(move |&tid| (tid, i)))
            .collect();

        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut replacements: HashMap<usize, AiNode> = HashMap::new();

        let mut next_id = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            // Outer Add(scaled, bias) -- one input is a bias param/graph-input,
            // one is the scaled computed tensor.
            if !matches!(node.op, AiOp::Add) || node.inputs.len() < 2 || node.outputs.is_empty() {
                continue;
            }

            let out_tid = node.outputs[0];

            // Try both orderings of (scaled_tid, bias_tid).
            let candidates = [
                (node.inputs[0], node.inputs[1]),
                (node.inputs[1], node.inputs[0]),
            ];
            for (scaled_tid, bias_tid) in candidates {
                // bias must be a param or graph input (not a computed tensor).
                let bias_is_param = graph.params.contains_key(&bias_tid);
                let bias_is_graph_input = graph.inputs.contains(&bias_tid);
                if !bias_is_param && !bias_is_graph_input {
                    continue;
                }
                if graph.params.contains_key(&scaled_tid) {
                    continue; // scaled must be a computed tensor
                }

                if let Some((x_tid, weight_tid, epsilon, inner_idxs)) =
                    match_layernorm_inner(scaled_tid, &tid_to_node, &graph)
                {
                    for idx in &inner_idxs {
                        to_remove.insert(*idx);
                    }
                    to_remove.insert(node_idx);

                    let fused = AiNode::new(
                        next_id,
                        AiOp::LayerNorm {
                            axis: -1,
                            epsilon,
                        },
                        vec![x_tid, weight_tid, bias_tid],
                        vec![out_tid],
                    );
                    next_id += 1;
                    replacements.insert(node_idx, fused);

                    tracing::debug!(
                        x = x_tid,
                        weight = weight_tid,
                        bias = bias_tid,
                        epsilon,
                        out = out_tid,
                        "LayerNormFusion: fused LayerNorm chain"
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
        graph.invalidate_topo_cache();

        tracing::info!("LayerNormFusion: fused {fused_count} LayerNorm chain(s)");
        Ok(graph)
    }
}

// -- Pattern matching ---------------------------------------------------------

/// Try to match:
/// ```text
/// scaled_tid <- Mul(normed, weight)
/// normed     <- Div(centered, Sqrt(Add(ReduceMean(Pow(centered, 2)), eps)))
/// centered   <- Sub(X, ReduceMean(X))
/// ```
///
/// Returns `(x_tid, weight_tid, epsilon, consumed_node_indices)` on success.
fn match_layernorm_inner(
    scaled_tid: TensorId,
    tid_to_node: &HashMap<TensorId, usize>,
    graph: &AiGraph,
) -> Option<(TensorId, TensorId, f32, Vec<usize>)> {
    if graph.params.contains_key(&scaled_tid) {
        return None;
    }

    // scaled_tid <- Mul(normed, weight)
    let mul_idx = *tid_to_node.get(&scaled_tid)?;
    let mul_node = &graph.nodes[mul_idx];
    if !matches!(mul_node.op, AiOp::Mul) || mul_node.inputs.len() < 2 {
        return None;
    }

    // Try both orderings: (normed, weight) and (weight, normed).
    let mul_candidates = [
        (mul_node.inputs[0], mul_node.inputs[1]),
        (mul_node.inputs[1], mul_node.inputs[0]),
    ];
    for (normed_tid, weight_tid) in mul_candidates {
        let weight_is_param = graph.params.contains_key(&weight_tid);
        let weight_is_graph_input = graph.inputs.contains(&weight_tid);
        if !weight_is_param && !weight_is_graph_input {
            continue;
        }
        if graph.params.contains_key(&normed_tid) {
            continue;
        }

        if let Some((x_tid, epsilon, mut inner_idxs)) =
            match_layernorm_core(normed_tid, tid_to_node, graph)
        {
            inner_idxs.push(mul_idx);
            return Some((x_tid, weight_tid, epsilon, inner_idxs));
        }
    }
    None
}

/// Match the core LayerNorm pattern from the normed output back to X:
/// ```text
/// normed   <- Div(centered, std)
/// std      <- Sqrt(biased)
/// biased   <- Add(var, eps)
/// var      <- ReduceMean(pow2)
/// pow2     <- Pow(centered, 2.0)
/// centered <- Sub(X, mean)
/// mean     <- ReduceMean(X)
/// ```
///
/// Returns `(x_tid, epsilon, consumed_node_indices)`.
fn match_layernorm_core(
    normed_tid: TensorId,
    tid_to_node: &HashMap<TensorId, usize>,
    graph: &AiGraph,
) -> Option<(TensorId, f32, Vec<usize>)> {
    if graph.params.contains_key(&normed_tid) {
        return None;
    }

    // normed <- Div(centered, std)
    let div_idx = *tid_to_node.get(&normed_tid)?;
    let div_node = &graph.nodes[div_idx];
    if !matches!(div_node.op, AiOp::Div) || div_node.inputs.len() < 2 {
        return None;
    }
    let centered_tid = div_node.inputs[0];
    let std_tid = div_node.inputs[1];

    let mut consumed = vec![div_idx];

    // std <- Sqrt(biased)
    let sqrt_idx = *tid_to_node.get(&std_tid)?;
    let sqrt_node = &graph.nodes[sqrt_idx];
    if !matches!(sqrt_node.op, AiOp::Sqrt) || sqrt_node.inputs.is_empty() {
        return None;
    }
    consumed.push(sqrt_idx);
    let biased_tid = sqrt_node.inputs[0];

    // biased <- Add(var, eps) or Add(eps, var)
    let add_idx = *tid_to_node.get(&biased_tid)?;
    let add_node = &graph.nodes[add_idx];
    if !matches!(add_node.op, AiOp::Add) || add_node.inputs.len() < 2 {
        return None;
    }
    consumed.push(add_idx);

    let (var_tid, epsilon) = if let Some(eps) = scalar_f32_param(add_node.inputs[0], graph) {
        (add_node.inputs[1], eps)
    } else if let Some(eps) = scalar_f32_param(add_node.inputs[1], graph) {
        (add_node.inputs[0], eps)
    } else {
        return None;
    };

    // var <- ReduceMean(pow2)
    let var_reduce_idx = *tid_to_node.get(&var_tid)?;
    let var_reduce_node = &graph.nodes[var_reduce_idx];
    if !matches!(var_reduce_node.op, AiOp::ReduceMean { .. }) || var_reduce_node.inputs.is_empty() {
        return None;
    }
    consumed.push(var_reduce_idx);
    let pow2_tid = var_reduce_node.inputs[0];

    // pow2 <- Pow(centered, 2.0)
    let pow_idx = *tid_to_node.get(&pow2_tid)?;
    let pow_node = &graph.nodes[pow_idx];
    let centered_from_pow = match &pow_node.op {
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

    // The centered tensor from Pow must match the one from Div.
    if centered_from_pow != centered_tid {
        return None;
    }

    // centered <- Sub(X, mean)
    if graph.params.contains_key(&centered_tid) {
        return None;
    }
    let sub_idx = *tid_to_node.get(&centered_tid)?;
    let sub_node = &graph.nodes[sub_idx];
    if !matches!(sub_node.op, AiOp::Sub) || sub_node.inputs.len() < 2 {
        return None;
    }
    consumed.push(sub_idx);
    let x_tid = sub_node.inputs[0];
    let mean_tid = sub_node.inputs[1];

    // mean <- ReduceMean(X)
    let mean_reduce_idx = *tid_to_node.get(&mean_tid)?;
    let mean_reduce_node = &graph.nodes[mean_reduce_idx];
    if !matches!(mean_reduce_node.op, AiOp::ReduceMean { .. })
        || mean_reduce_node.inputs.is_empty()
    {
        return None;
    }
    consumed.push(mean_reduce_idx);

    // The input to ReduceMean(mean) must be the same X as Sub(X, mean).
    if mean_reduce_node.inputs[0] != x_tid {
        return None;
    }

    Some((x_tid, epsilon, consumed))
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

// -- Tests --------------------------------------------------------------------

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

    /// Build a standard LayerNorm graph:
    ///   mean(2) = ReduceMean(X(0))
    ///   centered(3) = Sub(X, mean)
    ///   pow2(5) = Pow(centered, TWO(4))
    ///   var(6) = ReduceMean(pow2)
    ///   biased(8) = Add(var, EPS(7))
    ///   std(9) = Sqrt(biased)
    ///   normed(10) = Div(centered, std)
    ///   scaled(12) = Mul(normed, W(11))
    ///   output(14) = Add(scaled, B(13))
    fn build_layernorm_graph(hidden: u64) -> AiGraph {
        let x_tid: TensorId = 0;
        let mean_tid: TensorId = 2;
        let centered_tid: TensorId = 3;
        let two_tid: TensorId = 4;
        let pow2_tid: TensorId = 5;
        let var_tid: TensorId = 6;
        let eps_tid: TensorId = 7;
        let biased_tid: TensorId = 8;
        let std_tid: TensorId = 9;
        let normed_tid: TensorId = 10;
        let w_tid: TensorId = 11;
        let scaled_tid: TensorId = 12;
        let b_tid: TensorId = 13;
        let out_tid: TensorId = 14;

        let x_shape = shape_from_concrete(&[2, hidden]);
        let scalar_shape = shape_from_concrete(&[]);
        let reduced_shape = shape_from_concrete(&[2, 1]);
        let hidden_shape = shape_from_concrete(&[hidden]);

        let mut tensor_info: HashMap<TensorId, TensorInfo> = HashMap::new();
        tensor_info.insert(x_tid, TensorInfo::new(DType::F32, x_shape.clone()));
        tensor_info.insert(
            mean_tid,
            TensorInfo::new(DType::F32, reduced_shape.clone()),
        );
        tensor_info.insert(
            centered_tid,
            TensorInfo::new(DType::F32, x_shape.clone()),
        );
        tensor_info.insert(two_tid, TensorInfo::new(DType::F32, scalar_shape.clone()));
        tensor_info.insert(pow2_tid, TensorInfo::new(DType::F32, x_shape.clone()));
        tensor_info.insert(
            var_tid,
            TensorInfo::new(DType::F32, reduced_shape.clone()),
        );
        tensor_info.insert(eps_tid, TensorInfo::new(DType::F32, scalar_shape));
        tensor_info.insert(biased_tid, TensorInfo::new(DType::F32, reduced_shape.clone()));
        tensor_info.insert(std_tid, TensorInfo::new(DType::F32, reduced_shape));
        tensor_info.insert(normed_tid, TensorInfo::new(DType::F32, x_shape.clone()));
        tensor_info.insert(w_tid, TensorInfo::new(DType::F32, hidden_shape.clone()));
        tensor_info.insert(scaled_tid, TensorInfo::new(DType::F32, x_shape.clone()));
        tensor_info.insert(b_tid, TensorInfo::new(DType::F32, hidden_shape));
        tensor_info.insert(out_tid, TensorInfo::new(DType::F32, x_shape));

        let mut params: HashMap<TensorId, AiParam> = HashMap::new();
        params.insert(two_tid, f32_param(2.0));
        params.insert(eps_tid, f32_param(1e-5));
        params.insert(w_tid, vec_param(&vec![1.0f32; hidden as usize]));
        params.insert(b_tid, vec_param(&vec![0.0f32; hidden as usize]));

        let nodes = vec![
            AiNode::new(
                0,
                AiOp::ReduceMean {
                    axes: vec![-1],
                    keepdims: true,
                },
                vec![x_tid],
                vec![mean_tid],
            ),
            AiNode::new(1, AiOp::Sub, vec![x_tid, mean_tid], vec![centered_tid]),
            AiNode::new(
                2,
                AiOp::Pow,
                vec![centered_tid, two_tid],
                vec![pow2_tid],
            ),
            AiNode::new(
                3,
                AiOp::ReduceMean {
                    axes: vec![-1],
                    keepdims: true,
                },
                vec![pow2_tid],
                vec![var_tid],
            ),
            AiNode::new(4, AiOp::Add, vec![var_tid, eps_tid], vec![biased_tid]),
            AiNode::new(5, AiOp::Sqrt, vec![biased_tid], vec![std_tid]),
            AiNode::new(
                6,
                AiOp::Div,
                vec![centered_tid, std_tid],
                vec![normed_tid],
            ),
            AiNode::new(7, AiOp::Mul, vec![normed_tid, w_tid], vec![scaled_tid]),
            AiNode::new(8, AiOp::Add, vec![scaled_tid, b_tid], vec![out_tid]),
        ];

        AiGraph {
            name: "test_layernorm".to_string(),
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
            topo_cache: Default::default(),
        }
    }

    #[test]
    fn fuses_layernorm_chain() {
        let graph = build_layernorm_graph(64);
        let pass = LayerNormFusion;
        let graph = pass.run(graph).expect("pass should succeed");

        assert_eq!(
            graph.nodes.len(),
            1,
            "expected 1 node after fusion, got {}: {:?}",
            graph.nodes.len(),
            graph.nodes.iter().map(|n| &n.op).collect::<Vec<_>>()
        );
        let node = &graph.nodes[0];
        assert!(
            matches!(node.op, AiOp::LayerNorm { axis: -1, epsilon } if (epsilon - 1e-5).abs() < 1e-8)
        );
        assert_eq!(node.inputs[0], 0); // x
        assert_eq!(node.inputs[1], 11); // weight
        assert_eq!(node.inputs[2], 13); // bias
        assert_eq!(node.outputs[0], 14); // out
    }

    #[test]
    fn fuses_layernorm_through_pipeline() {
        use crate::opt::pipeline::OptPipeline;
        let graph = build_layernorm_graph(64);
        let pipeline = OptPipeline::mvp();
        let graph = pipeline.run(graph).expect("pipeline should succeed");

        let ln_nodes: Vec<_> = graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, AiOp::LayerNorm { .. }))
            .collect();
        assert_eq!(
            ln_nodes.len(),
            1,
            "expected 1 LayerNorm after pipeline, got {}: {:?}",
            graph.nodes.len(),
            graph.nodes.iter().map(|n| &n.op).collect::<Vec<_>>()
        );
    }

    #[test]
    fn no_fusion_when_sub_missing() {
        let mut graph = build_layernorm_graph(64);
        // Replace the Sub node with Identity to break the pattern.
        graph.nodes[1].op = AiOp::Identity;
        let pass = LayerNormFusion;
        let graph = pass.run(graph).expect("pass should succeed");
        assert_eq!(graph.nodes.len(), 9, "no fusion should occur");
    }

    #[test]
    fn no_fusion_when_x_mismatch() {
        let mut graph = build_layernorm_graph(64);
        // Make the first ReduceMean operate on a different input.
        graph.nodes[0].inputs[0] = 99;
        graph.tensor_info.insert(
            99,
            TensorInfo::new(DType::F32, shape_from_concrete(&[2, 64])),
        );
        let pass = LayerNormFusion;
        let graph = pass.run(graph).expect("pass should succeed");
        assert_eq!(graph.nodes.len(), 9, "no fusion should occur");
    }
}
