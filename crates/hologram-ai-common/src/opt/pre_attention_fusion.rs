//! Pre-attention fusion: fold QK-norm and RoPE into the Attention op.
//!
//! # Patterns
//!
//! **QK-Norm** (RmsNorm on Q and/or K before attention):
//! ```text
//! Q' = RmsNorm(Q, weight_q)
//! K' = RmsNorm(K, weight_k)
//! out = Attention(Q', K', V)    → Attention(Q, K, V, weight_q, weight_k) with qk_norm=true
//! ```
//!
//! **RoPE** (RotaryEmbedding on Q and/or K before attention):
//! ```text
//! Q' = RotaryEmbedding(Q)
//! K' = RotaryEmbedding(K)
//! out = Attention(Q', K', V)    → Attention(Q, K, V) with rope=true
//! ```
//!
//! The fused Attention op carries flags (`qk_norm`, `rope`, `rope_base`)
//! that the hologram base kernel reads to apply these transformations
//! inline, eliminating intermediate buffers and dispatches.

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, TensorId};
use std::collections::{HashMap, HashSet};
use tracing::info;

/// Fuse QK-Norm and RoPE into `GroupedQueryAttention`.
pub struct PreAttentionFusion;

impl Pass for PreAttentionFusion {
    fn name(&self) -> &str {
        "PreAttentionFusion"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let tid_to_node: HashMap<TensorId, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.outputs.iter().map(move |&tid| (tid, i)))
            .collect();

        let mut consumer_count: HashMap<TensorId, usize> = HashMap::new();
        for n in &graph.nodes {
            for &tid in &n.inputs {
                *consumer_count.entry(tid).or_default() += 1;
            }
        }

        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut replacements: HashMap<usize, AiNode> = HashMap::new();
        let mut fused_rope = 0u32;
        let mut fused_qk_norm = 0u32;

        for (attn_idx, attn_node) in graph.nodes.iter().enumerate() {
            let (num_heads, num_kv_heads, head_dim, scale, causal, heads_first) =
                match &attn_node.op {
                    AiOp::GroupedQueryAttention {
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        scale,
                        causal,
                        heads_first,
                        ..
                    } => (*num_heads, *num_kv_heads, *head_dim, *scale, *causal, *heads_first),
                    _ => continue,
                };

            if attn_node.inputs.len() < 3 {
                continue;
            }

            let q_tid = attn_node.inputs[0];
            let k_tid = attn_node.inputs[1];
            let v_tid = attn_node.inputs[2];

            let mut new_q_tid = q_tid;
            let mut new_k_tid = k_tid;
            let mut has_rope = false;
            let mut rope_base_val: f32 = 10000.0;
            let mut has_qk_norm = false;
            let mut remove_nodes: Vec<usize> = Vec::new();
            let mut extra_inputs: Vec<TensorId> = Vec::new();

            // Phase 1: Peel off RoPE on Q and K (must be on both).
            let (q_rope_source, q_rope_idx, q_rope_base) =
                try_peel_rope(&graph, q_tid, &tid_to_node, &consumer_count);
            let (k_rope_source, k_rope_idx, k_rope_base) =
                try_peel_rope(&graph, k_tid, &tid_to_node, &consumer_count);

            if let (Some(q_src), Some(q_idx), Some(k_src), Some(k_idx)) =
                (q_rope_source, q_rope_idx, k_rope_source, k_rope_idx)
            {
                new_q_tid = q_src;
                new_k_tid = k_src;
                has_rope = true;
                rope_base_val = q_rope_base.unwrap_or(k_rope_base.unwrap_or(10000.0));
                remove_nodes.push(q_idx);
                remove_nodes.push(k_idx);
                fused_rope += 1;
            }

            // Phase 2: Peel off RmsNorm (QK-Norm) on the (possibly updated) Q and K.
            let (q_norm_source, q_norm_idx, q_norm_weight) =
                try_peel_rmsnorm(&graph, new_q_tid, &tid_to_node, &consumer_count);
            let (k_norm_source, k_norm_idx, k_norm_weight) =
                try_peel_rmsnorm(&graph, new_k_tid, &tid_to_node, &consumer_count);

            if let (
                Some(q_src),
                Some(q_idx),
                Some(q_w),
                Some(k_src),
                Some(k_idx),
                Some(k_w),
            ) = (
                q_norm_source,
                q_norm_idx,
                q_norm_weight,
                k_norm_source,
                k_norm_idx,
                k_norm_weight,
            ) {
                new_q_tid = q_src;
                new_k_tid = k_src;
                has_qk_norm = true;
                extra_inputs.push(q_w);
                extra_inputs.push(k_w);
                remove_nodes.push(q_idx);
                remove_nodes.push(k_idx);
                fused_qk_norm += 1;
            }

            if !has_rope && !has_qk_norm {
                continue;
            }

            let mut new_inputs = vec![new_q_tid, new_k_tid, v_tid];
            // Preserve optional mask input.
            if attn_node.inputs.len() > 3 {
                new_inputs.push(attn_node.inputs[3]);
            }
            new_inputs.extend(extra_inputs);

            let fused_node = AiNode::new(
                attn_node.id,
                AiOp::GroupedQueryAttention {
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    causal,
                    heads_first,
                    qk_norm: has_qk_norm,
                    rope: has_rope,
                    rope_base: rope_base_val,
                },
                new_inputs,
                attn_node.outputs.clone(),
            );

            replacements.insert(attn_idx, fused_node);
            for idx in remove_nodes {
                to_remove.insert(idx);
            }
        }

        for (idx, node) in replacements {
            graph.nodes[idx] = node;
        }

        if !to_remove.is_empty() {
            graph.nodes = graph
                .nodes
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !to_remove.contains(i))
                .map(|(_, n)| n)
                .collect();
        }

        if fused_rope > 0 || fused_qk_norm > 0 {
            info!(
                rope_fused = fused_rope,
                qk_norm_fused = fused_qk_norm,
                "pre-attention fusion"
            );
        }

        Ok(graph)
    }
}

/// Try to peel a single-consumer RotaryEmbedding off a tensor.
fn try_peel_rope(
    graph: &AiGraph,
    tid: TensorId,
    tid_to_node: &HashMap<TensorId, usize>,
    consumer_count: &HashMap<TensorId, usize>,
) -> (Option<TensorId>, Option<usize>, Option<f32>) {
    let node_idx = match tid_to_node.get(&tid) {
        Some(&idx) => idx,
        None => return (None, None, None),
    };
    let node = &graph.nodes[node_idx];
    match &node.op {
        AiOp::RotaryEmbedding { base, .. } => {
            let output_tid = node.outputs.first().copied().unwrap_or(tid);
            if consumer_count.get(&output_tid).copied().unwrap_or(0) != 1 {
                return (None, None, None);
            }
            let source = node.inputs.first().copied();
            (source, Some(node_idx), Some(*base))
        }
        _ => (None, None, None),
    }
}

/// Try to peel a single-consumer RmsNorm off a tensor (QK-norm pattern).
fn try_peel_rmsnorm(
    graph: &AiGraph,
    tid: TensorId,
    tid_to_node: &HashMap<TensorId, usize>,
    consumer_count: &HashMap<TensorId, usize>,
) -> (Option<TensorId>, Option<usize>, Option<TensorId>) {
    let node_idx = match tid_to_node.get(&tid) {
        Some(&idx) => idx,
        None => return (None, None, None),
    };
    let node = &graph.nodes[node_idx];
    match &node.op {
        AiOp::RmsNorm { .. } => {
            let output_tid = node.outputs.first().copied().unwrap_or(tid);
            if consumer_count.get(&output_tid).copied().unwrap_or(0) != 1 {
                return (None, None, None);
            }
            if node.inputs.len() < 2 {
                return (None, None, None);
            }
            let source = Some(node.inputs[0]);
            let weight = Some(node.inputs[1]);
            (source, Some(node_idx), weight)
        }
        _ => (None, None, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AiGraph, AiNode, AiOp};

    fn empty_graph() -> AiGraph {
        AiGraph {
            name: String::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_names: Vec::new(),
            output_names: Vec::new(),
            params: HashMap::new(),
            tensor_info: HashMap::new(),
            metadata: HashMap::new(),
            warnings: Vec::new(),
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        }
    }

    fn rope_node(id: u32, input: TensorId, output: TensorId, base: f32) -> AiNode {
        AiNode::new(
            id,
            AiOp::RotaryEmbedding { base, dim: 64 },
            vec![input],
            vec![output],
        )
    }

    fn rmsnorm_node(id: u32, data: TensorId, weight: TensorId, output: TensorId) -> AiNode {
        AiNode::new(
            id,
            AiOp::RmsNorm { epsilon: 1e-6 },
            vec![data, weight],
            vec![output],
        )
    }

    fn gqa_node(id: u32, q: TensorId, k: TensorId, v: TensorId, output: TensorId) -> AiNode {
        AiNode::new(
            id,
            AiOp::GroupedQueryAttention {
                num_heads: 4,
                num_kv_heads: 4,
                head_dim: 64,
                scale: None,
                causal: true,
                heads_first: true,
                qk_norm: false,
                rope: false,
                rope_base: 0.0,
            },
            vec![q, k, v],
            vec![output],
        )
    }

    fn input_node(id: u32, output: TensorId) -> AiNode {
        AiNode::new(id, AiOp::Identity, vec![], vec![output])
    }

    #[test]
    fn fuses_rope_on_q_and_k() {
        let mut graph = empty_graph();
        graph.nodes = vec![
            input_node(0, 100),
            input_node(1, 101),
            input_node(2, 102),
            rope_node(3, 100, 200, 10000.0),
            rope_node(4, 101, 201, 10000.0),
            gqa_node(5, 200, 201, 102, 300),
        ];

        let pass = PreAttentionFusion;
        let result = pass.run(graph).expect("fusion should succeed");

        // RoPE nodes removed: 3 inputs + 1 fused attention = 4.
        assert_eq!(result.nodes.len(), 4);

        let attn = result.nodes.last().expect("should have attention");
        assert_eq!(attn.inputs[0], 100); // Q_raw
        assert_eq!(attn.inputs[1], 101); // K_raw
        assert_eq!(attn.inputs[2], 102); // V

        match &attn.op {
            AiOp::GroupedQueryAttention { rope, rope_base, .. } => {
                assert!(*rope);
                assert_eq!(*rope_base, 10000.0);
            }
            _ => panic!("expected GQA"),
        }
    }

    #[test]
    fn fuses_qk_norm_on_q_and_k() {
        let mut graph = empty_graph();
        graph.nodes = vec![
            input_node(0, 100),
            input_node(1, 101),
            input_node(2, 102),
            input_node(3, 103), // w_q
            input_node(4, 104), // w_k
            rmsnorm_node(5, 100, 103, 200),
            rmsnorm_node(6, 101, 104, 201),
            gqa_node(7, 200, 201, 102, 300),
        ];

        let pass = PreAttentionFusion;
        let result = pass.run(graph).expect("fusion should succeed");

        // 5 inputs + 1 fused attention = 6.
        assert_eq!(result.nodes.len(), 6);

        let attn = result.nodes.last().expect("should have attention");
        assert_eq!(attn.inputs[0], 100); // Q_raw
        assert_eq!(attn.inputs[1], 101); // K_raw
        assert_eq!(attn.inputs[2], 102); // V
        assert_eq!(attn.inputs[3], 103); // w_q
        assert_eq!(attn.inputs[4], 104); // w_k

        match &attn.op {
            AiOp::GroupedQueryAttention { qk_norm, .. } => assert!(*qk_norm),
            _ => panic!("expected GQA"),
        }
    }

    #[test]
    fn no_fusion_when_rope_has_multiple_consumers() {
        let mut graph = empty_graph();
        graph.nodes = vec![
            input_node(0, 100),
            input_node(1, 101),
            input_node(2, 102),
            rope_node(3, 100, 200, 10000.0),
            rope_node(4, 101, 201, 10000.0),
            gqa_node(5, 200, 201, 102, 300),
            // Extra consumer of RoPE(Q) output:
            AiNode::new(6, AiOp::Relu, vec![200], vec![301]),
        ];

        let pass = PreAttentionFusion;
        let result = pass.run(graph).expect("should succeed");
        assert_eq!(result.nodes.len(), 7); // No fusion
    }

    #[test]
    fn no_fusion_when_only_q_has_rope() {
        let mut graph = empty_graph();
        graph.nodes = vec![
            input_node(0, 100),
            input_node(1, 101),
            input_node(2, 102),
            rope_node(3, 100, 200, 10000.0),
            gqa_node(4, 200, 101, 102, 300),
        ];

        let pass = PreAttentionFusion;
        let result = pass.run(graph).expect("should succeed");
        assert_eq!(result.nodes.len(), 5); // No fusion
    }
}
