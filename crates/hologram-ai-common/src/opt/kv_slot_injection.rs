//! KV-cache slot injection for ONNX models.
//!
//! After `AttentionFusion` has fused decomposed SDPA chains into
//! `AiOp::GroupedQueryAttention` nodes, this pass inserts
//! `AiOp::KvSlotWrite` nodes on the K and V inputs of each attention
//! layer. This enables the runtime KV cache (decode mode) for ONNX models.
//!
//! The GGUF builder already injects these during graph construction
//! (see `arch/llama.rs`). This pass is idempotent — if K/V inputs
//! already come from `KvSlotWrite` nodes, the layer is skipped.

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, TensorId};
use std::collections::HashMap;

/// Inject `KvSlotWrite` nodes on K/V inputs of `GroupedQueryAttention`.
pub struct KvSlotInjection;

impl Pass for KvSlotInjection {
    fn name(&self) -> &str {
        "KvSlotInjection"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        // Build producer map: TensorId → node index that outputs it.
        let tid_producer: HashMap<TensorId, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.outputs.iter().map(move |&tid| (tid, i)))
            .collect();

        // Find all GroupedQueryAttention nodes.
        let gqa_indices: Vec<usize> = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n.op, AiOp::GroupedQueryAttention { .. }))
            .map(|(i, _)| i)
            .collect();

        if gqa_indices.is_empty() {
            return Ok(graph);
        }

        // Check if KvSlotWrite already exists (GGUF path). If the K input
        // of the first GQA node already comes from a KvSlotWrite, skip.
        if let Some(&first_gqa) = gqa_indices.first() {
            let node = &graph.nodes[first_gqa];
            if node.inputs.len() >= 2 {
                let k_tid = node.inputs[1];
                if let Some(&producer_idx) = tid_producer.get(&k_tid) {
                    if matches!(graph.nodes[producer_idx].op, AiOp::KvSlotWrite { .. }) {
                        tracing::debug!("KvSlotInjection: KvSlotWrite already present, skipping");
                        return Ok(graph);
                    }
                }
            }
        }

        // Allocate IDs above existing maximums.
        let mut next_node_id = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        let mut next_tid: TensorId = graph
            .nodes
            .iter()
            .flat_map(|n| n.inputs.iter().chain(n.outputs.iter()))
            .copied()
            .max()
            .unwrap_or(0)
            + 1;
        if let Some(&max_info) = graph.tensor_info.keys().max() {
            next_tid = next_tid.max(max_info + 1);
        }
        if let Some(&max_param) = graph.params.keys().max() {
            next_tid = next_tid.max(max_param + 1);
        }

        // Collect (gqa_index, layer, k_node, v_node) for each attention layer.
        // We also track already-created KvSlotWrite nodes per input TensorId
        // to avoid duplicates.
        let mut tid_remap: HashMap<TensorId, TensorId> = HashMap::new();

        struct Injection {
            gqa_idx: usize,
            k_node: AiNode,
            v_node: AiNode,
        }
        let mut injections: Vec<Injection> = Vec::new();

        for (layer, &gqa_idx) in gqa_indices.iter().enumerate() {
            let node = &graph.nodes[gqa_idx];
            if node.inputs.len() < 3 {
                tracing::warn!(layer, "KvSlotInjection: GQA node has <3 inputs, skipping");
                continue;
            }

            // Extract architecture params from the GQA node.
            // KV layout is always SeqFirst for ONNX: the attention fusion
            // traces K/V back past the head-reshaping Transpose, so the data
            // arriving at KvWrite is the flat projection output [seq, kv_heads*head_dim]
            // which is seq-first. No transpose needed for cache storage.
            let (nkv, hd, layout) = match &node.op {
                AiOp::GroupedQueryAttention { num_kv_heads, head_dim, .. } => {
                    (*num_kv_heads, *head_dim, crate::ir::KvLayout::SeqFirst)
                }
                _ => (0, 0, crate::ir::KvLayout::SeqFirst),
            };

            let k_tid = node.inputs[1];
            let v_tid = node.inputs[2];

            // Create or reuse KvSlotWrite for K.
            let k_out = *tid_remap.entry(k_tid).or_insert_with(|| {
                let out = next_tid;
                next_tid += 1;
                out
            });
            let k_node = AiNode::new(
                next_node_id,
                AiOp::KvSlotWrite { layer, is_key: true, n_kv_heads: nkv, head_dim: hd, layout },
                vec![k_tid],
                vec![k_out],
            );
            next_node_id += 1;

            // Create KvSlotWrite for V.
            let v_out = *tid_remap.entry(v_tid).or_insert_with(|| {
                let out = next_tid;
                next_tid += 1;
                out
            });
            let v_node = AiNode::new(
                next_node_id,
                AiOp::KvSlotWrite { layer, is_key: false, n_kv_heads: nkv, head_dim: hd, layout },
                vec![v_tid],
                vec![v_out],
            );
            next_node_id += 1;

            // Copy tensor_info from original K/V tensors to new outputs.
            if let Some(info) = graph.tensor_info.get(&k_tid).cloned() {
                graph.tensor_info.insert(k_out, info);
            }
            if let Some(info) = graph.tensor_info.get(&v_tid).cloned() {
                graph.tensor_info.insert(v_out, info);
            }

            // Rewire GQA inputs to use cached K/V.
            graph.nodes[gqa_idx].inputs[1] = k_out;
            graph.nodes[gqa_idx].inputs[2] = v_out;

            injections.push(Injection { gqa_idx, k_node, v_node });
        }

        // Insert KvSlotWrite nodes just before their GQA nodes.
        // Process in reverse so earlier insertions don't shift later indices.
        for inj in injections.into_iter().rev() {
            graph.nodes.insert(inj.gqa_idx, inj.v_node);
            graph.nodes.insert(inj.gqa_idx, inj.k_node);
        }

        let layer_count = gqa_indices.len();
        tracing::info!(
            layers = layer_count,
            "KvSlotInjection: injected KvSlotWrite for {layer_count} attention layers"
        );

        Ok(graph)
    }
}
