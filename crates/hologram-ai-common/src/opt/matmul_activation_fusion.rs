//! MatMul + Activation fusion.
//!
//! Detects `MatMul → Activation` chains and fuses them into single ops
//! (`MatMulSilu`, `MatMulGelu`, `MatMulRelu`). The lowering in `strategy.rs`
//! maps these to `GraphOp::FusedMatMulActivation`, and the tape builder maps
//! to `InlineMatMulActivation` which applies the activation in-register
//! after matmul writeback — eliminating the intermediate buffer.
//!
//! # Pattern
//!
//! ```text
//! matmul_out = MatMul(A, W)
//! out        = SiLU(matmul_out)   — or GeLU / ReLU
//! ```
//!
//! Fused into:
//!
//! ```text
//! out = MatMulSilu(A, W)          — matmul + activation in one op
//! ```
//!
//! This pattern appears in every transformer FFN (gate/up projections with
//! SiLU in LLaMA/Mistral, GeLU in GPT/BERT). Fusing saves ~2x memory
//! traffic by avoiding the intermediate activation buffer.

use super::graph_utils::{build_consumer_map, build_producer_map, has_single_consumer, remove_nodes};
use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp};
use std::collections::HashSet;

/// Fuse `MatMul + Activation` into `AiOp::MatMulSilu/Gelu/Relu`.
pub struct MatMulActivationFusion;

impl Pass for MatMulActivationFusion {
    fn name(&self) -> &str {
        "MatMulActivationFusion"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let tid_to_node = build_producer_map(&graph);
        let consumers = build_consumer_map(&graph);

        // Collect fusion candidates: (matmul_idx, act_idx, fused_op).
        let mut fusions: Vec<(usize, usize, AiOp)> = Vec::new();
        let mut claimed: HashSet<usize> = HashSet::new();

        for (act_idx, act_node) in graph.nodes.iter().enumerate() {
            let fused_op = match &act_node.op {
                AiOp::Silu => AiOp::MatMulSilu,
                AiOp::Gelu => AiOp::MatMulGelu,
                AiOp::Relu => AiOp::MatMulRelu,
                _ => continue,
            };

            if act_node.inputs.len() != 1 {
                continue;
            }

            let act_input_tid = act_node.inputs[0];

            let matmul_idx = match tid_to_node.get(&act_input_tid) {
                Some(&idx) => idx,
                None => continue,
            };
            if claimed.contains(&matmul_idx) || claimed.contains(&act_idx) {
                continue;
            }
            let matmul_node = &graph.nodes[matmul_idx];
            if !matches!(matmul_node.op, AiOp::MatMul) {
                continue;
            }

            // MatMul output must have exactly one consumer (this activation).
            let matmul_out_tid = matmul_node.outputs[0];
            if !has_single_consumer(matmul_out_tid, &consumers) {
                continue;
            }

            fusions.push((matmul_idx, act_idx, fused_op));
            claimed.insert(matmul_idx);
            claimed.insert(act_idx);
        }

        // Apply fusions.
        let mut to_remove: HashSet<usize> = HashSet::new();
        for (matmul_idx, act_idx, fused_op) in &fusions {
            let act_outputs = graph.nodes[*act_idx].outputs.clone();
            let matmul_inputs = graph.nodes[*matmul_idx].inputs.clone();
            let matmul_id = graph.nodes[*matmul_idx].id;

            graph.nodes[*matmul_idx] =
                AiNode::new(matmul_id, fused_op.clone(), matmul_inputs, act_outputs);
            to_remove.insert(*act_idx);
        }

        let fused_count = fusions.len();

        // Remove dead activation nodes.
        remove_nodes(&mut graph, &to_remove);

        tracing::info!("MatMulActivationFusion: fused {fused_count} MatMul+Activation pair(s)");
        Ok(graph)
    }
}
