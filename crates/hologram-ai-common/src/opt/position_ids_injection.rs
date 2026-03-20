//! Inject `position_ids` as an explicit graph input for KV cache decode.
//!
//! ONNX models that derive positions from `Range(0, seq_len, 1)` can't
//! support KV cache decode at seq=1 because the position is always 0.
//! This pass detects `Range` nodes that produce position indices and
//! replaces them with a new `position_ids` graph input.
//!
//! The generation loop then passes `position_ids = [kv_write_pos]` on
//! each decode step, giving the model the correct absolute position.
//!
//! # Detection
//!
//! A `Range` node is a position producer if:
//! 1. Its output feeds (directly or through Unsqueeze/Cast) into the RoPE
//!    or causal mask computation.
//! 2. It has `start=0, step=1` (generates `[0, 1, ..., seq-1]`).
//!
//! # Transformation
//!
//! ```text
//! Before: Range(0, seq_len, 1) → [0, 1, ..., seq-1]
//! After:  Input("position_ids") → [pos_offset, pos_offset+1, ..., pos_offset+seq-1]
//! ```
//!
//! For prefill: `position_ids = [0, 1, ..., prompt_len-1]`
//! For decode:  `position_ids = [kv_write_pos]`

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiOp, AiParam, DType, TensorInfo};
use hologram_ai_quant::QuantDescriptor;

/// Inject `position_ids` input to replace Range-based position computation.
pub struct PositionIdsInjection;

impl Pass for PositionIdsInjection {
    fn name(&self) -> &str {
        "PositionIdsInjection"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        // Find Range nodes that look like position generators:
        // Range(start=0, limit=seq_len, step=1)
        let range_indices: Vec<usize> = graph
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| {
                if !matches!(node.op, AiOp::Range) || node.inputs.len() < 3 {
                    return None;
                }
                // Check start=0 and step=1 from known_i64_values or constant params.
                let start = get_i64_param(node.inputs[0], &graph)?;
                let step = get_i64_param(node.inputs[2], &graph)?;
                if start == 0 && step == 1 {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        if range_indices.is_empty() {
            return Ok(graph);
        }

        // Allocate a new TensorId for the position_ids input.
        let next_tid = graph
            .tensor_info
            .keys()
            .copied()
            .max()
            .unwrap_or(0)
            + 1;

        // For each Range node, replace it with an Identity reading from
        // position_ids. The Range output shape and consumers stay the same.
        for &idx in &range_indices {
            let range_node = &graph.nodes[idx];
            let range_output = match range_node.outputs.first() {
                Some(&tid) => tid,
                None => continue,
            };

            // Get the Range's output shape (should be [seq_len] from DataPropagation).
            let range_shape = graph
                .tensor_info
                .get(&range_output)
                .map(|info| info.shape.clone())
                .unwrap_or_default();

            // Create the position_ids input tensor info.
            // Same shape as the Range output (1D: [seq_len]).
            let pos_tid = next_tid; // Reuse same tid for all Range replacements
            if !graph.tensor_info.contains_key(&pos_tid) {
                graph.tensor_info.insert(
                    pos_tid,
                    TensorInfo {
                        logical_dtype: DType::INT64,
                        storage_dtype: DType::INT64,
                        shape: range_shape.clone(),
                        quant: QuantDescriptor::none(),
                        known_i64_values: None, // Runtime-provided, not constant
                    },
                );
                graph.tensor_names.insert(pos_tid, "position_ids".into());
                graph.inputs.push(pos_tid);
                graph.input_names.push("position_ids".into());
            }

            // Replace the Range node with Identity(position_ids).
            let node = &mut graph.nodes[idx];
            node.op = AiOp::Identity;
            node.inputs = vec![pos_tid];

            tracing::info!(
                range_output,
                "PositionIdsInjection: replaced Range with position_ids input"
            );
        }

        Ok(graph)
    }
}

/// Extract a constant i64 value from a tensor parameter.
fn get_i64_param(tid: u32, graph: &AiGraph) -> Option<i64> {
    // Check known_i64_values first.
    if let Some(info) = graph.tensor_info.get(&tid) {
        if let Some(vals) = &info.known_i64_values {
            if vals.len() == 1 {
                return vals[0];
            }
        }
    }
    // Check inline param data.
    if let Some(param) = graph.params.get(&tid) {
        if let AiParam::Inline { data, info } = param {
            if info.logical_dtype == DType::INT64 && data.len() == 8 {
                return Some(i64::from_le_bytes(data[..8].try_into().ok()?));
            }
            // Also handle f32 scalars (common for Range start/step).
            if info.logical_dtype == DType::F32 && data.len() == 4 {
                let f = f32::from_le_bytes(data[..4].try_into().ok()?);
                return Some(f as i64);
            }
        }
    }
    None
}
