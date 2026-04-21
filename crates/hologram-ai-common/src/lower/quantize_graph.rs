//! Post-lowering quantization pass (Plan 076).
//!
//! Walks a compiled `hologram::Graph` and converts eligible `Float(MatMul)`
//! nodes to `MatMulLut4` or `MatMulLut8` with pre-serialized quantized weights.
//!
//! This replaces the 9+ quantization hooks previously scattered across
//! `builder.rs`. All quantization decisions happen in one place.

use crate::lower::QuantStrategy;
use hologram::{ConstantData, ConstantId, Graph, GraphOp, NodeId};
use hologram::FloatOp;
use std::collections::HashMap;

/// Statistics from the quantization pass.
#[derive(Debug, Default)]
pub struct QuantizeStats {
    /// Number of MatMul/Gemm nodes successfully quantized.
    pub quantized: usize,
    /// Number of nodes skipped (too small, error too high, etc.).
    pub skipped: usize,
    /// Total weight bytes saved (f32 original - quantized).
    pub bytes_saved: u64,
}

/// Run the post-lowering quantization pass on a Graph.
///
/// Walks all nodes, finds `Float(MatMul)` and `Float(Gemm)` ops whose weight
/// input is a `Constant` node with f32 data, quantizes the weight, and replaces
/// the node with `MatMulLut4` or `MatMulLut8`.
///
/// `quant_cache` is shared across multiple graphs (prefill/decode/verify) to
/// avoid re-quantizing the same weight constant.
pub fn quantize_graph(
    graph: &mut Graph,
    strategy: QuantStrategy,
    total_params: u64,
    quant_cache: &mut HashMap<ConstantId, (ConstantId, QuantLevel)>,
) -> anyhow::Result<QuantizeStats> {
    if matches!(strategy, QuantStrategy::None | QuantStrategy::Auto) {
        return Ok(QuantizeStats::default());
    }

    // Determine effective quantization level based on model size.
    let q4_min_params: u64 = 750_000_000;
    let effective_level = if matches!(strategy, QuantStrategy::Q4_0) && total_params < q4_min_params
    {
        tracing::info!(
            total_params,
            threshold = q4_min_params,
            "model too small for Q4 — using Q8 uniform"
        );
        QuantLevel::Q8
    } else {
        match strategy {
            QuantStrategy::Q4_0 => QuantLevel::Q4,
            QuantStrategy::Q8_0 => QuantLevel::Q8,
            QuantStrategy::Q2_0 => QuantLevel::Q2,
            _ => return Ok(QuantizeStats::default()),
        }
    };

    // Adaptive error threshold for Q4.
    let size_scale = (total_params as f64 / 1e9).sqrt().max(0.3) as f32;
    let q4_error_threshold = 0.15 * size_scale;

    tracing::info!(
        total_nodes = graph.node_count(),
        ?effective_level,
        "quantize_graph: scanning"
    );

    // Collect all MatMul/Gemm nodes and their weight constant predecessors.
    // We collect first, then mutate — avoids borrow issues.
    let candidates: Vec<(NodeId, ConstantId, GraphOp)> = graph
        .node_ids()
        .into_iter()
        .filter_map(|nid| {
            let node = graph.get(nid)?;
            match &node.op {
                GraphOp::Float(FloatOp::MatMul { .. })
                | GraphOp::Float(FloatOp::Gemm { .. }) => {}
                _ => return None,
            }
            // Find the weight input: predecessor at slot 1 that is a Constant node.
            let preds = graph.predecessors(nid);
            if preds.len() < 2 {
                return None;
            }
            let weight_pred = preds[1];
            let weight_node = graph.get(weight_pred)?;
            let weight_cid = match &weight_node.op {
                GraphOp::Constant(cid) => *cid,
                _ => return None,
            };
            Some((nid, weight_cid, node.op.clone()))
        })
        .collect();

    let mut stats = QuantizeStats::default();

    for (node_id, weight_cid, original_op) in candidates {
        // Check cache first (shared across pipeline graphs).
        if let Some(&(quantized_cid, cached_level)) = quant_cache.get(&weight_cid) {
            // Already quantized — just replace the node.
            let new_op = make_quantized_op(cached_level, quantized_cid);
            replace_matmul_with_lut(graph, node_id, new_op);
            stats.quantized += 1;
            continue;
        }

        // Read the weight constant data.
        let weight_bytes = match graph.get_constant(weight_cid) {
            Some(ConstantData::Bytes(b)) => b.clone(),
            _ => {
                stats.skipped += 1;
                continue;
            }
        };

        // Determine weight shape from the MatMul/Gemm parameters.
        let (k, n) = match &original_op {
            GraphOp::Float(FloatOp::MatMul { k, n, .. }) => (*k as usize, *n as usize),
            GraphOp::Float(FloatOp::Gemm { k, n, .. }) => (*k as usize, *n as usize),
            _ => {
                stats.skipped += 1;
                continue;
            }
        };

        // Eligibility: 2D, both dims >= 256, f32.
        if k < 256 || n < 256 {
            stats.skipped += 1;
            continue;
        }
        let expected_bytes = k * n * 4;
        if weight_bytes.len() != expected_bytes {
            stats.skipped += 1;
            continue;
        }

        // Convert to f32.
        let f32_weights: Vec<f32> = weight_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();

        // Quantize.
        let (serialized, level, original_bytes) =
            match quantize_weight(&f32_weights, k, n, effective_level, q4_error_threshold) {
                Some(result) => result,
                None => {
                    stats.skipped += 1;
                    continue;
                }
            };

        // Register the quantized constant and replace the node.
        let quantized_cid = graph.add_constant(ConstantData::Bytes(serialized.clone()));
        quant_cache.insert(weight_cid, (quantized_cid, level));

        let new_op = make_quantized_op(level, quantized_cid);
        replace_matmul_with_lut(graph, node_id, new_op);

        stats.quantized += 1;
        stats.bytes_saved += original_bytes as u64 - serialized.len() as u64;

        tracing::debug!(
            ?node_id,
            k,
            n,
            ?level,
            saved_bytes = original_bytes - serialized.len(),
            "quantized weight"
        );
    }

    Ok(stats)
}

/// Quantization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantLevel {
    Q2,
    Q4,
    Q8,
}

fn make_quantized_op(level: QuantLevel, cid: ConstantId) -> GraphOp {
    match level {
        QuantLevel::Q2 => GraphOp::MatMulLut2(cid),
        QuantLevel::Q4 => GraphOp::MatMulLut4(cid),
        QuantLevel::Q8 => GraphOp::MatMulLut8(cid),
    }
}

/// Replace a Float(MatMul) node with a MatMulLut* node.
///
/// The MatMulLut node takes only the activation input (slot 0). The weight
/// input (slot 1) is embedded in the MatMulLut's ConstantId.
///
/// We replace the op in-place. The old weight Constant predecessor becomes
/// a dead node (cleaned up by hologram's dead-node elimination during compile).
/// The edge structure doesn't need modification — the tape builder only looks
/// at the GraphOp variant and the first predecessor for MatMulLut nodes.
fn replace_matmul_with_lut(graph: &mut Graph, node_id: NodeId, new_op: GraphOp) {
    graph.replace_op(node_id, new_op);
}

/// Quantize a weight matrix and return (serialized_bytes, level, original_byte_count).
fn quantize_weight(
    weights: &[f32],
    rows: usize,
    cols: usize,
    level: QuantLevel,
    q4_threshold: f32,
) -> Option<(Vec<u8>, QuantLevel, usize)> {
    use hologram::hologram_exec::lut_gemm::quantize::{
        dequantize_error_q4, quantize_4bit, quantize_8bit_uniform,
    };

    let original_bytes = rows * cols * 4;

    match level {
        QuantLevel::Q4 => {
            let qw = quantize_4bit(weights, rows as u32, cols as u32);
            let err = dequantize_error_q4(weights, &qw);
            if err > q4_threshold {
                tracing::debug!(
                    rows,
                    cols,
                    err,
                    threshold = q4_threshold,
                    "Q4 error too high, skipping"
                );
                return None;
            }
            let serialized = rkyv::to_bytes::<rkyv::rancor::Error>(&qw).ok()?;
            Some((serialized.to_vec(), QuantLevel::Q4, original_bytes))
        }
        QuantLevel::Q8 => {
            let qw = quantize_8bit_uniform(weights, rows as u32, cols as u32);
            let serialized = rkyv::to_bytes::<rkyv::rancor::Error>(&qw).ok()?;
            Some((serialized.to_vec(), QuantLevel::Q8, original_bytes))
        }
        QuantLevel::Q2 => {
            // Q2 uses the same quantize_4bit path but with 4 centroids.
            // For now, delegate to Q4 — Q2 support is future work.
            None
        }
    }
}
