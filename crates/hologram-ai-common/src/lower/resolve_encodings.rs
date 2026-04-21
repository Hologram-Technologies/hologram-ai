//! UOR encoding resolution pass (Plan 077).
//!
//! Replaces `quantize_graph.rs` (Plan 076). Walks a compiled `hologram::Graph`
//! and converts eligible `Float(MatMul)`, `Float(Gemm)`, and `Float(Conv2d)`
//! nodes to quantized LUT-GEMM variants with content-addressed constants.
//!
//! Key difference from Plan 076: constants are registered as
//! `ConstantData::ContentAddressed` with BLAKE3 digest, enabling unified
//! resolution across mmap/in-memory/streaming paths.

use crate::lower::QuantStrategy;
use hologram::hologram_graph::constant::ConstantData;
use hologram::{ConstantId, FloatOp, Graph, GraphOp, NodeId};
use std::collections::HashMap;

/// A content-address entry for one encoded weight tensor.
#[derive(Debug, Clone)]
pub struct EncodedWeightEntry {
    /// BLAKE3 digest of the encoded bytes.
    pub digest: [u8; 32],
    /// Byte size of the encoded data.
    pub byte_size: u64,
}

/// Statistics from the encoding resolution pass.
#[derive(Debug, Default)]
pub struct ResolveStats {
    /// Number of nodes successfully encoded.
    pub encoded: usize,
    /// Number of nodes skipped (too small, error too high, etc.).
    pub skipped: usize,
    /// Total weight bytes saved (f32 original - encoded).
    pub bytes_saved: u64,
    /// Content-address entries for all encoded weights (for building ContentAddressIndex).
    pub content_entries: Vec<EncodedWeightEntry>,
}

/// Quantization level (maps to UOR quantum level projections).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantLevel {
    /// 2-bit: Q3 -> sub-Q0 with 4 centroids.
    Q2,
    /// 4-bit: Q3 -> sub-Q0 with 16 centroids.
    Q4,
    /// 8-bit: Q3 -> Q0 with 256-level uniform scale.
    Q8,
}

/// Optional weight file for streaming compilation where constants
/// are `ConstantData::Deferred` (mmap'd from a temp file).
pub type WeightFile<'a> = Option<&'a [u8]>;

/// Run the UOR encoding resolution pass on a Graph.
///
/// For each eligible `Float(MatMul)`, `Float(Gemm)`, or `Float(Conv2d)` node:
/// 1. Reads weight bytes (from `Bytes`, `Deferred` via mmap, or cached)
/// 2. Encodes using k-means (Q4) or uniform (Q8) quantization
/// 3. Computes BLAKE3 digest of encoded bytes
/// 4. Registers as `ConstantData::ContentAddressed`
/// 5. Replaces node with `MatMulLut4`/`MatMulLut8`/`MatMulLut2`
///
/// `quant_cache` is shared across pipeline graphs (prefill/decode/verify).
pub fn resolve_encodings(
    graph: &mut Graph,
    strategy: QuantStrategy,
    total_params: u64,
    quant_cache: &mut HashMap<ConstantId, (ConstantId, QuantLevel)>,
    weight_file: WeightFile<'_>,
) -> anyhow::Result<ResolveStats> {
    if matches!(strategy, QuantStrategy::None | QuantStrategy::Auto) {
        return Ok(ResolveStats::default());
    }

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
            _ => return Ok(ResolveStats::default()),
        }
    };

    let size_scale = (total_params as f64 / 1e9).sqrt().max(0.3) as f32;
    let q4_error_threshold = 0.15 * size_scale;

    tracing::info!(
        total_nodes = graph.node_count(),
        ?effective_level,
        "resolve_encodings: scanning"
    );

    // Collect candidates: MatMul/Gemm/Conv2d nodes with weight constant predecessors.
    let candidates: Vec<(NodeId, ConstantId, GraphOp)> = graph
        .node_ids()
        .into_iter()
        .filter_map(|nid| {
            let node = graph.get(nid)?;
            match &node.op {
                GraphOp::Float(FloatOp::MatMul { .. })
                | GraphOp::Float(FloatOp::Gemm { .. }) => {}
                // TODO: Conv2d support will be added here.
                _ => return None,
            }
            let preds = graph.predecessors(nid);
            let weight_cid = if preds.len() >= 2 {
                graph.get(preds[1]).and_then(|wn| match &wn.op {
                    GraphOp::Constant(cid) => Some(*cid),
                    _ => None,
                })
            } else {
                None
            };
            weight_cid.map(|cid| (nid, cid, node.op.clone()))
        })
        .collect();

    tracing::info!(
        candidates = candidates.len(),
        "resolve_encodings: found candidates"
    );

    let mut stats = ResolveStats::default();

    for (node_id, weight_cid, original_op) in candidates {
        // Check cache first.
        if let Some(&(quantized_cid, cached_level)) = quant_cache.get(&weight_cid) {
            let new_op = make_quantized_op(cached_level, quantized_cid);
            graph.replace_op(node_id, new_op);
            stats.encoded += 1;
            continue;
        }

        // Read weight bytes — unified path for Bytes, Deferred, and ContentAddressed.
        let weight_data = match read_weight_bytes(graph, weight_cid, weight_file) {
            Some(data) => data,
            None => {
                stats.skipped += 1;
                continue;
            }
        };

        // Extract shape from op parameters.
        let (k, n) = match &original_op {
            GraphOp::Float(FloatOp::MatMul { k, n, .. }) => (*k as usize, *n as usize),
            GraphOp::Float(FloatOp::Gemm { k, n, .. }) => (*k as usize, *n as usize),
            _ => {
                stats.skipped += 1;
                continue;
            }
        };

        // Eligibility: both dims >= 256, f32 data.
        if k < 256 || n < 256 {
            stats.skipped += 1;
            continue;
        }
        let expected_bytes = k * n * 4;
        if weight_data.len() != expected_bytes {
            stats.skipped += 1;
            continue;
        }

        let f32_weights: Vec<f32> = weight_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();

        // Quantize and get serialized bytes.
        let (serialized, level, original_bytes) =
            match quantize_weight(&f32_weights, k, n, effective_level, q4_error_threshold) {
                Some(result) => result,
                None => {
                    stats.skipped += 1;
                    continue;
                }
            };

        // Compute BLAKE3 digest of encoded bytes.
        let digest: [u8; 32] = blake3::hash(&serialized).into();

        // Track content-address entry for the ContentAddressIndex section.
        stats.content_entries.push(EncodedWeightEntry {
            digest,
            byte_size: serialized.len() as u64,
        });

        // Register the encoded bytes as a constant. The graph uses Bytes
        // for now; the archive writer builds the ContentAddressIndex from
        // stats.content_entries after all graphs are compiled.
        let effective_cid = graph.add_constant(ConstantData::Bytes(serialized.clone()));

        // Replace the original f32 Deferred constant with empty Bytes so
        // trim_weight_blob excludes it from the trimmed blob.
        if graph.get_constant(weight_cid).map_or(false, |c| c.is_deferred()) {
            graph.replace_constant(weight_cid, ConstantData::Bytes(vec![]));
        }

        quant_cache.insert(weight_cid, (effective_cid, level));

        let new_op = make_quantized_op(level, effective_cid);
        graph.replace_op(node_id, new_op);

        stats.encoded += 1;
        stats.bytes_saved += original_bytes as u64 - serialized.len() as u64;

        tracing::debug!(
            ?node_id,
            k,
            n,
            ?level,
            digest = %format_args!("{:016x}", u64::from_be_bytes(digest[..8].try_into().expect("8 bytes"))),
            saved_bytes = original_bytes - serialized.len(),
            "encoded weight"
        );
    }

    // Second pass: inline ALL remaining Deferred constants from the weight file.
    // This eliminates the streaming weight blob entirely — every constant is
    // either a quantized Bytes (from the loop above) or an inlined Bytes (here).
    if let Some(wf) = weight_file {
        let store = graph.constant_store();
        let n = store.len();
        let mut inlined = 0usize;
        let mut inlined_bytes = 0u64;
        // Collect Deferred constants first to avoid borrow issues.
        let deferred: Vec<(ConstantId, u64, u64)> = (0..n)
            .filter_map(|i| {
                let cid = ConstantId::new(i as u32);
                match store.get(cid)? {
                    ConstantData::Deferred {
                        byte_size,
                        source_id,
                    } if *byte_size > 0 => Some((cid, *source_id, *byte_size)),
                    _ => None,
                }
            })
            .collect();

        for (cid, source_id, byte_size) in deferred {
            let start = source_id as usize;
            let end = start + byte_size as usize;
            if end <= wf.len() {
                let data = wf[start..end].to_vec();
                graph.replace_constant(cid, ConstantData::Bytes(data));
                inlined += 1;
                inlined_bytes += byte_size;
            } else {
                tracing::warn!(
                    cid = cid.raw(),
                    start,
                    end,
                    file_len = wf.len(),
                    "inline: deferred constant out of bounds"
                );
            }
        }

        if inlined > 0 {
            tracing::info!(
                inlined,
                inlined_mb = inlined_bytes / (1024 * 1024),
                "inlined remaining Deferred constants"
            );
        }
    }

    Ok(stats)
}

/// Read weight bytes from any ConstantData variant.
fn read_weight_bytes(graph: &Graph, cid: ConstantId, weight_file: WeightFile<'_>) -> Option<Vec<u8>> {
    match graph.get_constant(cid)? {
        ConstantData::Bytes(b) => Some(b.clone()),
        ConstantData::Deferred {
            byte_size,
            source_id,
        } => {
            let size = *byte_size as usize;
            let offset = *source_id as usize;
            let wf = weight_file?;
            if offset + size <= wf.len() {
                Some(wf[offset..offset + size].to_vec())
            } else {
                tracing::warn!(
                    offset,
                    size,
                    file_len = wf.len(),
                    "deferred constant out of bounds"
                );
                None
            }
        }
        ConstantData::ContentAddressed { .. } => {
            // Already encoded — skip re-encoding.
            None
        }
    }
}

fn make_quantized_op(level: QuantLevel, cid: ConstantId) -> GraphOp {
    match level {
        QuantLevel::Q2 => GraphOp::MatMulLut2(cid),
        QuantLevel::Q4 => GraphOp::MatMulLut4(cid),
        QuantLevel::Q8 => GraphOp::MatMulLut8(cid),
    }
}

/// Build a trimmed weight blob containing only the Deferred constant ranges
/// still referenced in the graph. Updates Deferred offsets to point into the
/// new contiguous blob. Returns `None` if no Deferred constants remain.
///
/// This eliminates the streaming archive bloat where quantized (now Bytes)
/// weights caused the full f32 blob to be copied unnecessarily.
pub fn trim_weight_blob(graph: &mut Graph, weight_file: &[u8]) -> Option<Vec<u8>> {
    let store = graph.constant_store();
    let n = store.len();

    // Collect all Deferred constant ranges: (cid, source_id, byte_size).
    let mut ranges: Vec<(ConstantId, u64, u64)> = Vec::new();
    for i in 0..n {
        let cid = ConstantId::new(i as u32);
        if let Some(ConstantData::Deferred {
            byte_size,
            source_id,
        }) = store.get(cid)
        {
            if *byte_size > 0 {
                ranges.push((cid, *source_id, *byte_size));
            }
        }
    }

    if ranges.is_empty() {
        return None;
    }

    // Sort by source offset for sequential reads.
    ranges.sort_by_key(|&(_, offset, _)| offset);

    // Build the trimmed blob: pack ranges contiguously.
    let total_trimmed: u64 = ranges.iter().map(|&(_, _, size)| size).sum();
    let mut blob = Vec::with_capacity(total_trimmed as usize);
    let mut new_offsets: Vec<(ConstantId, u64)> = Vec::with_capacity(ranges.len());

    for &(cid, source_offset, byte_size) in &ranges {
        let start = source_offset as usize;
        let end = start + byte_size as usize;
        if end > weight_file.len() {
            tracing::warn!(
                cid = cid.raw(),
                start,
                end,
                file_len = weight_file.len(),
                "trim_weight_blob: deferred constant out of bounds, skipping"
            );
            continue;
        }
        let new_offset = blob.len() as u64;
        blob.extend_from_slice(&weight_file[start..end]);
        new_offsets.push((cid, new_offset));
    }

    // Update Deferred constants with new offsets.
    for (cid, new_offset) in new_offsets {
        if let Some(ConstantData::Deferred { byte_size, .. }) = graph.get_constant(cid) {
            let byte_size = *byte_size;
            graph.replace_constant(
                cid,
                ConstantData::Deferred {
                    byte_size,
                    source_id: new_offset,
                },
            );
        }
    }

    let original_mb = weight_file.len() / (1024 * 1024);
    let trimmed_mb = blob.len() / (1024 * 1024);
    tracing::info!(
        deferred_count = ranges.len(),
        original_mb,
        trimmed_mb,
        saved_mb = original_mb - trimmed_mb,
        "trimmed weight blob"
    );

    Some(blob)
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
        QuantLevel::Q2 => None,
    }
}
