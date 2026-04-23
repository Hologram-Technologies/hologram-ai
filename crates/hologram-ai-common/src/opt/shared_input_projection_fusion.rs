//! Shared-input projection fusion.
//!
//! Detects multiple MatMul nodes sharing the same input tensor and fuses them
//! into a single MatMul with concatenated weights + Slice nodes to split the output.
//!
//! # QKV Projection Fusion (3-way)
//!
//! ```text
//! hidden → MatMul(hidden, W_q) → q_out   (2048)
//!        → MatMul(hidden, W_k) → k_out   (256)
//!        → MatMul(hidden, W_v) → v_out   (256)
//! ```
//!
//! Fused into:
//!
//! ```text
//! hidden → MatMul(hidden, W_qkv) → Slice(0..2048) → q_out
//!                                 → Slice(2048..2304) → k_out
//!                                 → Slice(2304..2560) → v_out
//! ```
//!
//! # Gate+Up FFN Fusion (2-way)
//!
//! ```text
//! hidden → MatMul(hidden, W_gate) → gate_out  (5632)
//!        → MatMul(hidden, W_up)   → up_out    (5632)
//! ```
//!
//! Fused into:
//!
//! ```text
//! hidden → MatMul(hidden, W_gate_up) → Slice(0..5632) → gate_out
//!                                    → Slice(5632..11264) → up_out
//! ```
//!
//! Saves 66 BLAS calls per decode step (44 from QKV + 22 from gate+up).

use super::graph_utils::next_node_id;
use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, AiParam, DType, Dim, SemanticHint, TensorId, TensorInfo};
use std::collections::{HashMap, HashSet};

/// Fuse shared-input MatMul projections into single MatMul + Slices.
pub struct SharedInputProjectionFusion;

impl Pass for SharedInputProjectionFusion {
    fn name(&self) -> &str {
        "SharedInputProjectionFusion"
    }

    fn run(&self, graph: AiGraph) -> anyhow::Result<AiGraph> {
        // Disabled: profiling shows fused matmuls are slower on Apple Silicon AMX
        // (single large sgemm is slower than multiple small ones for M=1 decode).
        // Also gate+up fusion hits rkyv overflow for large Q4 weights.
        // Infrastructure preserved for non-AMX platforms and future use.
        //
        // To re-enable: remove the early return and uncomment the fusion paths below.
        return Ok(graph);

        #[allow(
            unreachable_code,
            clippy::overly_complex_bool_expr,
            clippy::needless_borrow,
            clippy::iter_over_hash_type,
            clippy::for_kv_map,
            unused_variables,
            unused_mut
        )]
        {
            let mut graph = graph;
            // Collect MatMul nodes grouped by their shared input tensor.
            // Key: input[0] tid, Value: vec of (node_idx, weight_tid, K, N).
            let mut shared_input_groups: HashMap<TensorId, Vec<MatMulInfo>> = HashMap::new();

            for (idx, node) in graph.nodes.iter().enumerate() {
                if !matches!(node.op, AiOp::MatMul) || node.inputs.len() < 2 {
                    continue;
                }
                let input_tid = node.inputs[0];
                let weight_tid = node.inputs[1];

                // Weight must be a parameter (not an intermediate activation).
                if !graph.params.contains_key(&weight_tid) {
                    continue;
                }

                // Get weight shape — must be 2D with concrete dims.
                let (k, n) = match get_2d_shape(&graph, weight_tid) {
                    Some(dims) => dims,
                    None => continue,
                };

                // Skip tiny weights.
                if k < 256 || n < 256 {
                    continue;
                }

                let output_tid = match node.outputs.first() {
                    Some(&tid) => tid,
                    None => continue,
                };

                shared_input_groups
                    .entry(input_tid)
                    .or_default()
                    .push(MatMulInfo {
                        node_idx: idx,
                        weight_tid,
                        output_tid,
                        k,
                        n,
                    });
            }

            let mut to_remove: HashSet<usize> = HashSet::new();
            let mut new_nodes: Vec<(usize, Vec<AiNode>)> = Vec::new(); // (insert_before_idx, nodes)
            let mut fused_qkv = 0u32;
            let mut fused_gate_up = 0u32;

            // Next available node ID and tensor ID.
            let mut next_node_id = next_node_id(&graph);
            let mut next_tid = graph
                .nodes
                .iter()
                .flat_map(|n| n.outputs.iter().chain(n.inputs.iter()))
                .copied()
                .max()
                .unwrap_or(0)
                + 1;
            // Also consider param tensor IDs.
            if let Some(max_param_tid) = graph.params.keys().max() {
                next_tid = next_tid.max(*max_param_tid + 1);
            }

            for (_input_tid, group) in &shared_input_groups {
                // All members must have the same K dimension.
                let k = group[0].k;
                if !group.iter().all(|m| m.k == k) {
                    continue;
                }

                // QKV fusion: disabled for now — profiling shows the fused 2048×2560
                // sgemm is slower than 3 separate calls on Apple Silicon AMX.
                // The per-call overhead is negligible for M=1 vecmat on this hardware.
                // Re-enable for non-AMX platforms or when M>1 (prefill).
                if false && group.len() == 3 {
                    if let Some(qkv) = try_classify_qkv(group) {
                        if to_remove.contains(&qkv.q.node_idx)
                            || to_remove.contains(&qkv.k.node_idx)
                            || to_remove.contains(&qkv.v.node_idx)
                        {
                            continue;
                        }

                        match fuse_matmuls(
                            &mut graph,
                            &[&qkv.q, &qkv.k, &qkv.v],
                            &mut next_node_id,
                            &mut next_tid,
                        ) {
                            Ok(result) => {
                                to_remove.insert(qkv.q.node_idx);
                                to_remove.insert(qkv.k.node_idx);
                                to_remove.insert(qkv.v.node_idx);
                                let insert_idx =
                                    qkv.q.node_idx.min(qkv.k.node_idx).min(qkv.v.node_idx);
                                new_nodes.push((insert_idx, result));
                                fused_qkv += 1;
                                tracing::debug!(
                                    k,
                                    n_q = qkv.q.n,
                                    n_k = qkv.k.n,
                                    n_v = qkv.v.n,
                                    "SharedInputProjectionFusion: fused QKV projections"
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "SharedInputProjectionFusion: QKV fusion failed: {e}"
                                );
                            }
                        }
                        continue;
                    }
                }

                // Try gate+up fusion: exactly 2 members with equal N.
                // TODO: re-enable after fixing rkyv serialization for large Q4 weights
                if false && group.len() >= 2 {
                    // Find pairs with equal N.
                    let mut n_groups: HashMap<usize, Vec<&MatMulInfo>> = HashMap::new();
                    for m in group {
                        n_groups.entry(m.n).or_default().push(m);
                    }
                    for (_n_val, members) in &n_groups {
                        if members.len() == 2 {
                            let a = members[0];
                            let b = members[1];
                            if to_remove.contains(&a.node_idx) || to_remove.contains(&b.node_idx) {
                                continue;
                            }

                            match fuse_matmuls(
                                &mut graph,
                                &[a, b],
                                &mut next_node_id,
                                &mut next_tid,
                            ) {
                                Ok(result) => {
                                    to_remove.insert(a.node_idx);
                                    to_remove.insert(b.node_idx);
                                    let insert_idx = a.node_idx.min(b.node_idx);
                                    new_nodes.push((insert_idx, result));
                                    fused_gate_up += 1;
                                    tracing::debug!(
                                        k,
                                        n = a.n,
                                        "SharedInputProjectionFusion: fused gate+up projections"
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "SharedInputProjectionFusion: gate+up fusion failed: {e}"
                                    );
                                }
                            }
                        }
                    }
                }
            }

            if fused_qkv > 0 || fused_gate_up > 0 {
                tracing::info!(
                    fused_qkv,
                    fused_gate_up,
                    "SharedInputProjectionFusion: fused projection groups"
                );
            }

            // Apply: remove old nodes, insert new ones.
            // Sort new_nodes by insert position (ascending) for stable insertion.
            new_nodes.sort_by_key(|(idx, _)| *idx);

            let mut result_nodes = Vec::with_capacity(graph.nodes.len());
            let mut insert_iter = new_nodes.into_iter().peekable();

            for (i, node) in graph.nodes.into_iter().enumerate() {
                // Insert any new nodes that should appear before this index.
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
            // Remaining insertions after all original nodes.
            for (_, nodes) in insert_iter {
                result_nodes.extend(nodes);
            }

            graph.nodes = result_nodes;
            graph.invalidate_topo_cache();
            Ok(graph)
        } // end #[allow(unreachable_code)]
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct MatMulInfo {
    node_idx: usize,
    weight_tid: TensorId,
    output_tid: TensorId,
    k: usize,
    n: usize,
}

struct QkvTriple<'a> {
    q: &'a MatMulInfo,
    k: &'a MatMulInfo,
    v: &'a MatMulInfo,
}

/// Classify 3 MatMuls as Q/K/V: one with the largest N (Q), two with equal
/// smaller N (K, V). Returns None if pattern doesn't match.
fn try_classify_qkv(group: &[MatMulInfo]) -> Option<QkvTriple<'_>> {
    debug_assert_eq!(group.len(), 3);
    let mut sorted: Vec<&MatMulInfo> = group.iter().collect();
    sorted.sort_by_key(|m| std::cmp::Reverse(m.n));

    let q = sorted[0];
    let k = sorted[1];
    let v = sorted[2];

    // K and V must have equal dimensions, and Q must be larger.
    if k.n != v.n || q.n <= k.n {
        return None;
    }

    Some(QkvTriple { q, k, v })
}

/// Get concrete 2D shape (rows, cols) of a tensor.
fn get_2d_shape(graph: &AiGraph, tid: TensorId) -> Option<(usize, usize)> {
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

/// Read parameter bytes as owned Vec<u8>.
fn param_bytes(param: &AiParam) -> anyhow::Result<Vec<u8>> {
    use anyhow::Context;
    match param {
        AiParam::Inline { data, .. } => Ok(data.as_ref().clone()),
        AiParam::Mmap {
            path, offset, len, ..
        } => {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = std::fs::File::open(path)
                .with_context(|| format!("opening mmap param at {path:?}"))?;
            f.seek(SeekFrom::Start(*offset))?;
            let mut buf = vec![0u8; *len as usize];
            f.read_exact(&mut buf)?;
            Ok(buf)
        }
    }
}

/// Concatenate weight matrices along the N (column) axis.
/// All weights must have shape [K, N_i] in row-major f32 layout.
/// Returns concatenated bytes with shape [K, sum(N_i)].
fn concat_weights_along_n(
    graph: &AiGraph,
    matmuls: &[&MatMulInfo],
) -> anyhow::Result<(Vec<u8>, usize)> {
    let k = matmuls[0].k;
    let n_total: usize = matmuls.iter().map(|m| m.n).sum();

    // Read all weight data.
    let weight_data: Vec<Vec<u8>> = matmuls
        .iter()
        .map(|m| {
            let param = graph.params.get(&m.weight_tid).ok_or_else(|| {
                anyhow::anyhow!("weight param not found for tid {}", m.weight_tid)
            })?;
            param_bytes(param)
        })
        .collect::<anyhow::Result<_>>()?;

    // Concatenate row-by-row: for each row i, append N_0 + N_1 + ... elements.
    let mut result = vec![0u8; k * n_total * 4];
    for row in 0..k {
        let mut col_offset = 0usize;
        for (m_idx, m) in matmuls.iter().enumerate() {
            let src_start = row * m.n * 4;
            let src_end = src_start + m.n * 4;
            let dst_start = (row * n_total + col_offset) * 4;
            result[dst_start..dst_start + m.n * 4]
                .copy_from_slice(&weight_data[m_idx][src_start..src_end]);
            col_offset += m.n;
        }
    }

    Ok((result, n_total))
}

/// Fuse a group of MatMul nodes into 1 MatMul + N Slice nodes.
/// Returns the new nodes to insert. Registers new params and tensor_info in the graph.
fn fuse_matmuls(
    graph: &mut AiGraph,
    matmuls: &[&MatMulInfo],
    next_node_id: &mut u32,
    next_tid: &mut TensorId,
) -> anyhow::Result<Vec<AiNode>> {
    let k = matmuls[0].k;
    let input_tid = graph.nodes[matmuls[0].node_idx].inputs[0];

    // Concatenate weights.
    let (concat_bytes, n_total) = concat_weights_along_n(graph, matmuls)?;

    // Allocate TIDs for the concatenated weight and fused matmul output.
    let weight_tid = *next_tid;
    *next_tid += 1;
    let fused_output_tid = *next_tid;
    *next_tid += 1;

    // Register concatenated weight as inline param.
    let weight_shape = crate::ir::shape_from_concrete(&[k as u64, n_total as u64]);
    let weight_info = TensorInfo {
        shape: weight_shape.clone(),
        logical_dtype: DType::F32,
        storage_dtype: DType::F32,
        quant: hologram_ai_quant::QuantDescriptor::none(),
        known_i64_values: None,
        semantic: SemanticHint::Unknown,
    };
    graph.params.insert(
        weight_tid,
        AiParam::Inline {
            data: std::sync::Arc::new(concat_bytes),
            info: weight_info.clone(),
        },
    );
    graph.tensor_info.insert(weight_tid, weight_info);

    // Register fused output tensor info.
    // Copy shape from original input, replace last dim with n_total.
    let fused_output_info = if let Some(input_info) = graph.tensor_info.get(&input_tid) {
        let mut shape = input_info.shape.clone();
        if let Some(last) = shape.last_mut() {
            *last = Dim::Concrete(n_total as u64);
        }
        TensorInfo {
            shape,
            logical_dtype: DType::F32,
            storage_dtype: DType::F32,
            quant: hologram_ai_quant::QuantDescriptor::none(),
            known_i64_values: None,
            semantic: SemanticHint::Unknown,
        }
    } else {
        TensorInfo {
            shape: crate::ir::shape_from_concrete(&[1, n_total as u64]),
            logical_dtype: DType::F32,
            storage_dtype: DType::F32,
            quant: hologram_ai_quant::QuantDescriptor::none(),
            known_i64_values: None,
            semantic: SemanticHint::Unknown,
        }
    };
    graph
        .tensor_info
        .insert(fused_output_tid, fused_output_info);

    // Create fused MatMul node.
    let matmul_node_id = *next_node_id;
    *next_node_id += 1;
    let matmul_node = AiNode::new(
        matmul_node_id,
        AiOp::MatMul,
        vec![input_tid, weight_tid],
        vec![fused_output_tid],
    );

    let mut nodes = vec![matmul_node];

    // Create Slice nodes for each original MatMul's output.
    let mut col_offset = 0i64;
    for m in matmuls {
        let slice_node_id = *next_node_id;
        *next_node_id += 1;

        let start = col_offset;
        let end = col_offset + m.n as i64;

        let slice_node = AiNode::new(
            slice_node_id,
            AiOp::Slice {
                axes: vec![-1],
                starts: vec![start],
                ends: vec![end],
                steps: vec![1],
            },
            vec![fused_output_tid],
            vec![m.output_tid], // Reuse original output TID — no rewiring needed.
        );

        // Preserve original tensor info for the slice output.
        // (Already exists from the original MatMul output.)

        nodes.push(slice_node);
        col_offset = end;
    }

    Ok(nodes)
}
