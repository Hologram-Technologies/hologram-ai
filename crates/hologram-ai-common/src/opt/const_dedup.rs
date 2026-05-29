//! Content-addressed constant deduplication.
//!
//! Addresses every `AiParam::Inline` weight to its **uor-addr content
//! fingerprint** — the canonical BLAKE3 content address
//! ([`hologram_archive::WeightFingerprint`], the same address hologram's
//! `WeightStore` dedups by and the same hasher family as a model's κ-label) —
//! and merges weights that share an address by remapping TensorIds. A weight's
//! identity is thus its uor-address, not its bytes: structurally-equal weights
//! collapse to one, consistent with the canonical-forms model (the IR operates
//! over addresses, not values). This catches cross-layer duplication that
//! op+inputTID CSE cannot (e.g. 22 transformer layers independently
//! materializing the same RoPE table, or tied embed/unembed weights).
//!
//! Runs after ConstantEvaluation + ConstantFolding so that all materializable
//! constants have been produced and dead intermediates removed.

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiParam, DType, Shape, TensorId};
use hologram_archive::WeightFingerprint;
use std::collections::HashMap;

/// Deduplicate inline constants by their uor-addr content fingerprint.
pub struct ConstantDeduplication;

impl Pass for ConstantDeduplication {
    fn name(&self) -> &str {
        "ConstantDeduplication"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        // Canonical owner per (content address, dtype, shape). Equal address
        // ⇒ equal bytes (BLAKE3 is collision-resistant — the premise of
        // content addressing, and exactly what hologram's WeightStore relies
        // on), so no byte re-comparison is needed. The dtype + shape scope
        // the address so two weights with identical bytes but different
        // *interpretations* stay distinct: a `[32, 64]` and a `[64, 32]`
        // tensor full of zeros are byte-identical but semantically different
        // tensors (transposes of each other), and downstream ops read them
        // as their declared shape (the matmul's k = A.dim(1) must equal
        // B.dim(0)). Merging them would silently rewrite a MatMul's B
        // operand to a wrong-shape constant — caught loud by hologram's
        // rank-2 k-mismatch check, but the right fix is to never merge
        // them in the first place. Tensor identity = (content, dtype,
        // shape).
        let mut canonical: HashMap<(WeightFingerprint, DType, Shape), TensorId> = HashMap::new();
        let mut remap: HashMap<TensorId, TensorId> = HashMap::new();

        // Collect param TIDs sorted so we get deterministic canonical choices.
        let mut param_tids: Vec<TensorId> = graph.params.keys().copied().collect();
        param_tids.sort();

        for &tid in &param_tids {
            let (data, info) = match graph.params.get(&tid) {
                Some(AiParam::Inline { data, info }) => (data.as_slice(), info),
                _ => continue,
            };

            // Skip small constants (dedup overhead not worth it).
            if data.len() < 256 {
                continue;
            }

            let key = (
                WeightFingerprint::of(data),
                info.logical_dtype,
                info.shape.clone(),
            );

            if let Some(&canon_tid) = canonical.get(&key) {
                remap.insert(tid, canon_tid);
            } else {
                canonical.insert(key, tid);
            }
        }

        if remap.is_empty() {
            return Ok(graph);
        }

        let dedup_count = remap.len();
        let saved_bytes: usize = remap
            .keys()
            .filter_map(|tid| match graph.params.get(tid) {
                Some(AiParam::Inline { data, .. }) => Some(data.len()),
                _ => None,
            })
            .sum();

        // Remove duplicate params.
        for &tid in remap.keys() {
            graph.params.remove(&tid);
        }

        // Remap all node inputs/outputs.
        for node in &mut graph.nodes {
            for tid in &mut node.inputs {
                if let Some(&target) = remap.get(tid) {
                    *tid = target;
                }
            }
            for tid in &mut node.outputs {
                if let Some(&target) = remap.get(tid) {
                    *tid = target;
                }
            }
        }

        // Remap graph-level inputs/outputs.
        for tid in &mut graph.inputs {
            if let Some(&target) = remap.get(tid) {
                *tid = target;
            }
        }
        for tid in &mut graph.outputs {
            if let Some(&target) = remap.get(tid) {
                *tid = target;
            }
        }

        tracing::debug!(
            dedup_count,
            saved_mb = saved_bytes as f64 / 1_048_576.0,
            "const-dedup: deduplicated constants"
        );

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};
    use std::collections::HashMap;

    fn make_graph(
        nodes: Vec<AiNode>,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
        params: HashMap<TensorId, AiParam>,
        tensor_info: HashMap<TensorId, TensorInfo>,
    ) -> AiGraph {
        AiGraph {
            name: "test".into(),
            nodes,
            inputs,
            outputs,
            input_names: vec![],
            output_names: vec![],
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
    fn dedup_identical_constants() {
        // Two params (T10, T20) with identical content.
        // Node0: Relu(T10 → T11), Node1: Relu(T20 → T21)
        // After dedup: T20 remapped to T10, Node1 reads from T10.
        let data = vec![0u8; 1024]; // >= 256 bytes threshold
        let info = TensorInfo::new(DType::F32, shape_from_concrete(&[256]));

        let mut params = HashMap::new();
        params.insert(10u32, AiParam::inline(data.clone(), info.clone()));
        params.insert(20u32, AiParam::inline(data.clone(), info.clone()));

        let mut ti = HashMap::new();
        ti.insert(10u32, info.clone());
        ti.insert(11u32, info.clone());
        ti.insert(20u32, info.clone());
        ti.insert(21u32, info.clone());

        let g = make_graph(
            vec![
                AiNode::new(0, AiOp::Relu, vec![10], vec![11]),
                AiNode::new(1, AiOp::Relu, vec![20], vec![21]),
            ],
            vec![],
            vec![11, 21],
            params,
            ti,
        );

        let g2 = ConstantDeduplication.run(g).unwrap();
        // T20 should be remapped to T10.
        assert_eq!(g2.params.len(), 1);
        assert!(g2.params.contains_key(&10));
        // Node1's input should now be T10.
        assert_eq!(g2.nodes[1].inputs, vec![10]);
    }

    #[test]
    fn no_dedup_different_constants() {
        let data_a = vec![1u8; 1024];
        let data_b = vec![2u8; 1024];
        let info = TensorInfo::new(DType::F32, shape_from_concrete(&[256]));

        let mut params = HashMap::new();
        params.insert(10u32, AiParam::inline(data_a, info.clone()));
        params.insert(20u32, AiParam::inline(data_b, info.clone()));

        let mut ti = HashMap::new();
        ti.insert(10u32, info.clone());
        ti.insert(20u32, info.clone());

        let g = make_graph(vec![], vec![], vec![], params, ti);

        let g2 = ConstantDeduplication.run(g).unwrap();
        assert_eq!(g2.params.len(), 2);
    }

    #[test]
    fn no_dedup_same_bytes_different_shapes() {
        // Regression: a `[32, 64]` and a `[64, 32]` tensor full of zeros are
        // byte-identical but semantically different — transposes of each
        // other. Merging them would silently rewrite a MatMul's B operand
        // to a wrong-shape constant (caught loud by hologram's k-mismatch
        // check, but the right fix is to never merge them). Tensor identity
        // = (content, dtype, shape).
        let data = vec![0u8; 32 * 64 * 4]; // 8192 bytes, >= 256 threshold
        let info_a = TensorInfo::new(DType::F32, shape_from_concrete(&[32, 64]));
        let info_b = TensorInfo::new(DType::F32, shape_from_concrete(&[64, 32]));

        let mut params = HashMap::new();
        params.insert(10u32, AiParam::inline(data.clone(), info_a.clone()));
        params.insert(20u32, AiParam::inline(data, info_b.clone()));

        let mut ti = HashMap::new();
        ti.insert(10u32, info_a);
        ti.insert(20u32, info_b);

        let g = make_graph(vec![], vec![], vec![], params, ti);

        let g2 = ConstantDeduplication.run(g).unwrap();
        // Both must remain — shape disagreement blocks merging.
        assert_eq!(g2.params.len(), 2);
        assert!(g2.params.contains_key(&10));
        assert!(g2.params.contains_key(&20));
    }

    #[test]
    fn skip_small_constants() {
        // Constants < 256 bytes should not be deduplicated.
        let data = vec![0u8; 100];
        let info = TensorInfo::new(DType::F32, shape_from_concrete(&[25]));

        let mut params = HashMap::new();
        params.insert(10u32, AiParam::inline(data.clone(), info.clone()));
        params.insert(20u32, AiParam::inline(data, info.clone()));

        let g = make_graph(vec![], vec![], vec![], params, HashMap::new());

        let g2 = ConstantDeduplication.run(g).unwrap();
        // Both should remain (too small to dedup).
        assert_eq!(g2.params.len(), 2);
    }
}
