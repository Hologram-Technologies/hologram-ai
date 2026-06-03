//! Compile-time weight quantization pass: rewrites MatMul f32-weight constants
//! into i8 + per-channel scale + a `Dequantize` node. Gated on `QuantStrategy`.

use std::sync::Arc;

use anyhow::{bail, Result};

use hologram_ai_quant::encode_int8_per_channel;

use crate::ir::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorId, TensorInfo};
use crate::lower::QuantStrategy;

/// Read a tensor's concrete 2D dims `[k, n]`, or `None` if not rank-2 concrete.
fn concrete_2d(info: &TensorInfo) -> Option<(usize, usize)> {
    let k = info.shape.first()?.as_concrete()?;
    let n = info.shape.get(1)?.as_concrete()?;
    if info.shape.get(2).is_some() {
        return None; // higher rank — skip (batched matmul weight)
    }
    Some((k as usize, n as usize))
}

/// Rewrite each MatMul whose B (weight) operand is an inline f32 rank-2 constant
/// into `Dequantize(i8_weight, per_column_scale) → MatMul`. No-op unless
/// `strategy == Int8`. `Int4` is rejected until the int4 spec lands.
///
/// The emitted `Dequantize` node uses `axis: 1` (per output column for a
/// `[k, n]` weight) and carries only two operands — weight and scale; no
/// explicit zero-point operand is included (the lowering pass treats a missing
/// zero-point as all-zeros, matching the reference test in
/// `hologram-ai/tests/quantized_weight_memory.rs`).
pub fn quantize_weights(graph: &mut AiGraph, strategy: QuantStrategy) -> Result<()> {
    match strategy {
        QuantStrategy::Int8 => {}
        QuantStrategy::Int4 => bail!("int4 quantization is not yet implemented"),
        _ => return Ok(()),
    }

    let mut next_tid: TensorId = graph
        .tensor_info
        .keys()
        .chain(graph.params.keys())
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    let mut next_nid = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;

    let mut new_nodes: Vec<AiNode> = Vec::new();
    let mut dead_params: Vec<TensorId> = Vec::new();
    let mut changed = false;

    for idx in 0..graph.nodes.len() {
        if !matches!(graph.nodes[idx].op, AiOp::MatMul) {
            continue;
        }
        let b_tid = match graph.nodes[idx].inputs.get(1).copied() {
            Some(t) => t,
            None => continue,
        };
        let (data, info) = match graph.params.get(&b_tid) {
            Some(AiParam::Inline { data, info }) => (data.clone(), info.clone()),
            _ => continue, // mmap or non-constant B: skip in this baseline
        };
        if info.logical_dtype != DType::F32 {
            continue;
        }
        let (k, n) = match concrete_2d(&info) {
            Some(kn) => kn,
            None => continue,
        };
        if data.len() != k * n * 4 {
            continue;
        }

        let wf: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (q, scales) = encode_int8_per_channel(&wf, k, n);

        let wq_tid = next_tid;
        let scale_tid = next_tid + 1;
        let deq_tid = next_tid + 2;
        next_tid += 3;

        // i8 weight constant [k, n].
        let q_bytes: Vec<u8> = q.iter().map(|&v| v as u8).collect();
        let wq_info = TensorInfo::new(DType::INT8, info.shape.clone());
        graph.params.insert(
            wq_tid,
            AiParam::Inline {
                data: Arc::new(q_bytes),
                info: wq_info.clone(),
            },
        );
        graph.tensor_info.insert(wq_tid, wq_info);

        // Per-column scale constant [n] f32.
        let scale_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
        let scale_info = TensorInfo::new(DType::F32, shape_from_concrete(&[n as u64]));
        graph.params.insert(
            scale_tid,
            AiParam::Inline {
                data: Arc::new(scale_bytes),
                info: scale_info.clone(),
            },
        );
        graph.tensor_info.insert(scale_tid, scale_info);

        // Dequant output tensor [k, n] f32.
        graph
            .tensor_info
            .insert(deq_tid, TensorInfo::new(DType::F32, info.shape.clone()));

        // Dequantize node (zero-point omitted — lowering fills zeros), axis=1
        // (per output column for a [k, n] weight). This matches the per-channel
        // reference graph in `quantized_weight_memory.rs::per_channel_graph()`.
        new_nodes.push(AiNode::new(
            next_nid,
            AiOp::Dequantize { axis: 1 },
            vec![wq_tid, scale_tid],
            vec![deq_tid],
        ));
        next_nid += 1;

        // Rewire MatMul B → dequant output; retire the f32 weight constant.
        graph.nodes[idx].inputs[1] = deq_tid;
        dead_params.push(b_tid);
        changed = true;
    }

    graph.nodes.extend(new_nodes);
    for t in dead_params {
        graph.params.remove(&t);
        graph.tensor_info.remove(&t);
    }
    if changed {
        graph.invalidate_topo_cache();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{shape_from_concrete, DimVarTable, TensorInfo};
    use std::collections::HashMap;

    /// A minimal MatMul graph: `X[1,4] · W[4,2] → Y[1,2]` where W is an inline
    /// f32 constant.
    fn f32_weight_matmul_graph() -> AiGraph {
        let mut ti: HashMap<TensorId, TensorInfo> = HashMap::new();
        let mut params: HashMap<TensorId, AiParam> = HashMap::new();

        // X = input [1, 4]
        ti.insert(0, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 4])));

        // W = f32 weight [4, 2]
        let w: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_info = TensorInfo::new(DType::F32, shape_from_concrete(&[4, 2]));
        params.insert(1, AiParam::inline(w_bytes, w_info.clone()));
        ti.insert(1, w_info);

        // Y = output [1, 2]
        ti.insert(2, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 2])));

        AiGraph {
            name: "qtest".into(),
            nodes: vec![AiNode::new(0, AiOp::MatMul, vec![0, 1], vec![2])],
            inputs: vec![0],
            outputs: vec![2],
            input_names: Vec::new(),
            output_names: Vec::new(),
            params,
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: Vec::new(),
            dim_vars: DimVarTable::default(),
            shape_constraints: crate::ir::ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        }
    }

    #[test]
    fn none_is_noop() {
        let mut g = f32_weight_matmul_graph();
        let before = g.nodes.len();
        quantize_weights(&mut g, QuantStrategy::None).unwrap();
        assert_eq!(g.nodes.len(), before);
        assert!(matches!(g.params.get(&1), Some(AiParam::Inline { .. })));
    }

    #[test]
    fn int8_rewrites_matmul_weight() {
        let mut g = f32_weight_matmul_graph();
        quantize_weights(&mut g, QuantStrategy::Int8).unwrap();

        // A Dequantize node was added.
        let deq = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, AiOp::Dequantize { .. }))
            .expect("dequant node inserted");

        // Its weight operand is now an i8 constant of the original shape [4,2].
        let wq_tid = deq.inputs[0];
        match g.params.get(&wq_tid) {
            Some(AiParam::Inline { info, data }) => {
                assert_eq!(info.logical_dtype, DType::INT8);
                assert_eq!(data.len(), 4 * 2); // 1 byte/elem
            }
            _ => panic!("i8 weight constant missing"),
        }

        // Scale operand is an f32 vector of length n=2.
        let scale_tid = deq.inputs[1];
        match g.params.get(&scale_tid) {
            Some(AiParam::Inline { info, data }) => {
                assert_eq!(info.logical_dtype, DType::F32);
                assert_eq!(data.len(), 2 * 4);
            }
            _ => panic!("scale constant missing"),
        }

        // The MatMul's B now points at the dequant output; the old f32 const is gone.
        let mm = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, AiOp::MatMul))
            .unwrap();
        assert_eq!(mm.inputs[1], deq.outputs[0]);
        assert!(!g.params.contains_key(&1), "old f32 weight retired");
    }

    #[test]
    fn int4_errors() {
        let mut g = f32_weight_matmul_graph();
        assert!(quantize_weights(&mut g, QuantStrategy::Int4).is_err());
    }
}
