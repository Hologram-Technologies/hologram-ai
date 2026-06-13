//! Resolve Slice parameters from constant inputs.
//!
//! ONNX opset 10+ encodes Slice axes/starts/ends/steps as input tensors
//! rather than op attributes. The ONNX importer creates placeholder
//! `AiOp::Slice { axes: [], starts: [], ends: [], steps: [] }` nodes.
//!
//! This pass reads the constant param inputs and fills in the AiOp struct
//! fields, enabling `SliceToGather` and the lowering strategy to handle them.

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiOp, AiParam, TensorId};

pub struct ResolveSliceParams;

impl Pass for ResolveSliceParams {
    fn name(&self) -> &str {
        "ResolveSliceParams"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        for idx in 0..graph.nodes.len() {
            let inputs = graph.nodes[idx].inputs.clone();
            let existing = match &graph.nodes[idx].op {
                AiOp::Slice {
                    axes,
                    starts,
                    ends,
                    steps,
                } => Some((axes.clone(), starts.clone(), ends.clone(), steps.clone())),
                _ => None,
            };
            let Some((existing_axes, existing_starts, existing_ends, existing_steps)) = existing
            else {
                continue;
            };

            // ONNX Slice inputs: [data, starts, ends, axes?, steps?].
            // Read from known_i64_values first (DataPropagation may have
            // materialized dynamic bounds there), then from constant params.
            let resolved_starts = read_i64_input(&graph, inputs.get(1).copied())
                .or_else(|| (!existing_starts.is_empty()).then_some(existing_starts.clone()));
            let Some(starts) = resolved_starts else {
                continue;
            };
            let n = starts.len();
            let ends = read_i64_input(&graph, inputs.get(2).copied())
                .or_else(|| (!existing_ends.is_empty()).then_some(existing_ends.clone()))
                .unwrap_or_else(|| vec![i64::MAX; n]);
            let axes = read_i64_input(&graph, inputs.get(3).copied())
                .or_else(|| (!existing_axes.is_empty()).then_some(existing_axes.clone()))
                .unwrap_or_else(|| (0..n as i64).collect());
            let steps = read_i64_input(&graph, inputs.get(4).copied())
                .or_else(|| (!existing_steps.is_empty()).then_some(existing_steps.clone()))
                .unwrap_or_else(|| vec![1; n]);

            if let AiOp::Slice {
                axes: node_axes,
                starts: node_starts,
                ends: node_ends,
                steps: node_steps,
            } = &mut graph.nodes[idx].op
            {
                *node_axes = axes;
                *node_starts = starts;
                *node_ends = ends;
                *node_steps = steps;
            }
        }

        Ok(graph)
    }
}

/// Read an i64 vector from a known-value tensor or a constant parameter tensor.
fn read_i64_input(graph: &AiGraph, tid: Option<TensorId>) -> Option<Vec<i64>> {
    let tid = tid?;
    if let Some(values) = graph
        .tensor_info
        .get(&tid)
        .and_then(|info| info.known_i64_values.as_ref())
    {
        let resolved: Option<Vec<i64>> = values.iter().copied().collect();
        if resolved.is_some() {
            return resolved;
        }
    }
    let param = graph.params.get(&tid)?;
    let (data, info) = match param {
        AiParam::Inline { data, info } => (data.as_slice(), info),
        AiParam::Mmap { .. } => return None, // Can't read mmap at compile time easily
    };
    if data.is_empty() {
        return None;
    }
    match info.logical_dtype {
        crate::ir::DType::INT64 => {
            let values: Vec<i64> = data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().expect("chunk is 8 bytes")))
                .collect();
            Some(values)
        }
        crate::ir::DType::INT32 => {
            let values: Vec<i64> = data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")) as i64)
                .collect();
            Some(values)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::ResolveSliceParams;
    use crate::ir::{
        shape_from_concrete, AiGraph, AiNode, AiOp, ConstraintStore, DType, DimVarTable, TensorInfo,
    };
    use crate::Pass;
    use std::collections::HashMap;

    #[test]
    fn updates_dynamic_slice_end_from_known_i64_values() {
        let mut graph = AiGraph {
            name: "slice".into(),
            nodes: vec![AiNode::new(
                0,
                AiOp::Slice {
                    axes: vec![1],
                    starts: vec![0],
                    ends: vec![i64::MAX],
                    steps: vec![1],
                },
                vec![0, 1, 2, 3],
                vec![4],
            )],
            inputs: vec![0],
            outputs: vec![4],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: HashMap::new(),
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: DimVarTable::default(),
            shape_constraints: ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };
        graph.tensor_info.insert(
            0,
            TensorInfo::new(DType::INT64, shape_from_concrete(&[1, 512])),
        );
        graph
            .tensor_info
            .insert(1, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        let mut end_info = TensorInfo::new(DType::INT64, shape_from_concrete(&[1]));
        end_info.known_i64_values = Some(vec![Some(8)]);
        graph.tensor_info.insert(2, end_info);
        graph
            .tensor_info
            .insert(3, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        graph.tensor_info.insert(
            4,
            TensorInfo::new(DType::INT64, shape_from_concrete(&[1, 512])),
        );

        let out = ResolveSliceParams.run(graph).expect("pass succeeds");
        let AiOp::Slice { ends, .. } = &out.nodes[0].op else {
            panic!("expected slice");
        };
        assert_eq!(ends, &vec![8]);
    }
}
