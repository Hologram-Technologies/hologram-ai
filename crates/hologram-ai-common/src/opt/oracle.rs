//! Shape oracle pass.
//!
//! Seeds tensor shapes from an external oracle (e.g., ONNX `value_info`).
//! Applied as the first pass after import to ensure that all shapes
//! available from the source format are injected into the graph before
//! any inference-based propagation runs.

use super::pipeline::Pass;
use crate::ir::{AiGraph, TensorId, TensorInfo};
use std::collections::HashMap;

/// Seeds empty tensor shapes from a pre-computed oracle.
///
/// The oracle is a map from `TensorId` to `TensorInfo` populated by the
/// importer (e.g., ONNX `value_info` covers every intermediate tensor).
/// Any tensor whose shape is currently empty receives the oracle's shape.
/// Tensors with non-empty shapes (params, graph inputs, already inferred)
/// are left unchanged — the oracle never overwrites.
///
/// This pass is op-agnostic: it works for any model regardless of which
/// ops are used. When a future op is added, the oracle automatically
/// provides its output shapes if the source format annotates them.
pub struct ShapeOraclePass {
    oracle: HashMap<TensorId, TensorInfo>,
}

impl ShapeOraclePass {
    pub fn new(oracle: HashMap<TensorId, TensorInfo>) -> Self {
        Self { oracle }
    }

    pub fn is_empty(&self) -> bool {
        self.oracle.is_empty()
    }

    pub fn len(&self) -> usize {
        self.oracle.len()
    }
}

impl Pass for ShapeOraclePass {
    fn name(&self) -> &str {
        "ShapeOracle"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let mut applied = 0usize;

        for (&tid, oracle_info) in &self.oracle {
            if let Some(existing) = graph.tensor_info.get_mut(&tid) {
                if existing.shape.is_empty() {
                    existing.shape = oracle_info.shape.clone();
                    // Apply dtype when the existing entry is the default F32
                    // and the oracle has more specific information.
                    if existing.logical_dtype == crate::ir::DType::F32
                        && oracle_info.logical_dtype != crate::ir::DType::F32
                    {
                        existing.logical_dtype = oracle_info.logical_dtype;
                        existing.storage_dtype = oracle_info.storage_dtype;
                    }
                    applied += 1;
                }
            }
        }

        tracing::debug!(
            applied,
            total = self.oracle.len(),
            "ShapeOracle: seeded shapes from oracle"
        );
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        shape::{shape_from_concrete, Shape},
        AiGraph, AiNode, AiOp, DType, DimVarTable, ConstraintStore,
    };

    fn make_graph(tids: &[(u32, Shape)]) -> AiGraph {
        let mut tensor_info = HashMap::new();
        for &(tid, ref shape) in tids {
            tensor_info.insert(tid, TensorInfo::new(DType::F32, shape.clone()));
        }
        AiGraph {
            name: "test".into(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: DimVarTable::default(),
            shape_constraints: ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        }
    }

    #[test]
    fn oracle_fills_empty_shapes() {
        let graph = make_graph(&[(0, Shape::new()), (1, shape_from_concrete(&[4, 8]))]);
        let mut oracle = HashMap::new();
        oracle.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 3])));
        let pass = ShapeOraclePass::new(oracle);
        let graph = pass.run(graph).unwrap();
        // Tensor 0: empty → filled by oracle.
        assert_eq!(
            graph.tensor_info[&0].shape.as_slice(),
            shape_from_concrete(&[2, 3]).as_slice()
        );
    }

    #[test]
    fn oracle_skips_nonempty_shapes() {
        let graph = make_graph(&[(0, shape_from_concrete(&[4, 8]))]);
        let mut oracle = HashMap::new();
        oracle.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 3])));
        let pass = ShapeOraclePass::new(oracle);
        let graph = pass.run(graph).unwrap();
        // Tensor 0: already had shape [4,8] — oracle must not overwrite.
        assert_eq!(
            graph.tensor_info[&0].shape.as_slice(),
            shape_from_concrete(&[4, 8]).as_slice()
        );
    }

    #[test]
    fn oracle_applies_dtype_when_default() {
        let graph = make_graph(&[(0, Shape::new())]);
        let mut oracle = HashMap::new();
        oracle.insert(0u32, TensorInfo::new(DType::INT64, shape_from_concrete(&[3])));
        let pass = ShapeOraclePass::new(oracle);
        let graph = pass.run(graph).unwrap();
        // Both shape and dtype should be updated.
        assert_eq!(graph.tensor_info[&0].logical_dtype, DType::INT64);
        assert_eq!(
            graph.tensor_info[&0].shape.as_slice(),
            shape_from_concrete(&[3]).as_slice()
        );
    }

    #[test]
    fn oracle_does_not_replace_existing_dtype() {
        let mut ti = TensorInfo::new(DType::INT32, Shape::new());
        let mut tensor_info = HashMap::new();
        tensor_info.insert(0u32, ti.clone());

        let mut graph = make_graph(&[]);
        // Override with INT32 entry.
        ti.shape = Shape::new();
        graph.tensor_info.insert(0, ti);

        let mut oracle = HashMap::new();
        oracle.insert(0u32, TensorInfo::new(DType::INT64, shape_from_concrete(&[3])));
        let pass = ShapeOraclePass::new(oracle);
        let graph = pass.run(graph).unwrap();
        // Shape is filled by oracle, but dtype should stay INT32 (not default F32).
        assert_eq!(graph.tensor_info[&0].logical_dtype, DType::INT32);
    }
}
