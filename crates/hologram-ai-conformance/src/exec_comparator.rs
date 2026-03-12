//! Node-by-node execution comparison between ORT and hologram.
//!
//! Compares intermediate tensors from ORT execution against hologram executor
//! outputs, using the `DebugMap` (ONNX name → builder index) to correlate
//! tensors across the two systems.
//!
//! Gated behind the `conformance` feature.

use crate::tolerance::{compare_outputs, ComparisonResult, Tolerance};
use std::collections::HashMap;

/// Result of comparing a single node's output between ORT and hologram.
#[derive(Debug)]
pub struct NodeComparison {
    pub name: String,
    pub ort_shape: Vec<usize>,
    pub holo_shape: Option<Vec<usize>>,
    pub result: NodeComparisonResult,
}

/// Outcome for a single node comparison.
#[derive(Debug)]
pub enum NodeComparisonResult {
    /// Shapes and values match within tolerance.
    Pass(ComparisonResult),
    /// Shape mismatch (values not compared).
    ShapeMismatch {
        ort_shape: Vec<usize>,
        holo_shape: Vec<usize>,
    },
    /// Node exists in ORT output but not found in hologram.
    MissingInHologram,
    /// Comparison failed (values diverge beyond tolerance).
    ValueMismatch(ComparisonResult),
}

/// Full execution comparison report.
#[derive(Debug)]
pub struct ExecutionReport {
    pub node_results: Vec<NodeComparison>,
    pub total_nodes: usize,
    pub passed: usize,
    pub failed: usize,
    pub missing: usize,
    /// Name of the first node that diverged (if any).
    pub first_divergence: Option<String>,
}

impl ExecutionReport {
    /// Whether all compared nodes passed.
    pub fn all_passed(&self) -> bool {
        self.failed == 0 && self.missing == 0
    }
}

/// An intermediate tensor from hologram execution (byte buffer + shape).
pub struct HologramTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub elem_size: usize,
}

impl HologramTensor {
    /// Interpret the raw bytes as f32 values.
    pub fn as_f32(&self) -> Vec<f32> {
        if self.elem_size != 4 || !self.data.len().is_multiple_of(4) {
            return Vec::new();
        }
        self.data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }
}

/// Compare ORT intermediate outputs against hologram executor outputs.
///
/// - `ort_outputs`: named tensors from ORT (via `run_onnx_all_outputs`)
/// - `holo_outputs`: hologram buffers keyed by builder index
///   (from `execute_with_intermediates`)
/// - `debug_map`: ONNX tensor name → builder index mapping
///   (from `compile_with_debug_info`)
/// - `tolerance`: comparison tolerance for values
pub fn compare_execution(
    ort_outputs: &[(String, Vec<usize>, Vec<f32>)],
    holo_outputs: &HashMap<usize, HologramTensor>,
    debug_map: &HashMap<String, usize>,
    tolerance: Tolerance,
) -> ExecutionReport {
    let mut node_results = Vec::new();
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut missing = 0usize;
    let mut first_divergence = None;

    for (name, ort_shape, ort_data) in ort_outputs {
        // Look up the hologram builder index for this ONNX tensor name.
        let holo_idx = match debug_map.get(name) {
            Some(&idx) => idx,
            None => {
                missing += 1;
                if first_divergence.is_none() {
                    first_divergence = Some(name.clone());
                }
                node_results.push(NodeComparison {
                    name: name.clone(),
                    ort_shape: ort_shape.clone(),
                    holo_shape: None,
                    result: NodeComparisonResult::MissingInHologram,
                });
                continue;
            }
        };

        let holo_tensor = match holo_outputs.get(&holo_idx) {
            Some(t) => t,
            None => {
                missing += 1;
                if first_divergence.is_none() {
                    first_divergence = Some(name.clone());
                }
                node_results.push(NodeComparison {
                    name: name.clone(),
                    ort_shape: ort_shape.clone(),
                    holo_shape: None,
                    result: NodeComparisonResult::MissingInHologram,
                });
                continue;
            }
        };

        let holo_data = holo_tensor.as_f32();
        let holo_shape = &holo_tensor.shape;

        // Check shape match first.
        if ort_shape != holo_shape {
            failed += 1;
            if first_divergence.is_none() {
                first_divergence = Some(name.clone());
            }
            node_results.push(NodeComparison {
                name: name.clone(),
                ort_shape: ort_shape.clone(),
                holo_shape: Some(holo_shape.clone()),
                result: NodeComparisonResult::ShapeMismatch {
                    ort_shape: ort_shape.clone(),
                    holo_shape: holo_shape.clone(),
                },
            });
            continue;
        }

        // Compare values.
        let cmp = compare_outputs(&holo_data, ort_data, tolerance);
        if cmp.passed {
            passed += 1;
            node_results.push(NodeComparison {
                name: name.clone(),
                ort_shape: ort_shape.clone(),
                holo_shape: Some(holo_shape.clone()),
                result: NodeComparisonResult::Pass(cmp),
            });
        } else {
            failed += 1;
            if first_divergence.is_none() {
                first_divergence = Some(name.clone());
            }
            node_results.push(NodeComparison {
                name: name.clone(),
                ort_shape: ort_shape.clone(),
                holo_shape: Some(holo_shape.clone()),
                result: NodeComparisonResult::ValueMismatch(cmp),
            });
        }
    }

    ExecutionReport {
        total_nodes: node_results.len(),
        node_results,
        passed,
        failed,
        missing,
        first_divergence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tol() -> Tolerance {
        Tolerance {
            atol: 1e-5,
            rtol: 1e-4,
        }
    }

    /// Verify that ONNX-imported RmsNorm models have correct structure for fusion.
    #[cfg(feature = "conformance")]
    #[test]
    fn rmsnorm_onnx_import_and_fusion() {
        use crate::ort_runner::onnx_builder;

        let model_bytes = onnx_builder::rms_norm(2, 16, 1e-6);
        let graph = hologram_ai_onnx::import_onnx(&model_bytes, Default::default())
            .expect("import failed");

        eprintln!("Imported {} nodes, {} params", graph.nodes.len(), graph.params.len());
        for node in &graph.nodes {
            eprintln!("  id={} {:?} inputs={:?} outputs={:?}", node.id, node.op, node.inputs, node.outputs);
        }
        for (&tid, param) in &graph.params {
            match param {
                hologram_ai_common::AiParam::Inline { data, info } => {
                    let val_str = if data.len() == 4 {
                        format!("= {}", f32::from_le_bytes(data[..4].try_into().unwrap()))
                    } else {
                        format!("{} bytes", data.len())
                    };
                    eprintln!("  param tid={}: dtype={:?} shape={:?} {}", tid, info.logical_dtype, info.shape, val_str);
                }
                _ => eprintln!("  param tid={}: non-inline", tid),
            }
        }

        // Run pipeline.
        let pipeline = hologram_ai_common::opt::pipeline::OptPipeline::mvp();
        let graph = pipeline.run(graph).unwrap();

        eprintln!("\nAfter optimization: {} nodes", graph.nodes.len());
        for node in &graph.nodes {
            eprintln!("  {:?} inputs={:?}", node.op, node.inputs);
        }

        let has_rmsnorm = graph.nodes.iter().any(|n| matches!(n.op, hologram_ai_common::AiOp::RmsNorm { .. }));
        assert!(has_rmsnorm, "RmsNorm fusion should have fired on ONNX-imported model");
    }

    #[test]
    fn matching_outputs_pass() {
        let ort = vec![
            ("relu_out".to_string(), vec![1, 4], vec![0.0, 1.0, 2.0, 3.0]),
        ];
        let mut holo = HashMap::new();
        holo.insert(
            2,
            HologramTensor {
                data: bytemuck::cast_slice(&[0.0f32, 1.0, 2.0, 3.0]).to_vec(),
                shape: vec![1, 4],
                elem_size: 4,
            },
        );
        let mut debug_map = HashMap::new();
        debug_map.insert("relu_out".to_string(), 2);

        let report = compare_execution(&ort, &holo, &debug_map, tol());
        assert!(report.all_passed());
        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 0);
    }

    #[test]
    fn shape_mismatch_detected() {
        let ort = vec![
            ("out".to_string(), vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]),
        ];
        let mut holo = HashMap::new();
        holo.insert(
            0,
            HologramTensor {
                data: bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_vec(),
                shape: vec![2, 2], // different shape
                elem_size: 4,
            },
        );
        let mut debug_map = HashMap::new();
        debug_map.insert("out".to_string(), 0);

        let report = compare_execution(&ort, &holo, &debug_map, tol());
        assert!(!report.all_passed());
        assert_eq!(report.failed, 1);
        assert!(matches!(
            &report.node_results[0].result,
            NodeComparisonResult::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn missing_node_detected() {
        let ort = vec![
            ("unknown_tensor".to_string(), vec![1, 2], vec![1.0, 2.0]),
        ];
        let holo = HashMap::new();
        let debug_map = HashMap::new();

        let report = compare_execution(&ort, &holo, &debug_map, tol());
        assert!(!report.all_passed());
        assert_eq!(report.missing, 1);
        assert_eq!(report.first_divergence.as_deref(), Some("unknown_tensor"));
    }

    #[test]
    fn value_mismatch_detected() {
        let ort = vec![
            ("out".to_string(), vec![1, 2], vec![1.0, 2.0]),
        ];
        let mut holo = HashMap::new();
        holo.insert(
            0,
            HologramTensor {
                data: bytemuck::cast_slice(&[1.0f32, 999.0]).to_vec(), // wrong value
                shape: vec![1, 2],
                elem_size: 4,
            },
        );
        let mut debug_map = HashMap::new();
        debug_map.insert("out".to_string(), 0);

        let report = compare_execution(&ort, &holo, &debug_map, tol());
        assert!(!report.all_passed());
        assert_eq!(report.failed, 1);
        assert!(matches!(
            &report.node_results[0].result,
            NodeComparisonResult::ValueMismatch(_)
        ));
    }

    #[test]
    fn multi_node_mixed_results() {
        let ort = vec![
            ("a".to_string(), vec![2], vec![1.0, 2.0]),
            ("b".to_string(), vec![2], vec![3.0, 4.0]),
            ("c".to_string(), vec![2], vec![5.0, 6.0]),
        ];
        let mut holo = HashMap::new();
        holo.insert(0, HologramTensor {
            data: bytemuck::cast_slice(&[1.0f32, 2.0]).to_vec(),
            shape: vec![2],
            elem_size: 4,
        });
        // "b" maps to idx 1 but idx 1 has wrong values
        holo.insert(1, HologramTensor {
            data: bytemuck::cast_slice(&[99.0f32, 100.0]).to_vec(),
            shape: vec![2],
            elem_size: 4,
        });
        // "c" is missing from holo
        let mut debug_map = HashMap::new();
        debug_map.insert("a".to_string(), 0);
        debug_map.insert("b".to_string(), 1);
        // "c" not in debug_map

        let report = compare_execution(&ort, &holo, &debug_map, tol());
        assert!(!report.all_passed());
        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);
        assert_eq!(report.missing, 1);
        assert_eq!(report.first_divergence.as_deref(), Some("b"));
    }
}
