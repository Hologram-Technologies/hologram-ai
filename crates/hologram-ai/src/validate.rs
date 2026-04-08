//! Model validation: import → optimize → lower → report.
//!
//! Validates that a model can be fully compiled without errors and reports
//! per-node coverage. Future: ORT-based node-by-node output comparison.

use crate::compiler::{ModelCompiler, ModelSource};
use std::collections::HashMap;
use std::path::Path;

/// Validation result for a single model.
pub struct ValidationReport {
    /// Model file path.
    pub model_path: String,
    /// Total number of AiGraph nodes after optimization.
    pub total_nodes: usize,
    /// Total number of parameters.
    pub total_params: usize,
    /// Number of input tensors.
    pub num_inputs: usize,
    /// Number of output tensors.
    pub num_outputs: usize,
    /// Per-op-type counts.
    pub op_counts: HashMap<String, usize>,
    /// Compilation succeeded.
    pub compilation_ok: bool,
    /// Compiled node count (from the .holo archive).
    pub compiled_node_count: usize,
    /// Total weight bytes in the compiled archive.
    pub compiled_weight_bytes: u64,
    /// Import warnings count.
    pub import_warnings: usize,
    /// Error message if compilation failed.
    pub error: Option<String>,
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Model Validation Report ===")?;
        writeln!(f, "Model: {}", self.model_path)?;
        writeln!(f)?;
        writeln!(f, "--- Import ---")?;
        writeln!(f, "  nodes:    {}", self.total_nodes)?;
        writeln!(f, "  params:   {}", self.total_params)?;
        writeln!(f, "  inputs:   {}", self.num_inputs)?;
        writeln!(f, "  outputs:  {}", self.num_outputs)?;
        if self.import_warnings > 0 {
            writeln!(f, "  warnings: {}", self.import_warnings)?;
        }
        writeln!(f)?;

        writeln!(f, "--- Op Coverage ({} types) ---", self.op_counts.len())?;
        let mut sorted: Vec<_> = self.op_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (op, count) in &sorted {
            writeln!(f, "  {op:<30} {count:>4}")?;
        }
        writeln!(f)?;

        writeln!(f, "--- Compilation ---")?;
        if self.compilation_ok {
            writeln!(f, "  status:  PASS")?;
            writeln!(f, "  nodes:   {}", self.compiled_node_count)?;
            writeln!(f, "  weights: {} bytes", self.compiled_weight_bytes)?;
        } else {
            writeln!(f, "  status:  FAIL")?;
            if let Some(err) = &self.error {
                writeln!(f, "  error:   {err}")?;
            }
        }

        Ok(())
    }
}

/// Validate a model file: import, optimize, compile, and report results.
pub fn validate_model(path: &Path) -> ValidationReport {
    let model_path = path.display().to_string();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let source = match ext {
        "onnx" => ModelSource::OnnxPath(path.to_owned()),
        other => {
            return ValidationReport {
                model_path,
                total_nodes: 0,
                total_params: 0,
                num_inputs: 0,
                num_outputs: 0,
                op_counts: HashMap::new(),
                compilation_ok: false,
                compiled_node_count: 0,
                compiled_weight_bytes: 0,
                import_warnings: 0,
                error: Some(format!("unsupported format: '.{other}'")),
            };
        }
    };

    // Step 1: Import to get AiGraph stats before compilation.
    let ai_graph = match import_model(path, ext) {
        Ok(g) => g,
        Err(e) => {
            return ValidationReport {
                model_path,
                total_nodes: 0,
                total_params: 0,
                num_inputs: 0,
                num_outputs: 0,
                op_counts: HashMap::new(),
                compilation_ok: false,
                compiled_node_count: 0,
                compiled_weight_bytes: 0,
                import_warnings: 0,
                error: Some(format!("import failed: {e}")),
            };
        }
    };

    let total_nodes = ai_graph.nodes.len();
    let total_params = ai_graph.params.len();
    let num_inputs = ai_graph.inputs.len();
    let num_outputs = ai_graph.outputs.len();

    let mut op_counts = HashMap::new();
    for node in &ai_graph.nodes {
        let name = format!("{:?}", node.op);
        // Truncate at first '{' or '(' to get just the op name
        let name = name.split(['{', '(']).next().unwrap_or(&name).trim();
        *op_counts.entry(name.to_string()).or_insert(0) += 1;
    }

    // Step 2: Full compilation.
    match ModelCompiler::default().compile(source) {
        Ok(compiled) => ValidationReport {
            model_path,
            total_nodes,
            total_params,
            num_inputs,
            num_outputs,
            op_counts,
            compilation_ok: true,
            compiled_node_count: compiled.stats.node_count,
            compiled_weight_bytes: compiled.stats.total_weight_bytes,
            import_warnings: compiled.stats.import_warnings,
            error: None,
        },
        Err(e) => ValidationReport {
            model_path,
            total_nodes,
            total_params,
            num_inputs,
            num_outputs,
            op_counts,
            compilation_ok: false,
            compiled_node_count: 0,
            compiled_weight_bytes: 0,
            import_warnings: 0,
            error: Some(format!("{e:#}")),
        },
    }
}

/// Validate an in-memory AiGraph (no file needed — suitable for CI tests).
pub fn validate_graph(graph: hologram_ai_common::AiGraph) -> ValidationReport {
    let total_nodes = graph.nodes.len();
    let total_params = graph.params.len();
    let num_inputs = graph.inputs.len();
    let num_outputs = graph.outputs.len();

    let mut op_counts = HashMap::new();
    for node in &graph.nodes {
        let name = format!("{:?}", node.op);
        let name = name.split(['{', '(']).next().unwrap_or(&name).trim();
        *op_counts.entry(name.to_string()).or_insert(0) += 1;
    }

    let source = ModelSource::AiGraph(graph);
    match ModelCompiler::default().compile(source) {
        Ok(compiled) => ValidationReport {
            model_path: "<in-memory>".into(),
            total_nodes,
            total_params,
            num_inputs,
            num_outputs,
            op_counts,
            compilation_ok: true,
            compiled_node_count: compiled.stats.node_count,
            compiled_weight_bytes: compiled.stats.total_weight_bytes,
            import_warnings: compiled.stats.import_warnings,
            error: None,
        },
        Err(e) => ValidationReport {
            model_path: "<in-memory>".into(),
            total_nodes,
            total_params,
            num_inputs,
            num_outputs,
            op_counts,
            compilation_ok: false,
            compiled_node_count: 0,
            compiled_weight_bytes: 0,
            import_warnings: 0,
            error: Some(format!("{e:#}")),
        },
    }
}

fn import_model(path: &Path, ext: &str) -> anyhow::Result<hologram_ai_common::AiGraph> {
    match ext {
        "onnx" => Ok(hologram_ai_onnx::import_onnx_path(
            path,
            Default::default(),
        )?),
        other => anyhow::bail!("unsupported: .{other}"),
    }
}
