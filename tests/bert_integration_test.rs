//! BERT model integration tests.
//!
//! Tests the compilation pipeline for BERT transformer models:
//! - ONNX parsing → IR translation → decomposition → .holo serialization
//! - Variable sequence length support
//! - Attention mechanism decomposition
//!
//! Note: BERT model must be downloaded separately. Tests will skip if not available.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use hologram_compiler::ir::IRBuilder;
use hologram_compiler::shapes::Dim;
use hologram_onnx_core::{
    OnnxConfig, SymbolicShape, extract_opset_version, parse_model, validate_model,
};
use hologram_onnx_ops::translate_onnx_op;
use tempfile::TempDir;

// ============================================================================
// Test Fixtures
// ============================================================================

/// Possible BERT model paths.
fn bert_model_paths() -> Vec<PathBuf> {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    vec![
        base.join("models/bert-base-uncased.onnx"),
        base.join("models/bert-base.onnx"),
        base.join("models/bert.onnx"),
        // HuggingFace optimum export path
        base.join("models/bert-base-uncased/model.onnx"),
    ]
}

/// Find available BERT model.
fn find_bert_model() -> Option<PathBuf> {
    bert_model_paths().into_iter().find(|p| p.exists())
}

/// Check if any BERT model exists.
fn has_bert_model() -> bool {
    find_bert_model().is_some()
}

/// Load BERT model bytes.
fn load_bert_model() -> Option<Vec<u8>> {
    find_bert_model().and_then(|path| fs::read(&path).ok())
}

// ============================================================================
// BERT Compilation Pipeline Tests
// ============================================================================

/// Test BERT model parsing succeeds.
#[test]
fn test_bert_parsing() {
    let Some(onnx_bytes) = load_bert_model() else {
        eprintln!("Skipping test_bert_parsing: BERT model not found");
        eprintln!("Download a BERT ONNX model to one of these paths:");
        for path in bert_model_paths() {
            eprintln!("  - {:?}", path);
        }
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse BERT model");

    let graph = model.graph.as_ref().expect("Model should have graph");
    assert!(!graph.node.is_empty(), "Graph should have nodes");
    assert!(!graph.input.is_empty(), "Graph should have inputs");
    assert!(!graph.output.is_empty(), "Graph should have outputs");

    eprintln!(
        "BERT parsed: {} nodes, {} initializers",
        graph.node.len(),
        graph.initializer.len()
    );
}

/// Test BERT model validation.
#[test]
fn test_bert_validation() {
    let Some(onnx_bytes) = load_bert_model() else {
        eprintln!("Skipping test_bert_validation: BERT model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse BERT model");
    let result = validate_model(&model);

    assert!(
        result.is_ok(),
        "BERT validation should pass: {:?}",
        result.err()
    );
}

/// Test BERT opset version.
#[test]
fn test_bert_opset_version() {
    let Some(onnx_bytes) = load_bert_model() else {
        eprintln!("Skipping test_bert_opset_version: BERT model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse BERT model");
    let opset = extract_opset_version(&model);

    // BERT models typically use opset 11+
    eprintln!("BERT opset version: {}", opset);
}

/// Test BERT input shape with symbolic sequence length.
#[test]
fn test_bert_symbolic_sequence_length() {
    let Some(onnx_bytes) = load_bert_model() else {
        eprintln!("Skipping test_bert_symbolic_sequence_length: BERT model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse BERT model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // BERT typically has input_ids, attention_mask, token_type_ids
    for input in &graph.input {
        // Skip initializers
        if graph.initializer.iter().any(|i| i.name == input.name) {
            continue;
        }

        if let Ok(shape) = SymbolicShape::from_value_info(input) {
            let dims = shape.dims();
            eprintln!("BERT input '{}': {:?}", input.name, dims);

            // Check for symbolic dimensions
            for (i, dim) in dims.iter().enumerate() {
                match dim {
                    Dim::Var(name) => {
                        eprintln!("  Dim {} is symbolic: {}", i, name);
                    }
                    Dim::Concrete(size) => {
                        eprintln!("  Dim {} is concrete: {}", i, size);
                    }
                    Dim::Expr(expr) => {
                        eprintln!("  Dim {} is expression: {}", i, expr);
                    }
                }
            }
        }
    }
}

/// Test BERT with variable sequence lengths.
#[test]
fn test_bert_variable_sequence_lengths() {
    let Some(onnx_bytes) = load_bert_model() else {
        eprintln!("Skipping test_bert_variable_sequence_lengths: BERT model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse BERT model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Find input_ids input
    let input_ids = graph
        .input
        .iter()
        .find(|i| i.name.contains("input") || i.name.contains("ids"))
        .and_then(|i| SymbolicShape::from_value_info(i).ok());

    let Some(shape) = input_ids else {
        eprintln!("Could not find input_ids shape");
        return;
    };

    // Test various sequence lengths
    let seq_lengths = [32, 64, 128, 256, 512];

    for seq_len in seq_lengths {
        let mut concrete_dims: Vec<Dim> = Vec::new();
        let dims = shape.dims();

        for (i, dim) in dims.iter().enumerate() {
            if i == 0 {
                // Batch dimension
                concrete_dims.push(Dim::Concrete(1));
            } else if i == 1 {
                // Sequence length dimension
                concrete_dims.push(Dim::Concrete(seq_len));
            } else {
                concrete_dims.push(dim.clone());
            }
        }

        let concrete_shape = SymbolicShape::new(
            concrete_dims
                .into_iter()
                .map(|d| match d {
                    Dim::Concrete(n) => hologram_onnx_core::Dim::Concrete(n),
                    Dim::Var(name) => hologram_onnx_core::Dim::Var(name),
                    Dim::Expr(expr) => hologram_onnx_core::Dim::Expr(expr),
                })
                .collect(),
        );

        eprintln!("BERT with seq_len {}: {:?}", seq_len, concrete_shape.dims());
    }
}

/// Test BERT operation coverage.
#[test]
fn test_bert_operation_coverage() {
    let Some(onnx_bytes) = load_bert_model() else {
        eprintln!("Skipping test_bert_operation_coverage: BERT model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse BERT model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    let mut builder = IRBuilder::new("bert_test");
    let shapes: HashMap<String, SymbolicShape> = HashMap::new();

    let mut op_counts: HashMap<String, usize> = HashMap::new();
    let mut supported_count = 0;
    let mut unsupported_ops: Vec<String> = Vec::new();

    for node in &graph.node {
        *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;

        let result = translate_onnx_op(&node.op_type, &[], &node.attribute, &shapes, &mut builder);

        match &result {
            Err(hologram_onnx_core::OnnxError::UnsupportedOp { op_type, .. }) => {
                if !unsupported_ops.contains(op_type) {
                    unsupported_ops.push(op_type.clone());
                }
            }
            _ => {
                supported_count += 1;
            }
        }
    }

    eprintln!("BERT operation breakdown:");
    let mut sorted_ops: Vec<_> = op_counts.iter().collect();
    sorted_ops.sort_by(|a, b| b.1.cmp(a.1));
    for (op, count) in &sorted_ops[..sorted_ops.len().min(10)] {
        eprintln!("  {}: {}", op, count);
    }

    if !unsupported_ops.is_empty() {
        eprintln!("Unsupported operations: {:?}", unsupported_ops);
    }

    let support_ratio = supported_count as f64 / graph.node.len() as f64;
    eprintln!("Support ratio: {:.1}%", support_ratio * 100.0);
}

/// Test BERT attention operations.
#[test]
fn test_bert_attention_operations() {
    let Some(onnx_bytes) = load_bert_model() else {
        eprintln!("Skipping test_bert_attention_operations: BERT model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse BERT model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Count attention-related operations
    let matmul_count = graph.node.iter().filter(|n| n.op_type == "MatMul").count();
    let softmax_count = graph.node.iter().filter(|n| n.op_type == "Softmax").count();
    let attention_count = graph
        .node
        .iter()
        .filter(|n| n.op_type == "Attention")
        .count();

    eprintln!("BERT attention metrics:");
    eprintln!("  MatMul operations: {}", matmul_count);
    eprintln!("  Softmax operations: {}", softmax_count);
    eprintln!("  Attention operations: {}", attention_count);

    // BERT-base has 12 attention layers, each with Q, K, V projections and output projection
    // That's at least 12 * 4 = 48 MatMuls for attention alone
    if attention_count == 0 {
        // Decomposed BERT (no fused Attention op)
        assert!(
            matmul_count >= 24,
            "BERT should have many MatMul ops for attention"
        );
        assert!(softmax_count >= 12, "BERT should have 12+ Softmax ops");
    }
}

/// Test BERT layer normalization operations.
#[test]
fn test_bert_layer_normalization() {
    let Some(onnx_bytes) = load_bert_model() else {
        eprintln!("Skipping test_bert_layer_normalization: BERT model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse BERT model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Count LayerNormalization operations
    let layer_norm_count = graph
        .node
        .iter()
        .filter(|n| n.op_type == "LayerNormalization")
        .count();

    eprintln!("BERT LayerNormalization count: {}", layer_norm_count);

    // BERT-base has LayerNorm after each attention and FFN block
    // 12 layers * 2 = 24 LayerNorms (plus embeddings)
    if layer_norm_count > 0 {
        assert!(layer_norm_count >= 24, "BERT should have 24+ LayerNorm ops");
    }
}

/// Test BERT hidden dimension.
#[test]
fn test_bert_hidden_dimension() {
    let Some(onnx_bytes) = load_bert_model() else {
        eprintln!("Skipping test_bert_hidden_dimension: BERT model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse BERT model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Find hidden dimension from initializer shapes
    // Look for dense/embedding weights
    let mut hidden_dims: Vec<usize> = Vec::new();

    for init in &graph.initializer {
        if init.dims.len() == 2 {
            let (d0, d1) = (init.dims[0] as usize, init.dims[1] as usize);
            // Common BERT dimensions: 768 (base), 1024 (large), 512 (small)
            for &dim in &[768, 1024, 512, 256, 384] {
                if (d0 == dim || d1 == dim) && !hidden_dims.contains(&dim) {
                    hidden_dims.push(dim);
                }
            }
        }
    }

    if !hidden_dims.is_empty() {
        eprintln!("Detected BERT hidden dimensions: {:?}", hidden_dims);
        // Most common is 768 for BERT-base
        assert!(
            hidden_dims
                .iter()
                .any(|&d| d == 768 || d == 1024 || d == 512),
            "BERT should have standard hidden dimension"
        );
    }
}

/// Test full BERT compilation.
#[test]
fn test_bert_full_compilation() {
    let bert_path = match find_bert_model() {
        Some(path) => path,
        None => {
            eprintln!("Skipping test_bert_full_compilation: BERT model not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("bert_compiled");

    use std::process::Command;

    let status = Command::new(env!("CARGO_BIN_EXE_hologram-onnx"))
        .args([
            "compile",
            bert_path.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(status.success(), "BERT compilation should succeed");

    let holo_path = output.with_extension("holo");
    assert!(holo_path.exists(), ".holo file should be created");

    let holo_content = fs::read(&holo_path).expect("Should read .holo file");
    assert!(!holo_content.is_empty(), ".holo file should not be empty");
    assert_eq!(
        &holo_content[0..4],
        b"HOLO",
        "Should have HOLO magic header"
    );

    eprintln!("BERT compiled successfully: {} bytes", holo_content.len());
}

/// Test BERT attention decomposition.
#[test]
fn test_bert_attention_decomposition() {
    let bert_path = match find_bert_model() {
        Some(path) => path,
        None => {
            eprintln!("Skipping test_bert_attention_decomposition: BERT model not found");
            return;
        }
    };

    // Configuration for attention decomposition
    let _config = OnnxConfig {
        weight_threshold: 4096,
        enable_partitioning: false,
        partition_size: 500,
        decompose_conv2d: true,
        decompose_pooling: true,
        memory_budget: Some(1024), // 1GB limit for BERT
    };

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("bert_decomposed");

    use std::process::Command;

    let status = Command::new(env!("CARGO_BIN_EXE_hologram-onnx"))
        .args([
            "compile",
            bert_path.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--memory-budget",
            "1024",
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(
        status.success(),
        "BERT compilation with attention decomposition should succeed"
    );

    let holo_path = output.with_extension("holo");
    let holo_content = fs::read(&holo_path).expect("Should read .holo file");

    eprintln!("BERT decomposed .holo size: {} bytes", holo_content.len());
}

/// Test BERT weight count and size.
#[test]
fn test_bert_weight_extraction() {
    let Some(onnx_bytes) = load_bert_model() else {
        eprintln!("Skipping test_bert_weight_extraction: BERT model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse BERT model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    let weight_count = graph.initializer.len();

    let total_weight_bytes: usize = graph
        .initializer
        .iter()
        .map(|init| {
            if !init.raw_data.is_empty() {
                init.raw_data.len()
            } else {
                let elem_size = match init.data_type {
                    1 => 4, // FLOAT
                    7 => 8, // INT64
                    _ => 4,
                };
                let num_elements: usize = init.dims.iter().map(|&d| d as usize).product();
                num_elements * elem_size
            }
        })
        .sum();

    eprintln!(
        "BERT weights: {} initializers, {} MB total",
        weight_count,
        total_weight_bytes / (1024 * 1024)
    );

    // BERT-base has ~110M parameters
    // This is just informational, no strict assertion
}
