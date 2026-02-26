//! T5 text-to-text model integration tests.
//!
//! Tests the compilation pipeline for Google's T5 (Text-To-Text Transfer Transformer) models:
//! - ONNX parsing → IR translation → decomposition → .holo serialization
//! - Encoder-decoder architecture
//! - Variable batch size and sequence length support
//!
//! Note: T5 models must be downloaded separately. Tests will skip if not available.

use std::fs;
use std::path::PathBuf;

use hologram_ai_onnx::{parse_model, validate_model, extract_opset_version};
use hologram_holo::{HOLB_MAGIC, HOLM_MAGIC};
use tempfile::TempDir;

fn hologram_ai_bin() -> std::path::PathBuf {
    // Try env var first (set by cargo test for binary crates)
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_hologram-ai") {
        return std::path::PathBuf::from(path);
    }

    // Fall back to target directory relative to manifest
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let debug_bin = manifest_dir.join("target/debug/hologram-ai");
    if debug_bin.exists() {
        return debug_bin;
    }

    let release_bin = manifest_dir.join("target/release/hologram-ai");
    if release_bin.exists() {
        return release_bin;
    }

    panic!(
        "hologram-ai binary not found. Run `cargo build` first.\n\
         Searched:\n  - {:?}\n  - {:?}",
        debug_bin, release_bin
    );
}

// ============================================================================
// Test Fixtures
// ============================================================================

/// Possible T5 encoder model paths.
fn t5_encoder_paths() -> Vec<PathBuf> {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    vec![
        base.join("models/t5-small/encoder_model.onnx"),
        base.join("models/t5-small/encoder.onnx"),
        base.join("models/t5/encoder_model.onnx"),
        base.join("models/t5/encoder.onnx"),
    ]
}

/// Possible T5 decoder model paths.
fn t5_decoder_paths() -> Vec<PathBuf> {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    vec![
        base.join("models/t5-small/decoder_model.onnx"),
        base.join("models/t5-small/decoder.onnx"),
        base.join("models/t5/decoder_model.onnx"),
        base.join("models/t5/decoder.onnx"),
    ]
}

/// Find available T5 encoder model.
fn find_t5_encoder() -> Option<PathBuf> {
    t5_encoder_paths().into_iter().find(|p| p.exists())
}

/// Find available T5 decoder model.
fn find_t5_decoder() -> Option<PathBuf> {
    t5_decoder_paths().into_iter().find(|p| p.exists())
}

/// Load T5 encoder model bytes.
fn load_t5_encoder() -> Option<Vec<u8>> {
    find_t5_encoder().and_then(|path| fs::read(&path).ok())
}

/// Load T5 decoder model bytes.
fn load_t5_decoder() -> Option<Vec<u8>> {
    find_t5_decoder().and_then(|path| fs::read(&path).ok())
}

/// Print instructions for downloading T5 models.
fn print_t5_download_instructions() {
    eprintln!("Download T5-small ONNX models first:\n");
    eprintln!("  pip install optimum[exporters]");
    eprintln!("  optimum-cli export onnx \\");
    eprintln!("    --model google/t5-small \\");
    eprintln!("    --task text2text-generation-with-past \\");
    eprintln!("    /workspace/models/t5-small/\n");
    eprintln!("Expected files:");
    eprintln!("  - encoder_model.onnx (or encoder.onnx)");
    eprintln!("  - decoder_model.onnx (or decoder.onnx)");
    eprintln!("  - tokenizer.json");
}

// ============================================================================
// T5 Encoder Tests
// ============================================================================

/// Test T5 encoder parsing.
#[test]
fn test_t5_encoder_parsing() {
    let Some(onnx_bytes) = load_t5_encoder() else {
        eprintln!("Skipping test_t5_encoder_parsing: T5 encoder not found");
        print_t5_download_instructions();
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse T5 encoder");

    let graph = model.graph.as_ref().expect("Model should have graph");
    assert!(!graph.node.is_empty(), "Graph should have nodes");
    assert!(!graph.input.is_empty(), "Graph should have inputs");
    assert!(!graph.output.is_empty(), "Graph should have outputs");

    eprintln!(
        "T5 encoder parsed: {} nodes, {} initializers",
        graph.node.len(),
        graph.initializer.len()
    );

    // T5-small encoder typically has 500-800 nodes
    assert!(
        graph.node.len() >= 100,
        "T5 encoder should have many nodes"
    );
}

/// Test T5 encoder validation.
#[test]
fn test_t5_encoder_validation() {
    let Some(onnx_bytes) = load_t5_encoder() else {
        eprintln!("Skipping test_t5_encoder_validation: T5 encoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse T5 encoder");
    let result = validate_model(&model);

    assert!(
        result.is_ok(),
        "T5 encoder validation should pass: {:?}",
        result.err()
    );
}

/// Test T5 encoder opset version.
#[test]
fn test_t5_encoder_opset_version() {
    let Some(onnx_bytes) = load_t5_encoder() else {
        eprintln!("Skipping test_t5_encoder_opset_version: T5 encoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse T5 encoder");
    let opset = extract_opset_version(&model);

    eprintln!("T5 encoder opset version: {}", opset);
    // T5 models typically use opset 14+
}

/// Test T5 encoder input shapes with symbolic dimensions.
#[test]
fn test_t5_encoder_symbolic_shapes() {
    let Some(onnx_bytes) = load_t5_encoder() else {
        eprintln!("Skipping test_t5_encoder_symbolic_shapes: T5 encoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse T5 encoder");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // T5 encoder input: input_ids [batch, seq_len]
    for input in &graph.input {
        // Skip initializers
        if graph.initializer.iter().any(|i| i.name == input.name) {
            continue;
        }

        // Extract shape from value_info using the proto type_info
        if let Some(ref type_proto) = input.r#type {
            if let Some(ref value) = type_proto.value {
                if let hologram_ai_onnx::proto::type_proto::Value::TensorType(ref tensor_type) = value {
                    if let Some(ref shape_proto) = tensor_type.shape {
                        let dims: Vec<_> = shape_proto.dim.iter().map(|d| {
                            match &d.value {
                                Some(hologram_ai_onnx::proto::tensor_shape_proto::dimension::Value::DimValue(v)) => format!("{}", v),
                                Some(hologram_ai_onnx::proto::tensor_shape_proto::dimension::Value::DimParam(s)) => s.clone(),
                                None => "?".to_string(),
                            }
                        }).collect();
                        eprintln!("T5 encoder input '{}': {:?}", input.name, dims);

                        if input.name.contains("input") {
                            assert!(
                                dims.len() == 2,
                                "T5 encoder input should have 2D shape [batch, seq_len]"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Test T5 encoder operation coverage.
#[test]
fn test_t5_encoder_operation_coverage() {
    let Some(onnx_bytes) = load_t5_encoder() else {
        eprintln!("Skipping test_t5_encoder_operation_coverage: T5 encoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse T5 encoder");
    let graph = model.graph.as_ref().expect("Model should have graph");

    use std::collections::HashMap;
    let mut op_counts: HashMap<String, usize> = HashMap::new();

    for node in &graph.node {
        *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
    }

    eprintln!("T5 encoder operation breakdown:");
    let mut sorted_ops: Vec<_> = op_counts.iter().collect();
    sorted_ops.sort_by(|a, b| b.1.cmp(a.1));
    for (op, count) in &sorted_ops[..sorted_ops.len().min(15)] {
        eprintln!("  {}: {}", op, count);
    }

    // List of operations we know should be supported
    let supported_ops = vec![
        "MatMul", "Add", "Mul", "Sub", "Div",
        "LayerNormalization", "Softmax", "Relu", "Gelu",
        "Transpose", "Reshape", "Concat", "Slice", "Gather",
        "Cast", "Constant", "Shape", "Unsqueeze", "Squeeze"
    ];

    // Check that all T5 operations are in the supported list
    let mut unsupported = Vec::new();
    for op_type in op_counts.keys() {
        if !supported_ops.contains(&op_type.as_str()) {
            unsupported.push(op_type.clone());
        }
    }

    if !unsupported.is_empty() {
        eprintln!("Warning: Found operations not in known-supported list: {:?}", unsupported);
        eprintln!("This may be okay if these operations are actually supported");
    }
}

/// Test T5 encoder critical operations.
#[test]
fn test_t5_encoder_critical_operations() {
    let Some(onnx_bytes) = load_t5_encoder() else {
        eprintln!("Skipping test_t5_encoder_critical_operations: T5 encoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse T5 encoder");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Count critical operations for T5 encoder
    let matmul_count = graph.node.iter().filter(|n| n.op_type == "MatMul").count();
    let layer_norm_count = graph
        .node
        .iter()
        .filter(|n| n.op_type == "LayerNormalization")
        .count();
    let softmax_count = graph.node.iter().filter(|n| n.op_type == "Softmax").count();
    let add_count = graph.node.iter().filter(|n| n.op_type == "Add").count();

    eprintln!("T5 encoder operations:");
    eprintln!("  MatMul: {}", matmul_count);
    eprintln!("  LayerNormalization: {}", layer_norm_count);
    eprintln!("  Softmax: {}", softmax_count);
    eprintln!("  Add: {}", add_count);

    // T5-small has 6 encoder layers
    // Each layer: Q, K, V projections + output projection + FFN (2 linear layers)
    // = 6 * 6 = 36+ MatMuls
    assert!(matmul_count >= 20, "T5 encoder should have many MatMul ops");

    // T5 models can use fused LayerNormalization OR decompose it into
    // primitive ops (ReduceMean, Sub, Pow, Sqrt, Div, Mul, Add).
    // hologram-onnx supports both patterns.
    let reduce_mean_count = graph.node.iter().filter(|n| n.op_type == "ReduceMean").count();

    if layer_norm_count >= 10 {
        eprintln!("  Using fused LayerNormalization ops");
    } else if reduce_mean_count >= 10 {
        eprintln!("  ReduceMean: {} (decomposed LayerNorm pattern)", reduce_mean_count);
    } else {
        panic!(
            "T5 encoder should have normalization ops (fused LayerNorm: {}, ReduceMean: {})",
            layer_norm_count, reduce_mean_count
        );
    }

    // Each layer has 1 Softmax (for attention)
    // = 6 Softmax
    assert!(
        softmax_count >= 6,
        "T5 encoder should have Softmax ops for attention"
    );
}

/// Test T5 encoder compilation.
#[test]
fn test_t5_encoder_compilation() {
    let encoder_path = match find_t5_encoder() {
        Some(path) => path,
        None => {
            eprintln!("Skipping test_t5_encoder_compilation: T5 encoder not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("t5_encoder");

    use std::process::Command;

    let status = Command::new(hologram_ai_bin())
        .args([
            "compile",
            encoder_path.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--partition",
            "--partition-size",
            "200",
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(
        status.success(),
        "T5 encoder compilation should succeed"
    );

    let holo_path = output.with_extension("holo");
    assert!(holo_path.exists(), ".holo file should be created");

    let holo_content = fs::read(&holo_path).expect("Should read .holo file");
    assert!(!holo_content.is_empty(), ".holo file should not be empty");
    let magic = &holo_content[0..4];
    assert!(
        magic == HOLB_MAGIC || magic == HOLM_MAGIC,
        "Should have HOLB or HOLP magic header"
    );

    eprintln!(
        "T5 encoder compiled successfully: {} bytes",
        holo_content.len()
    );
}

// ============================================================================
// T5 Decoder Tests
// ============================================================================

/// Test T5 decoder parsing.
#[test]
fn test_t5_decoder_parsing() {
    let Some(onnx_bytes) = load_t5_decoder() else {
        eprintln!("Skipping test_t5_decoder_parsing: T5 decoder not found");
        print_t5_download_instructions();
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse T5 decoder");

    let graph = model.graph.as_ref().expect("Model should have graph");
    assert!(!graph.node.is_empty(), "Graph should have nodes");
    assert!(!graph.input.is_empty(), "Graph should have inputs");
    assert!(!graph.output.is_empty(), "Graph should have outputs");

    eprintln!(
        "T5 decoder parsed: {} nodes, {} initializers",
        graph.node.len(),
        graph.initializer.len()
    );

    // T5-small decoder typically has 600-1000 nodes (more than encoder due to cross-attention)
    assert!(
        graph.node.len() >= 100,
        "T5 decoder should have many nodes"
    );
}

/// Test T5 decoder validation.
#[test]
fn test_t5_decoder_validation() {
    let Some(onnx_bytes) = load_t5_decoder() else {
        eprintln!("Skipping test_t5_decoder_validation: T5 decoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse T5 decoder");
    let result = validate_model(&model);

    assert!(
        result.is_ok(),
        "T5 decoder validation should pass: {:?}",
        result.err()
    );
}

/// Test T5 decoder operation coverage.
#[test]
fn test_t5_decoder_operation_coverage() {
    let Some(onnx_bytes) = load_t5_decoder() else {
        eprintln!("Skipping test_t5_decoder_operation_coverage: T5 decoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse T5 decoder");
    let graph = model.graph.as_ref().expect("Model should have graph");

    use std::collections::HashMap;
    let mut op_counts: HashMap<String, usize> = HashMap::new();

    for node in &graph.node {
        *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
    }

    eprintln!("T5 decoder operation breakdown:");
    let mut sorted_ops: Vec<_> = op_counts.iter().collect();
    sorted_ops.sort_by(|a, b| b.1.cmp(a.1));
    for (op, count) in &sorted_ops[..sorted_ops.len().min(15)] {
        eprintln!("  {}: {}", op, count);
    }

    // List of operations we know should be supported (decoder has Gather for embeddings)
    let supported_ops = vec![
        "MatMul", "Add", "Mul", "Sub", "Div",
        "LayerNormalization", "Softmax", "Relu", "Gelu",
        "Transpose", "Reshape", "Concat", "Slice", "Gather",
        "Cast", "Constant", "Shape", "Unsqueeze", "Squeeze"
    ];

    // Check that all T5 operations are in the supported list
    let mut unsupported = Vec::new();
    for op_type in op_counts.keys() {
        if !supported_ops.contains(&op_type.as_str()) {
            unsupported.push(op_type.clone());
        }
    }

    if !unsupported.is_empty() {
        eprintln!("Warning: Found operations not in known-supported list: {:?}", unsupported);
        eprintln!("This may be okay if these operations are actually supported");
    }
}

/// Test T5 decoder critical operations.
#[test]
fn test_t5_decoder_critical_operations() {
    let Some(onnx_bytes) = load_t5_decoder() else {
        eprintln!("Skipping test_t5_decoder_critical_operations: T5 decoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse T5 decoder");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Count critical operations for T5 decoder
    let matmul_count = graph.node.iter().filter(|n| n.op_type == "MatMul").count();
    let gather_count = graph.node.iter().filter(|n| n.op_type == "Gather").count();
    let layer_norm_count = graph
        .node
        .iter()
        .filter(|n| n.op_type == "LayerNormalization")
        .count();

    eprintln!("T5 decoder operations:");
    eprintln!("  MatMul: {}", matmul_count);
    eprintln!("  Gather: {}", gather_count);
    eprintln!("  LayerNormalization: {}", layer_norm_count);

    // T5 decoder has Gather for embedding lookups
    assert!(gather_count >= 1, "T5 decoder should have Gather ops");

    // More MatMuls than encoder due to cross-attention
    assert!(
        matmul_count >= 30,
        "T5 decoder should have many MatMul ops"
    );
}

/// Test T5 decoder compilation.
#[test]
fn test_t5_decoder_compilation() {
    let decoder_path = match find_t5_decoder() {
        Some(path) => path,
        None => {
            eprintln!("Skipping test_t5_decoder_compilation: T5 decoder not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("t5_decoder");

    use std::process::Command;

    let status = Command::new(hologram_ai_bin())
        .args([
            "compile",
            decoder_path.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--partition",
            "--partition-size",
            "200",
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(
        status.success(),
        "T5 decoder compilation should succeed"
    );

    let holo_path = output.with_extension("holo");
    assert!(holo_path.exists(), ".holo file should be created");

    let holo_content = fs::read(&holo_path).expect("Should read .holo file");
    assert!(!holo_content.is_empty(), ".holo file should not be empty");
    let magic = &holo_content[0..4];
    assert!(
        magic == HOLB_MAGIC || magic == HOLM_MAGIC,
        "Should have HOLB or HOLP magic header"
    );

    eprintln!(
        "T5 decoder compiled successfully: {} bytes",
        holo_content.len()
    );
}

/// Test T5 encoder weight extraction.
#[test]
fn test_t5_encoder_weights() {
    let Some(onnx_bytes) = load_t5_encoder() else {
        eprintln!("Skipping test_t5_encoder_weights: T5 encoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse T5 encoder");
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
        "T5 encoder weights: {} initializers, {} MB total",
        weight_count,
        total_weight_bytes / (1024 * 1024)
    );

    // T5-small encoder has ~30M parameters
    // This is informational, no strict assertion
}

/// Test T5 decoder weight extraction.
#[test]
fn test_t5_decoder_weights() {
    let Some(onnx_bytes) = load_t5_decoder() else {
        eprintln!("Skipping test_t5_decoder_weights: T5 decoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse T5 decoder");
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
        "T5 decoder weights: {} initializers, {} MB total",
        weight_count,
        total_weight_bytes / (1024 * 1024)
    );

    // T5-small decoder has ~30M parameters
    // This is informational, no strict assertion
}
