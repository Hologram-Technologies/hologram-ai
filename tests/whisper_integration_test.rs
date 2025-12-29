//! Whisper model integration tests.
//!
//! Tests the compilation pipeline for OpenAI Whisper speech recognition models:
//! - ONNX parsing → IR translation → decomposition → .holo serialization
//! - Audio encoder/decoder architecture
//! - Variable audio length support
//!
//! Note: Whisper model must be downloaded separately. Tests will skip if not available.

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

/// Possible Whisper model paths.
fn whisper_model_paths() -> Vec<PathBuf> {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    vec![
        base.join("models/whisper-tiny.onnx"),
        base.join("models/whisper-base.onnx"),
        base.join("models/whisper-small.onnx"),
        base.join("models/whisper.onnx"),
        // Encoder/decoder components
        base.join("models/whisper-tiny-encoder.onnx"),
        base.join("models/whisper-tiny-decoder.onnx"),
        base.join("models/whisper/encoder.onnx"),
        base.join("models/whisper/decoder.onnx"),
    ]
}

/// Find available Whisper model.
fn find_whisper_model() -> Option<PathBuf> {
    whisper_model_paths().into_iter().find(|p| p.exists())
}

/// Check if any Whisper model exists.
fn has_whisper_model() -> bool {
    find_whisper_model().is_some()
}

/// Load Whisper model bytes.
fn load_whisper_model() -> Option<Vec<u8>> {
    find_whisper_model().and_then(|path| fs::read(&path).ok())
}

// ============================================================================
// Whisper Compilation Pipeline Tests
// ============================================================================

/// Test Whisper model parsing succeeds.
#[test]
fn test_whisper_parsing() {
    let Some(onnx_bytes) = load_whisper_model() else {
        eprintln!("Skipping test_whisper_parsing: Whisper model not found");
        eprintln!("Download a Whisper ONNX model to one of these paths:");
        for path in whisper_model_paths() {
            eprintln!("  - {:?}", path);
        }
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse Whisper model");

    let graph = model.graph.as_ref().expect("Model should have graph");
    assert!(!graph.node.is_empty(), "Graph should have nodes");

    eprintln!(
        "Whisper parsed: {} nodes, {} initializers",
        graph.node.len(),
        graph.initializer.len()
    );
}

/// Test Whisper model validation.
#[test]
fn test_whisper_validation() {
    let Some(onnx_bytes) = load_whisper_model() else {
        eprintln!("Skipping test_whisper_validation: Whisper model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse Whisper model");
    let result = validate_model(&model);

    assert!(
        result.is_ok(),
        "Whisper validation should pass: {:?}",
        result.err()
    );
}

/// Test Whisper opset version.
#[test]
fn test_whisper_opset_version() {
    let Some(onnx_bytes) = load_whisper_model() else {
        eprintln!("Skipping test_whisper_opset_version: Whisper model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse Whisper model");
    let opset = extract_opset_version(&model);

    eprintln!("Whisper opset version: {}", opset);
}

/// Test Whisper input shapes (audio features).
#[test]
fn test_whisper_input_shapes() {
    let Some(onnx_bytes) = load_whisper_model() else {
        eprintln!("Skipping test_whisper_input_shapes: Whisper model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse Whisper model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    for input in &graph.input {
        // Skip initializers
        if graph.initializer.iter().any(|i| i.name == input.name) {
            continue;
        }

        if let Ok(shape) = SymbolicShape::from_value_info(input) {
            let dims = shape.dims();
            eprintln!("Whisper input '{}': {:?}", input.name, dims);

            // Check for audio-related dimensions
            // Whisper encoder expects [batch, n_mels, audio_frames]
            // n_mels is typically 80 or 128
            for (i, dim) in dims.iter().enumerate() {
                match dim {
                    Dim::Concrete(80) | Dim::Concrete(128) => {
                        eprintln!("  Found mel spectrogram dimension at index {}", i);
                    }
                    Dim::Var(name) if name.contains("audio") || name.contains("time") => {
                        eprintln!("  Found audio time dimension: {}", name);
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Test Whisper with variable audio lengths.
#[test]
fn test_whisper_variable_audio_length() {
    let Some(onnx_bytes) = load_whisper_model() else {
        eprintln!("Skipping test_whisper_variable_audio_length: Whisper model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse Whisper model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Find audio input
    let audio_input = graph
        .input
        .iter()
        .filter(|i| !graph.initializer.iter().any(|init| init.name == i.name))
        .find_map(|i| SymbolicShape::from_value_info(i).ok());

    let Some(shape) = audio_input else {
        eprintln!("Could not find audio input shape");
        return;
    };

    // Test various audio frame counts
    // 30 seconds of audio at 16kHz = 480000 samples
    // With 160 hop length = 3000 frames
    let audio_frame_counts = [1500, 3000, 6000]; // 15s, 30s, 60s

    for frames in audio_frame_counts {
        let mut concrete_dims: Vec<Dim> = Vec::new();
        let dims = shape.dims();

        for (i, dim) in dims.iter().enumerate() {
            if i == 0 {
                // Batch dimension
                concrete_dims.push(Dim::Concrete(1));
            } else if matches!(dim, Dim::Var(_)) {
                // Assume variable dimension is audio length
                concrete_dims.push(Dim::Concrete(frames));
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

        eprintln!(
            "Whisper with {} audio frames: {:?}",
            frames,
            concrete_shape.dims()
        );
    }
}

/// Test Whisper operation coverage.
#[test]
fn test_whisper_operation_coverage() {
    let Some(onnx_bytes) = load_whisper_model() else {
        eprintln!("Skipping test_whisper_operation_coverage: Whisper model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse Whisper model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    let mut builder = IRBuilder::new("whisper_test");
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

    eprintln!("Whisper operation breakdown:");
    let mut sorted_ops: Vec<_> = op_counts.iter().collect();
    sorted_ops.sort_by(|a, b| b.1.cmp(a.1));
    for (op, count) in &sorted_ops[..sorted_ops.len().min(15)] {
        eprintln!("  {}: {}", op, count);
    }

    if !unsupported_ops.is_empty() {
        eprintln!("Unsupported operations: {:?}", unsupported_ops);
    }

    let support_ratio = supported_count as f64 / graph.node.len() as f64;
    eprintln!("Support ratio: {:.1}%", support_ratio * 100.0);
}

/// Test Whisper Conv1D operations (for audio processing).
#[test]
fn test_whisper_conv1d_operations() {
    let Some(onnx_bytes) = load_whisper_model() else {
        eprintln!("Skipping test_whisper_conv1d_operations: Whisper model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse Whisper model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Count Conv operations
    let conv_count = graph.node.iter().filter(|n| n.op_type == "Conv").count();

    eprintln!("Whisper Conv operations: {}", conv_count);

    // Whisper uses Conv1D in the audio encoder
    // The encoder has 2 Conv1D layers for audio processing
}

/// Test Whisper attention operations.
#[test]
fn test_whisper_attention_operations() {
    let Some(onnx_bytes) = load_whisper_model() else {
        eprintln!("Skipping test_whisper_attention_operations: Whisper model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse Whisper model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Count attention-related operations
    let matmul_count = graph.node.iter().filter(|n| n.op_type == "MatMul").count();
    let softmax_count = graph.node.iter().filter(|n| n.op_type == "Softmax").count();

    eprintln!("Whisper attention metrics:");
    eprintln!("  MatMul operations: {}", matmul_count);
    eprintln!("  Softmax operations: {}", softmax_count);

    // Whisper uses transformer architecture with self and cross attention
}

/// Test full Whisper compilation.
#[test]
fn test_whisper_full_compilation() {
    let whisper_path = match find_whisper_model() {
        Some(path) => path,
        None => {
            eprintln!("Skipping test_whisper_full_compilation: Whisper model not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("whisper_compiled");

    use std::process::Command;

    let status = Command::new(env!("CARGO_BIN_EXE_hologram-onnx"))
        .args([
            "compile",
            whisper_path.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(status.success(), "Whisper compilation should succeed");

    let holo_path = output.with_extension("holo");
    assert!(holo_path.exists(), ".holo file should be created");

    let holo_content = fs::read(&holo_path).expect("Should read .holo file");
    assert!(!holo_content.is_empty(), ".holo file should not be empty");
    assert_eq!(
        &holo_content[0..4],
        b"HOLO",
        "Should have HOLO magic header"
    );

    eprintln!(
        "Whisper compiled successfully: {} bytes",
        holo_content.len()
    );
}

/// Test Whisper compilation with partitioning.
#[test]
fn test_whisper_compilation_with_partitioning() {
    let whisper_path = match find_whisper_model() {
        Some(path) => path,
        None => {
            eprintln!(
                "Skipping test_whisper_compilation_with_partitioning: Whisper model not found"
            );
            return;
        }
    };

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("whisper_partitioned");

    use std::process::Command;

    let status = Command::new(env!("CARGO_BIN_EXE_hologram-onnx"))
        .args([
            "compile",
            whisper_path.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--partition",
            "--partition-size",
            "100",
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(
        status.success(),
        "Whisper compilation with partitioning should succeed"
    );

    let holo_path = output.with_extension("holo");
    assert!(holo_path.exists(), ".holo file should be created");

    eprintln!("Whisper compiled with partitioning successfully");
}

/// Test Whisper weight extraction.
#[test]
fn test_whisper_weight_extraction() {
    let Some(onnx_bytes) = load_whisper_model() else {
        eprintln!("Skipping test_whisper_weight_extraction: Whisper model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse Whisper model");
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
        "Whisper weights: {} initializers, {} MB total",
        weight_count,
        total_weight_bytes / (1024 * 1024)
    );

    // Whisper-tiny has ~39M parameters (~156MB float32)
    // Whisper-base has ~74M parameters (~296MB float32)
}

/// Test Whisper configuration validation.
#[test]
fn test_whisper_config_validation() {
    // Test configuration for Whisper
    let config = OnnxConfig {
        weight_threshold: 4096,
        enable_partitioning: true, // Whisper benefits from partitioning
        partition_size: 200,
        decompose_conv2d: true,
        decompose_pooling: true,
        memory_budget: Some(512), // 512MB for Whisper-tiny
    };

    assert!(config.validate().is_ok(), "Whisper config should be valid");
}

/// Test Whisper output shape (token predictions).
#[test]
fn test_whisper_output_shape() {
    let Some(onnx_bytes) = load_whisper_model() else {
        eprintln!("Skipping test_whisper_output_shape: Whisper model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse Whisper model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    for output in &graph.output {
        if let Ok(shape) = SymbolicShape::from_value_info(output) {
            let dims = shape.dims();
            eprintln!("Whisper output '{}': {:?}", output.name, dims);
        }
    }
}
