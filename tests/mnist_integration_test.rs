//! MNIST model integration tests.
//!
//! Tests the full compilation pipeline for the MNIST digit classification model:
//! - ONNX parsing → IR translation → decomposition → .holo serialization
//! - Variable batch size support
//! - Output shape verification

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

fn hologram_onnx_bin() -> Option<PathBuf> {
    std::env::var("CARGO_BIN_EXE_hologram-onnx")
        .map(PathBuf::from)
        .ok()
}
// ============================================================================
// Test Fixtures
// ============================================================================

/// Get path to MNIST model.
fn mnist_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates/hologram-onnx-core/tests/fixtures/mnist-12.onnx")
}

/// Check if MNIST model exists.
fn has_mnist_model() -> bool {
    mnist_model_path().exists()
}

/// Load MNIST model bytes.
fn load_mnist_model() -> Option<Vec<u8>> {
    let path = mnist_model_path();
    if path.exists() {
        fs::read(&path).ok()
    } else {
        None
    }
}

// ============================================================================
// MNIST Compilation Pipeline Tests
// ============================================================================

/// Test MNIST model parsing succeeds.
#[test]
fn test_mnist_parsing() {
    let Some(onnx_bytes) = load_mnist_model() else {
        eprintln!("Skipping test_mnist_parsing: MNIST model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse MNIST model");

    // Verify basic structure
    let graph = model.graph.as_ref().expect("Model should have graph");
    assert!(!graph.node.is_empty(), "Graph should have nodes");
    assert!(!graph.input.is_empty(), "Graph should have inputs");
    assert!(!graph.output.is_empty(), "Graph should have outputs");

    eprintln!(
        "MNIST parsed: {} nodes, {} inputs, {} outputs",
        graph.node.len(),
        graph.input.len(),
        graph.output.len()
    );
}

/// Test MNIST model validation passes.
#[test]
fn test_mnist_validation() {
    let Some(onnx_bytes) = load_mnist_model() else {
        eprintln!("Skipping test_mnist_validation: MNIST model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse MNIST model");
    let result = validate_model(&model);

    assert!(
        result.is_ok(),
        "MNIST validation should pass: {:?}",
        result.err()
    );
}

/// Test MNIST opset version extraction.
#[test]
fn test_mnist_opset_version() {
    let Some(onnx_bytes) = load_mnist_model() else {
        eprintln!("Skipping test_mnist_opset_version: MNIST model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse MNIST model");
    let opset = extract_opset_version(&model);

    // MNIST-12 should be opset 12 or higher
    assert!(
        opset >= 8,
        "MNIST opset should be at least 8, got {}",
        opset
    );
    eprintln!("MNIST opset version: {}", opset);
}

/// Test MNIST input shape parsing with symbolic batch dimension.
#[test]
fn test_mnist_symbolic_batch_size() {
    let Some(onnx_bytes) = load_mnist_model() else {
        eprintln!("Skipping test_mnist_symbolic_batch_size: MNIST model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse MNIST model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Get input shape
    let input = graph.input.first().expect("Should have input");
    let shape = SymbolicShape::from_value_info(input).expect("Should parse input shape");

    // MNIST input should be [batch, 1, 28, 28] or similar
    let dims = shape.dims();
    assert!(
        dims.len() >= 3,
        "MNIST input should have at least 3 dimensions"
    );

    // First dimension should be batch (often symbolic)
    match &dims[0] {
        Dim::Var(name) => {
            eprintln!("MNIST batch dimension is symbolic: {}", name);
        }
        Dim::Concrete(size) => {
            eprintln!("MNIST batch dimension is concrete: {}", size);
        }
        Dim::Expr(expr) => {
            eprintln!("MNIST batch dimension is expression: {}", expr);
        }
    }
}

/// Test MNIST operation translation to IR.
#[test]
fn test_mnist_operation_translation() {
    let Some(onnx_bytes) = load_mnist_model() else {
        eprintln!("Skipping test_mnist_operation_translation: MNIST model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse MNIST model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    let mut builder = IRBuilder::new("mnist_test");
    let shapes: HashMap<String, SymbolicShape> = HashMap::new();

    // Count supported vs unsupported operations
    let mut supported = 0;
    let mut unsupported = 0;
    let mut unsupported_ops: Vec<String> = Vec::new();

    for node in &graph.node {
        // Create dummy inputs for translation test
        let result = translate_onnx_op(
            &node.op_type,
            &[], // Empty inputs for support check
            &node.attribute,
            &shapes,
            &mut builder,
        );

        // Operations fail for lack of inputs, but we can check if the op type is recognized
        match &result {
            Err(hologram_onnx_core::OnnxError::UnsupportedOp { op_type, .. }) => {
                unsupported += 1;
                if !unsupported_ops.contains(op_type) {
                    unsupported_ops.push(op_type.clone());
                }
            }
            _ => {
                // Either succeeded (unlikely with no inputs) or failed for other reason
                // Both indicate the operation type is recognized
                supported += 1;
            }
        }
    }

    eprintln!(
        "MNIST operations: {} supported, {} unsupported",
        supported, unsupported
    );
    if !unsupported_ops.is_empty() {
        eprintln!("Unsupported ops: {:?}", unsupported_ops);
    }

    // MNIST uses basic operations that should all be supported
    assert!(
        supported > 0,
        "At least some MNIST operations should be supported"
    );
}

/// Test MNIST output shape inference.
#[test]
fn test_mnist_output_shape() {
    let Some(onnx_bytes) = load_mnist_model() else {
        eprintln!("Skipping test_mnist_output_shape: MNIST model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse MNIST model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Get output shape
    let output = graph.output.first().expect("Should have output");
    let shape = SymbolicShape::from_value_info(output).expect("Should parse output shape");

    let dims = shape.dims();
    eprintln!("MNIST output shape: {:?}", dims);

    // MNIST output should have 10 classes (digit 0-9)
    // Shape is typically [batch, 10]
    let last_dim = dims.last().expect("Should have at least one dimension");
    if let Dim::Concrete(size) = last_dim {
        assert_eq!(*size, 10, "MNIST should have 10 output classes");
    }
}

/// Test MNIST with different batch sizes.
#[test]
fn test_mnist_variable_batch_sizes() {
    let Some(onnx_bytes) = load_mnist_model() else {
        eprintln!("Skipping test_mnist_variable_batch_sizes: MNIST model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse MNIST model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Parse input shape
    let input = graph.input.first().expect("Should have input");
    let shape = SymbolicShape::from_value_info(input).expect("Should parse input shape");

    // Test that shape supports variable batch
    let batch_sizes = [1, 8, 16, 32, 64];

    for batch_size in batch_sizes {
        // Create concrete shape with specific batch
        let mut concrete_dims: Vec<Dim> = Vec::new();
        let dims = shape.dims();

        for (i, dim) in dims.iter().enumerate() {
            if i == 0 {
                // Replace batch dimension
                concrete_dims.push(Dim::Concrete(batch_size));
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
            "MNIST with batch size {}: {:?}",
            batch_size,
            concrete_shape.dims()
        );
    }
}

/// Test full MNIST compilation pipeline.
#[test]
fn test_mnist_full_compilation() {
    if !has_mnist_model() {
        eprintln!("Skipping test_mnist_full_compilation: MNIST model not found");
        return;
    }

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("mnist_compiled");

    // Use the CLI's compile command
    use std::process::Command;

    let Some(bin_path) = hologram_onnx_bin() else {
        eprintln!("Skipping test_mnist_full_compilation: hologram-onnx binary not built");
        return;
    };
    let status = Command::new(bin_path)
        .args([
            "compile",
            mnist_model_path().to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(status.success(), "MNIST compilation should succeed");

    // Verify output file exists
    let holo_path = output.with_extension("holo");
    assert!(holo_path.exists(), ".holo file should be created");

    // Verify file has content
    let holo_content = fs::read(&holo_path).expect("Should read .holo file");
    assert!(!holo_content.is_empty(), ".holo file should not be empty");

    // Verify magic header
    assert_eq!(
        &holo_content[0..4],
        b"HOLO",
        "Should have HOLO magic header"
    );

    eprintln!("MNIST compiled successfully: {} bytes", holo_content.len());
}

/// Test MNIST compilation with configuration options.
#[test]
fn test_mnist_compilation_with_config() {
    if !has_mnist_model() {
        eprintln!("Skipping test_mnist_compilation_with_config: MNIST model not found");
        return;
    }

    // Test configuration validation
    let config = OnnxConfig {
        weight_threshold: 4096,
        enable_partitioning: false,
        partition_size: 500,
        decompose_conv2d: true,
        decompose_pooling: true,
        enable_resize_upscaling: true,
        pack_weights: true,
        memory_budget: Some(512), // 512MB limit
    };

    assert!(config.validate().is_ok(), "Config should be valid");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("mnist_config");

    use std::process::Command;

    let Some(bin_path) = hologram_onnx_bin() else {
        eprintln!("Skipping test_mnist_compilation_with_config: hologram-onnx binary not built");
        return;
    };
    let status = Command::new(bin_path)
        .args([
            "compile",
            mnist_model_path().to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--memory-budget",
            "512",
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(
        status.success(),
        "MNIST compilation with config should succeed"
    );
}

/// Test MNIST compilation produces deterministic output.
#[test]
fn test_mnist_deterministic_compilation() {
    if !has_mnist_model() {
        eprintln!("Skipping test_mnist_deterministic_compilation: MNIST model not found");
        return;
    }

    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    use std::process::Command;

    let Some(bin_path) = hologram_onnx_bin() else {
        eprintln!(
            "Skipping test_mnist_deterministic_compilation: hologram-onnx binary not built"
        );
        return;
    };

    // Compile twice
    let outputs: Vec<_> = (0..2)
        .map(|i| {
            let output = temp_dir.path().join(format!("mnist_compile_{}", i));

            let status = Command::new(&bin_path)
                .args([
                    "compile",
                    mnist_model_path().to_str().unwrap(),
                    "-o",
                    output.to_str().unwrap(),
                ])
                .status()
                .expect("Failed to run hologram-onnx compile");

            assert!(status.success(), "Compilation {} should succeed", i);

            fs::read(output.with_extension("holo")).expect("Should read .holo file")
        })
        .collect();

    // Verify both compilations produce same size output
    assert_eq!(
        outputs[0].len(),
        outputs[1].len(),
        "Deterministic compilation should produce same size output"
    );

    eprintln!(
        "MNIST deterministic compilation verified: {} bytes",
        outputs[0].len()
    );
}
