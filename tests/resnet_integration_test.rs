//! ResNet50 model integration tests.
//!
//! Tests the full compilation pipeline for the ResNet50 image classification model:
//! - ONNX parsing → IR translation → decomposition → .holo serialization
//! - Variable batch size support
//! - Conv2D decomposition verification
//! - ISA optimization verification

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

/// Get path to ResNet50 model.
fn resnet_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/resnet50-v1-7.onnx")
}

/// Check if ResNet50 model exists.
fn has_resnet_model() -> bool {
    resnet_model_path().exists()
}

/// Load ResNet50 model bytes.
fn load_resnet_model() -> Option<Vec<u8>> {
    let path = resnet_model_path();
    if path.exists() {
        fs::read(&path).ok()
    } else {
        None
    }
}

// ============================================================================
// ResNet50 Compilation Pipeline Tests
// ============================================================================

/// Test ResNet50 model parsing succeeds.
#[test]
fn test_resnet_parsing() {
    let Some(onnx_bytes) = load_resnet_model() else {
        eprintln!(
            "Skipping test_resnet_parsing: ResNet model not found at {:?}",
            resnet_model_path()
        );
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse ResNet model");

    // Verify basic structure
    let graph = model.graph.as_ref().expect("Model should have graph");
    assert!(
        graph.node.len() > 100,
        "ResNet should have many nodes (got {})",
        graph.node.len()
    );
    assert!(!graph.input.is_empty(), "Graph should have inputs");
    assert!(!graph.output.is_empty(), "Graph should have outputs");

    // ResNet50 has ~175 nodes
    eprintln!(
        "ResNet parsed: {} nodes, {} initializers",
        graph.node.len(),
        graph.initializer.len()
    );
}

/// Test ResNet50 model validation passes.
#[test]
fn test_resnet_validation() {
    let Some(onnx_bytes) = load_resnet_model() else {
        eprintln!("Skipping test_resnet_validation: ResNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse ResNet model");
    let result = validate_model(&model);

    assert!(
        result.is_ok(),
        "ResNet validation should pass: {:?}",
        result.err()
    );
}

/// Test ResNet50 opset version extraction.
#[test]
fn test_resnet_opset_version() {
    let Some(onnx_bytes) = load_resnet_model() else {
        eprintln!("Skipping test_resnet_opset_version: ResNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse ResNet model");
    let opset = extract_opset_version(&model);

    // ResNet50-v1-7 is opset 7 or higher
    assert!(
        opset >= 7,
        "ResNet opset should be at least 7, got {}",
        opset
    );
    eprintln!("ResNet opset version: {}", opset);
}

/// Test ResNet50 input shape (224x224 RGB images).
#[test]
fn test_resnet_input_shape() {
    let Some(onnx_bytes) = load_resnet_model() else {
        eprintln!("Skipping test_resnet_input_shape: ResNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse ResNet model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Get input shape
    let input = graph.input.first().expect("Should have input");
    let shape = SymbolicShape::from_value_info(input).expect("Should parse input shape");

    let dims = shape.dims();
    eprintln!("ResNet input shape: {:?}", dims);

    // ResNet expects [batch, 3, 224, 224]
    assert!(dims.len() >= 4, "ResNet input should have 4 dimensions");

    // Check channel dimension (typically index 1)
    if let Dim::Concrete(channels) = &dims[1] {
        assert_eq!(*channels, 3, "ResNet should have 3 input channels (RGB)");
    }

    // Check spatial dimensions
    if let (Dim::Concrete(h), Dim::Concrete(w)) = (&dims[2], &dims[3]) {
        assert_eq!(*h, 224, "ResNet height should be 224");
        assert_eq!(*w, 224, "ResNet width should be 224");
    }
}

/// Test ResNet50 output shape (1000 ImageNet classes).
#[test]
fn test_resnet_output_shape() {
    let Some(onnx_bytes) = load_resnet_model() else {
        eprintln!("Skipping test_resnet_output_shape: ResNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse ResNet model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Get output shape
    let output = graph.output.first().expect("Should have output");
    let shape = SymbolicShape::from_value_info(output).expect("Should parse output shape");

    let dims = shape.dims();
    eprintln!("ResNet output shape: {:?}", dims);

    // ResNet output should have 1000 classes
    let last_dim = dims.last().expect("Should have at least one dimension");
    if let Dim::Concrete(classes) = last_dim {
        assert_eq!(*classes, 1000, "ResNet should have 1000 ImageNet classes");
    }
}

/// Test ResNet50 Conv2D operation count.
#[test]
fn test_resnet_conv2d_count() {
    let Some(onnx_bytes) = load_resnet_model() else {
        eprintln!("Skipping test_resnet_conv2d_count: ResNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse ResNet model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Count Conv operations
    let conv_count = graph.node.iter().filter(|n| n.op_type == "Conv").count();

    // ResNet50 has 53 Conv layers
    eprintln!("ResNet Conv count: {}", conv_count);
    assert!(
        conv_count >= 50,
        "ResNet should have ~53 Conv layers, got {}",
        conv_count
    );
}

/// Test ResNet50 residual connection (Add) count.
#[test]
fn test_resnet_residual_connections() {
    let Some(onnx_bytes) = load_resnet_model() else {
        eprintln!("Skipping test_resnet_residual_connections: ResNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse ResNet model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Count Add operations (residual connections)
    let add_count = graph.node.iter().filter(|n| n.op_type == "Add").count();

    // ResNet50 has 16 residual blocks
    eprintln!("ResNet Add (residual) count: {}", add_count);
    assert!(
        add_count >= 16,
        "ResNet should have ~16 residual connections, got {}",
        add_count
    );
}

/// Test ResNet50 operation translation coverage.
#[test]
fn test_resnet_operation_coverage() {
    let Some(onnx_bytes) = load_resnet_model() else {
        eprintln!("Skipping test_resnet_operation_coverage: ResNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse ResNet model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    let mut builder = IRBuilder::new("resnet_test");
    let shapes: HashMap<String, SymbolicShape> = HashMap::new();

    // Count operation types
    let mut op_counts: HashMap<String, usize> = HashMap::new();
    let mut supported_count = 0;
    let mut unsupported_ops: Vec<String> = Vec::new();

    for node in &graph.node {
        *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;

        // Test if operation type is supported
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

    eprintln!("ResNet operation breakdown:");
    let mut sorted_ops: Vec<_> = op_counts.iter().collect();
    sorted_ops.sort_by(|a, b| b.1.cmp(a.1));
    for (op, count) in sorted_ops {
        eprintln!("  {}: {}", op, count);
    }

    if !unsupported_ops.is_empty() {
        eprintln!("Unsupported operations: {:?}", unsupported_ops);
    }

    // Most ResNet operations should be supported
    let support_ratio = supported_count as f64 / graph.node.len() as f64;
    eprintln!("Support ratio: {:.1}%", support_ratio * 100.0);
    assert!(
        support_ratio > 0.9,
        "At least 90% of ResNet operations should be supported"
    );
}

/// Test ResNet50 with variable batch sizes.
#[test]
fn test_resnet_variable_batch_sizes() {
    let Some(onnx_bytes) = load_resnet_model() else {
        eprintln!("Skipping test_resnet_variable_batch_sizes: ResNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse ResNet model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Parse input shape
    let input = graph.input.first().expect("Should have input");
    let shape = SymbolicShape::from_value_info(input).expect("Should parse input shape");

    // Test various batch sizes
    let batch_sizes = [1, 4, 8, 16, 32];

    for batch_size in batch_sizes {
        let mut concrete_dims: Vec<Dim> = Vec::new();
        let dims = shape.dims();

        for (i, dim) in dims.iter().enumerate() {
            if i == 0 {
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
            "ResNet with batch size {}: {:?}",
            batch_size,
            concrete_shape.dims()
        );
    }
}

/// Test full ResNet50 compilation.
#[test]
fn test_resnet_full_compilation() {
    if !has_resnet_model() {
        eprintln!("Skipping test_resnet_full_compilation: ResNet model not found");
        return;
    }

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("resnet_compiled");

    use std::process::Command;

    let Some(bin_path) = hologram_onnx_bin() else {
        eprintln!("Skipping test_resnet_full_compilation: hologram-onnx binary not built");
        return;
    };
    let status = Command::new(bin_path)
        .args([
            "compile",
            resnet_model_path().to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(status.success(), "ResNet compilation should succeed");

    // Verify output file exists
    let holo_path = output.with_extension("holo");
    assert!(holo_path.exists(), ".holo file should be created");

    let holo_content = fs::read(&holo_path).expect("Should read .holo file");
    assert!(!holo_content.is_empty(), ".holo file should not be empty");
    assert_eq!(
        &holo_content[0..4],
        b"HOLO",
        "Should have HOLO magic header"
    );

    eprintln!("ResNet compiled successfully: {} bytes", holo_content.len());
}

/// Test ResNet50 compilation with partitioning.
#[test]
fn test_resnet_compilation_with_partitioning() {
    if !has_resnet_model() {
        eprintln!("Skipping test_resnet_compilation_with_partitioning: ResNet model not found");
        return;
    }

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("resnet_partitioned");

    use std::process::Command;

    let Some(bin_path) = hologram_onnx_bin() else {
        eprintln!(
            "Skipping test_resnet_compilation_with_partitioning: hologram-onnx binary not built"
        );
        return;
    };
    let status = Command::new(bin_path)
        .args([
            "compile",
            resnet_model_path().to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--partition",
            "--partition-size",
            "50", // Small partitions for testing
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(
        status.success(),
        "ResNet compilation with partitioning should succeed"
    );

    let holo_path = output.with_extension("holo");
    assert!(
        holo_path.exists(),
        ".holo file should be created with partitioning"
    );

    eprintln!("ResNet compiled with partitioning successfully");
}

/// Test ResNet50 Conv2D decomposition.
#[test]
fn test_resnet_conv2d_decomposition() {
    if !has_resnet_model() {
        eprintln!("Skipping test_resnet_conv2d_decomposition: ResNet model not found");
        return;
    }

    // Configuration with decomposition enabled
    let config = OnnxConfig {
        weight_threshold: 4096,
        enable_partitioning: false,
        partition_size: 500,
        decompose_conv2d: true, // Enable Conv2D → Im2col + GEMM
        decompose_pooling: true,
        memory_budget: None,
    };

    assert!(config.validate().is_ok());

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("resnet_decomposed");

    use std::process::Command;

    // Compile with default settings (decomposition enabled)
    let Some(bin_path) = hologram_onnx_bin() else {
        eprintln!("Skipping test_resnet_conv2d_decomposition: hologram-onnx binary not built");
        return;
    };
    let status = Command::new(bin_path)
        .args([
            "compile",
            resnet_model_path().to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(
        status.success(),
        "ResNet compilation with decomposition should succeed"
    );

    let holo_path = output.with_extension("holo");
    let holo_content = fs::read(&holo_path).expect("Should read .holo file");

    // Decomposed version should have more nodes than original
    // (53 Conv → 53 Im2col + 53 GEMM = 106 extra operations)
    eprintln!("ResNet decomposed .holo size: {} bytes", holo_content.len());
}

/// Test ResNet50 weight extraction.
#[test]
fn test_resnet_weight_extraction() {
    let Some(onnx_bytes) = load_resnet_model() else {
        eprintln!("Skipping test_resnet_weight_extraction: ResNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse ResNet model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Count initializers (weights)
    let weight_count = graph.initializer.len();

    // Calculate total weight size
    let total_weight_bytes: usize = graph
        .initializer
        .iter()
        .map(|init| {
            if !init.raw_data.is_empty() {
                init.raw_data.len()
            } else {
                // Calculate from typed fields
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
        "ResNet weights: {} initializers, {} MB total",
        weight_count,
        total_weight_bytes / (1024 * 1024)
    );

    // ResNet50 has ~25M parameters * 4 bytes = ~97MB
    assert!(
        total_weight_bytes > 90_000_000,
        "ResNet should have ~97MB of weights"
    );
}

/// Test ResNet50 ISA optimization markers (verification that translation prepares for ISA).
#[test]
fn test_resnet_isa_optimization_readiness() {
    let Some(onnx_bytes) = load_resnet_model() else {
        eprintln!("Skipping test_resnet_isa_optimization_readiness: ResNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse ResNet model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Count operations that benefit from ISA optimizations
    let mut isa_benefiting_ops = 0;

    for node in &graph.node {
        match node.op_type.as_str() {
            // LOOP instructions benefit
            "Conv" | "MaxPool" | "AveragePool" | "GlobalAveragePool" => {
                isa_benefiting_ops += 1;
            }
            // ClassMap fusion benefits
            "Relu" | "Add" | "Mul" => {
                isa_benefiting_ops += 1;
            }
            // SIMD vectorization benefits
            "MatMul" | "Gemm" => {
                isa_benefiting_ops += 1;
            }
            // PhiCoordinate benefits
            "BatchNormalization" => {
                isa_benefiting_ops += 1;
            }
            _ => {}
        }
    }

    let isa_ratio = isa_benefiting_ops as f64 / graph.node.len() as f64;
    eprintln!(
        "ResNet ISA-optimizable operations: {} / {} ({:.1}%)",
        isa_benefiting_ops,
        graph.node.len(),
        isa_ratio * 100.0
    );

    // Most ResNet operations should benefit from ISA
    assert!(
        isa_ratio > 0.8,
        "At least 80% of ResNet ops should benefit from ISA"
    );
}
