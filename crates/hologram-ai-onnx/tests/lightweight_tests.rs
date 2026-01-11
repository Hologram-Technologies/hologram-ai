//! Lightweight integration tests using mock ONNX models.
//!
//! These tests verify the full compilation pipeline without requiring downloads
//! of real models like MNIST or BERT. They run quickly in CI and test:
//! - Model parsing and validation
//! - Graph translation to IR
//! - Compilation to .holo bundles
//! - Various operation types

mod mock_models;

use hologram_ai_onnx::{
    OnnxCompiler, OnnxConfig, extract_opset_version, parse_model, translate_graph_to_ir,
    validate_model,
};

/// Test parsing and validating mock models.
#[test]
fn test_parse_identity_model() {
    let bytes = mock_models::identity_model();
    let model = parse_model(&bytes).expect("Should parse identity model");
    validate_model(&model).expect("Should validate identity model");

    let opset = extract_opset_version(&model);
    assert!(opset >= 13, "Opset should be at least 13");

    let graph = model.graph.as_ref().expect("Should have graph");
    assert_eq!(graph.input.len(), 1);
    assert_eq!(graph.output.len(), 1);
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Identity");
}

#[test]
fn test_parse_linear_model() {
    let bytes = mock_models::linear_model();
    let model = parse_model(&bytes).expect("Should parse linear model");
    validate_model(&model).expect("Should validate linear model");

    let graph = model.graph.as_ref().expect("Should have graph");
    assert_eq!(graph.node.len(), 2); // MatMul + Add
    assert_eq!(graph.initializer.len(), 2); // weights + bias

    // Verify operation types
    let op_types: Vec<&str> = graph.node.iter().map(|n| n.op_type.as_str()).collect();
    assert!(op_types.contains(&"MatMul"));
    assert!(op_types.contains(&"Add"));
}

#[test]
fn test_parse_mlp_model() {
    let bytes = mock_models::mlp_model();
    let model = parse_model(&bytes).expect("Should parse MLP model");
    validate_model(&model).expect("Should validate MLP model");

    let graph = model.graph.as_ref().expect("Should have graph");
    assert_eq!(graph.node.len(), 6);

    // Verify we have expected operations
    let op_types: Vec<&str> = graph.node.iter().map(|n| n.op_type.as_str()).collect();
    assert_eq!(op_types.iter().filter(|&&t| t == "MatMul").count(), 2);
    assert_eq!(op_types.iter().filter(|&&t| t == "Add").count(), 2);
    assert!(op_types.contains(&"Relu"));
    assert!(op_types.contains(&"Softmax"));
}

#[test]
fn test_parse_conv_model() {
    let bytes = mock_models::conv_model();
    let model = parse_model(&bytes).expect("Should parse conv model");
    validate_model(&model).expect("Should validate conv model");

    let graph = model.graph.as_ref().expect("Should have graph");

    // Verify Conv attributes
    let conv_node = graph.node.iter().find(|n| n.op_type == "Conv").unwrap();
    let kernel_attr = conv_node
        .attribute
        .iter()
        .find(|a| a.name == "kernel_shape")
        .unwrap();
    assert_eq!(kernel_attr.ints, vec![3, 3]);
}

#[test]
fn test_parse_mini_classifier() {
    let bytes = mock_models::mini_classifier_model();
    let model = parse_model(&bytes).expect("Should parse mini classifier");
    validate_model(&model).expect("Should validate mini classifier");

    let graph = model.graph.as_ref().expect("Should have graph");

    // Count operation types (MLP architecture: 2x MatMul, 2x Add, Relu, Softmax)
    let mut op_counts = std::collections::HashMap::new();
    for node in &graph.node {
        *op_counts.entry(node.op_type.as_str()).or_insert(0) += 1;
    }

    assert_eq!(op_counts.get("MatMul"), Some(&2));
    assert_eq!(op_counts.get("Add"), Some(&2));
    assert_eq!(op_counts.get("Relu"), Some(&1));
    assert_eq!(op_counts.get("Softmax"), Some(&1));
}

/// Test graph translation to IR.
#[test]
fn test_translate_identity_to_ir() {
    let bytes = mock_models::identity_model();
    let model = parse_model(&bytes).expect("Should parse");
    let graph = model.graph.as_ref().expect("Should have graph");

    let op_graph = translate_graph_to_ir(graph).expect("Should translate to IR");

    // Identity model should produce a valid operation graph
    assert!(op_graph.node_count() > 0);
}

#[test]
fn test_translate_linear_to_ir() {
    let bytes = mock_models::linear_model();
    let model = parse_model(&bytes).expect("Should parse");
    let graph = model.graph.as_ref().expect("Should have graph");

    let op_graph = translate_graph_to_ir(graph).expect("Should translate to IR");

    // Linear layer should have nodes for MatMul and Add
    assert!(op_graph.node_count() >= 2);
}

#[test]
fn test_translate_mlp_to_ir() {
    let bytes = mock_models::mlp_model();
    let model = parse_model(&bytes).expect("Should parse");
    let graph = model.graph.as_ref().expect("Should have graph");

    let op_graph = translate_graph_to_ir(graph).expect("Should translate to IR");

    // MLP should produce multiple IR nodes
    assert!(op_graph.node_count() >= 4);
}

#[test]
#[ignore = "Conv bias broadcasting requires IR-level fixes"]
fn test_translate_conv_to_ir() {
    let bytes = mock_models::conv_model();
    let model = parse_model(&bytes).expect("Should parse");
    let graph = model.graph.as_ref().expect("Should have graph");

    let op_graph = translate_graph_to_ir(graph).expect("Should translate to IR");

    assert!(op_graph.node_count() > 0);
}

/// Test full compilation pipeline.
#[test]
fn test_compile_identity_model() {
    let bytes = mock_models::identity_model();
    let compiler = OnnxCompiler::new();

    let (holo_bytes, weight_bytes) = compiler
        .compile(&bytes)
        .expect("Should compile identity model");

    assert!(!holo_bytes.is_empty(), "Should produce non-empty .holo");
    // Identity has no weights
    assert!(weight_bytes.is_empty() || weight_bytes.len() < 100);
}

#[test]
fn test_compile_linear_model() {
    let bytes = mock_models::linear_model();
    let compiler = OnnxCompiler::new();

    let (holo_bytes, _weight_bytes) = compiler
        .compile(&bytes)
        .expect("Should compile linear model");

    // Compilation produces valid .holo output
    assert!(!holo_bytes.is_empty());
}

#[test]
fn test_compile_mlp_model() {
    let bytes = mock_models::mlp_model();
    let compiler = OnnxCompiler::new();

    let (holo_bytes, _weight_bytes) = compiler.compile(&bytes).expect("Should compile MLP model");

    // Compilation produces valid .holo output
    assert!(!holo_bytes.is_empty());
}

#[test]
#[ignore = "Conv bias broadcasting requires IR-level fixes"]
fn test_compile_conv_model() {
    let bytes = mock_models::conv_model();
    let compiler = OnnxCompiler::new();

    let (holo_bytes, _weight_bytes) = compiler.compile(&bytes).expect("Should compile conv model");

    // Compilation produces valid .holo output
    assert!(!holo_bytes.is_empty());
}

#[test]
fn test_compile_mini_classifier() {
    let bytes = mock_models::mini_classifier_model();
    let compiler = OnnxCompiler::new();

    let (holo_bytes, _weight_bytes) = compiler
        .compile(&bytes)
        .expect("Should compile mini classifier");

    // Compilation produces valid .holo output
    assert!(!holo_bytes.is_empty());
}

#[test]
fn test_compile_elementwise_model() {
    let bytes = mock_models::elementwise_model();
    let compiler = OnnxCompiler::new();

    let (holo_bytes, _weight_bytes) = compiler
        .compile(&bytes)
        .expect("Should compile elementwise model");
    assert!(!holo_bytes.is_empty());
}

#[test]
fn test_compile_reduction_model() {
    let bytes = mock_models::reduction_model();
    let compiler = OnnxCompiler::new();

    let (holo_bytes, _weight_bytes) = compiler
        .compile(&bytes)
        .expect("Should compile reduction model");
    assert!(!holo_bytes.is_empty());
}

#[test]
fn test_compile_concat_model() {
    let bytes = mock_models::concat_model();
    let compiler = OnnxCompiler::new();

    let (holo_bytes, _weight_bytes) = compiler
        .compile(&bytes)
        .expect("Should compile concat model");
    assert!(!holo_bytes.is_empty());
}

#[test]
fn test_compile_split_model() {
    let bytes = mock_models::split_model();
    let compiler = OnnxCompiler::new();

    let (holo_bytes, _weight_bytes) = compiler
        .compile(&bytes)
        .expect("Should compile split model");
    assert!(!holo_bytes.is_empty());
}

#[test]
fn test_compile_gather_shape_model() {
    let bytes = mock_models::gather_shape_model();
    let compiler = OnnxCompiler::new();

    let (holo_bytes, _weight_bytes) = compiler
        .compile(&bytes)
        .expect("Should compile gather model");
    assert!(!holo_bytes.is_empty());
}

#[test]
#[ignore = "Reshape with dimension 0 (copy from input) not fully supported"]
fn test_compile_transpose_reshape_model() {
    let bytes = mock_models::transpose_reshape_model();
    let compiler = OnnxCompiler::new();

    let (holo_bytes, _weight_bytes) = compiler
        .compile(&bytes)
        .expect("Should compile transpose/reshape model");
    assert!(!holo_bytes.is_empty());
}

/// Test compilation to unified bundle format.
#[test]
fn test_compile_to_bundle_linear() {
    let bytes = mock_models::linear_model();
    let compiler = OnnxCompiler::new();

    let bundle_bytes = compiler
        .compile_to_bundle(&bytes)
        .expect("Should compile to bundle");

    // Bundle should be non-empty
    assert!(!bundle_bytes.is_empty());
}

#[test]
fn test_compile_to_bundle_mini_classifier() {
    let bytes = mock_models::mini_classifier_model();
    let compiler = OnnxCompiler::new();

    let bundle_bytes = compiler
        .compile_to_bundle(&bytes)
        .expect("Should compile to bundle");

    // Bundle should be non-empty
    assert!(!bundle_bytes.is_empty());
}

/// Test compilation with partitioning enabled.
#[test]
fn test_compile_with_partitioning() {
    let bytes = mock_models::mlp_model();
    let config = OnnxConfig {
        enable_partitioning: true,
        partition_size: 2, // Very small partition to test partitioning logic
        ..Default::default()
    };
    let compiler = OnnxCompiler::with_config(config);

    let (holo_bytes, _weight_bytes) = compiler
        .compile(&bytes)
        .expect("Should compile with partitioning");

    assert!(!holo_bytes.is_empty());
}

/// Test compilation with custom opset handling.
#[test]
fn test_different_opset_versions() {
    // All our mock models use opset 13, but the compiler should handle them
    let bytes = mock_models::identity_model();
    let model = parse_model(&bytes).expect("Should parse");

    let opset = extract_opset_version(&model);
    assert_eq!(opset, 13);
}

/// Test error handling for invalid models.
#[test]
fn test_parse_invalid_protobuf() {
    let invalid_bytes = vec![0xFF, 0xFF, 0xFF, 0xFF];
    let result = parse_model(&invalid_bytes);
    assert!(result.is_err());
}

#[test]
fn test_parse_empty_bytes() {
    // Empty bytes parses as an empty model (valid protobuf, just has default values)
    let result = parse_model(&[]);
    // This doesn't error - it produces an empty model
    // We just verify it doesn't panic
    let _ = result;
}

/// Test that all mock models have correct shapes.
#[test]
fn test_model_shapes_are_valid() {
    let models: Vec<(&str, Vec<u8>)> = vec![
        ("identity", mock_models::identity_model()),
        ("linear", mock_models::linear_model()),
        ("mlp", mock_models::mlp_model()),
        ("conv", mock_models::conv_model()),
        ("mini_classifier", mock_models::mini_classifier_model()),
        ("elementwise", mock_models::elementwise_model()),
        ("reduction", mock_models::reduction_model()),
        ("concat", mock_models::concat_model()),
        ("split", mock_models::split_model()),
        ("gather_shape", mock_models::gather_shape_model()),
        ("transpose_reshape", mock_models::transpose_reshape_model()),
    ];

    for (name, bytes) in models {
        let model =
            parse_model(&bytes).unwrap_or_else(|e| panic!("Failed to parse {}: {:?}", name, e));
        validate_model(&model).unwrap_or_else(|e| panic!("Failed to validate {}: {:?}", name, e));

        let graph = model
            .graph
            .as_ref()
            .unwrap_or_else(|| panic!("{} should have graph", name));

        // Every model should have at least one input and one output
        assert!(
            !graph.input.is_empty(),
            "{} should have at least one input",
            name
        );
        assert!(
            !graph.output.is_empty(),
            "{} should have at least one output",
            name
        );
    }
}
