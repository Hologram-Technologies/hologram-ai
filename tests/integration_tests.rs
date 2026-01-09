//! Integration tests for hologram-onnx-core.
//!
//! These tests verify the complete ONNX compilation pipeline from parsing
//! through weight extraction and validation.

use hologram_onnx::{
    OnnxConfig, OnnxError, SymbolicShape, WeightData, extract_opset_version, parse_model,
    validate_model,
};
use hologram_onnx::proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto, TypeProto,
    ValueInfoProto,
};
use prost::Message;
use tempfile::NamedTempFile;

/// Helper to create a ValueInfoProto with a tensor type.
fn make_value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
    use hologram_onnx::proto::tensor_shape_proto::Dimension;
    use hologram_onnx::proto::type_proto::Value;

    let shape_dims: Vec<Dimension> = dims
        .iter()
        .map(|&d| Dimension {
            value: Some(hologram_onnx::proto::tensor_shape_proto::dimension::Value::DimValue(d)),
            ..Default::default()
        })
        .collect();

    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(Value::TensorType(hologram_onnx::proto::type_proto::Tensor {
                elem_type: 1, // FLOAT
                shape: Some(TensorShapeProto { dim: shape_dims }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Helper to create a symbolic ValueInfoProto with a variable dimension.
fn make_symbolic_value_info(
    name: &str,
    symbolic_dim_name: &str,
    concrete_dims: &[i64],
) -> ValueInfoProto {
    use hologram_onnx::proto::tensor_shape_proto::Dimension;
    use hologram_onnx::proto::type_proto::Value;

    let mut shape_dims: Vec<Dimension> = Vec::new();

    // First dimension is symbolic (e.g., batch size)
    shape_dims.push(Dimension {
        value: Some(
            hologram_onnx::proto::tensor_shape_proto::dimension::Value::DimParam(
                symbolic_dim_name.to_string(),
            ),
        ),
        ..Default::default()
    });

    // Rest are concrete
    for &d in concrete_dims {
        shape_dims.push(Dimension {
            value: Some(hologram_onnx::proto::tensor_shape_proto::dimension::Value::DimValue(d)),
            ..Default::default()
        });
    }

    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(Value::TensorType(hologram_onnx::proto::type_proto::Tensor {
                elem_type: 1, // FLOAT
                shape: Some(TensorShapeProto { dim: shape_dims }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Helper to create a TensorProto initializer (weight).
fn make_initializer(name: &str, dims: &[i64], data: Vec<f32>) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: 1, // FLOAT
        dims: dims.to_vec(),
        float_data: data,
        ..Default::default()
    }
}

/// Helper to create an AttributeProto with an integer value.
fn make_int_attr(name: &str, value: i64) -> AttributeProto {
    use hologram_onnx::proto::attribute_proto::AttributeType;
    AttributeProto {
        name: name.to_string(),
        i: value,
        r#type: AttributeType::Int as i32,
        ..Default::default()
    }
}

/// Helper to create an AttributeProto with an integer list.
#[allow(dead_code)]
fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
    use hologram_onnx::proto::attribute_proto::AttributeType;
    AttributeProto {
        name: name.to_string(),
        ints: values,
        r#type: AttributeType::Ints as i32,
        ..Default::default()
    }
}

/// Create a minimal valid ONNX model with a single Add node.
fn create_minimal_model() -> ModelProto {
    let graph = GraphProto {
        name: "minimal_graph".to_string(),
        input: vec![make_value_info("A", &[1, 3]), make_value_info("B", &[1, 3])],
        output: vec![make_value_info("C", &[1, 3])],
        node: vec![NodeProto {
            name: "add_node".to_string(),
            op_type: "Add".to_string(),
            input: vec!["A".to_string(), "B".to_string()],
            output: vec!["C".to_string()],
            ..Default::default()
        }],
        ..Default::default()
    };

    ModelProto {
        ir_version: 9,
        opset_import: vec![hologram_onnx::proto::OperatorSetIdProto {
            domain: "".to_string(),
            version: 17,
        }],
        producer_name: "hologram-onnx-test".to_string(),
        producer_version: "1.0".to_string(),
        model_version: 1,
        graph: Some(graph),
        ..Default::default()
    }
}

/// Create a simple linear model: Y = X @ W + B
fn create_linear_model() -> ModelProto {
    // Weight matrix: 4x2 (input_dim=4, output_dim=2)
    let weight_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
    // Bias: 2
    let bias_data: Vec<f32> = vec![0.5, -0.5];

    let graph = GraphProto {
        name: "linear_graph".to_string(),
        input: vec![make_value_info("X", &[1, 4])],
        output: vec![make_value_info("Y", &[1, 2])],
        initializer: vec![
            make_initializer("W", &[4, 2], weight_data),
            make_initializer("B", &[2], bias_data),
        ],
        node: vec![
            NodeProto {
                name: "matmul_node".to_string(),
                op_type: "MatMul".to_string(),
                input: vec!["X".to_string(), "W".to_string()],
                output: vec!["XW".to_string()],
                ..Default::default()
            },
            NodeProto {
                name: "add_node".to_string(),
                op_type: "Add".to_string(),
                input: vec!["XW".to_string(), "B".to_string()],
                output: vec!["Y".to_string()],
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    ModelProto {
        ir_version: 9,
        opset_import: vec![hologram_onnx::proto::OperatorSetIdProto {
            domain: "".to_string(),
            version: 17,
        }],
        producer_name: "hologram-onnx-test".to_string(),
        producer_version: "1.0".to_string(),
        model_version: 1,
        graph: Some(graph),
        ..Default::default()
    }
}

/// Create a model with symbolic batch size for MNIST-like input.
fn create_symbolic_batch_model() -> ModelProto {
    // Symbolic batch dimension, concrete feature dimension
    let graph = GraphProto {
        name: "symbolic_batch_graph".to_string(),
        input: vec![make_symbolic_value_info("X", "batch", &[784])],
        output: vec![make_symbolic_value_info("Y", "batch", &[10])],
        initializer: vec![
            make_initializer("W", &[784, 10], vec![0.01; 784 * 10]),
            make_initializer("B", &[10], vec![0.0; 10]),
        ],
        node: vec![
            NodeProto {
                name: "matmul_node".to_string(),
                op_type: "MatMul".to_string(),
                input: vec!["X".to_string(), "W".to_string()],
                output: vec!["XW".to_string()],
                ..Default::default()
            },
            NodeProto {
                name: "add_node".to_string(),
                op_type: "Add".to_string(),
                input: vec!["XW".to_string(), "B".to_string()],
                output: vec!["pre_softmax".to_string()],
                ..Default::default()
            },
            NodeProto {
                name: "softmax_node".to_string(),
                op_type: "Softmax".to_string(),
                input: vec!["pre_softmax".to_string()],
                output: vec!["Y".to_string()],
                attribute: vec![make_int_attr("axis", 1)],
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    ModelProto {
        ir_version: 9,
        opset_import: vec![hologram_onnx::proto::OperatorSetIdProto {
            domain: "".to_string(),
            version: 17,
        }],
        producer_name: "hologram-onnx-test".to_string(),
        producer_version: "1.0".to_string(),
        model_version: 1,
        graph: Some(graph),
        ..Default::default()
    }
}

/// Create an invalid model (missing graph).
fn create_invalid_model_no_graph() -> ModelProto {
    ModelProto {
        ir_version: 9,
        opset_import: vec![hologram_onnx::proto::OperatorSetIdProto {
            domain: "".to_string(),
            version: 17,
        }],
        producer_name: "invalid-test".to_string(),
        graph: None,
        ..Default::default()
    }
}

/// Create an invalid model (missing inputs).
fn create_invalid_model_no_inputs() -> ModelProto {
    let graph = GraphProto {
        name: "no_inputs_graph".to_string(),
        input: vec![], // Missing inputs
        output: vec![make_value_info("Y", &[1, 2])],
        node: vec![NodeProto {
            name: "relu".to_string(),
            op_type: "Relu".to_string(),
            input: vec!["X".to_string()],
            output: vec!["Y".to_string()],
            ..Default::default()
        }],
        ..Default::default()
    };

    ModelProto {
        ir_version: 9,
        opset_import: vec![hologram_onnx::proto::OperatorSetIdProto {
            domain: "".to_string(),
            version: 17,
        }],
        graph: Some(graph),
        ..Default::default()
    }
}

/// Create an invalid model (missing outputs).
fn create_invalid_model_no_outputs() -> ModelProto {
    let graph = GraphProto {
        name: "no_outputs_graph".to_string(),
        input: vec![make_value_info("X", &[1, 2])],
        output: vec![], // Missing outputs
        node: vec![NodeProto {
            name: "relu".to_string(),
            op_type: "Relu".to_string(),
            input: vec!["X".to_string()],
            output: vec!["Y".to_string()],
            ..Default::default()
        }],
        ..Default::default()
    };

    ModelProto {
        ir_version: 9,
        opset_import: vec![hologram_onnx::proto::OperatorSetIdProto {
            domain: "".to_string(),
            version: 17,
        }],
        graph: Some(graph),
        ..Default::default()
    }
}

/// Encode a ModelProto to bytes.
fn encode_model(model: &ModelProto) -> Vec<u8> {
    let mut buf = Vec::new();
    model.encode(&mut buf).expect("Failed to encode model");
    buf
}

// =============================================================================
// Test: Model Parsing
// =============================================================================

#[test]
fn test_parse_minimal_model() {
    let model = create_minimal_model();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).expect("Failed to parse model");

    assert!(parsed.graph.is_some());
    let graph = parsed.graph.as_ref().unwrap();
    assert_eq!(graph.name, "minimal_graph");
    assert_eq!(graph.input.len(), 2);
    assert_eq!(graph.output.len(), 1);
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Add");
}

#[test]
fn test_parse_linear_model() {
    let model = create_linear_model();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).expect("Failed to parse model");

    let graph = parsed.graph.as_ref().unwrap();
    assert_eq!(graph.name, "linear_graph");
    assert_eq!(graph.node.len(), 2);
    assert_eq!(graph.initializer.len(), 2);

    // Verify weight names
    let weight_names: Vec<_> = graph.initializer.iter().map(|i| &i.name).collect();
    assert!(weight_names.contains(&&"W".to_string()));
    assert!(weight_names.contains(&&"B".to_string()));
}

#[test]
fn test_parse_symbolic_batch_model() {
    let model = create_symbolic_batch_model();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).expect("Failed to parse model");

    let graph = parsed.graph.as_ref().unwrap();
    assert_eq!(graph.input.len(), 1);

    // Verify symbolic shape is preserved
    let input_type = graph.input[0].r#type.as_ref().unwrap();
    if let Some(hologram_onnx::proto::type_proto::Value::TensorType(tensor)) = &input_type.value {
        let shape = tensor.shape.as_ref().unwrap();
        assert_eq!(shape.dim.len(), 2);

        // First dim should be symbolic
        if let Some(hologram_onnx::proto::tensor_shape_proto::dimension::Value::DimParam(name)) =
            &shape.dim[0].value
        {
            assert_eq!(name, "batch");
        } else {
            panic!("Expected symbolic batch dimension");
        }
    }
}

#[test]
fn test_parse_invalid_bytes() {
    let invalid_bytes = b"not a valid protobuf";

    let result = parse_model(invalid_bytes);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), OnnxError::ParseError(_)));
}

#[test]
fn test_parse_empty_bytes() {
    let empty_bytes: &[u8] = &[];

    // Empty bytes decode to empty model (all defaults)
    let result = parse_model(empty_bytes);
    assert!(result.is_ok());
    let model = result.unwrap();
    // Model will have no graph
    assert!(model.graph.is_none());
}

// =============================================================================
// Test: Model Validation
// =============================================================================

#[test]
fn test_validate_minimal_model() {
    let model = create_minimal_model();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let result = validate_model(&parsed);

    assert!(result.is_ok());
}

#[test]
fn test_validate_linear_model() {
    let model = create_linear_model();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let result = validate_model(&parsed);

    assert!(result.is_ok());
}

#[test]
fn test_validate_symbolic_model() {
    let model = create_symbolic_batch_model();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let result = validate_model(&parsed);

    assert!(result.is_ok());
}

#[test]
fn test_validate_no_graph() {
    let model = create_invalid_model_no_graph();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let result = validate_model(&parsed);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
}

#[test]
fn test_validate_no_inputs() {
    let model = create_invalid_model_no_inputs();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let result = validate_model(&parsed);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
}

#[test]
fn test_validate_no_outputs() {
    let model = create_invalid_model_no_outputs();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let result = validate_model(&parsed);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
}

// =============================================================================
// Test: Opset Version Extraction
// =============================================================================

#[test]
fn test_extract_opset_version() {
    let model = create_minimal_model();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let version = extract_opset_version(&parsed);

    assert_eq!(version, 17);
}

#[test]
fn test_extract_opset_version_no_imports() {
    let model = ModelProto {
        ir_version: 9,
        opset_import: vec![], // No opset imports
        graph: Some(GraphProto::default()),
        ..Default::default()
    };
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let version = extract_opset_version(&parsed);

    // Should return default version (e.g., 1 or 0) when no imports
    assert!(version <= 1);
}

// =============================================================================
// Test: Weight Extraction
// =============================================================================

#[test]
fn test_weight_extraction_from_model() {
    let model = create_linear_model();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let graph = parsed.graph.as_ref().unwrap();

    let mut weights = WeightData::new();
    for init in &graph.initializer {
        let data = WeightData::extract_tensor_data(init).expect("Failed to extract tensor");
        weights.add_weight(&init.name, data);
    }

    // Should have 2 weights: W and B
    assert_eq!(weights.len(), 2);

    // W is 4x2 = 8 floats = 32 bytes
    // B is 2 floats = 8 bytes
    // Total = 40 bytes
    assert_eq!(weights.buffer_size(), 40);

    // Verify refs exist
    assert!(weights.get_ref("W").is_some());
    assert!(weights.get_ref("B").is_some());
}

#[test]
fn test_weight_file_writing() {
    let model = create_linear_model();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let graph = parsed.graph.as_ref().unwrap();

    let mut weights = WeightData::new();
    for init in &graph.initializer {
        let data = WeightData::extract_tensor_data(init).expect("Failed to extract tensor");
        weights.add_weight(&init.name, data);
    }

    // Write to temp file
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    weights
        .write_to_file(temp_file.path())
        .expect("Failed to write weights");

    // Verify file size
    let metadata = std::fs::metadata(temp_file.path()).expect("Failed to get metadata");
    assert_eq!(metadata.len(), 40);

    // Verify file content can be read back
    let content = std::fs::read(temp_file.path()).expect("Failed to read file");
    assert_eq!(content.len(), 40);

    // Verify it's valid f32 data
    let floats: &[f32] = bytemuck::cast_slice(&content);
    assert_eq!(floats.len(), 10); // 8 from W + 2 from B
}

#[test]
fn test_weight_deduplication_in_model() {
    // Create model with duplicate weights
    let graph = GraphProto {
        name: "dedup_test".to_string(),
        input: vec![make_value_info("X", &[1, 4])],
        output: vec![make_value_info("Y", &[1, 4])],
        initializer: vec![
            make_initializer("W1", &[4], vec![1.0, 2.0, 3.0, 4.0]),
            make_initializer("W2", &[4], vec![1.0, 2.0, 3.0, 4.0]), // Duplicate!
        ],
        node: vec![
            NodeProto {
                name: "add1".to_string(),
                op_type: "Add".to_string(),
                input: vec!["X".to_string(), "W1".to_string()],
                output: vec!["Y1".to_string()],
                ..Default::default()
            },
            NodeProto {
                name: "add2".to_string(),
                op_type: "Add".to_string(),
                input: vec!["Y1".to_string(), "W2".to_string()],
                output: vec!["Y".to_string()],
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    let model = ModelProto {
        ir_version: 9,
        graph: Some(graph),
        ..Default::default()
    };
    let bytes = encode_model(&model);
    let parsed = parse_model(&bytes).unwrap();
    let graph = parsed.graph.as_ref().unwrap();

    let mut weights = WeightData::new();
    for init in &graph.initializer {
        let data = WeightData::extract_tensor_data(init).expect("Failed to extract tensor");
        weights.add_weight(&init.name, data);
    }

    // Should have 2 named weights but only 16 bytes (4 floats) due to deduplication
    assert_eq!(weights.len(), 2);
    assert_eq!(weights.buffer_size(), 16); // Not 32!

    // Both refs should point to same offset
    let ref1 = weights.get_ref("W1").unwrap();
    let ref2 = weights.get_ref("W2").unwrap();
    assert_eq!(ref1.offset, ref2.offset);
}

// =============================================================================
// Test: Symbolic Shape Parsing
// =============================================================================

#[test]
fn test_symbolic_shape_from_model() {
    let model = create_symbolic_batch_model();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let graph = parsed.graph.as_ref().unwrap();

    // Parse input shape as SymbolicShape
    let input = &graph.input[0];
    let shape = SymbolicShape::from_value_info(input).expect("Failed to parse shape");

    // Should have 2 dimensions: batch (symbolic) and 784 (concrete)
    assert_eq!(shape.rank(), 2);
    assert!(shape.is_partially_symbolic());
    assert!(!shape.is_fully_concrete());
}

#[test]
fn test_concrete_shape_from_model() {
    let model = create_linear_model();
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let graph = parsed.graph.as_ref().unwrap();

    // Parse input shape as SymbolicShape
    let input = &graph.input[0];
    let shape = SymbolicShape::from_value_info(input).expect("Failed to parse shape");

    // Should have 2 dimensions: 1 and 4 (both concrete)
    assert_eq!(shape.rank(), 2);
    assert!(shape.is_fully_concrete());
}

// =============================================================================
// Test: Parse and Validate Integration
// =============================================================================
// Note: OnnxCompiler integration tests are in the top-level hologram-onnx crate
// since OnnxCompiler requires both core (parsing) and ops (translation).

#[test]
fn test_core_parse_and_validate() {
    let model = create_linear_model();
    let bytes = encode_model(&model);

    // Core provides parsing and validation
    let parsed = parse_model(&bytes).unwrap();
    validate_model(&parsed).unwrap();
}

#[test]
fn test_config_structure() {
    // Verify OnnxConfig can be created and accessed
    let config = OnnxConfig {
        weight_threshold: 8192,
        enable_partitioning: true,
        partition_size: 100,
        decompose_conv2d: true,
        decompose_pooling: true,
        pack_weights: true,
        memory_budget: Some(1024),
        enable_resize_upscaling: true,
    };

    assert_eq!(config.weight_threshold, 8192);
    assert!(config.enable_partitioning);
    assert_eq!(config.partition_size, 100);
    assert!(config.decompose_conv2d);
    assert!(config.decompose_pooling);
    assert!(config.pack_weights);
    assert_eq!(config.memory_budget, Some(1024));
}

#[test]
fn test_parse_validates_model_structure() {
    let model = create_minimal_model();
    let bytes = encode_model(&model);

    // Parse should succeed for valid model
    let parsed = parse_model(&bytes).unwrap();

    // Validate should succeed for well-formed model
    validate_model(&parsed).unwrap();

    // Extract opset version
    let opset = extract_opset_version(&parsed);
    assert!(opset >= 1, "Opset version should be at least 1");
}

// =============================================================================
// Test: End-to-End Pipeline (what works today)
// =============================================================================

#[test]
fn test_end_to_end_parsing_and_weight_extraction() {
    // This test verifies the complete flow that currently works:
    // 1. Parse ONNX bytes
    // 2. Validate model
    // 3. Extract opset version
    // 4. Extract weights
    // 5. Write weights to file

    let model = create_symbolic_batch_model();
    let bytes = encode_model(&model);

    // Step 1: Parse
    let parsed = parse_model(&bytes).expect("Parsing should succeed");

    // Step 2: Validate
    validate_model(&parsed).expect("Validation should succeed");

    // Step 3: Extract opset
    let opset = extract_opset_version(&parsed);
    assert_eq!(opset, 17);

    // Step 4: Extract weights
    let graph = parsed.graph.as_ref().unwrap();
    let mut weights = WeightData::new();
    for init in &graph.initializer {
        let data = WeightData::extract_tensor_data(init).expect("Weight extraction should succeed");
        weights.add_weight(&init.name, data);
    }

    // W is 784*10 = 7840 floats = 31360 bytes
    // B is 10 floats = 40 bytes
    // Total = 31400 bytes
    assert_eq!(weights.len(), 2);
    assert_eq!(weights.buffer_size(), 31400);

    // Step 5: Write to file
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    weights
        .write_to_file(temp_file.path())
        .expect("Weight writing should succeed");

    let metadata = std::fs::metadata(temp_file.path()).unwrap();
    assert_eq!(metadata.len(), 31400);
}

#[test]
fn test_holo_file_placeholder() {
    // Since the translator is a stub, we test the file writing infrastructure
    // by verifying WeightData can write valid files

    let mut weights = WeightData::new();
    weights.add_weight("test_weight", vec![1.0, 2.0, 3.0, 4.0]);

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    weights
        .write_to_file(temp_file.path())
        .expect("Should write successfully");

    // Verify file was created
    assert!(temp_file.path().exists());

    // Verify content is correct
    let content = std::fs::read(temp_file.path()).unwrap();
    let floats: &[f32] = bytemuck::cast_slice(&content);
    assert_eq!(floats, &[1.0, 2.0, 3.0, 4.0]);
}

// =============================================================================
// Test: Large Model Handling
// =============================================================================

#[test]
fn test_large_model_parsing() {
    // Create a model with many nodes to verify parsing scales
    let num_nodes = 100;
    let mut nodes = Vec::with_capacity(num_nodes);
    let mut current_input = "X".to_string();

    for i in 0..num_nodes {
        let output = if i == num_nodes - 1 {
            "Y".to_string()
        } else {
            format!("intermediate_{}", i)
        };

        nodes.push(NodeProto {
            name: format!("relu_{}", i),
            op_type: "Relu".to_string(),
            input: vec![current_input.clone()],
            output: vec![output.clone()],
            ..Default::default()
        });

        current_input = output;
    }

    let graph = GraphProto {
        name: "large_graph".to_string(),
        input: vec![make_value_info("X", &[1, 64])],
        output: vec![make_value_info("Y", &[1, 64])],
        node: nodes,
        ..Default::default()
    };

    let model = ModelProto {
        ir_version: 9,
        opset_import: vec![hologram_onnx::proto::OperatorSetIdProto {
            domain: "".to_string(),
            version: 17,
        }],
        graph: Some(graph),
        ..Default::default()
    };

    let bytes = encode_model(&model);

    // Should parse successfully
    let parsed = parse_model(&bytes).expect("Large model parsing should succeed");
    validate_model(&parsed).expect("Large model validation should succeed");

    let graph = parsed.graph.as_ref().unwrap();
    assert_eq!(graph.node.len(), num_nodes);
}

// =============================================================================
// Test: Real ONNX Models from Model Zoo
// =============================================================================

/// Path to test fixtures directory
fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Load MNIST model from fixtures
fn load_mnist_model() -> Option<Vec<u8>> {
    let path = fixtures_dir().join("mnist-12.onnx");
    if path.exists() {
        std::fs::read(&path).ok()
    } else {
        None
    }
}

#[test]
fn test_real_mnist_model_parsing() {
    let bytes = match load_mnist_model() {
        Some(b) => b,
        None => {
            eprintln!("MNIST model not found in fixtures, skipping test");
            return;
        }
    };

    // Parse the real MNIST model
    let model = parse_model(&bytes).expect("MNIST model should parse successfully");

    // Verify basic structure
    assert!(model.graph.is_some());
    let graph = model.graph.as_ref().unwrap();

    // MNIST model should have inputs and outputs
    assert!(!graph.input.is_empty(), "MNIST should have inputs");
    assert!(!graph.output.is_empty(), "MNIST should have outputs");

    // Should have multiple nodes (conv, relu, pool, etc.)
    assert!(
        graph.node.len() > 5,
        "MNIST should have multiple operations"
    );

    // Should have weights (initializers)
    assert!(!graph.initializer.is_empty(), "MNIST should have weights");
}

#[test]
fn test_real_mnist_model_validation() {
    let bytes = match load_mnist_model() {
        Some(b) => b,
        None => {
            eprintln!("MNIST model not found in fixtures, skipping test");
            return;
        }
    };

    let model = parse_model(&bytes).expect("Failed to parse model");

    // Validation should pass for a valid model from the ONNX model zoo
    validate_model(&model).expect("MNIST model validation should succeed");
}

#[test]
fn test_real_mnist_model_opset() {
    let bytes = match load_mnist_model() {
        Some(b) => b,
        None => {
            eprintln!("MNIST model not found in fixtures, skipping test");
            return;
        }
    };

    let model = parse_model(&bytes).expect("Failed to parse model");
    let opset = extract_opset_version(&model);

    // MNIST model should have a valid opset version (typically 12+)
    assert!(opset >= 8, "MNIST opset should be >= 8, got {}", opset);
}

#[test]
fn test_real_mnist_model_weight_extraction() {
    let bytes = match load_mnist_model() {
        Some(b) => b,
        None => {
            eprintln!("MNIST model not found in fixtures, skipping test");
            return;
        }
    };

    let model = parse_model(&bytes).expect("Failed to parse model");
    let graph = model.graph.as_ref().unwrap();

    // Extract all weights
    let mut weights = WeightData::new();
    let mut extracted_count = 0;
    let mut failed_count = 0;

    for init in &graph.initializer {
        match WeightData::extract_tensor_data(init) {
            Ok(data) => {
                weights.add_weight(&init.name, data);
                extracted_count += 1;
            }
            Err(e) => {
                eprintln!("Warning: Failed to extract tensor '{}': {}", init.name, e);
                failed_count += 1;
            }
        }
    }

    // Should have extracted most weights successfully
    assert!(extracted_count > 0, "Should extract at least some weights");
    assert!(
        failed_count == 0,
        "All weights should be extractable, but {} failed",
        failed_count
    );

    // Weights buffer should have non-zero size
    assert!(
        weights.buffer_size() > 0,
        "Should have extracted weight data"
    );
}

#[test]
fn test_real_mnist_model_weight_file_output() {
    let bytes = match load_mnist_model() {
        Some(b) => b,
        None => {
            eprintln!("MNIST model not found in fixtures, skipping test");
            return;
        }
    };

    let model = parse_model(&bytes).expect("Failed to parse model");
    let graph = model.graph.as_ref().unwrap();

    // Extract weights
    let mut weights = WeightData::new();
    for init in &graph.initializer {
        if let Ok(data) = WeightData::extract_tensor_data(init) {
            weights.add_weight(&init.name, data);
        }
    }

    // Write to temp file
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    weights
        .write_to_file(temp_file.path())
        .expect("Should write MNIST weights successfully");

    // Verify file was created with correct size
    let metadata = std::fs::metadata(temp_file.path()).expect("Failed to get metadata");
    assert_eq!(
        metadata.len() as usize,
        weights.buffer_size(),
        "File size should match buffer size"
    );

    // Verify content is valid float data
    let content = std::fs::read(temp_file.path()).expect("Failed to read file");
    let floats: &[f32] = bytemuck::cast_slice(&content);

    // All values should be finite (not NaN or Inf)
    for (i, &val) in floats.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Weight at index {} is not finite: {}",
            i,
            val
        );
    }
}

#[test]
fn test_real_mnist_model_shape_parsing() {
    let bytes = match load_mnist_model() {
        Some(b) => b,
        None => {
            eprintln!("MNIST model not found in fixtures, skipping test");
            return;
        }
    };

    let model = parse_model(&bytes).expect("Failed to parse model");
    let graph = model.graph.as_ref().unwrap();

    // Parse input shape
    if let Some(shape) = graph
        .input
        .first()
        .and_then(|i| SymbolicShape::from_value_info(i).ok())
    {
        // MNIST input is typically [batch, 1, 28, 28] or similar
        assert!(
            shape.rank() >= 3,
            "MNIST input should have at least 3 dimensions"
        );
    }

    // Parse output shape
    if let Some(shape) = graph
        .output
        .first()
        .and_then(|o| SymbolicShape::from_value_info(o).ok())
    {
        // MNIST output is typically [batch, 10] (10 digit classes)
        assert!(
            shape.rank() >= 1,
            "MNIST output should have at least 1 dimension"
        );
    }
}

#[test]
fn test_real_mnist_model_operation_types() {
    let bytes = match load_mnist_model() {
        Some(b) => b,
        None => {
            eprintln!("MNIST model not found in fixtures, skipping test");
            return;
        }
    };

    let model = parse_model(&bytes).expect("Failed to parse model");
    let graph = model.graph.as_ref().unwrap();

    // Collect operation types
    let op_types: std::collections::HashSet<_> =
        graph.node.iter().map(|n| n.op_type.as_str()).collect();

    // MNIST should contain common operations
    // Typical ops: Conv, Relu, MaxPool, Reshape, MatMul, Add, Softmax
    let expected_ops = ["Conv", "Relu", "MaxPool", "Reshape", "MatMul", "Add"];

    let mut found_count = 0;
    for op in &expected_ops {
        if op_types.contains(*op) {
            found_count += 1;
        }
    }

    // Should have at least some of the expected operations
    assert!(
        found_count >= 3,
        "MNIST should have at least 3 of the expected ops. Found ops: {:?}",
        op_types
    );
}

#[test]
fn test_real_mnist_full_pipeline() {
    let bytes = match load_mnist_model() {
        Some(b) => b,
        None => {
            eprintln!("MNIST model not found in fixtures, skipping test");
            return;
        }
    };

    // Full pipeline test: parse -> validate -> extract opset -> extract weights -> write files

    // Step 1: Parse
    let model = parse_model(&bytes).expect("Parse failed");

    // Step 2: Validate
    validate_model(&model).expect("Validation failed");

    // Step 3: Extract opset
    let opset = extract_opset_version(&model);
    assert!(opset >= 8, "Opset too old: {}", opset);

    // Step 4: Extract weights
    let graph = model.graph.as_ref().unwrap();
    let mut weights = WeightData::new();
    for init in &graph.initializer {
        if let Ok(data) = WeightData::extract_tensor_data(init) {
            weights.add_weight(&init.name, data);
        }
    }

    // Step 5: Write weights file
    let temp_file = NamedTempFile::new().expect("Temp file creation failed");
    weights
        .write_to_file(temp_file.path())
        .expect("Weight file writing failed");

    // Verify
    let file_size = std::fs::metadata(temp_file.path()).unwrap().len();
    assert!(file_size > 0, "Weight file should not be empty");

    eprintln!(
        "MNIST full pipeline test passed: {} nodes, {} weights, {} bytes",
        graph.node.len(),
        weights.len(),
        file_size
    );
}

// =============================================================================
// Test: ResNet Model from Model Zoo
// =============================================================================

/// Load ResNet50 model from workspace models directory
fn load_resnet_model() -> Option<Vec<u8>> {
    // ResNet is in workspace root/models directory
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("models/resnet50-v1-7.onnx"));

    path.filter(|p| p.exists())
        .and_then(|p| std::fs::read(p).ok())
}

#[test]
fn test_real_resnet_model_parsing() {
    let bytes = match load_resnet_model() {
        Some(b) => b,
        None => {
            eprintln!("ResNet model not found, skipping test");
            return;
        }
    };

    // Parse the real ResNet50 model
    let model = parse_model(&bytes).expect("ResNet model should parse successfully");

    // Verify basic structure
    assert!(model.graph.is_some());
    let graph = model.graph.as_ref().unwrap();

    // ResNet should have inputs and outputs
    assert!(!graph.input.is_empty(), "ResNet should have inputs");
    assert!(!graph.output.is_empty(), "ResNet should have outputs");

    // Should have many nodes (ResNet50 has ~120+ operations)
    assert!(
        graph.node.len() > 50,
        "ResNet should have many operations, got {}",
        graph.node.len()
    );

    // Should have many weights (initializers) - ResNet50 has ~100+ weight tensors
    assert!(
        graph.initializer.len() > 50,
        "ResNet should have many weights, got {}",
        graph.initializer.len()
    );

    eprintln!(
        "ResNet parsing: {} nodes, {} initializers",
        graph.node.len(),
        graph.initializer.len()
    );
}

#[test]
fn test_real_resnet_model_validation() {
    let bytes = match load_resnet_model() {
        Some(b) => b,
        None => {
            eprintln!("ResNet model not found, skipping test");
            return;
        }
    };

    let model = parse_model(&bytes).expect("Failed to parse model");

    // Validation should pass for a valid model from the ONNX model zoo
    validate_model(&model).expect("ResNet model validation should succeed");
}

#[test]
fn test_real_resnet_model_opset() {
    let bytes = match load_resnet_model() {
        Some(b) => b,
        None => {
            eprintln!("ResNet model not found, skipping test");
            return;
        }
    };

    let model = parse_model(&bytes).expect("Failed to parse model");
    let opset = extract_opset_version(&model);

    // ResNet model should have a valid opset version
    assert!(opset >= 7, "ResNet opset should be >= 7, got {}", opset);
    eprintln!("ResNet opset version: {}", opset);
}

#[test]
fn test_real_resnet_model_weight_extraction() {
    let bytes = match load_resnet_model() {
        Some(b) => b,
        None => {
            eprintln!("ResNet model not found, skipping test");
            return;
        }
    };

    let model = parse_model(&bytes).expect("Failed to parse model");
    let graph = model.graph.as_ref().unwrap();

    // Extract all weights
    let mut weights = WeightData::new();
    let mut extracted_count = 0;
    let mut failed_count = 0;

    for init in &graph.initializer {
        match WeightData::extract_tensor_data(init) {
            Ok(data) => {
                weights.add_weight(&init.name, data);
                extracted_count += 1;
            }
            Err(e) => {
                eprintln!("Warning: Failed to extract tensor '{}': {}", init.name, e);
                failed_count += 1;
            }
        }
    }

    // Should have extracted most weights successfully
    assert!(
        extracted_count > 50,
        "Should extract at least 50 weights, got {}",
        extracted_count
    );
    assert!(
        failed_count == 0,
        "All weights should be extractable, but {} failed",
        failed_count
    );

    // ResNet50 weights should be substantial (>90MB)
    assert!(
        weights.buffer_size() > 90_000_000,
        "ResNet weights should be >90MB, got {} bytes",
        weights.buffer_size()
    );

    eprintln!(
        "ResNet weight extraction: {} weights, {} MB",
        extracted_count,
        weights.buffer_size() / (1024 * 1024)
    );
}

#[test]
fn test_real_resnet_model_shape_parsing() {
    let bytes = match load_resnet_model() {
        Some(b) => b,
        None => {
            eprintln!("ResNet model not found, skipping test");
            return;
        }
    };

    let model = parse_model(&bytes).expect("Failed to parse model");
    let graph = model.graph.as_ref().unwrap();

    // Parse input shape
    if let Some(shape) = graph
        .input
        .first()
        .and_then(|i| SymbolicShape::from_value_info(i).ok())
    {
        // ResNet input is typically [batch, 3, 224, 224] (NCHW format)
        assert_eq!(shape.rank(), 4, "ResNet input should have 4 dimensions");
        eprintln!("ResNet input shape: {:?}", shape.dims());
    }

    // Parse output shape
    if let Some(shape) = graph
        .output
        .first()
        .and_then(|o| SymbolicShape::from_value_info(o).ok())
    {
        // ResNet output is typically [batch, 1000] (1000 ImageNet classes)
        assert!(
            shape.rank() >= 1,
            "ResNet output should have at least 1 dimension"
        );
        eprintln!("ResNet output shape: {:?}", shape.dims());
    }
}

#[test]
fn test_real_resnet_model_operation_types() {
    let bytes = match load_resnet_model() {
        Some(b) => b,
        None => {
            eprintln!("ResNet model not found, skipping test");
            return;
        }
    };

    let model = parse_model(&bytes).expect("Failed to parse model");
    let graph = model.graph.as_ref().unwrap();

    // Collect operation types with counts
    let mut op_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for node in &graph.node {
        *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
    }

    // ResNet should contain these operations
    let expected_ops = [
        "Conv",
        "Relu",
        "BatchNormalization",
        "Add",
        "GlobalAveragePool",
        "MaxPool",
    ];

    let mut found_count = 0;
    for op in &expected_ops {
        if op_counts.contains_key(*op) {
            found_count += 1;
        }
    }

    // Should have most of the expected operations
    assert!(
        found_count >= 4,
        "ResNet should have at least 4 of the expected ops. Found ops: {:?}",
        op_counts.keys().collect::<Vec<_>>()
    );

    // Print operation breakdown
    eprintln!("ResNet operation types:");
    for (op, count) in &op_counts {
        eprintln!("  {}: {}", op, count);
    }
}

#[test]
fn test_real_resnet_model_residual_connections() {
    let bytes = match load_resnet_model() {
        Some(b) => b,
        None => {
            eprintln!("ResNet model not found, skipping test");
            return;
        }
    };

    let model = parse_model(&bytes).expect("Failed to parse model");
    let graph = model.graph.as_ref().unwrap();

    // Count Add operations (residual connections)
    let add_count = graph.node.iter().filter(|n| n.op_type == "Add").count();

    // ResNet50 should have many residual connections (Add operations)
    // ResNet50 has 16 bottleneck blocks, each with an Add for the skip connection
    assert!(
        add_count >= 10,
        "ResNet should have residual Add operations, got {}",
        add_count
    );

    eprintln!("ResNet residual connections (Add ops): {}", add_count);
}

#[test]
fn test_real_resnet_full_pipeline() {
    let bytes = match load_resnet_model() {
        Some(b) => b,
        None => {
            eprintln!("ResNet model not found, skipping test");
            return;
        }
    };

    // Full pipeline test: parse -> validate -> extract opset -> extract weights -> write files

    // Step 1: Parse
    let model = parse_model(&bytes).expect("Parse failed");

    // Step 2: Validate
    validate_model(&model).expect("Validation failed");

    // Step 3: Extract opset
    let opset = extract_opset_version(&model);
    assert!(opset >= 7, "Opset too old: {}", opset);

    // Step 4: Extract weights
    let graph = model.graph.as_ref().unwrap();
    let mut weights = WeightData::new();
    for init in &graph.initializer {
        if let Ok(data) = WeightData::extract_tensor_data(init) {
            weights.add_weight(&init.name, data);
        }
    }

    // Step 5: Write weights file
    let temp_file = NamedTempFile::new().expect("Temp file creation failed");
    weights
        .write_to_file(temp_file.path())
        .expect("Weight file writing failed");

    // Verify
    let file_size = std::fs::metadata(temp_file.path()).unwrap().len();
    assert!(
        file_size > 90_000_000,
        "ResNet weight file should be >90MB, got {} bytes",
        file_size
    );

    eprintln!(
        "ResNet full pipeline test passed: {} nodes, {} weights, {} MB",
        graph.node.len(),
        weights.len(),
        file_size / (1024 * 1024)
    );
}
