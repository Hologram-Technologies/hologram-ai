//! Integration tests for the interpreter.

use hologram_onnx_core::{DimSpec, Interpreter, load_holo_file};
use std::path::Path;

/// Test loading a .holo model
#[test]
fn test_load_holo_model() {
    let holo_path = Path::new("/tmp/test-mnist.holo");

    // Skip if the compiled model doesn't exist
    if !holo_path.exists() {
        eprintln!("Skipping test: /tmp/test-mnist.holo not found");
        return;
    }

    let model = load_holo_file(holo_path).expect("Failed to load model");

    println!("Loaded model: {}", model.metadata.name);
    println!("Nodes: {}", model.graph.nodes.len());
    println!("Weights: {}", model.weight_entries.len());

    // Verify model structure
    assert_eq!(model.metadata.name, "CNTKGraph");
    assert_eq!(model.graph.nodes.len(), 31);
    assert_eq!(model.weight_entries.len(), 8);

    // Verify weights are loadable
    for entry in &model.weight_entries {
        let weight = model.get_weight(entry.id);
        assert!(weight.is_some(), "Weight {} not accessible", entry.id);
        assert_eq!(
            weight.unwrap().len(),
            entry.size,
            "Weight {} size mismatch",
            entry.id
        );
    }
}

/// Test interpreter creation and input setting
#[test]
fn test_interpreter_setup() {
    let holo_path = Path::new("/tmp/test-mnist.holo");

    if !holo_path.exists() {
        eprintln!("Skipping test: /tmp/test-mnist.holo not found");
        return;
    }

    let model = load_holo_file(holo_path).expect("Failed to load model");
    let mut interpreter = Interpreter::new(&model).expect("Failed to create interpreter");

    // Create test input (28x28 grayscale image)
    let input_data = vec![0.5f32; 28 * 28];

    // Set input by name (Input3)
    interpreter
        .set_input("Input3", &input_data)
        .expect("Failed to set input");

    // Input was set successfully
    println!("Interpreter set up successfully with input shape [1, 1, 28, 28]");
}

/// Test basic interpreter execution with a simple synthetic model
/// Note: The MNIST model has Im2Col kernel size mismatches from translation,
/// so we test with a simpler synthetic model here.
#[test]
fn test_simple_interpreter_execution() {
    use hologram_onnx_core::serialization::{
        FORMAT_VERSION, HoloHeader, HoloMetadata, HoloModel, InputSpec, OutputSpec, SerGraph,
        SerNode, SerNodeKind, WeightEntry,
    };

    // Create a minimal synthetic model: input -> add(input, weight) -> output
    let weight_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let model = HoloModel {
        header: HoloHeader {
            version: FORMAT_VERSION,
            flags: 0,
            metadata_offset: 40,
            graph_offset: 40,
            weights_offset: 0,
        },
        metadata: HoloMetadata {
            name: "test_add".to_string(),
            inputs: vec![InputSpec {
                name: "x".to_string(),
                dtype: "f32".to_string(),
                shape: vec![DimSpec::Concrete(2), DimSpec::Concrete(2)],
            }],
            outputs: vec![OutputSpec {
                node_id: 2,
                dtype: "f32".to_string(),
                shape: vec![DimSpec::Concrete(2), DimSpec::Concrete(2)],
            }],
            embedded_weight_size: 16,
            external_weight_size: 0,
            node_count: 3,
        },
        graph: SerGraph {
            nodes: vec![
                SerNode {
                    id: 0,
                    node: SerNodeKind::Input {
                        name: "x".to_string(),
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2), DimSpec::Concrete(2)]),
                },
                SerNode {
                    id: 1,
                    node: SerNodeKind::Constant { weight_id: 0 },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2), DimSpec::Concrete(2)]),
                },
                SerNode {
                    id: 2,
                    node: SerNodeKind::BinaryOp {
                        op: "add".to_string(),
                        lhs: 0,
                        rhs: 1,
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2), DimSpec::Concrete(2)]),
                },
            ],
            outputs: vec![2],
        },
        weight_entries: vec![WeightEntry {
            id: 0,
            name: "weight".to_string(),
            shape: vec![2, 2],
            dtype: "f32".to_string(),
            offset: 0,
            size: 16,
            external: false,
        }],
        embedded_weights: weight_data,
        external_weights: vec![],
    };

    let mut interpreter = Interpreter::new(&model).expect("Failed to create interpreter");

    // Input: [0, 0, 0, 0]
    let input = vec![0.0f32, 0.0, 0.0, 0.0];
    interpreter
        .set_input("x", &input)
        .expect("Failed to set input");

    interpreter.run().expect("Failed to run");

    let output = interpreter.get_output(0).expect("Failed to get output");
    let result = output.to_vec();

    // Expected: [0+1, 0+2, 0+3, 0+4] = [1, 2, 3, 4]
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    println!("Simple add test passed: {:?}", result);
}

/// Test dynamic reshape via Call(onnx.Reshape).
#[test]
fn test_dynamic_reshape_call_execution() {
    use hologram_onnx_core::serialization::{
        FORMAT_VERSION, HoloHeader, HoloMetadata, HoloModel, InputSpec, OutputSpec, SerGraph,
        SerNode, SerNodeKind, WeightEntry,
    };

    let shape_bytes: Vec<u8> = [3i64, 2]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let model = HoloModel {
        header: HoloHeader {
            version: FORMAT_VERSION,
            flags: 0,
            metadata_offset: 40,
            graph_offset: 40,
            weights_offset: 0,
        },
        metadata: HoloMetadata {
            name: "test_dynamic_reshape".to_string(),
            inputs: vec![InputSpec {
                name: "x".to_string(),
                dtype: "f32".to_string(),
                shape: vec![DimSpec::Concrete(2), DimSpec::Concrete(3)],
            }],
            outputs: vec![OutputSpec {
                node_id: 2,
                dtype: "f32".to_string(),
                shape: vec![DimSpec::Concrete(3), DimSpec::Concrete(2)],
            }],
            embedded_weight_size: shape_bytes.len() as u64,
            external_weight_size: 0,
            node_count: 3,
        },
        graph: SerGraph {
            nodes: vec![
                SerNode {
                    id: 0,
                    node: SerNodeKind::Input {
                        name: "x".to_string(),
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2), DimSpec::Concrete(3)]),
                },
                SerNode {
                    id: 1,
                    node: SerNodeKind::Constant { weight_id: 0 },
                    dtype: Some("i64".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2)]),
                },
                SerNode {
                    id: 2,
                    node: SerNodeKind::Call {
                        func: "onnx.Reshape".to_string(),
                        args: vec![0, 1],
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(3), DimSpec::Concrete(2)]),
                },
            ],
            outputs: vec![2],
        },
        weight_entries: vec![WeightEntry {
            id: 0,
            name: "shape".to_string(),
            shape: vec![2],
            dtype: "i64".to_string(),
            offset: 0,
            size: shape_bytes.len(),
            external: false,
        }],
        embedded_weights: shape_bytes,
        external_weights: vec![],
    };

    let mut interpreter = Interpreter::new(&model).expect("Failed to create interpreter");
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    interpreter
        .set_input("x", &input)
        .expect("Failed to set input");

    interpreter.run().expect("Failed to run");

    let output = interpreter.get_output(0).expect("Failed to get output");
    let result = output.to_vec();
    assert_eq!(result, input);
}

/// Test matmul operation
#[test]
fn test_matmul_execution() {
    use hologram_onnx_core::serialization::{
        FORMAT_VERSION, HoloHeader, HoloMetadata, HoloModel, InputSpec, OutputSpec, SerGraph,
        SerNode, SerNodeKind, WeightEntry,
    };

    // Create model: input [2,3] @ weight [3,2] -> output [2,2]
    // Weight: [[1,2], [3,4], [5,6]]
    let weight_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let model = HoloModel {
        header: HoloHeader {
            version: FORMAT_VERSION,
            flags: 0,
            metadata_offset: 40,
            graph_offset: 40,
            weights_offset: 0,
        },
        metadata: HoloMetadata {
            name: "test_matmul".to_string(),
            inputs: vec![InputSpec {
                name: "x".to_string(),
                dtype: "f32".to_string(),
                shape: vec![DimSpec::Concrete(2), DimSpec::Concrete(3)],
            }],
            outputs: vec![OutputSpec {
                node_id: 2,
                dtype: "f32".to_string(),
                shape: vec![DimSpec::Concrete(2), DimSpec::Concrete(2)],
            }],
            embedded_weight_size: 24,
            external_weight_size: 0,
            node_count: 3,
        },
        graph: SerGraph {
            nodes: vec![
                SerNode {
                    id: 0,
                    node: SerNodeKind::Input {
                        name: "x".to_string(),
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2), DimSpec::Concrete(3)]),
                },
                SerNode {
                    id: 1,
                    node: SerNodeKind::Constant { weight_id: 0 },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(3), DimSpec::Concrete(2)]),
                },
                SerNode {
                    id: 2,
                    node: SerNodeKind::MatMul { lhs: 0, rhs: 1 },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2), DimSpec::Concrete(2)]),
                },
            ],
            outputs: vec![2],
        },
        weight_entries: vec![WeightEntry {
            id: 0,
            name: "weight".to_string(),
            shape: vec![3, 2],
            dtype: "f32".to_string(),
            offset: 0,
            size: 24,
            external: false,
        }],
        embedded_weights: weight_data,
        external_weights: vec![],
    };

    let mut interpreter = Interpreter::new(&model).expect("Failed to create interpreter");

    // Input: [[1,0,0], [0,1,0]] (identity-ish pattern)
    let input = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    interpreter
        .set_input("x", &input)
        .expect("Failed to set input");

    interpreter.run().expect("Failed to run");

    let output = interpreter.get_output(0).expect("Failed to get output");
    let result = output.to_vec();

    // Expected:
    // [[1,0,0], [0,1,0]] @ [[1,2], [3,4], [5,6]]
    // = [[1*1+0*3+0*5, 1*2+0*4+0*6], [0*1+1*3+0*5, 0*2+1*4+0*6]]
    // = [[1, 2], [3, 4]]
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    println!("MatMul test passed: {:?}", result);
}

#[test]
fn test_call_shape() {
    use hologram_onnx_core::serialization::{
        FORMAT_VERSION, HoloHeader, HoloMetadata, HoloModel, InputSpec, OutputSpec, SerGraph,
        SerNode, SerNodeKind,
    };

    let model = HoloModel {
        header: HoloHeader {
            version: FORMAT_VERSION,
            flags: 0,
            metadata_offset: 40,
            graph_offset: 40,
            weights_offset: 0,
        },
        metadata: HoloMetadata {
            name: "test_shape".to_string(),
            inputs: vec![InputSpec {
                name: "x".to_string(),
                dtype: "f32".to_string(),
                shape: vec![
                    DimSpec::Concrete(2),
                    DimSpec::Concrete(3),
                    DimSpec::Concrete(4),
                ],
            }],
            outputs: vec![OutputSpec {
                node_id: 1,
                dtype: "f32".to_string(),
                shape: vec![DimSpec::Concrete(3)],
            }],
            embedded_weight_size: 0,
            external_weight_size: 0,
            node_count: 2,
        },
        graph: SerGraph {
            nodes: vec![
                SerNode {
                    id: 0,
                    node: SerNodeKind::Input {
                        name: "x".to_string(),
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![
                        DimSpec::Concrete(2),
                        DimSpec::Concrete(3),
                        DimSpec::Concrete(4),
                    ]),
                },
                SerNode {
                    id: 1,
                    node: SerNodeKind::Call {
                        func: "onnx.Shape".to_string(),
                        args: vec![0],
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(3)]),
                },
            ],
            outputs: vec![1],
        },
        weight_entries: vec![],
        embedded_weights: vec![],
        external_weights: vec![],
    };

    let mut interpreter = Interpreter::new(&model).expect("Failed to create interpreter");
    let input = vec![0.0f32; 2 * 3 * 4];
    interpreter
        .set_input("x", &input)
        .expect("Failed to set input");
    interpreter.run().expect("Failed to run");

    let output = interpreter.get_output(0).expect("Failed to get output");
    assert_eq!(output.to_vec(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_call_constant_of_shape() {
    use hologram_onnx_core::serialization::{
        FORMAT_VERSION, HoloHeader, HoloMetadata, HoloModel, InputSpec, OutputSpec, SerGraph,
        SerNode, SerNodeKind, WeightEntry,
    };

    let shape_data: Vec<u8> = [2.0f32, 3.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let model = HoloModel {
        header: HoloHeader {
            version: FORMAT_VERSION,
            flags: 0,
            metadata_offset: 40,
            graph_offset: 40,
            weights_offset: 0,
        },
        metadata: HoloMetadata {
            name: "test_constant_of_shape".to_string(),
            inputs: vec![InputSpec {
                name: "shape".to_string(),
                dtype: "f32".to_string(),
                shape: vec![DimSpec::Concrete(2)],
            }],
            outputs: vec![OutputSpec {
                node_id: 2,
                dtype: "f32".to_string(),
                shape: vec![DimSpec::Concrete(2), DimSpec::Concrete(3)],
            }],
            embedded_weight_size: 8,
            external_weight_size: 0,
            node_count: 3,
        },
        graph: SerGraph {
            nodes: vec![
                SerNode {
                    id: 0,
                    node: SerNodeKind::Constant { weight_id: 0 },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2)]),
                },
                SerNode {
                    id: 1,
                    node: SerNodeKind::Input {
                        name: "shape".to_string(),
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2)]),
                },
                SerNode {
                    id: 2,
                    node: SerNodeKind::Call {
                        func: "onnx.ConstantOfShape".to_string(),
                        args: vec![1],
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2), DimSpec::Concrete(3)]),
                },
            ],
            outputs: vec![2],
        },
        weight_entries: vec![WeightEntry {
            id: 0,
            name: "shape".to_string(),
            shape: vec![2],
            dtype: "f32".to_string(),
            offset: 0,
            size: 8,
            external: false,
        }],
        embedded_weights: shape_data.clone(),
        external_weights: vec![],
    };

    let mut interpreter = Interpreter::new(&model).expect("Failed to create interpreter");
    interpreter
        .set_input("shape", &[2.0, 3.0])
        .expect("Failed to set input");
    interpreter.run().expect("Failed to run");

    let output = interpreter.get_output(0).expect("Failed to get output");
    assert_eq!(output.shape(), &[2, 3]);
    assert!(output.to_vec().iter().all(|&v| v == 0.0));
}

#[test]
fn test_call_group_normalization() {
    use hologram_onnx_core::serialization::{
        FORMAT_VERSION, HoloHeader, HoloMetadata, HoloModel, InputSpec, OutputSpec, SerGraph,
        SerNode, SerNodeKind, WeightEntry,
    };

    let scale_data: Vec<u8> = [1.0f32, 1.0, 1.0, 1.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let bias_data: Vec<u8> = [0.0f32, 0.0, 0.0, 0.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let model = HoloModel {
        header: HoloHeader {
            version: FORMAT_VERSION,
            flags: 0,
            metadata_offset: 40,
            graph_offset: 40,
            weights_offset: 0,
        },
        metadata: HoloMetadata {
            name: "test_group_norm".to_string(),
            inputs: vec![InputSpec {
                name: "x".to_string(),
                dtype: "f32".to_string(),
                shape: vec![
                    DimSpec::Concrete(1),
                    DimSpec::Concrete(4),
                    DimSpec::Concrete(2),
                    DimSpec::Concrete(2),
                ],
            }],
            outputs: vec![OutputSpec {
                node_id: 5,
                dtype: "f32".to_string(),
                shape: vec![
                    DimSpec::Concrete(1),
                    DimSpec::Concrete(4),
                    DimSpec::Concrete(2),
                    DimSpec::Concrete(2),
                ],
            }],
            embedded_weight_size: 32,
            external_weight_size: 0,
            node_count: 6,
        },
        graph: SerGraph {
            nodes: vec![
                SerNode {
                    id: 0,
                    node: SerNodeKind::Input {
                        name: "x".to_string(),
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![
                        DimSpec::Concrete(1),
                        DimSpec::Concrete(4),
                        DimSpec::Concrete(2),
                        DimSpec::Concrete(2),
                    ]),
                },
                SerNode {
                    id: 1,
                    node: SerNodeKind::Constant { weight_id: 0 },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(4)]),
                },
                SerNode {
                    id: 2,
                    node: SerNodeKind::Constant { weight_id: 1 },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(4)]),
                },
                SerNode {
                    id: 3,
                    node: SerNodeKind::ScalarConst { value: 2.0 },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![]),
                },
                SerNode {
                    id: 4,
                    node: SerNodeKind::ScalarConst { value: 1e-5 },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![]),
                },
                SerNode {
                    id: 5,
                    node: SerNodeKind::Call {
                        func: "onnx.GroupNormalization".to_string(),
                        args: vec![0, 1, 2, 3, 4],
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![
                        DimSpec::Concrete(1),
                        DimSpec::Concrete(4),
                        DimSpec::Concrete(2),
                        DimSpec::Concrete(2),
                    ]),
                },
            ],
            outputs: vec![5],
        },
        weight_entries: vec![
            WeightEntry {
                id: 0,
                name: "scale".to_string(),
                shape: vec![4],
                dtype: "f32".to_string(),
                offset: 0,
                size: 16,
                external: false,
            },
            WeightEntry {
                id: 1,
                name: "bias".to_string(),
                shape: vec![4],
                dtype: "f32".to_string(),
                offset: 16,
                size: 16,
                external: false,
            },
        ],
        embedded_weights: {
            let mut data = Vec::new();
            data.extend(scale_data);
            data.extend(bias_data);
            data
        },
        external_weights: vec![],
    };

    let mut interpreter = Interpreter::new(&model).expect("Failed to create interpreter");
    let input = vec![1.0f32; 4 * 2 * 2];
    interpreter
        .set_input("x", &input)
        .expect("Failed to set input");
    interpreter.run().expect("Failed to run");

    let output = interpreter.get_output(0).expect("Failed to get output");
    assert_eq!(output.shape(), &[1, 4, 2, 2]);
    assert!(output.to_vec().iter().all(|&v| v.abs() < 1e-6));
}

#[test]
fn test_static_reshape_symbolic_flatten() {
    use hologram_onnx_core::serialization::{
        FORMAT_VERSION, HoloHeader, HoloMetadata, HoloModel, InputSpec, OutputSpec, SerGraph,
        SerNode, SerNodeKind,
    };

    let model = HoloModel {
        header: HoloHeader {
            version: FORMAT_VERSION,
            flags: 0,
            metadata_offset: 40,
            graph_offset: 40,
            weights_offset: 0,
        },
        metadata: HoloMetadata {
            name: "test_reshape_flatten".to_string(),
            inputs: vec![InputSpec {
                name: "x".to_string(),
                dtype: "f32".to_string(),
                shape: vec![
                    DimSpec::Concrete(2),
                    DimSpec::Concrete(3),
                    DimSpec::Concrete(4),
                ],
            }],
            outputs: vec![OutputSpec {
                node_id: 1,
                dtype: "f32".to_string(),
                shape: vec![DimSpec::Concrete(2), DimSpec::Concrete(12)],
            }],
            embedded_weight_size: 0,
            external_weight_size: 0,
            node_count: 2,
        },
        graph: SerGraph {
            nodes: vec![
                SerNode {
                    id: 0,
                    node: SerNodeKind::Input {
                        name: "x".to_string(),
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![
                        DimSpec::Concrete(2),
                        DimSpec::Concrete(3),
                        DimSpec::Concrete(4),
                    ]),
                },
                SerNode {
                    id: 1,
                    node: SerNodeKind::Reshape {
                        input: 0,
                        shape: vec![
                            DimSpec::Symbolic("batch".to_string()),
                            DimSpec::Symbolic("flatten_features".to_string()),
                        ],
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2), DimSpec::Concrete(12)]),
                },
            ],
            outputs: vec![1],
        },
        weight_entries: vec![],
        embedded_weights: vec![],
        external_weights: vec![],
    };

    let mut interpreter = Interpreter::new(&model).expect("Failed to create interpreter");
    let input = vec![1.0f32; 2 * 3 * 4];
    interpreter
        .set_input("x", &input)
        .expect("Failed to set input");
    interpreter.run().expect("Failed to run");

    let output = interpreter.get_output(0).expect("Failed to get output");
    assert_eq!(output.shape(), &[2, 12]);
}

#[test]
fn test_call_reshape() {
    use hologram_onnx_core::serialization::{
        FORMAT_VERSION, HoloHeader, HoloMetadata, HoloModel, InputSpec, OutputSpec, SerGraph,
        SerNode, SerNodeKind, WeightEntry,
    };

    let shape_data: Vec<u8> = [-1.0f32, 12.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let model = HoloModel {
        header: HoloHeader {
            version: FORMAT_VERSION,
            flags: 0,
            metadata_offset: 40,
            graph_offset: 40,
            weights_offset: 0,
        },
        metadata: HoloMetadata {
            name: "test_call_reshape".to_string(),
            inputs: vec![InputSpec {
                name: "x".to_string(),
                dtype: "f32".to_string(),
                shape: vec![
                    DimSpec::Concrete(2),
                    DimSpec::Concrete(3),
                    DimSpec::Concrete(4),
                ],
            }],
            outputs: vec![OutputSpec {
                node_id: 2,
                dtype: "f32".to_string(),
                shape: vec![DimSpec::Concrete(2), DimSpec::Concrete(12)],
            }],
            embedded_weight_size: 8,
            external_weight_size: 0,
            node_count: 3,
        },
        graph: SerGraph {
            nodes: vec![
                SerNode {
                    id: 0,
                    node: SerNodeKind::Input {
                        name: "x".to_string(),
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![
                        DimSpec::Concrete(2),
                        DimSpec::Concrete(3),
                        DimSpec::Concrete(4),
                    ]),
                },
                SerNode {
                    id: 1,
                    node: SerNodeKind::Constant { weight_id: 0 },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2)]),
                },
                SerNode {
                    id: 2,
                    node: SerNodeKind::Call {
                        func: "onnx.Reshape".to_string(),
                        args: vec![0, 1],
                    },
                    dtype: Some("f32".to_string()),
                    shape: Some(vec![DimSpec::Concrete(2), DimSpec::Concrete(12)]),
                },
            ],
            outputs: vec![2],
        },
        weight_entries: vec![WeightEntry {
            id: 0,
            name: "shape".to_string(),
            shape: vec![2],
            dtype: "f32".to_string(),
            offset: 0,
            size: 8,
            external: false,
        }],
        embedded_weights: shape_data,
        external_weights: vec![],
    };

    let mut interpreter = Interpreter::new(&model).expect("Failed to create interpreter");
    let input = vec![1.0f32; 2 * 3 * 4];
    interpreter
        .set_input("x", &input)
        .expect("Failed to set input");
    interpreter.run().expect("Failed to run");

    let output = interpreter.get_output(0).expect("Failed to get output");
    assert_eq!(output.shape(), &[2, 12]);
}
