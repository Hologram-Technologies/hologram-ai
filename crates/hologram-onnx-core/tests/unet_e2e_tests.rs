//! End-to-end compilation tests for UNet-style models.
//!
//! These tests verify the complete compilation pipeline works correctly
//! for encoder-decoder architectures like UNet, which have:
//! - Skip connections between encoder and decoder
//! - Multiple resolution levels
//! - Complex dependency graphs
//!
//! # Test Organization
//!
//! - **CI Tests**: Use smaller graphs (500-1000 nodes) that run quickly
//! - **Large Model Tests**: Use `#[ignore]` for 3000+ node tests, run with `--ignored`
//!
//! Run ignored tests: `cargo test -- --ignored`

use hologram_onnx_core::{
    extract_opset_version, parse_model, validate_model, GraphPartitioner,
    OnnxConfig, SymbolicShape, WeightData,
};
use hologram_onnx_spec::{
    AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto, TypeProto,
    ValueInfoProto,
};
use prost::Message;
use std::collections::HashSet;
use tempfile::NamedTempFile;

// ============================================================================
// UNet Graph Generator
// ============================================================================

/// Create a realistic UNet-style graph with configurable depth and width.
///
/// This generates a graph that closely mimics real UNet architectures:
/// - Encoder path with downsampling (MaxPool)
/// - Decoder path with upsampling (ConvTranspose)
/// - Skip connections from encoder to decoder at each level
/// - Batch normalization and ReLU after convolutions
///
/// # Node Count Formula
///
/// Approximate nodes = `depth * nodes_per_level * 6 + depth * 4`
/// - depth=4, nodes_per_level=10 → ~250 nodes
/// - depth=4, nodes_per_level=30 → ~750 nodes
/// - depth=5, nodes_per_level=50 → ~3000 nodes
fn create_unet_graph(depth: usize, nodes_per_level: usize) -> GraphProto {
    let mut graph = GraphProto::default();
    graph.name = format!("unet_d{}_n{}", depth, nodes_per_level);

    // Input: batch, channels, height, width (NCHW)
    graph.input.push(make_value_info("input", &[1, 3, 512, 512]));

    let mut encoder_skip_outputs = Vec::new();
    let mut current_tensor = "input".to_string();
    let mut current_channels: i64 = 3;

    // ===== ENCODER PATH =====
    for level in 0..depth {
        let out_channels = 64 * (1 << level.min(4)) as i64;

        for block in 0..nodes_per_level {
            let is_first = block == 0;
            let in_ch = if is_first { current_channels } else { out_channels };

            // Conv
            let conv_out = format!("enc_l{}_conv{}", level, block);
            let weight_name = format!("enc_l{}_conv{}_weight", level, block);
            graph.initializer.push(make_conv_weight(&weight_name, out_channels, in_ch, 3, 3));

            let mut conv = NodeProto::default();
            conv.name = conv_out.clone();
            conv.op_type = "Conv".to_string();
            conv.input.push(current_tensor.clone());
            conv.input.push(weight_name);
            conv.output.push(conv_out.clone());
            conv.attribute.push(make_ints_attr("kernel_shape", vec![3, 3]));
            conv.attribute.push(make_ints_attr("pads", vec![1, 1, 1, 1]));
            graph.node.push(conv);

            // BatchNorm
            let bn_out = format!("enc_l{}_bn{}", level, block);
            let bn_scale = format!("enc_l{}_bn{}_scale", level, block);
            let bn_bias = format!("enc_l{}_bn{}_bias", level, block);
            let bn_mean = format!("enc_l{}_bn{}_mean", level, block);
            let bn_var = format!("enc_l{}_bn{}_var", level, block);

            graph.initializer.push(make_1d_weight(&bn_scale, out_channels, 1.0));
            graph.initializer.push(make_1d_weight(&bn_bias, out_channels, 0.0));
            graph.initializer.push(make_1d_weight(&bn_mean, out_channels, 0.0));
            graph.initializer.push(make_1d_weight(&bn_var, out_channels, 1.0));

            let mut bn = NodeProto::default();
            bn.name = bn_out.clone();
            bn.op_type = "BatchNormalization".to_string();
            bn.input.push(conv_out);
            bn.input.push(bn_scale);
            bn.input.push(bn_bias);
            bn.input.push(bn_mean);
            bn.input.push(bn_var);
            bn.output.push(bn_out.clone());
            graph.node.push(bn);

            // ReLU
            let relu_out = format!("enc_l{}_relu{}", level, block);
            let mut relu = NodeProto::default();
            relu.name = relu_out.clone();
            relu.op_type = "Relu".to_string();
            relu.input.push(bn_out);
            relu.output.push(relu_out.clone());
            graph.node.push(relu);

            current_tensor = relu_out;
        }

        current_channels = out_channels;
        encoder_skip_outputs.push(current_tensor.clone());

        // MaxPool (except at bottom)
        if level < depth - 1 {
            let pool_out = format!("enc_pool_{}", level);
            let mut pool = NodeProto::default();
            pool.name = pool_out.clone();
            pool.op_type = "MaxPool".to_string();
            pool.input.push(current_tensor.clone());
            pool.output.push(pool_out.clone());
            pool.attribute.push(make_ints_attr("kernel_shape", vec![2, 2]));
            pool.attribute.push(make_ints_attr("strides", vec![2, 2]));
            graph.node.push(pool);
            current_tensor = pool_out;
        }
    }

    // ===== DECODER PATH =====
    for level in (0..depth - 1).rev() {
        let out_channels = 64 * (1 << level.min(4)) as i64;

        // Upsample (ConvTranspose)
        let upsample_out = format!("dec_upsample_{}", level);
        let upsample_weight = format!("dec_upsample_{}_weight", level);
        graph.initializer.push(make_conv_weight(&upsample_weight, current_channels, current_channels, 2, 2));

        let mut upsample = NodeProto::default();
        upsample.name = upsample_out.clone();
        upsample.op_type = "ConvTranspose".to_string();
        upsample.input.push(current_tensor.clone());
        upsample.input.push(upsample_weight);
        upsample.output.push(upsample_out.clone());
        upsample.attribute.push(make_ints_attr("kernel_shape", vec![2, 2]));
        upsample.attribute.push(make_ints_attr("strides", vec![2, 2]));
        graph.node.push(upsample);

        // Concat with skip connection
        let concat_out = format!("dec_concat_{}", level);
        let mut concat = NodeProto::default();
        concat.name = concat_out.clone();
        concat.op_type = "Concat".to_string();
        concat.input.push(upsample_out);
        concat.input.push(encoder_skip_outputs[level].clone());
        concat.output.push(concat_out.clone());
        concat.attribute.push(make_int_attr("axis", 1));
        graph.node.push(concat);

        let concat_channels = current_channels + 64 * (1 << level.min(4)) as i64;
        current_tensor = concat_out;

        // Decoder conv blocks
        for block in 0..nodes_per_level {
            let is_first = block == 0;
            let in_ch = if is_first { concat_channels } else { out_channels };

            // Conv
            let conv_out = format!("dec_l{}_conv{}", level, block);
            let weight_name = format!("dec_l{}_conv{}_weight", level, block);
            graph.initializer.push(make_conv_weight(&weight_name, out_channels, in_ch, 3, 3));

            let mut conv = NodeProto::default();
            conv.name = conv_out.clone();
            conv.op_type = "Conv".to_string();
            conv.input.push(current_tensor.clone());
            conv.input.push(weight_name);
            conv.output.push(conv_out.clone());
            conv.attribute.push(make_ints_attr("kernel_shape", vec![3, 3]));
            conv.attribute.push(make_ints_attr("pads", vec![1, 1, 1, 1]));
            graph.node.push(conv);

            // BatchNorm
            let bn_out = format!("dec_l{}_bn{}", level, block);
            let bn_scale = format!("dec_l{}_bn{}_scale", level, block);
            let bn_bias = format!("dec_l{}_bn{}_bias", level, block);
            let bn_mean = format!("dec_l{}_bn{}_mean", level, block);
            let bn_var = format!("dec_l{}_bn{}_var", level, block);

            graph.initializer.push(make_1d_weight(&bn_scale, out_channels, 1.0));
            graph.initializer.push(make_1d_weight(&bn_bias, out_channels, 0.0));
            graph.initializer.push(make_1d_weight(&bn_mean, out_channels, 0.0));
            graph.initializer.push(make_1d_weight(&bn_var, out_channels, 1.0));

            let mut bn = NodeProto::default();
            bn.name = bn_out.clone();
            bn.op_type = "BatchNormalization".to_string();
            bn.input.push(conv_out);
            bn.input.push(bn_scale);
            bn.input.push(bn_bias);
            bn.input.push(bn_mean);
            bn.input.push(bn_var);
            bn.output.push(bn_out.clone());
            graph.node.push(bn);

            // ReLU
            let relu_out = format!("dec_l{}_relu{}", level, block);
            let mut relu = NodeProto::default();
            relu.name = relu_out.clone();
            relu.op_type = "Relu".to_string();
            relu.input.push(bn_out);
            relu.output.push(relu_out.clone());
            graph.node.push(relu);

            current_tensor = relu_out;
        }

        current_channels = out_channels;
    }

    // Final 1x1 conv to output classes
    let final_weight = "final_conv_weight".to_string();
    graph.initializer.push(make_conv_weight(&final_weight, 1, current_channels, 1, 1));

    let mut final_conv = NodeProto::default();
    final_conv.name = "final_conv".to_string();
    final_conv.op_type = "Conv".to_string();
    final_conv.input.push(current_tensor);
    final_conv.input.push(final_weight);
    final_conv.output.push("logits".to_string());
    final_conv.attribute.push(make_ints_attr("kernel_shape", vec![1, 1]));
    graph.node.push(final_conv);

    // Sigmoid for segmentation output
    let mut sigmoid = NodeProto::default();
    sigmoid.name = "sigmoid".to_string();
    sigmoid.op_type = "Sigmoid".to_string();
    sigmoid.input.push("logits".to_string());
    sigmoid.output.push("output".to_string());
    graph.node.push(sigmoid);

    graph.output.push(make_value_info("output", &[1, 1, 512, 512]));

    graph
}

fn create_unet_model(depth: usize, nodes_per_level: usize) -> ModelProto {
    let graph = create_unet_graph(depth, nodes_per_level);

    ModelProto {
        ir_version: 9,
        opset_import: vec![hologram_onnx_spec::OperatorSetIdProto {
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

// ============================================================================
// Helper Functions
// ============================================================================

fn make_value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
    use hologram_onnx_spec::tensor_shape_proto::Dimension;
    use hologram_onnx_spec::type_proto::Value;

    let shape_dims: Vec<Dimension> = dims
        .iter()
        .map(|&d| Dimension {
            value: Some(hologram_onnx_spec::tensor_shape_proto::dimension::Value::DimValue(d)),
            ..Default::default()
        })
        .collect();

    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(Value::TensorType(hologram_onnx_spec::type_proto::Tensor {
                elem_type: 1,
                shape: Some(TensorShapeProto { dim: shape_dims }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn make_int_attr(name: &str, value: i64) -> AttributeProto {
    use hologram_onnx_spec::attribute_proto::AttributeType;
    AttributeProto {
        name: name.to_string(),
        i: value,
        r#type: AttributeType::Int as i32,
        ..Default::default()
    }
}

fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
    use hologram_onnx_spec::attribute_proto::AttributeType;
    AttributeProto {
        name: name.to_string(),
        ints: values,
        r#type: AttributeType::Ints as i32,
        ..Default::default()
    }
}

fn make_conv_weight(name: &str, out_ch: i64, in_ch: i64, kh: i64, kw: i64) -> TensorProto {
    let size = (out_ch * in_ch * kh * kw) as usize;
    let data: Vec<f32> = (0..size).map(|i| ((i % 100) as f32 - 50.0) * 0.01).collect();

    TensorProto {
        name: name.to_string(),
        data_type: 1,
        dims: vec![out_ch, in_ch, kh, kw],
        float_data: data,
        ..Default::default()
    }
}

fn make_1d_weight(name: &str, size: i64, value: f32) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: 1,
        dims: vec![size],
        float_data: vec![value; size as usize],
        ..Default::default()
    }
}

fn encode_model(model: &ModelProto) -> Vec<u8> {
    let mut buf = Vec::new();
    model.encode(&mut buf).expect("Failed to encode model");
    buf
}

// ============================================================================
// CI-Friendly Tests (smaller graphs, fast execution)
// ============================================================================

#[test]
fn test_unet_small_graph_creation() {
    // depth=3, nodes_per_level=10 → ~150 nodes
    let model = create_unet_model(3, 10);
    let graph = model.graph.as_ref().unwrap();

    assert!(graph.node.len() >= 100, "Got {} nodes", graph.node.len());
    assert!(!graph.input.is_empty());
    assert!(!graph.output.is_empty());
    assert!(!graph.initializer.is_empty());

    eprintln!("Small UNet: {} nodes", graph.node.len());
}

#[test]
fn test_unet_medium_graph_with_partitioning() {
    // depth=4, nodes_per_level=30 → ~750 nodes (requires partitioning)
    let model = create_unet_model(4, 30);
    let graph = model.graph.as_ref().unwrap();

    assert!(graph.node.len() >= 500, "Got {} nodes", graph.node.len());

    // Verify partitioning works
    let partitioner = GraphPartitioner::with_partition_size(200);
    let partitions = partitioner.partition(graph).expect("Partitioning failed");

    assert!(partitions.len() >= 3, "Expected 3+ partitions");

    // Verify all nodes accounted for
    let total: usize = partitions.iter().map(|p| p.node_count()).sum();
    assert_eq!(total, graph.node.len());

    eprintln!(
        "Medium UNet: {} nodes → {} partitions",
        graph.node.len(),
        partitions.len()
    );
}

#[test]
fn test_unet_parsing_and_validation() {
    let model = create_unet_model(4, 20);
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).expect("Parse failed");
    validate_model(&parsed).expect("Validation failed");

    let opset = extract_opset_version(&parsed);
    assert_eq!(opset, 17);
}

#[test]
fn test_unet_weight_extraction() {
    let model = create_unet_model(3, 15);
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let graph = parsed.graph.as_ref().unwrap();

    let mut weights = WeightData::new();
    for init in &graph.initializer {
        let data = WeightData::extract_tensor_data(init).expect("Failed to extract");
        weights.add_weight(&init.name, data);
    }

    assert_eq!(weights.len(), graph.initializer.len());
    assert!(weights.buffer_size() > 0);

    eprintln!(
        "UNet weights: {} tensors, {} bytes",
        weights.len(),
        weights.buffer_size()
    );
}

#[test]
fn test_unet_weight_file_output() {
    let model = create_unet_model(3, 10);
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let graph = parsed.graph.as_ref().unwrap();

    let mut weights = WeightData::new();
    for init in &graph.initializer {
        if let Ok(data) = WeightData::extract_tensor_data(init) {
            weights.add_weight(&init.name, data);
        }
    }

    let temp_file = NamedTempFile::new().unwrap();
    weights.write_to_file(temp_file.path()).expect("Write failed");

    let metadata = std::fs::metadata(temp_file.path()).unwrap();
    assert_eq!(metadata.len() as usize, weights.buffer_size());
}

#[test]
fn test_unet_partitioning_path() {
    let model = create_unet_model(4, 25);
    let bytes = encode_model(&model);

    // Parse and validate
    let parsed = parse_model(&bytes).unwrap();
    validate_model(&parsed).unwrap();

    let graph = parsed.graph.as_ref().unwrap();

    // Graph should be large enough to trigger partitioning
    assert!(graph.node.len() > 500);

    // Verify partitioning configuration works
    let config = OnnxConfig {
        enable_partitioning: true,
        partition_size: 200,
        ..Default::default()
    };
    assert!(config.enable_partitioning);
    assert_eq!(config.partition_size, 200);

    // Test partitioning directly with configured size
    let partitioner = GraphPartitioner::with_partition_size(config.partition_size);
    let partitions = partitioner.partition(graph).expect("Partitioning should succeed");

    // Should create multiple partitions for a graph > 500 nodes with partition_size=200
    assert!(partitions.len() > 1, "Expected multiple partitions for large graph");

    // All partitions should be reasonably sized (partition_size * 1.5 as tolerance)
    let max_allowed = (config.partition_size as f64 * 1.5) as usize;
    for partition in &partitions {
        assert!(
            partition.nodes.len() <= max_allowed,
            "Partition too large: {} nodes (max {})",
            partition.nodes.len(),
            max_allowed
        );
    }
}

#[test]
fn test_unet_shape_info() {
    let model = create_unet_model(3, 10);
    let bytes = encode_model(&model);

    let parsed = parse_model(&bytes).unwrap();
    let graph = parsed.graph.as_ref().unwrap();

    let input = &graph.input[0];
    let input_shape = SymbolicShape::from_value_info(input).unwrap();
    assert_eq!(input_shape.rank(), 4);
    assert!(input_shape.is_fully_concrete());

    let output = &graph.output[0];
    let output_shape = SymbolicShape::from_value_info(output).unwrap();
    assert_eq!(output_shape.rank(), 4);
}

#[test]
fn test_unet_operation_types() {
    let model = create_unet_model(3, 10);
    let graph = model.graph.as_ref().unwrap();

    let op_types: HashSet<_> = graph.node.iter().map(|n| n.op_type.as_str()).collect();

    assert!(op_types.contains("Conv"));
    assert!(op_types.contains("BatchNormalization"));
    assert!(op_types.contains("Relu"));
    assert!(op_types.contains("MaxPool"));
    assert!(op_types.contains("ConvTranspose"));
    assert!(op_types.contains("Concat"));
    assert!(op_types.contains("Sigmoid"));
}

#[test]
fn test_unet_skip_connection_boundaries() {
    let model = create_unet_model(4, 20);
    let graph = model.graph.as_ref().unwrap();

    let partitioner = GraphPartitioner::with_partition_size(100);
    let partitions = partitioner.partition(graph).unwrap();

    // UNet has skip connections → some partitions should have boundary tensors
    let has_boundaries = partitions.iter().any(|p| {
        !p.boundary_inputs.is_empty() || !p.boundary_outputs.is_empty()
    });

    assert!(has_boundaries, "UNet should have cross-partition boundaries");
}

#[test]
fn test_unet_pipeline_stages() {
    // Test all pipeline stages work correctly
    let model = create_unet_model(4, 20);
    let bytes = encode_model(&model);

    // Stage 1: Parse
    let parsed = parse_model(&bytes).expect("Parse failed");

    // Stage 2: Validate
    validate_model(&parsed).expect("Validation failed");

    // Stage 3: Opset
    let opset = extract_opset_version(&parsed);
    assert_eq!(opset, 17);

    // Stage 4: Partition
    let graph = parsed.graph.as_ref().unwrap();
    let partitioner = GraphPartitioner::with_partition_size(100);
    let partitions = partitioner.partition(graph).expect("Partitioning failed");
    assert!(partitions.len() >= 3);

    // Stage 5: Weight extraction
    let mut weights = WeightData::new();
    for init in &graph.initializer {
        if let Ok(data) = WeightData::extract_tensor_data(init) {
            weights.add_weight(&init.name, data);
        }
    }
    assert!(weights.buffer_size() > 0);

    eprintln!(
        "Pipeline: {} nodes → {} partitions, {} weight bytes",
        graph.node.len(),
        partitions.len(),
        weights.buffer_size()
    );
}

// ============================================================================
// Large Model Tests (require more memory, run with --ignored)
// ============================================================================

/// Test with 3000+ nodes like documented UNet case.
/// Run with: `cargo test test_unet_3000_nodes -- --ignored`
#[test]
#[ignore]
fn test_unet_3000_nodes_full_pipeline() {
    // depth=5, nodes_per_level=50 → ~3000 nodes (matches documented UNet)
    let model = create_unet_model(5, 50);
    let bytes = encode_model(&model);
    let graph = model.graph.as_ref().unwrap();

    assert!(graph.node.len() >= 3000, "Got {} nodes", graph.node.len());

    // Parse and validate
    let parsed = parse_model(&bytes).expect("Parse failed");
    validate_model(&parsed).expect("Validation failed");

    // Partition with 500-node chunks
    let partitioner = GraphPartitioner::with_partition_size(500);
    let partitions = partitioner.partition(graph).expect("Partitioning failed");

    // Should match documented ~7 partitions
    let expected = (graph.node.len() + 499) / 500;
    assert_eq!(partitions.len(), expected);

    // Extract weights
    let mut weights = WeightData::new();
    for init in &graph.initializer {
        if let Ok(data) = WeightData::extract_tensor_data(init) {
            weights.add_weight(&init.name, data);
        }
    }

    eprintln!(
        "UNet 3000 nodes:\n\
         - Nodes: {}\n\
         - Partitions: {}\n\
         - Weights: {} bytes",
        graph.node.len(),
        partitions.len(),
        weights.buffer_size()
    );
}

/// Memory efficiency test for large models.
/// Run with: `cargo test test_unet_memory_efficiency -- --ignored`
#[test]
#[ignore]
fn test_unet_memory_efficiency_large() {
    let model = create_unet_model(5, 50);
    let graph = model.graph.as_ref().unwrap();

    let node_count = graph.node.len();
    assert!(node_count >= 3000);

    // Partition into chunks
    let partitioner = GraphPartitioner::with_partition_size(500);
    let partitions = partitioner.partition(graph).unwrap();

    // Each partition should be bounded
    for partition in &partitions {
        assert!(partition.node_count() <= 500);
    }

    // Per memory profile: ~800 bytes per node overhead
    let estimated_per_partition = 500 * 800; // ~400 KB
    eprintln!(
        "Memory estimate: {} partitions × ~{} bytes = ~{} MB per partition",
        partitions.len(),
        estimated_per_partition,
        estimated_per_partition / 1024 / 1024
    );
}
