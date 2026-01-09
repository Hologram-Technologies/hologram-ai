//! Integration tests for graph partitioning in hologram-onnx-core.
//!
//! These tests verify the partitioning system works correctly with various
//! graph sizes and topologies, ensuring:
//! - Correct partitioning of large graphs
//! - Proper boundary detection between partitions
//! - Topological ordering is preserved
//! - Memory usage stays within bounds

use hologram_onnx::{GraphPartition, GraphPartitioner};
use hologram_onnx::proto::{
    AttributeProto, GraphProto, NodeProto, TensorShapeProto, TypeProto, ValueInfoProto,
};
use std::collections::HashSet;

// ============================================================================
// Test Fixtures and Helpers
// ============================================================================

/// Create a linear chain graph: node0 → node1 → node2 → ... → nodeN
fn create_linear_graph(node_count: usize) -> GraphProto {
    let mut graph = GraphProto {
        name: format!("linear_graph_{}", node_count),
        ..Default::default()
    };

    // Add input
    graph
        .input
        .push(make_value_info("input", &[1, 3, 224, 224]));

    for i in 0..node_count {
        let mut node = NodeProto {
            name: format!("node_{}", i),
            op_type: "Relu".to_string(),
            ..Default::default()
        };

        if i == 0 {
            node.input.push("input".to_string());
        } else {
            node.input.push(format!("tensor_{}", i - 1));
        }

        node.output.push(format!("tensor_{}", i));
        graph.node.push(node);
    }

    // Add output
    graph.output.push(make_value_info(
        &format!("tensor_{}", node_count - 1),
        &[1, 3, 224, 224],
    ));

    graph
}

/// Create a wide parallel graph with multiple independent branches.
///
/// Structure:
/// ```text
///      input
///    /   |   \
/// branch0 branch1 ... branchN
///    \   |   /
///     concat
/// ```
fn create_wide_graph(branch_count: usize, nodes_per_branch: usize) -> GraphProto {
    let mut graph = GraphProto {
        name: format!("wide_graph_{}x{}", branch_count, nodes_per_branch),
        ..Default::default()
    };

    // Add input
    graph
        .input
        .push(make_value_info("input", &[1, 3, 224, 224]));

    let mut branch_outputs = Vec::new();

    // Create parallel branches
    for b in 0..branch_count {
        for n in 0..nodes_per_branch {
            let mut node = NodeProto {
                name: format!("branch{}_{}", b, n),
                op_type: "Relu".to_string(),
                ..Default::default()
            };

            if n == 0 {
                node.input.push("input".to_string());
            } else {
                node.input.push(format!("branch{}_tensor_{}", b, n - 1));
            }

            let output_name = format!("branch{}_tensor_{}", b, n);
            node.output.push(output_name.clone());

            if n == nodes_per_branch - 1 {
                branch_outputs.push(output_name);
            }

            graph.node.push(node);
        }
    }

    // Create concat node that merges all branches
    let mut concat_node = NodeProto {
        name: "concat".to_string(),
        op_type: "Concat".to_string(),
        ..Default::default()
    };
    for output in &branch_outputs {
        concat_node.input.push(output.clone());
    }
    concat_node.output.push("output".to_string());

    // Add axis attribute for concat
    let axis_attr = AttributeProto {
        name: "axis".to_string(),
        i: 1,
        ..Default::default()
    };
    concat_node.attribute.push(axis_attr);

    graph.node.push(concat_node);

    // Add output
    graph.output.push(make_value_info(
        "output",
        &[1, 3 * branch_count as i64, 224, 224],
    ));

    graph
}

/// Create a diamond-shaped graph with converging and diverging paths.
///
/// Structure:
/// ```text
///        input
///          |
///        node0
///       /     \
///    node1   node2
///       \     /
///        node3
///          |
///        output
/// ```
fn create_diamond_graph() -> GraphProto {
    let mut graph = GraphProto {
        name: "diamond_graph".to_string(),
        ..Default::default()
    };

    graph.input.push(make_value_info("input", &[1, 64]));

    // node0: input → split into two paths
    let mut node0 = NodeProto {
        name: "node0".to_string(),
        op_type: "Relu".to_string(),
        ..Default::default()
    };
    node0.input.push("input".to_string());
    node0.output.push("t0".to_string());
    graph.node.push(node0);

    // node1: left branch
    let mut node1 = NodeProto {
        name: "node1".to_string(),
        op_type: "Sigmoid".to_string(),
        ..Default::default()
    };
    node1.input.push("t0".to_string());
    node1.output.push("t1".to_string());
    graph.node.push(node1);

    // node2: right branch
    let mut node2 = NodeProto {
        name: "node2".to_string(),
        op_type: "Tanh".to_string(),
        ..Default::default()
    };
    node2.input.push("t0".to_string());
    node2.output.push("t2".to_string());
    graph.node.push(node2);

    // node3: merge paths
    let mut node3 = NodeProto {
        name: "node3".to_string(),
        op_type: "Add".to_string(),
        ..Default::default()
    };
    node3.input.push("t1".to_string());
    node3.input.push("t2".to_string());
    node3.output.push("output".to_string());
    graph.node.push(node3);

    graph.output.push(make_value_info("output", &[1, 64]));

    graph
}

/// Create a ResNet-style graph with skip connections.
///
/// Structure:
/// ```text
/// input → conv1 → bn1 → relu1 ──────────────┐
///           ↓                                │
///         conv2 → bn2 → relu2                │
///           ↓                                │
///         conv3 → bn3 ─────────────────→ add → relu
/// ```
fn create_resnet_block_graph(num_blocks: usize) -> GraphProto {
    let mut graph = GraphProto {
        name: format!("resnet_blocks_{}", num_blocks),
        ..Default::default()
    };

    graph.input.push(make_value_info("input", &[1, 64, 56, 56]));

    let mut current_input = "input".to_string();

    for block in 0..num_blocks {
        // Main path: conv → bn → relu → conv → bn
        let conv1_out = format!("block{}_conv1", block);
        let bn1_out = format!("block{}_bn1", block);
        let relu1_out = format!("block{}_relu1", block);
        let conv2_out = format!("block{}_conv2", block);
        let bn2_out = format!("block{}_bn2", block);
        let add_out = format!("block{}_add", block);
        let relu2_out = format!("block{}_out", block);

        // Conv1
        let mut conv1 = NodeProto {
            name: format!("block{}_conv1", block),
            op_type: "Conv".to_string(),
            ..Default::default()
        };
        conv1.input.push(current_input.clone());
        conv1.input.push(format!("block{}_conv1_weight", block));
        conv1.output.push(conv1_out.clone());
        graph.node.push(conv1);

        // BN1
        let mut bn1 = NodeProto {
            name: format!("block{}_bn1", block),
            op_type: "BatchNormalization".to_string(),
            ..Default::default()
        };
        bn1.input.push(conv1_out);
        bn1.input.push(format!("block{}_bn1_scale", block));
        bn1.input.push(format!("block{}_bn1_bias", block));
        bn1.input.push(format!("block{}_bn1_mean", block));
        bn1.input.push(format!("block{}_bn1_var", block));
        bn1.output.push(bn1_out.clone());
        graph.node.push(bn1);

        // ReLU1
        let mut relu1 = NodeProto {
            name: format!("block{}_relu1", block),
            op_type: "Relu".to_string(),
            ..Default::default()
        };
        relu1.input.push(bn1_out);
        relu1.output.push(relu1_out.clone());
        graph.node.push(relu1);

        // Conv2
        let mut conv2 = NodeProto {
            name: format!("block{}_conv2", block),
            op_type: "Conv".to_string(),
            ..Default::default()
        };
        conv2.input.push(relu1_out);
        conv2.input.push(format!("block{}_conv2_weight", block));
        conv2.output.push(conv2_out.clone());
        graph.node.push(conv2);

        // BN2
        let mut bn2 = NodeProto {
            name: format!("block{}_bn2", block),
            op_type: "BatchNormalization".to_string(),
            ..Default::default()
        };
        bn2.input.push(conv2_out);
        bn2.input.push(format!("block{}_bn2_scale", block));
        bn2.input.push(format!("block{}_bn2_bias", block));
        bn2.input.push(format!("block{}_bn2_mean", block));
        bn2.input.push(format!("block{}_bn2_var", block));
        bn2.output.push(bn2_out.clone());
        graph.node.push(bn2);

        // Add (skip connection)
        let mut add = NodeProto {
            name: format!("block{}_add", block),
            op_type: "Add".to_string(),
            ..Default::default()
        };
        add.input.push(bn2_out);
        add.input.push(current_input.clone()); // Skip connection
        add.output.push(add_out.clone());
        graph.node.push(add);

        // ReLU2
        let mut relu2 = NodeProto {
            name: format!("block{}_relu2", block),
            op_type: "Relu".to_string(),
            ..Default::default()
        };
        relu2.input.push(add_out);
        relu2.output.push(relu2_out.clone());
        graph.node.push(relu2);

        current_input = relu2_out;
    }

    graph
        .output
        .push(make_value_info(&current_input, &[1, 64, 56, 56]));

    graph
}

/// Create a UNet-style encoder-decoder graph with skip connections.
fn create_unet_style_graph(depth: usize, nodes_per_level: usize) -> GraphProto {
    let mut graph = GraphProto {
        name: format!("unet_style_d{}_n{}", depth, nodes_per_level),
        ..Default::default()
    };

    graph
        .input
        .push(make_value_info("input", &[1, 3, 256, 256]));

    let mut encoder_outputs = Vec::new();
    let mut current_tensor = "input".to_string();

    // Encoder path
    for level in 0..depth {
        for node in 0..nodes_per_level {
            let output_name = format!("enc_l{}_n{}", level, node);

            let mut n = NodeProto {
                name: output_name.clone(),
                op_type: if node % 2 == 0 { "Conv" } else { "Relu" }.to_string(),
                ..Default::default()
            };
            n.input.push(current_tensor.clone());
            if n.op_type == "Conv" {
                n.input.push(format!("enc_l{}_n{}_weight", level, node));
            }
            n.output.push(output_name.clone());
            graph.node.push(n);

            current_tensor = output_name;
        }

        encoder_outputs.push(current_tensor.clone());

        // Downsample (except at bottom)
        if level < depth - 1 {
            let pool_out = format!("enc_pool_{}", level);
            let mut pool = NodeProto {
                name: pool_out.clone(),
                op_type: "MaxPool".to_string(),
                ..Default::default()
            };
            pool.input.push(current_tensor.clone());
            pool.output.push(pool_out.clone());
            graph.node.push(pool);
            current_tensor = pool_out;
        }
    }

    // Decoder path (with skip connections from encoder)
    for level in (0..depth - 1).rev() {
        // Upsample
        let upsample_out = format!("dec_upsample_{}", level);
        let mut upsample = NodeProto {
            name: upsample_out.clone(),
            op_type: "Resize".to_string(),
            ..Default::default()
        };
        upsample.input.push(current_tensor.clone());
        upsample.output.push(upsample_out.clone());
        graph.node.push(upsample);

        // Concat with encoder skip connection
        let concat_out = format!("dec_concat_{}", level);
        let mut concat = NodeProto {
            name: concat_out.clone(),
            op_type: "Concat".to_string(),
            ..Default::default()
        };
        concat.input.push(upsample_out);
        concat.input.push(encoder_outputs[level].clone()); // Skip connection
        concat.output.push(concat_out.clone());
        graph.node.push(concat);

        current_tensor = concat_out;

        // Decoder convolutions
        for node in 0..nodes_per_level {
            let output_name = format!("dec_l{}_n{}", level, node);

            let mut n = NodeProto {
                name: output_name.clone(),
                op_type: if node % 2 == 0 { "Conv" } else { "Relu" }.to_string(),
                ..Default::default()
            };
            n.input.push(current_tensor.clone());
            if n.op_type == "Conv" {
                n.input.push(format!("dec_l{}_n{}_weight", level, node));
            }
            n.output.push(output_name.clone());
            graph.node.push(n);

            current_tensor = output_name;
        }
    }

    // Final output conv
    let mut final_conv = NodeProto {
        name: "final_conv".to_string(),
        op_type: "Conv".to_string(),
        ..Default::default()
    };
    final_conv.input.push(current_tensor);
    final_conv.input.push("final_conv_weight".to_string());
    final_conv.output.push("output".to_string());
    graph.node.push(final_conv);

    graph
        .output
        .push(make_value_info("output", &[1, 1, 256, 256]));

    graph
}

/// Helper to create a ValueInfoProto.
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

/// Collect all tensor names produced by a graph's nodes.
#[allow(dead_code)]
fn collect_produced_tensors(graph: &GraphProto) -> HashSet<String> {
    graph
        .node
        .iter()
        .flat_map(|n| n.output.iter().cloned())
        .collect()
}

/// Collect all tensor names consumed by a graph's nodes.
#[allow(dead_code)]
fn collect_consumed_tensors(graph: &GraphProto) -> HashSet<String> {
    graph
        .node
        .iter()
        .flat_map(|n| n.input.iter().cloned())
        .filter(|s| !s.is_empty())
        .collect()
}

// ============================================================================
// Basic Partitioning Tests
// ============================================================================

#[test]
fn test_no_partitioning_needed_small_graph() {
    let graph = create_linear_graph(100);
    let partitioner = GraphPartitioner::new(); // Default 500 nodes

    let partitions = partitioner.partition(&graph).unwrap();

    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0].node_count(), 100);
    assert_eq!(partitions[0].partition_idx, 0);
}

#[test]
fn test_partitioning_large_linear_graph() {
    let graph = create_linear_graph(1000);
    let partitioner = GraphPartitioner::with_partition_size(200);

    let partitions = partitioner.partition(&graph).unwrap();

    // Should create 5 partitions of ~200 nodes each
    assert_eq!(partitions.len(), 5);

    // Total nodes should match
    let total_nodes: usize = partitions.iter().map(|p| p.node_count()).sum();
    assert_eq!(total_nodes, 1000);

    // Each partition should have ~200 nodes
    for partition in &partitions {
        assert!(partition.node_count() <= 200);
        assert!(partition.node_count() > 0);
    }
}

#[test]
fn test_partitioning_preserves_all_nodes() {
    let graph = create_linear_graph(750);
    let partitioner = GraphPartitioner::with_partition_size(100);

    let partitions = partitioner.partition(&graph).unwrap();

    // Collect all nodes from partitions
    let all_nodes: Vec<String> = partitions
        .iter()
        .flat_map(|p| p.nodes.iter().map(|n| n.name.clone()))
        .collect();

    // Should have all original nodes
    assert_eq!(all_nodes.len(), 750);

    // Check each original node is present
    for i in 0..750 {
        let expected_name = format!("node_{}", i);
        assert!(
            all_nodes.contains(&expected_name),
            "Missing node: {}",
            expected_name
        );
    }
}

#[test]
fn test_partition_indices_sequential() {
    let graph = create_linear_graph(500);
    let partitioner = GraphPartitioner::with_partition_size(100);

    let partitions = partitioner.partition(&graph).unwrap();

    for (i, partition) in partitions.iter().enumerate() {
        assert_eq!(partition.partition_idx, i);
    }
}

// ============================================================================
// Boundary Detection Tests
// ============================================================================

#[test]
fn test_boundary_detection_linear_graph() {
    let graph = create_linear_graph(400);
    let partitioner = GraphPartitioner::with_partition_size(100);

    let partitions = partitioner.partition(&graph).unwrap();

    assert_eq!(partitions.len(), 4);

    // First partition has boundary inputs from graph input ("input")
    // This is expected behavior - graph inputs are treated as external tensors
    assert!(partitions[0].has_boundary_inputs());
    assert!(partitions[0].boundary_inputs.contains_key("input"));

    // All partitions after first should have boundary inputs from previous partition
    for partition in partitions.iter().skip(1) {
        assert!(partition.has_boundary_inputs());
    }

    // All but last partition should have boundary outputs
    for partition in partitions.iter().take(partitions.len() - 1) {
        assert!(partition.has_boundary_outputs());
    }
}

#[test]
fn test_boundary_tensors_match() {
    let graph = create_linear_graph(200);
    let partitioner = GraphPartitioner::with_partition_size(50);

    let partitions = partitioner.partition(&graph).unwrap();

    // For each partition boundary, verify inputs match previous outputs
    for i in 1..partitions.len() {
        // Each boundary input should reference a tensor from a previous partition
        // or be a graph input
        for (external_tensor, _virtual_name) in &partitions[i].boundary_inputs {
            // The external tensor should be produced by a previous partition
            // or be the graph's original input
            let found_in_prev = partitions[..i]
                .iter()
                .any(|p| p.nodes.iter().any(|n| n.output.contains(external_tensor)));

            let is_graph_input = external_tensor == "input";

            assert!(
                found_in_prev || is_graph_input,
                "Boundary input '{}' not found in previous partitions or graph inputs",
                external_tensor
            );
        }
    }
}

// ============================================================================
// Graph Topology Tests
// ============================================================================

#[test]
fn test_partitioning_diamond_graph() {
    let graph = create_diamond_graph();
    let partitioner = GraphPartitioner::with_partition_size(2);

    let partitions = partitioner.partition(&graph).unwrap();

    // Diamond with 4 nodes, partition size 2 → 2 partitions
    assert!(!partitions.is_empty());

    // Total should be 4 nodes
    let total: usize = partitions.iter().map(|p| p.node_count()).sum();
    assert_eq!(total, 4);
}

#[test]
fn test_partitioning_wide_graph() {
    // 5 branches, 100 nodes each = 500 + 1 concat = 501 nodes
    let graph = create_wide_graph(5, 100);
    let partitioner = GraphPartitioner::with_partition_size(150);

    let partitions = partitioner.partition(&graph).unwrap();

    // Verify all nodes present
    let total: usize = partitions.iter().map(|p| p.node_count()).sum();
    assert_eq!(total, 501);

    // Should have multiple partitions
    assert!(partitions.len() > 1);
}

#[test]
fn test_partitioning_resnet_blocks() {
    // 50 ResNet blocks * 7 nodes per block = 350 nodes
    let graph = create_resnet_block_graph(50);
    let partitioner = GraphPartitioner::with_partition_size(100);

    let partitions = partitioner.partition(&graph).unwrap();

    // Verify all nodes present
    let total: usize = partitions.iter().map(|p| p.node_count()).sum();
    assert_eq!(total, 350);

    // Should have ~4 partitions
    assert!(partitions.len() >= 3);
    assert!(partitions.len() <= 5);
}

#[test]
fn test_partitioning_unet_style() {
    // UNet with 4 levels, 4 nodes per level
    // This creates a complex graph with skip connections
    let graph = create_unet_style_graph(4, 4);
    let node_count = graph.node.len();

    let partitioner = GraphPartitioner::with_partition_size(10);
    let partitions = partitioner.partition(&graph).unwrap();

    // Verify all nodes present
    let total: usize = partitions.iter().map(|p| p.node_count()).sum();
    assert_eq!(total, node_count);
}

// ============================================================================
// Large Graph Tests (1000+ nodes)
// ============================================================================

#[test]
fn test_large_linear_graph_1500_nodes() {
    let graph = create_linear_graph(1500);
    let partitioner = GraphPartitioner::with_partition_size(500);

    let partitions = partitioner.partition(&graph).unwrap();

    assert_eq!(partitions.len(), 3);

    let total: usize = partitions.iter().map(|p| p.node_count()).sum();
    assert_eq!(total, 1500);
}

#[test]
fn test_large_wide_graph_2000_nodes() {
    // 20 branches * 100 nodes = 2000 + 1 concat = 2001 nodes
    let graph = create_wide_graph(20, 100);
    let partitioner = GraphPartitioner::with_partition_size(500);

    let partitions = partitioner.partition(&graph).unwrap();

    let total: usize = partitions.iter().map(|p| p.node_count()).sum();
    assert_eq!(total, 2001);

    assert!(partitions.len() >= 4);
}

#[test]
fn test_large_resnet_style_graph() {
    // Simulating ~3000 nodes like UNet
    // 400 ResNet blocks * 7 nodes = 2800 nodes
    let graph = create_resnet_block_graph(400);
    let node_count = graph.node.len();
    assert!(node_count >= 2800);

    let partitioner = GraphPartitioner::with_partition_size(500);
    let partitions = partitioner.partition(&graph).unwrap();

    let total: usize = partitions.iter().map(|p| p.node_count()).sum();
    assert_eq!(total, node_count);

    // Should have ~6 partitions
    assert!(partitions.len() >= 5);
    assert!(partitions.len() <= 7);
}

// ============================================================================
// Topological Order Tests
// ============================================================================

#[test]
fn test_topological_order_preserved() {
    let graph = create_linear_graph(300);
    let partitioner = GraphPartitioner::with_partition_size(100);

    let partitions = partitioner.partition(&graph).unwrap();

    // In a linear graph, nodes should appear in order across partitions
    let mut prev_max_idx = 0;

    for partition in &partitions {
        let min_idx = partition
            .original_node_indices
            .iter()
            .min()
            .copied()
            .unwrap_or(0);
        let max_idx = partition
            .original_node_indices
            .iter()
            .max()
            .copied()
            .unwrap_or(0);

        // Each partition should have higher indices than the previous
        assert!(
            min_idx >= prev_max_idx || partition.partition_idx == 0,
            "Topological order violated: partition {} has min idx {} but prev max was {}",
            partition.partition_idx,
            min_idx,
            prev_max_idx
        );

        prev_max_idx = max_idx;
    }
}

// ============================================================================
// Original Node Index Tracking Tests
// ============================================================================

#[test]
fn test_original_indices_complete() {
    let graph = create_linear_graph(500);
    let partitioner = GraphPartitioner::with_partition_size(100);

    let partitions = partitioner.partition(&graph).unwrap();

    // Collect all original indices
    let all_indices: HashSet<usize> = partitions
        .iter()
        .flat_map(|p| p.original_node_indices.iter().copied())
        .collect();

    // Should have all indices 0..500
    assert_eq!(all_indices.len(), 500);
    for i in 0..500 {
        assert!(all_indices.contains(&i), "Missing original index: {}", i);
    }
}

#[test]
fn test_original_indices_no_duplicates() {
    let graph = create_linear_graph(300);
    let partitioner = GraphPartitioner::with_partition_size(75);

    let partitions = partitioner.partition(&graph).unwrap();

    // Collect all indices and check for duplicates
    let mut seen = HashSet::new();
    for partition in &partitions {
        for &idx in &partition.original_node_indices {
            assert!(
                seen.insert(idx),
                "Duplicate original index: {} in partition {}",
                idx,
                partition.partition_idx
            );
        }
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_single_node_graph() {
    let mut graph = GraphProto {
        name: "single_node".to_string(),
        ..Default::default()
    };

    let mut node = NodeProto {
        name: "node0".to_string(),
        op_type: "Relu".to_string(),
        ..Default::default()
    };
    node.input.push("input".to_string());
    node.output.push("output".to_string());
    graph.node.push(node);

    let partitioner = GraphPartitioner::new();
    let partitions = partitioner.partition(&graph).unwrap();

    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0].node_count(), 1);
}

#[test]
fn test_empty_graph() {
    let graph = GraphProto::default();
    let partitioner = GraphPartitioner::new();

    let partitions = partitioner.partition(&graph).unwrap();

    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0].node_count(), 0);
}

#[test]
fn test_exact_partition_size() {
    let graph = create_linear_graph(500);
    let partitioner = GraphPartitioner::with_partition_size(500);

    let partitions = partitioner.partition(&graph).unwrap();

    // Should create exactly 1 partition
    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0].node_count(), 500);
}

#[test]
fn test_partition_size_plus_one() {
    let graph = create_linear_graph(501);
    let partitioner = GraphPartitioner::with_partition_size(500);

    let partitions = partitioner.partition(&graph).unwrap();

    // Should create 2 partitions
    assert_eq!(partitions.len(), 2);

    let total: usize = partitions.iter().map(|p| p.node_count()).sum();
    assert_eq!(total, 501);
}

// ============================================================================
// Memory Usage Verification
// ============================================================================

#[test]
fn test_partition_memory_isolation() {
    // Large graph to ensure partitioning happens
    let graph = create_linear_graph(2000);
    let partitioner = GraphPartitioner::with_partition_size(500);

    let partitions = partitioner.partition(&graph).unwrap();

    // Each partition should be independent - can be processed and dropped
    for partition in partitions {
        // Verify partition is self-contained
        let produced: HashSet<_> = partition
            .nodes
            .iter()
            .flat_map(|n| n.output.iter())
            .collect();

        let consumed: HashSet<_> = partition
            .nodes
            .iter()
            .flat_map(|n| n.input.iter())
            .filter(|s| !s.is_empty())
            .collect();

        // Internal tensors should be either:
        // 1. Produced within the partition, OR
        // 2. A boundary input (virtual input from another partition)
        for tensor in consumed {
            let is_produced_internally = produced.contains(&tensor);
            let is_boundary_input = partition.boundary_inputs.contains_key(tensor);
            let is_graph_input = tensor == "input"; // Our test graphs use "input"

            assert!(
                is_produced_internally || is_boundary_input || is_graph_input,
                "Partition {} has unresolved dependency: {}",
                partition.partition_idx,
                tensor
            );
        }

        // Partition dropped here - memory freed
    }
}

#[test]
fn test_partition_size_bounds() {
    let graph = create_linear_graph(1000);
    let partitioner = GraphPartitioner::with_partition_size(200);

    let partitions = partitioner.partition(&graph).unwrap();

    for partition in &partitions {
        assert!(
            partition.node_count() <= 200,
            "Partition {} exceeds size limit: {} > 200",
            partition.partition_idx,
            partition.node_count()
        );
    }
}

// ============================================================================
// Custom Partition Size Tests
// ============================================================================

#[test]
fn test_small_partition_size() {
    let graph = create_linear_graph(100);
    let partitioner = GraphPartitioner::with_partition_size(10);

    let partitions = partitioner.partition(&graph).unwrap();

    assert_eq!(partitions.len(), 10);

    for partition in &partitions {
        assert_eq!(partition.node_count(), 10);
    }
}

#[test]
fn test_large_partition_size() {
    let graph = create_linear_graph(100);
    let partitioner = GraphPartitioner::with_partition_size(1000);

    let partitions = partitioner.partition(&graph).unwrap();

    // No partitioning needed - single partition
    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0].node_count(), 100);
}

#[test]
#[should_panic(expected = "Partition size must be > 0")]
fn test_zero_partition_size_panics() {
    GraphPartitioner::with_partition_size(0);
}

// ============================================================================
// GraphPartition API Tests
// ============================================================================

#[test]
fn test_graph_partition_from_full_graph() {
    let graph = create_linear_graph(50);
    let partition = GraphPartition::from_full_graph(&graph);

    assert_eq!(partition.partition_idx, 0);
    assert_eq!(partition.node_count(), 50);
    assert_eq!(partition.original_node_indices.len(), 50);
    assert!(!partition.has_boundary_inputs());
    assert!(partition.has_boundary_outputs());
}

#[test]
fn test_graph_partition_has_methods() {
    let graph = create_linear_graph(200);
    let partitioner = GraphPartitioner::with_partition_size(50);

    let partitions = partitioner.partition(&graph).unwrap();

    // First partition has boundary input from graph's "input" tensor
    let first = &partitions[0];
    assert!(first.has_boundary_inputs());
    assert!(first.boundary_inputs.contains_key("input"));

    // Last partition should also have boundary inputs (from previous partition)
    let last = partitions.last().unwrap();
    assert!(last.has_boundary_inputs());

    // Check has_boundary_outputs on non-last partitions
    for partition in partitions.iter().take(partitions.len() - 1) {
        assert!(partition.has_boundary_outputs());
    }
}
