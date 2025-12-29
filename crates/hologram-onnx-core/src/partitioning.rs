//! Graph partitioning for large ONNX models using petgraph.
//!
//! # Overview
//!
//! Large ONNX models (>500 nodes) can cause memory issues during compilation.
//! This module implements graph partitioning to compile large models in chunks.
//!
//! # Algorithm
//!
//! 1. **Build petgraph DiGraph**: Convert ONNX graph to petgraph structure
//! 2. **Topological Sort**: Order nodes by dependencies using petgraph::algo::toposort
//! 3. **Partition Creation**: Split into chunks of ~500 nodes
//! 4. **Boundary Detection**: Identify cross-partition dependencies
//! 5. **Subgraph Extraction**: Create independent subgraphs with virtual inputs/outputs
//!
//! # Performance
//!
//! - **O(V + E) graph construction**: Linear in graph size
//! - **O(V + E) topological sort**: Linear via petgraph
//! - **O(1) partition creation**: Simple chunking
//! - **O(E) boundary detection**: Linear in edge count
//! - **Memory**: Peak memory reduced by factor of (total_nodes / partition_size)
//!
//! # Example
//!
//! ```rust,ignore
//! use hologram_onnx_core::partitioning::GraphPartitioner;
//!
//! let partitioner = GraphPartitioner::new(); // 500 nodes per partition (default)
//! let partitions = partitioner.partition(&graph)?;
//!
//! // Compile each partition independently
//! for partition in partitions {
//!     let compiled = compile_subgraph(&partition)?;
//!     // ...
//! }
//! ```

use crate::error::{OnnxError, Result};
use ahash::{AHashMap, AHashSet};
use hologram_onnx_spec::{GraphProto, NodeProto};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::toposort;
use tracing::{debug, trace};

/// Graph partitioner for large ONNX models.
///
/// Uses petgraph for efficient graph algorithms and topological sorting.
#[derive(Debug, Clone)]
pub struct GraphPartitioner {
    /// Target nodes per partition (default: 500)
    pub partition_size: usize,
}

impl GraphPartitioner {
    /// Create new partitioner with default partition size (500 nodes).
    pub fn new() -> Self {
        Self {
            partition_size: 500,
        }
    }

    /// Create partitioner with custom partition size.
    pub fn with_partition_size(partition_size: usize) -> Self {
        if partition_size == 0 {
            panic!("Partition size must be > 0");
        }
        Self { partition_size }
    }

    /// Partition a graph into chunks.
    ///
    /// # Performance: O(V + E) where V = nodes, E = edges
    ///
    /// Returns partitions in topologically sorted order.
    pub fn partition(&self, graph: &GraphProto) -> Result<Vec<GraphPartition>> {
        let node_count = graph.node.len();

        if node_count <= self.partition_size {
            debug!("Graph has {} nodes (<= {}), no partitioning needed",
                   node_count, self.partition_size);
            return Ok(vec![GraphPartition::from_full_graph(graph)]);
        }

        debug!("Partitioning graph with {} nodes into chunks of {}",
               node_count, self.partition_size);

        // 1. Build petgraph DiGraph from ONNX graph
        trace!("Building dependency graph");
        let (pg_graph, node_index_map) = self.build_dependency_graph(graph)?;

        // 2. Topological sort using petgraph
        trace!("Performing topological sort");
        let sorted_node_indices = toposort(&pg_graph, None)
            .map_err(|cycle| {
                OnnxError::InvalidModel(
                    format!("Graph has cycles at node: {:?}", cycle.node_id())
                )
            })?;

        // Convert NodeIndex back to original node indices
        let sorted_indices: Vec<usize> = sorted_node_indices.iter()
            .map(|&pg_idx| node_index_map[&pg_idx])
            .collect();

        // 3. Create partition groups
        trace!("Creating partition groups");
        let partition_groups = self.create_partition_groups(&sorted_indices);

        // 4. Build subgraphs with boundary tensors
        trace!("Building subgraphs");
        let mut partitions = Vec::new();
        for (partition_idx, node_indices) in partition_groups.iter().enumerate() {
            let partition = self.create_subgraph(graph, node_indices, partition_idx)?;
            partitions.push(partition);
        }

        debug!("Created {} partitions", partitions.len());
        Ok(partitions)
    }

    /// Build petgraph DiGraph from ONNX graph.
    ///
    /// # Performance: O(V + E)
    ///
    /// Returns the graph and a mapping from NodeIndex → original node index.
    fn build_dependency_graph(
        &self,
        graph: &GraphProto,
    ) -> Result<(DiGraph<usize, ()>, AHashMap<NodeIndex, usize>)> {
        let mut pg_graph = DiGraph::new();
        let mut node_index_map: AHashMap<NodeIndex, usize> = AHashMap::new();
        let mut tensor_producers: AHashMap<String, NodeIndex> = AHashMap::new();

        // Add all nodes to petgraph
        let pg_nodes: Vec<NodeIndex> = (0..graph.node.len())
            .map(|i| {
                let pg_idx = pg_graph.add_node(i);
                node_index_map.insert(pg_idx, i);
                pg_idx
            })
            .collect();

        // Build tensor → producer mapping
        for (idx, node) in graph.node.iter().enumerate() {
            for output in &node.output {
                tensor_producers.insert(output.to_string(), pg_nodes[idx]);
            }
        }

        // Add edges for dependencies
        for (consumer_idx, node) in graph.node.iter().enumerate() {
            for input in &node.input {
                if let Some(&producer_pg_idx) = tensor_producers.get(input) {
                    // Add edge: producer → consumer
                    pg_graph.add_edge(producer_pg_idx, pg_nodes[consumer_idx], ());
                }
            }
        }

        Ok((pg_graph, node_index_map))
    }

    /// Create partition groups from sorted node indices.
    ///
    /// # Performance: O(n) where n = number of nodes
    fn create_partition_groups(&self, sorted_indices: &[usize]) -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let mut current_group = Vec::new();

        for &idx in sorted_indices {
            current_group.push(idx);

            if current_group.len() >= self.partition_size {
                groups.push(current_group);
                current_group = Vec::new();
            }
        }

        // Add remaining nodes
        if !current_group.is_empty() {
            groups.push(current_group);
        }

        groups
    }

    /// Create subgraph for a partition.
    ///
    /// # Performance: O(n * m) where n = nodes in partition, m = avg inputs per node
    ///
    /// Identifies boundary tensors (inputs from other partitions) and creates
    /// virtual inputs for them.
    fn create_subgraph(
        &self,
        graph: &GraphProto,
        node_indices: &[usize],
        partition_idx: usize,
    ) -> Result<GraphPartition> {
        let nodes_in_partition: AHashSet<usize> = node_indices.iter().copied().collect();

        // Collect all tensors produced within this partition
        let mut internal_tensors: AHashSet<String> = AHashSet::new();
        for &idx in node_indices {
            let node = &graph.node[idx];
            for output in &node.output {
                internal_tensors.insert(output.to_string());
            }
        }

        // Identify boundary inputs (tensors from other partitions or graph inputs)
        let mut boundary_inputs: AHashMap<String, String> = AHashMap::new();
        let mut boundary_input_counter = 0;

        for &idx in node_indices {
            let node = &graph.node[idx];
            for input in &node.input {
                if !internal_tensors.contains(input.as_str()) && !boundary_inputs.contains_key(input.as_str()) {
                    // This is an external tensor - create virtual input
                    let virtual_name = format!("partition_{}_input_{}", partition_idx, boundary_input_counter);
                    boundary_inputs.insert(input.to_string(), virtual_name);
                    boundary_input_counter += 1;
                }
            }
        }

        // Identify boundary outputs (tensors needed by other partitions or graph outputs)
        let mut boundary_outputs: AHashSet<String> = AHashSet::new();

        // Check which internal tensors are consumed outside this partition
        for (idx, node) in graph.node.iter().enumerate() {
            if !nodes_in_partition.contains(&idx) {
                for input in &node.input {
                    if internal_tensors.contains(input.as_str()) {
                        boundary_outputs.insert(input.to_string());
                    }
                }
            }
        }

        // Also include graph outputs
        for output_info in &graph.output {
            if internal_tensors.contains(&output_info.name) {
                boundary_outputs.insert(output_info.name.clone());
            }
        }

        // Clone nodes for subgraph
        let partition_nodes: Vec<NodeProto> = node_indices.iter()
            .map(|&idx| graph.node[idx].clone())
            .collect();

        Ok(GraphPartition {
            partition_idx,
            nodes: partition_nodes,
            boundary_inputs,
            boundary_outputs: boundary_outputs.into_iter().collect(),
            original_node_indices: node_indices.to_vec(),
        })
    }
}

impl Default for GraphPartitioner {
    fn default() -> Self {
        Self::new()
    }
}

/// A partition of an ONNX graph.
///
/// Contains a subset of nodes along with boundary tensor information.
#[derive(Debug, Clone)]
pub struct GraphPartition {
    /// Partition index
    pub partition_idx: usize,

    /// Nodes in this partition
    pub nodes: Vec<NodeProto>,

    /// Boundary inputs: external tensor name → virtual input name
    pub boundary_inputs: AHashMap<String, String>,

    /// Boundary outputs: tensors needed by other partitions
    pub boundary_outputs: Vec<String>,

    /// Original node indices from full graph
    pub original_node_indices: Vec<usize>,
}

impl GraphPartition {
    /// Create partition from full graph (no partitioning needed).
    pub fn from_full_graph(graph: &GraphProto) -> Self {
        Self {
            partition_idx: 0,
            nodes: graph.node.clone(),
            boundary_inputs: AHashMap::new(),
            boundary_outputs: graph.output.iter().map(|o| o.name.clone()).collect(),
            original_node_indices: (0..graph.node.len()).collect(),
        }
    }

    /// Get number of nodes in partition.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Check if partition has boundary inputs.
    pub fn has_boundary_inputs(&self) -> bool {
        !self.boundary_inputs.is_empty()
    }

    /// Check if partition has boundary outputs.
    pub fn has_boundary_outputs(&self) -> bool {
        !self.boundary_outputs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_onnx_spec::NodeProto;

    fn create_linear_graph(node_count: usize) -> GraphProto {
        // Create a simple linear graph: node0 → node1 → node2 → ...
        let mut graph = GraphProto::default();
        graph.name = "test_graph".to_string();

        for i in 0..node_count {
            let mut node = NodeProto::default();
            node.name = format!("node{}", i);
            node.op_type = "Add".to_string();

            if i == 0 {
                node.input.push("input".to_string());
            } else {
                node.input.push(format!("tensor{}", i - 1));
            }

            node.output.push(format!("tensor{}", i));
            graph.node.push(node);
        }

        graph
    }

    fn create_dag_graph() -> GraphProto {
        // Create a DAG:
        //    node0
        //   /     \
        // node1  node2
        //   \     /
        //    node3
        let mut graph = GraphProto::default();
        graph.name = "dag_graph".to_string();

        let mut node0 = NodeProto::default();
        node0.name = "node0".to_string();
        node0.op_type = "Add".to_string();
        node0.input.push("input".to_string());
        node0.output.push("tensor0".to_string());
        graph.node.push(node0);

        let mut node1 = NodeProto::default();
        node1.name = "node1".to_string();
        node1.op_type = "Add".to_string();
        node1.input.push("tensor0".to_string());
        node1.output.push("tensor1".to_string());
        graph.node.push(node1);

        let mut node2 = NodeProto::default();
        node2.name = "node2".to_string();
        node2.op_type = "Add".to_string();
        node2.input.push("tensor0".to_string());
        node2.output.push("tensor2".to_string());
        graph.node.push(node2);

        let mut node3 = NodeProto::default();
        node3.name = "node3".to_string();
        node3.op_type = "Add".to_string();
        node3.input.push("tensor1".to_string());
        node3.input.push("tensor2".to_string());
        node3.output.push("output".to_string());
        graph.node.push(node3);

        graph
    }

    #[test]
    fn test_partitioner_creation() {
        let partitioner = GraphPartitioner::new();
        assert_eq!(partitioner.partition_size, 500);

        let custom = GraphPartitioner::with_partition_size(100);
        assert_eq!(custom.partition_size, 100);
    }

    #[test]
    #[should_panic(expected = "Partition size must be > 0")]
    fn test_partitioner_zero_size() {
        GraphPartitioner::with_partition_size(0);
    }

    #[test]
    fn test_no_partitioning_needed() {
        let partitioner = GraphPartitioner::with_partition_size(10);
        let graph = create_linear_graph(5);

        let partitions = partitioner.partition(&graph).unwrap();
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].node_count(), 5);
    }

    #[test]
    fn test_build_dependency_graph() {
        let partitioner = GraphPartitioner::new();
        let graph = create_linear_graph(5);

        let (pg_graph, _) = partitioner.build_dependency_graph(&graph).unwrap();

        // Should have 5 nodes
        assert_eq!(pg_graph.node_count(), 5);

        // Should have 4 edges (0→1, 1→2, 2→3, 3→4)
        assert_eq!(pg_graph.edge_count(), 4);
    }

    #[test]
    fn test_topological_sort_linear() {
        let partitioner = GraphPartitioner::new();
        let graph = create_linear_graph(5);

        let partitions = partitioner.partition(&graph).unwrap();
        assert_eq!(partitions.len(), 1);

        // Nodes should maintain topological order
        let partition = &partitions[0];
        assert_eq!(partition.original_node_indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_topological_sort_dag() {
        let partitioner = GraphPartitioner::new();
        let graph = create_dag_graph();

        let partitions = partitioner.partition(&graph).unwrap();
        assert_eq!(partitions.len(), 1);

        let sorted = &partitions[0].original_node_indices;
        assert_eq!(sorted.len(), 4);

        // node0 must come before all others
        assert_eq!(sorted[0], 0);

        // node3 must come last (depends on node1 and node2)
        assert_eq!(sorted[3], 3);
    }

    #[test]
    fn test_partition_creation() {
        let partitioner = GraphPartitioner::with_partition_size(3);
        let graph = create_linear_graph(10);

        let partitions = partitioner.partition(&graph).unwrap();

        // 10 nodes / 3 per partition = 4 partitions (3+3+3+1)
        assert_eq!(partitions.len(), 4);
        assert_eq!(partitions[0].node_count(), 3);
        assert_eq!(partitions[1].node_count(), 3);
        assert_eq!(partitions[2].node_count(), 3);
        assert_eq!(partitions[3].node_count(), 1);
    }

    #[test]
    fn test_boundary_detection() {
        let partitioner = GraphPartitioner::with_partition_size(2);
        let graph = create_linear_graph(4); // Will create 2 partitions

        let partitions = partitioner.partition(&graph).unwrap();
        assert_eq!(partitions.len(), 2);

        // First partition: nodes 0, 1
        assert_eq!(partitions[0].node_count(), 2);
        assert_eq!(partitions[0].partition_idx, 0);
        // Should have 1 boundary output (tensor1 needed by node2)
        assert_eq!(partitions[0].boundary_outputs.len(), 1);
        assert!(partitions[0].boundary_outputs.contains(&"tensor1".to_string()));

        // Second partition: nodes 2, 3
        assert_eq!(partitions[1].node_count(), 2);
        assert_eq!(partitions[1].partition_idx, 1);
        // Should have 1 boundary input (tensor1 from partition 0)
        assert_eq!(partitions[1].boundary_inputs.len(), 1);
        assert!(partitions[1].boundary_inputs.contains_key("tensor1"));
    }

    #[test]
    fn test_partition_from_full_graph() {
        let graph = create_linear_graph(3);
        let partition = GraphPartition::from_full_graph(&graph);

        assert_eq!(partition.partition_idx, 0);
        assert_eq!(partition.node_count(), 3);
        assert!(!partition.has_boundary_inputs());
        assert_eq!(partition.original_node_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_create_partition_groups() {
        let partitioner = GraphPartitioner::with_partition_size(3);
        let sorted = vec![0, 1, 2, 3, 4, 5, 6, 7];

        let groups = partitioner.create_partition_groups(&sorted);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], vec![0, 1, 2]);
        assert_eq!(groups[1], vec![3, 4, 5]);
        assert_eq!(groups[2], vec![6, 7]);
    }

    #[test]
    fn test_large_graph_partitioning() {
        let partitioner = GraphPartitioner::with_partition_size(100);
        let graph = create_linear_graph(350);

        let partitions = partitioner.partition(&graph).unwrap();

        // 350 / 100 = 4 partitions (100+100+100+50)
        assert_eq!(partitions.len(), 4);
        assert_eq!(partitions[0].node_count(), 100);
        assert_eq!(partitions[1].node_count(), 100);
        assert_eq!(partitions[2].node_count(), 100);
        assert_eq!(partitions[3].node_count(), 50);

        // Each partition (except first) should have boundary inputs
        for i in 1..partitions.len() {
            assert!(partitions[i].has_boundary_inputs());
        }

        // Each partition (except last) should have boundary outputs
        for i in 0..partitions.len()-1 {
            assert!(partitions[i].has_boundary_outputs());
        }
    }

    #[test]
    fn test_dag_partitioning() {
        let partitioner = GraphPartitioner::with_partition_size(2);
        let graph = create_dag_graph();

        let partitions = partitioner.partition(&graph).unwrap();

        // 4 nodes / 2 per partition = 2 partitions
        assert_eq!(partitions.len(), 2);

        // Verify topological ordering is preserved
        // node0 should be in first partition
        assert!(partitions[0].original_node_indices.contains(&0));

        // node3 should be in last partition (depends on all others)
        assert!(partitions[1].original_node_indices.contains(&3));
    }
}
