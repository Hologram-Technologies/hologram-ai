//! Building hologram OperationGraph from ONNX models.
//!
//! This module provides [`GraphBuilder`], a builder pattern for converting ONNX models
//! to hologram's `OperationGraph` format. All state is encapsulated in the builder.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail};
use hologram::compiler::{OpKind, OpNode, OperationGraph};

use crate::{dtypes, ops, parser, proto};

/// Builder for constructing hologram `OperationGraph` from ONNX models.
///
/// Encapsulates all translation state:
/// - The hologram graph being built
/// - ONNX name → node ID mapping
/// - Node ID counter
///
/// # Example
///
/// ```ignore
/// let graph = GraphBuilder::from_onnx(&model)?;
/// ```
pub struct GraphBuilder {
    graph: OperationGraph,
    value_to_node: HashMap<String, u32>,
    next_id: u32,
}

impl GraphBuilder {
    /// Create a new empty graph builder.
    pub fn new() -> Self {
        Self {
            graph: OperationGraph::default(),
            value_to_node: HashMap::new(),
            next_id: 0,
        }
    }

    /// Build a hologram graph from an ONNX model.
    pub fn from_onnx(model: &proto::ModelProto) -> Result<OperationGraph> {
        let graph_proto = model.graph.as_ref().context("ONNX model has no graph")?;
        let mut builder = Self::new();
        builder.build_from_graph(graph_proto)?;
        Ok(builder.finish())
    }

    /// Build from an ONNX GraphProto.
    fn build_from_graph(&mut self, graph: &proto::GraphProto) -> Result<()> {
        let initializer_names: HashSet<_> =
            graph.initializer.iter().map(|i| i.name.as_str()).collect();

        // 1. Inputs (skip initializers)
        for input in &graph.input {
            if !initializer_names.contains(input.name.as_str()) {
                self.add_input(input)?;
            }
        }

        // 2. Initializers (constants/weights)
        for init in &graph.initializer {
            self.add_initializer(init)?;
        }

        // 3. Operations
        for node in &graph.node {
            self.add_operation(node)?;
        }

        // 4. Outputs
        for output in &graph.output {
            self.add_output(output)?;
        }

        Ok(())
    }

    fn add_input(&mut self, input: &proto::ValueInfoProto) -> Result<u32> {
        use hologram::compiler::DType;

        let name = &input.name;
        let shape = parser::extract_shape(input)?;
        let onnx_dtype = parser::extract_dtype(input)?;

        // Convert I64/I32 inputs to F32 for hologram runtime compatibility.
        // Token IDs like [13959, 1566, ...] work fine as F32 (sufficient precision for vocab indices).
        let dtype = match onnx_dtype {
            DType::I64 | DType::I32 => {
                tracing::debug!(
                    "ONNX input '{}': converting {:?} to F32 for hologram runtime",
                    name,
                    onnx_dtype
                );
                DType::F32
            }
            _ => onnx_dtype,
        };

        tracing::info!(
            "ONNX input '{}': shape {:?}, dtype {:?}",
            name,
            shape,
            dtype
        );

        let id = self.alloc_id();
        self.graph
            .add_node(OpNode::new(id, OpKind::Input, shape, dtype).with_name(name.clone()));
        self.graph.add_input(name, id);
        self.value_to_node.insert(name.clone(), id);
        Ok(id)
    }

    fn add_initializer(&mut self, init: &proto::TensorProto) -> Result<u32> {
        let name = &init.name;
        let shape: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
        let dtype = dtypes::from_onnx(init.data_type)?;

        let id = self.alloc_id();
        self.graph
            .add_node(OpNode::new(id, OpKind::Constant, shape, dtype).with_name(name.clone()));
        self.value_to_node.insert(name.clone(), id);
        // Link constant data to this specific node ID for proper serialization
        self.graph
            .add_constant_for_node(id, ops::extract_constant_data(init)?);
        Ok(id)
    }

    fn add_operation(&mut self, node: &proto::NodeProto) -> Result<()> {
        // Expansion ops (like Gemm) need special handling
        if ops::requires_expansion(&node.op_type) {
            return self.expand_operation(node);
        }

        let result = ops::translate_node_full(node, &self.value_to_node, &self.graph)?;
        let output_name = node.output.first().context("Node has no output")?.clone();

        // Handle broadcasting for binary operations
        if let Some(ref broadcast_info) = result.broadcast_info {
            return self.expand_broadcast_binary(node, &output_name, &result, broadcast_info);
        }

        let id = self.alloc_id();

        if let Some(const_data) = result.constant_data {
            // Constant-folded - link data to this node ID for proper serialization
            self.graph.add_node(
                OpNode::new(id, OpKind::Constant, result.shape, result.dtype)
                    .with_name(output_name.clone()),
            );
            self.graph.add_constant_for_node(id, const_data);
        } else {
            // Runtime operation
            self.graph.add_node(
                OpNode::new(id, result.op_kind, result.shape, result.dtype)
                    .with_name(output_name.clone()),
            );

            // Add edges from data inputs only
            let data_inputs = result.data_input_count.unwrap_or(node.input.len());
            for input_name in node.input.iter().take(data_inputs) {
                if let Some(&input_id) = self.value_to_node.get(input_name) {
                    self.graph.add_edge(input_id, id);
                }
            }
        }

        self.value_to_node.insert(output_name, id);
        Ok(())
    }

    fn expand_operation(&mut self, node: &proto::NodeProto) -> Result<()> {
        match node.op_type.as_str() {
            "Gemm" => self.expand_gemm(node),
            op => bail!("Unknown expansion operation: {}", op),
        }
    }

    fn expand_gemm(&mut self, node: &proto::NodeProto) -> Result<()> {
        let input_a = node.input.first().context("Gemm missing A")?;
        let input_b = node.input.get(1).context("Gemm missing B")?;
        let input_c = node.input.get(2);

        let trans_b = node
            .attribute
            .iter()
            .any(|a| a.name == "transB" && a.i == 1);

        // Optional transpose
        let weight_name = if trans_b {
            let w_id = *self
                .value_to_node
                .get(input_b)
                .context("Weight not found")?;
            let (w_shape, w_dtype) = {
                let w = &self.graph.nodes[w_id as usize];
                if w.shape.len() != 2 {
                    bail!("Gemm transB requires 2D weight, got {:?}", w.shape);
                }
                (w.shape.clone(), w.dtype)
            };

            let t_shape = vec![w_shape[1], w_shape[0]];
            let t_name = format!("{}_T", input_b);
            let t_id = self.alloc_id();

            self.graph.add_node(
                OpNode::new(
                    t_id,
                    OpKind::Transpose { perm: vec![1, 0] },
                    t_shape,
                    w_dtype,
                )
                .with_name(t_name.clone()),
            );
            self.graph.add_edge(w_id, t_id);
            self.value_to_node.insert(t_name.clone(), t_id);
            t_name
        } else {
            input_b.clone()
        };

        // MatMul
        let mm_proto = proto::NodeProto {
            input: vec![input_a.clone(), weight_name.clone()],
            output: vec![format!("{}_mm", node.name)],
            op_type: "MatMul".to_string(),
            ..Default::default()
        };
        let (mm_op, mm_shape, mm_dtype) =
            ops::translate_node(&mm_proto, &self.value_to_node, &self.graph)?;

        let mm_name = mm_proto.output[0].clone();
        let mm_id = self.alloc_id();
        self.graph
            .add_node(OpNode::new(mm_id, mm_op, mm_shape, mm_dtype).with_name(mm_name.clone()));

        if let Some(&a_id) = self.value_to_node.get(input_a) {
            self.graph.add_edge(a_id, mm_id);
        }
        if let Some(&b_id) = self.value_to_node.get(&weight_name) {
            self.graph.add_edge(b_id, mm_id);
        }
        self.value_to_node.insert(mm_name.clone(), mm_id);

        // Output name
        let out_name = node.output.first().context("Gemm missing output")?.clone();

        // Add bias if present
        if let Some(bias) = input_c {
            let add_proto = proto::NodeProto {
                input: vec![mm_name.clone(), bias.clone()],
                output: vec![out_name.clone()],
                op_type: "Add".to_string(),
                ..Default::default()
            };
            let (add_op, add_shape, add_dtype) =
                ops::translate_node(&add_proto, &self.value_to_node, &self.graph)?;

            let add_id = self.alloc_id();
            self.graph.add_node(
                OpNode::new(add_id, add_op, add_shape, add_dtype).with_name(out_name.clone()),
            );
            self.graph.add_edge(mm_id, add_id);
            if let Some(&b_id) = self.value_to_node.get(bias) {
                self.graph.add_edge(b_id, add_id);
            }
            self.value_to_node.insert(out_name, add_id);
        } else {
            self.value_to_node.insert(out_name, mm_id);
        }

        Ok(())
    }

    /// Expand a binary operation that requires broadcasting.
    ///
    /// This inserts Expand nodes for inputs that need to be broadcast to the output shape,
    /// then connects them to the binary operation.
    fn expand_broadcast_binary(
        &mut self,
        node: &proto::NodeProto,
        output_name: &str,
        result: &ops::TranslateResult,
        broadcast_info: &ops::BroadcastInfo,
    ) -> Result<()> {
        let input_a = node.input.first().context("Binary op missing input A")?;
        let input_b = node.input.get(1).context("Binary op missing input B")?;

        // Get the input node IDs
        let a_id = *self
            .value_to_node
            .get(input_a)
            .context("Input A not found")?;
        let b_id = *self
            .value_to_node
            .get(input_b)
            .context("Input B not found")?;

        // Determine the actual input IDs to connect to the binary op
        // (either original or expanded)
        let mut effective_a_id = a_id;
        let mut effective_b_id = b_id;

        // Insert Expand for input A if needed
        if broadcast_info.a_needs_broadcast {
            let expand_name = format!("{}_expand_a", output_name);
            let expand_id = self.alloc_id();

            tracing::debug!(
                "Inserting Expand for A: {:?} -> {:?}",
                broadcast_info.a_shape,
                broadcast_info.output_shape
            );

            self.graph.add_node(
                OpNode::new(
                    expand_id,
                    OpKind::Expand {
                        shape: broadcast_info.output_shape.clone(),
                    },
                    broadcast_info.output_shape.clone(),
                    result.dtype,
                )
                .with_name(expand_name.clone()),
            );
            self.graph.add_edge(a_id, expand_id);
            self.value_to_node.insert(expand_name, expand_id);
            effective_a_id = expand_id;
        }

        // Insert Expand for input B if needed
        if broadcast_info.b_needs_broadcast {
            let expand_name = format!("{}_expand_b", output_name);
            let expand_id = self.alloc_id();

            tracing::debug!(
                "Inserting Expand for B: {:?} -> {:?}",
                broadcast_info.b_shape,
                broadcast_info.output_shape
            );

            self.graph.add_node(
                OpNode::new(
                    expand_id,
                    OpKind::Expand {
                        shape: broadcast_info.output_shape.clone(),
                    },
                    broadcast_info.output_shape.clone(),
                    result.dtype,
                )
                .with_name(expand_name.clone()),
            );
            self.graph.add_edge(b_id, expand_id);
            self.value_to_node.insert(expand_name, expand_id);
            effective_b_id = expand_id;
        }

        // Now create the binary operation node
        let binary_id = self.alloc_id();
        self.graph.add_node(
            OpNode::new(
                binary_id,
                result.op_kind.clone(),
                result.shape.clone(),
                result.dtype,
            )
            .with_name(output_name.to_string()),
        );

        // Connect the (possibly expanded) inputs to the binary op
        self.graph.add_edge(effective_a_id, binary_id);
        self.graph.add_edge(effective_b_id, binary_id);

        self.value_to_node
            .insert(output_name.to_string(), binary_id);
        Ok(())
    }

    fn add_output(&mut self, output: &proto::ValueInfoProto) -> Result<()> {
        let name = &output.name;
        if let Some(&src_id) = self.value_to_node.get(name) {
            let (shape, dtype) = {
                let src = &self.graph.nodes[src_id as usize];
                (src.shape.clone(), src.dtype)
            };
            let id = self.alloc_id();

            self.graph
                .add_node(OpNode::new(id, OpKind::Output, shape, dtype).with_name(name.clone()));
            self.graph.add_edge(src_id, id);
            self.graph.add_output(name, id);
        }
        Ok(())
    }

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Consume builder and return the completed graph.
    pub fn finish(self) -> OperationGraph {
        self.graph
    }

    /// Get reference to graph (for TranslateContext).
    #[allow(dead_code)]
    pub fn graph(&self) -> &OperationGraph {
        &self.graph
    }

    /// Get reference to value mapping.
    #[allow(dead_code)]
    pub fn value_map(&self) -> &HashMap<String, u32> {
        &self.value_to_node
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a hologram OperationGraph from an ONNX ModelProto.
pub fn build_graph(model: &proto::ModelProto) -> Result<OperationGraph> {
    GraphBuilder::from_onnx(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_value_info(name: &str, dims: &[i64], dtype: i32) -> proto::ValueInfoProto {
        proto::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(proto::TypeProto {
                value: Some(proto::type_proto::Value::TensorType(
                    proto::type_proto::Tensor {
                        elem_type: dtype,
                        shape: Some(proto::TensorShapeProto {
                            dim: dims
                                .iter()
                                .map(|&d| proto::tensor_shape_proto::Dimension {
                                    value: Some(
                                        proto::tensor_shape_proto::dimension::Value::DimValue(d),
                                    ),
                                    ..Default::default()
                                })
                                .collect(),
                        }),
                    },
                )),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn test_simple_graph() {
        let model = proto::ModelProto {
            graph: Some(proto::GraphProto {
                input: vec![create_value_info("x", &[1, 10], 1)],
                output: vec![create_value_info("y", &[1, 10], 1)],
                node: vec![proto::NodeProto {
                    input: vec!["x".to_string()],
                    output: vec!["y".to_string()],
                    op_type: "Relu".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let graph = build_graph(&model).unwrap();
        assert_eq!(graph.nodes.len(), 3); // Input, Relu, Output
    }

    #[test]
    fn test_builder_id_allocation() {
        let mut builder = GraphBuilder::new();
        assert_eq!(builder.alloc_id(), 0);
        assert_eq!(builder.alloc_id(), 1);
        assert_eq!(builder.alloc_id(), 2);
    }

    fn create_tensor_initializer(name: &str, dims: &[i64], data: Vec<f32>) -> proto::TensorProto {
        proto::TensorProto {
            name: name.to_string(),
            dims: dims.to_vec(),
            data_type: 1, // F32
            float_data: data,
            ..Default::default()
        }
    }

    #[test]
    fn test_broadcast_sub_inserts_expand() {
        // Test case: [1, 512] - [1] should insert Expand for the scalar
        let model = proto::ModelProto {
            graph: Some(proto::GraphProto {
                input: vec![
                    create_value_info("hidden", &[1, 512], 1), // F32
                ],
                initializer: vec![
                    // Scalar initializer [1]
                    create_tensor_initializer("mean", &[1], vec![0.5]),
                ],
                output: vec![create_value_info("out", &[1, 512], 1)],
                node: vec![proto::NodeProto {
                    input: vec!["hidden".to_string(), "mean".to_string()],
                    output: vec!["out".to_string()],
                    op_type: "Sub".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let graph = build_graph(&model).unwrap();

        // Should have: Input, Constant(mean), Expand(mean->512), Sub, Output
        // Count Expand nodes
        let expand_count = graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, OpKind::Expand { .. }))
            .count();

        assert_eq!(
            expand_count, 1,
            "Should insert 1 Expand node for scalar broadcast"
        );

        // Verify the Expand node has the correct target shape
        let expand_node = graph
            .nodes
            .iter()
            .find(|n| matches!(n.op, OpKind::Expand { .. }))
            .expect("Should have Expand node");

        assert_eq!(
            expand_node.shape,
            vec![1, 512],
            "Expand should target [1, 512]"
        );
    }

    #[test]
    fn test_broadcast_add_both_inputs() {
        // Test case: [1, 5] + [5, 1] -> [5, 5]
        // Both inputs need expansion
        let model = proto::ModelProto {
            graph: Some(proto::GraphProto {
                input: vec![
                    create_value_info("a", &[1, 5], 1),
                    create_value_info("b", &[5, 1], 1),
                ],
                output: vec![create_value_info("out", &[5, 5], 1)],
                node: vec![proto::NodeProto {
                    input: vec!["a".to_string(), "b".to_string()],
                    output: vec!["out".to_string()],
                    op_type: "Add".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let graph = build_graph(&model).unwrap();

        // Should have: Input(a), Input(b), Expand(a), Expand(b), Add, Output
        let expand_count = graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, OpKind::Expand { .. }))
            .count();

        assert_eq!(
            expand_count, 2,
            "Should insert 2 Expand nodes when both inputs need broadcast"
        );

        // Both Expand nodes should target [5, 5]
        for node in &graph.nodes {
            if matches!(node.op, OpKind::Expand { .. }) {
                assert_eq!(node.shape, vec![5, 5], "Expand should target [5, 5]");
            }
        }
    }

    #[test]
    fn test_no_broadcast_same_shapes() {
        // Test case: [1, 10] + [1, 10] -> no Expand needed
        let model = proto::ModelProto {
            graph: Some(proto::GraphProto {
                input: vec![
                    create_value_info("a", &[1, 10], 1),
                    create_value_info("b", &[1, 10], 1),
                ],
                output: vec![create_value_info("out", &[1, 10], 1)],
                node: vec![proto::NodeProto {
                    input: vec!["a".to_string(), "b".to_string()],
                    output: vec!["out".to_string()],
                    op_type: "Add".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let graph = build_graph(&model).unwrap();

        // Should have: Input(a), Input(b), Add, Output - NO Expand
        let expand_count = graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, OpKind::Expand { .. }))
            .count();

        assert_eq!(expand_count, 0, "No Expand needed when shapes match");
    }
}
