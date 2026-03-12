//! ORT (ONNX Runtime) runner for cross-validation.
//!
//! Builds single-op ONNX graphs in memory and runs them through ORT to get
//! reference outputs. These are compared against `dispatch_float()` outputs.
//!
//! Gated behind the `conformance` feature.

#[cfg(feature = "conformance")]
pub mod runner {
    use anyhow::{Context, Result};
    use ort::{session::Session, value::Tensor};

    /// Input for ORT: name, shape, flat f32 data.
    pub struct OrtInput {
        pub name: String,
        pub shape: Vec<usize>,
        pub data: Vec<f32>,
    }

    /// Run an in-memory ONNX model (protobuf bytes) through ORT with the given inputs.
    pub fn run_onnx_bytes(model_bytes: &[u8], inputs: Vec<OrtInput>) -> Result<Vec<f32>> {
        let mut session = Session::builder()
            .context("failed to create ORT session builder")?
            .commit_from_memory(model_bytes)
            .context("failed to load ONNX model into ORT")?;

        let tensors: Vec<Tensor<f32>> = inputs
            .iter()
            .map(|inp| {
                let shape: Vec<i64> = inp.shape.iter().map(|&d| d as i64).collect();
                Tensor::<f32>::from_array((shape, inp.data.clone()))
                    .context("creating ORT input tensor")
            })
            .collect::<Result<Vec<_>>>()?;

        let ort_inputs: Vec<(String, ort::value::DynValue)> = inputs
            .iter()
            .zip(tensors)
            .map(|(inp, tensor)| (inp.name.clone(), tensor.into_dyn()))
            .collect();

        let outputs = session.run(ort_inputs).context("ORT session run failed")?;

        let output = outputs.values().next().context("no outputs from ORT")?;

        let (_shape, data) = output
            .try_extract_tensor::<f32>()
            .context("extracting f32 tensor from ORT output")?;

        Ok(data.to_vec())
    }
}

/// Build a minimal single-op ONNX model as raw protobuf bytes.
///
/// This module constructs ONNX protobuf by hand (no dependency on onnx crate)
/// since hologram-ai-onnx's proto module is private.
#[cfg(feature = "conformance")]
pub mod onnx_builder {
    /// Build a minimal ONNX model for a unary elementwise op.
    /// Graph: input "X" [1, size] -> op -> output "Y" [1, size]
    pub fn unary_op(op_type: &str, size: usize) -> Vec<u8> {
        build_model(
            op_type,
            &[("X", &[1, size])],
            &[("Y", &[1, size])],
            &[],
        )
    }

    /// Build a minimal ONNX model for a binary elementwise op.
    /// Graph: inputs "A" [1, size], "B" [1, size] -> op -> output "Y" [1, size]
    pub fn binary_op(op_type: &str, size: usize) -> Vec<u8> {
        build_model(
            op_type,
            &[("A", &[1, size]), ("B", &[1, size])],
            &[("Y", &[1, size])],
            &[],
        )
    }

    /// Build a Softmax ONNX model.
    /// Graph: input "X" [rows, size] -> Softmax(axis=-1) -> output "Y" [rows, size]
    pub fn softmax(rows: usize, size: usize) -> Vec<u8> {
        build_model(
            "Softmax",
            &[("input", &[rows, size])],
            &[("output", &[rows, size])],
            &[("axis", AttrVal::Int(-1))],
        )
    }

    /// Build a MatMul ONNX model.
    /// Graph: inputs "A" [m, k], "B" [k, n] -> MatMul -> output "Y" [m, n]
    pub fn matmul(m: usize, k: usize, n: usize) -> Vec<u8> {
        build_model(
            "MatMul",
            &[("A", &[m, k]), ("B", &[k, n])],
            &[("Y", &[m, n])],
            &[],
        )
    }

    /// Build a Gemm ONNX model.
    pub fn gemm(
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    ) -> Vec<u8> {
        let a_shape = if trans_a { vec![k, m] } else { vec![m, k] };
        let b_shape = if trans_b { vec![n, k] } else { vec![k, n] };
        build_model(
            "Gemm",
            &[("A", &a_shape), ("B", &b_shape), ("C", &[m, n])],
            &[("Y", &[m, n])],
            &[
                ("alpha", AttrVal::Float(alpha)),
                ("beta", AttrVal::Float(beta)),
                ("transA", AttrVal::Int(trans_a as i64)),
                ("transB", AttrVal::Int(trans_b as i64)),
            ],
        )
    }

    // ── Composite ops (multi-node ONNX graphs) ─────────────────────────────

    /// Build an RmsNorm ONNX model from primitives.
    /// RmsNorm(x, weight, eps) = x / sqrt(mean(x^2) + eps) * weight
    ///
    /// Graph:
    ///   X [rows, size], Weight [size] ->
    ///   Pow(X, 2) -> ReduceMean -> Add(eps) -> Sqrt -> Div(X, .) -> Mul(., Weight) -> Y
    pub fn rms_norm(rows: usize, size: usize, epsilon: f32) -> Vec<u8> {
        let nodes = vec![
            Node::new("Pow", &["X", "two"], &["x_sq"]),
            Node::with_attrs("ReduceMean", &["x_sq"], &["mean_sq"],
                &[("axes", AttrVal::Ints(vec![-1])), ("keepdims", AttrVal::Int(1))]),
            Node::new("Add", &["mean_sq", "eps"], &["mean_plus_eps"]),
            Node::new("Sqrt", &["mean_plus_eps"], &["rms"]),
            Node::new("Div", &["X", "rms"], &["normed"]),
            Node::new("Mul", &["normed", "Weight"], &["Y"]),
        ];
        let initializers = vec![
            Initializer::scalar("two", 2.0_f32),
            Initializer::scalar("eps", epsilon),
        ];
        build_multi_node_model(
            &nodes,
            &[("X", &[rows, size]), ("Weight", &[size])],
            &[("Y", &[rows, size])],
            &initializers,
        )
    }

    /// Build a LayerNorm ONNX model from primitives.
    /// LayerNorm(x, weight, bias, eps) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
    pub fn layer_norm(rows: usize, size: usize, epsilon: f32) -> Vec<u8> {
        let nodes = vec![
            Node::with_attrs("ReduceMean", &["X"], &["mean"],
                &[("axes", AttrVal::Ints(vec![-1])), ("keepdims", AttrVal::Int(1))]),
            Node::new("Sub", &["X", "mean"], &["x_centered"]),
            Node::new("Pow", &["x_centered", "two"], &["x_sq"]),
            Node::with_attrs("ReduceMean", &["x_sq"], &["var"],
                &[("axes", AttrVal::Ints(vec![-1])), ("keepdims", AttrVal::Int(1))]),
            Node::new("Add", &["var", "eps"], &["var_eps"]),
            Node::new("Sqrt", &["var_eps"], &["std"]),
            Node::new("Div", &["x_centered", "std"], &["normed"]),
            Node::new("Mul", &["normed", "Weight"], &["scaled"]),
            Node::new("Add", &["scaled", "Bias"], &["Y"]),
        ];
        let initializers = vec![
            Initializer::scalar("two", 2.0_f32),
            Initializer::scalar("eps", epsilon),
        ];
        build_multi_node_model(
            &nodes,
            &[("X", &[rows, size]), ("Weight", &[size]), ("Bias", &[size])],
            &[("Y", &[rows, size])],
            &initializers,
        )
    }

    // ── Protobuf encoding helpers ───────────────────────────────────────────

    enum AttrVal {
        Int(i64),
        Float(f32),
        Ints(Vec<i64>),
    }

    struct Node {
        op_type: &'static str,
        inputs: Vec<&'static str>,
        outputs: Vec<&'static str>,
        attrs: Vec<(&'static str, AttrVal)>,
    }

    impl Node {
        fn new(op_type: &'static str, inputs: &[&'static str], outputs: &[&'static str]) -> Self {
            Self {
                op_type,
                inputs: inputs.to_vec(),
                outputs: outputs.to_vec(),
                attrs: vec![],
            }
        }

        fn with_attrs(
            op_type: &'static str,
            inputs: &[&'static str],
            outputs: &[&'static str],
            attrs: &[(&'static str, AttrVal)],
        ) -> Self {
            Self {
                op_type,
                inputs: inputs.to_vec(),
                outputs: outputs.to_vec(),
                attrs: attrs.iter().map(|(n, v)| (*n, v.clone())).collect(),
            }
        }
    }

    struct Initializer {
        name: &'static str,
        data: Vec<f32>,
        shape: Vec<usize>,
    }

    impl Initializer {
        fn scalar(name: &'static str, val: f32) -> Self {
            Self { name, data: vec![val], shape: vec![] }
        }
    }

    impl Clone for AttrVal {
        fn clone(&self) -> Self {
            match self {
                Self::Int(v) => Self::Int(*v),
                Self::Float(v) => Self::Float(*v),
                Self::Ints(v) => Self::Ints(v.clone()),
            }
        }
    }

    fn build_model(
        op_type: &str,
        inputs: &[(&str, &[usize])],
        outputs: &[(&str, &[usize])],
        attrs: &[(&str, AttrVal)],
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // Single-node graph
        let node = encode_single_node(
            op_type,
            &inputs.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            &outputs.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            attrs,
        );

        let graph_inputs: Vec<Vec<u8>> = inputs
            .iter()
            .map(|(name, shape)| encode_value_info(name, shape))
            .collect();
        let graph_outputs: Vec<Vec<u8>> = outputs
            .iter()
            .map(|(name, shape)| encode_value_info(name, shape))
            .collect();

        let mut graph = Vec::new();
        write_field(&mut graph, 1, &node);
        write_string(&mut graph, 2, "test_graph");
        for inp in &graph_inputs {
            write_field(&mut graph, 11, inp);
        }
        for out in &graph_outputs {
            write_field(&mut graph, 12, out);
        }

        write_varint_field(&mut buf, 1, 8);
        write_field(&mut buf, 7, &graph);
        write_field(&mut buf, 8, &encode_opset(17));
        buf
    }

    fn encode_single_node(
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attrs: &[(&str, AttrVal)],
    ) -> Vec<u8> {
        let mut node = Vec::new();
        for name in inputs {
            write_string(&mut node, 1, name);
        }
        for name in outputs {
            write_string(&mut node, 2, name);
        }
        write_string(&mut node, 4, op_type);
        for (name, val) in attrs {
            let attr = encode_attr(name, val);
            write_field(&mut node, 5, &attr);
        }
        node
    }

    fn build_multi_node_model(
        nodes: &[Node],
        inputs: &[(&str, &[usize])],
        outputs: &[(&str, &[usize])],
        initializers: &[Initializer],
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // Graph inputs (ValueInfoProto)
        let graph_inputs: Vec<Vec<u8>> = inputs
            .iter()
            .map(|(name, shape)| encode_value_info(name, shape))
            .collect();

        // Graph outputs (ValueInfoProto)
        let graph_outputs: Vec<Vec<u8>> = outputs
            .iter()
            .map(|(name, shape)| encode_value_info(name, shape))
            .collect();

        // GraphProto (onnx.proto3 field numbers)
        let mut graph = Vec::new();
        // field 1: node (repeated)
        for node in nodes {
            let encoded = encode_node_struct(node);
            write_field(&mut graph, 1, &encoded);
        }
        // field 2: name
        write_string(&mut graph, 2, "test_graph");
        // field 5: initializer (repeated TensorProto)
        for init in initializers {
            let encoded = encode_initializer(init);
            write_field(&mut graph, 5, &encoded);
        }
        // field 11: input (repeated ValueInfoProto)
        for inp in &graph_inputs {
            write_field(&mut graph, 11, inp);
        }
        // field 12: output (repeated ValueInfoProto)
        for out in &graph_outputs {
            write_field(&mut graph, 12, out);
        }

        // ModelProto
        // field 1: ir_version (int64) = 8
        write_varint_field(&mut buf, 1, 8);
        // field 7: graph
        write_field(&mut buf, 7, &graph);
        // field 8: opset_import
        let opset = encode_opset(17);
        write_field(&mut buf, 8, &opset);

        buf
    }

    fn encode_node_struct(node: &Node) -> Vec<u8> {
        let mut buf = Vec::new();
        // field 1: input (repeated string)
        for name in &node.inputs {
            write_string(&mut buf, 1, name);
        }
        // field 2: output (repeated string)
        for name in &node.outputs {
            write_string(&mut buf, 2, name);
        }
        // field 4: op_type
        write_string(&mut buf, 4, node.op_type);
        // field 5: attribute (repeated)
        for (name, val) in &node.attrs {
            let attr = encode_attr(name, val);
            write_field(&mut buf, 5, &attr);
        }
        buf
    }

    fn encode_attr(name: &str, val: &AttrVal) -> Vec<u8> {
        let mut attr = Vec::new();
        // field 1: name
        write_string(&mut attr, 1, name);
        match val {
            AttrVal::Int(v) => {
                // field 3: i (int64)
                write_varint_field(&mut attr, 3, *v as u64);
                // field 20: type = INT (2)
                write_varint_field(&mut attr, 20, 2);
            }
            AttrVal::Float(v) => {
                // field 2: f (float)
                write_fixed32_field(&mut attr, 2, v.to_bits());
                // field 20: type = FLOAT (1)
                write_varint_field(&mut attr, 20, 1);
            }
            AttrVal::Ints(vals) => {
                // field 8: ints (repeated int64)
                for v in vals {
                    write_varint_field(&mut attr, 8, *v as u64);
                }
                // field 20: type = INTS (7)
                write_varint_field(&mut attr, 20, 7);
            }
        }
        attr
    }

    /// Encode a TensorProto initializer (constant tensor).
    fn encode_initializer(init: &Initializer) -> Vec<u8> {
        let mut tp = Vec::new();
        // field 1: dims (repeated int64)
        for &d in &init.shape {
            write_varint_field(&mut tp, 1, d as u64);
        }
        // field 2: data_type = FLOAT (1)
        write_varint_field(&mut tp, 2, 1);
        // field 4: float_data (repeated float, packed)
        // For small tensors, use field 4 (float_data) with packed encoding
        let mut float_bytes = Vec::new();
        for &v in &init.data {
            float_bytes.extend_from_slice(&v.to_le_bytes());
        }
        write_field(&mut tp, 4, &float_bytes);
        // field 8: name
        write_string(&mut tp, 8, init.name);
        tp
    }

    fn encode_value_info(name: &str, shape: &[usize]) -> Vec<u8> {
        let mut vi = Vec::new();
        // field 1: name
        write_string(&mut vi, 1, name);

        // field 2: type (TypeProto)
        let mut type_proto = Vec::new();
        // TypeProto field 1: tensor_type
        let mut tensor_type = Vec::new();
        // TensorTypeProto field 1: elem_type = FLOAT (1)
        write_varint_field(&mut tensor_type, 1, 1);
        // TensorTypeProto field 2: shape (TensorShapeProto)
        let mut shape_proto = Vec::new();
        for &dim in shape {
            // Dimension field 1: dim_value (int64)
            let mut dim_proto = Vec::new();
            write_varint_field(&mut dim_proto, 1, dim as u64);
            write_field(&mut shape_proto, 1, &dim_proto);
        }
        write_field(&mut tensor_type, 2, &shape_proto);
        write_field(&mut type_proto, 1, &tensor_type);
        write_field(&mut vi, 2, &type_proto);

        vi
    }

    fn encode_opset(version: u64) -> Vec<u8> {
        let mut opset = Vec::new();
        // field 1: domain (string, empty = default ONNX domain)
        write_string(&mut opset, 1, "");
        // field 2: version (int64)
        write_varint_field(&mut opset, 2, version);
        opset
    }

    // ── Low-level protobuf encoding ─────────────────────────────────────────

    fn write_varint(buf: &mut Vec<u8>, val: u64) {
        let mut v = val;
        loop {
            let byte = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                buf.push(byte);
                break;
            }
            buf.push(byte | 0x80);
        }
    }

    fn write_field(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
        // wire type 2 (length-delimited)
        write_varint(buf, ((field as u64) << 3) | 2);
        write_varint(buf, data.len() as u64);
        buf.extend_from_slice(data);
    }

    fn write_string(buf: &mut Vec<u8>, field: u32, s: &str) {
        write_field(buf, field, s.as_bytes());
    }

    fn write_varint_field(buf: &mut Vec<u8>, field: u32, val: u64) {
        // wire type 0 (varint)
        write_varint(buf, (field as u64) << 3);
        write_varint(buf, val);
    }

    fn write_fixed32_field(buf: &mut Vec<u8>, field: u32, val: u32) {
        // wire type 5 (32-bit)
        write_varint(buf, ((field as u64) << 3) | 5);
        buf.extend_from_slice(&val.to_le_bytes());
    }
}
