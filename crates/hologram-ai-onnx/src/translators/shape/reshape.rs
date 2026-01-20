//! Reshape operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{ConstantData, Dim, GraphBuilder, NodeIndex, NodeOp, Shape};

/// Translator for ONNX Reshape operation.
///
/// Reshape changes the dimensions of a tensor without changing its data.
///
/// # Inputs
/// - data: Input tensor to reshape
/// - shape: 1D tensor specifying the target shape
///
/// # Attributes
/// - allowzero (opset 14+): If 1, allows 0 in shape to mean "copy from input"
///
/// # Shape Semantics
/// - Positive values: Use as dimension size
/// - -1: Infer dimension from input size
/// - 0 (with allowzero=0): Copy dimension from input (default)
/// - 0 (with allowzero=1): Use 0 as actual dimension size
#[derive(Debug, Default)]
pub struct ReshapeTranslator;

impl OnnxTranslator for ReshapeTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Reshape"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(2)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let data = inputs[0];
        let shape_input = inputs[1];

        // Check for allowzero attribute (ONNX opset 14+)
        let allow_zero = node.get_int_or("allowzero", 0) != 0;

        // Get shape node to check if it's constant
        let shape_node = builder.graph().node(shape_input).ok_or_else(|| {
            TranslationError::IrBuilder("Reshape: shape input not found".to_string())
        })?;

        // Check if shape is constant for optimization
        let new_shape = match &shape_node.op.op {
            NodeOp::Constant { data } => match data {
                ConstantData::I64(values) => Some(values.clone()),
                ConstantData::I32(values) => Some(values.iter().map(|&v| v as i64).collect()),
                _ => None,
            },
            _ => None,
        };

        if let Some(shape_values) = new_shape {
            // Check for special values that require dynamic handling
            let has_infer = shape_values.contains(&-1);
            let has_zero = allow_zero && shape_values.contains(&0);

            if !has_infer && !has_zero {
                // Simple static reshape (no inference needed)
                tracing::debug!("Reshape: static path, new_shape = {:?}", shape_values);
                let result = builder
                    .reshape(data, shape_values)
                    .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
                return Ok(vec![result]);
            }
        } else if let Some(shape_values) =
            try_resolve_shape_from_graph(builder, data, shape_input, allow_zero)
        {
            tracing::debug!(
                "Reshape: resolved shape graph, new_shape = {:?}",
                shape_values
            );
            let result = builder
                .reshape(data, shape_values)
                .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
            return Ok(vec![result]);
        }

        // Dynamic reshape path (supports runtime shapes, -1 inference, and allowzero)
        tracing::debug!("Reshape: dynamic path, allow_zero = {}", allow_zero);
        let result = builder
            .reshape_dynamic(data, shape_input, allow_zero)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        Ok(vec![result])
    }
}

fn try_resolve_shape_from_graph(
    builder: &GraphBuilder,
    data: NodeIndex,
    shape_input: NodeIndex,
    allow_zero: bool,
) -> Option<Vec<i64>> {
    let data_node = builder.graph().node(data)?;
    let input_shape = &data_node.op.shape;

    let mut evaluator = ShapeEvaluator::new(builder, input_shape);
    let mut dims = evaluator.eval(shape_input)?;

    if dims.is_empty() {
        return None;
    }

    // Handle zero dimensions based on allow_zero flag
    if !allow_zero {
        for (i, dim) in dims.iter_mut().enumerate() {
            if matches!(dim, Dim::Static(0)) {
                let replacement = input_shape.dims.get(i)?;
                *dim = replacement.clone();
            }
        }
    } else if dims.iter().any(|dim| matches!(dim, Dim::Static(0))) {
        return None;
    }

    // If any dimension is symbolic, fall through to reshape_dynamic
    if dims.iter().any(|d| matches!(d, Dim::Symbolic(_))) {
        return None;
    }

    convert_dims_to_shape_values(dims)
}

/// Convert evaluated dims to shape values for reshape.
fn convert_dims_to_shape_values(dims: Vec<Dim>) -> Option<Vec<i64>> {
    let mut dynamic_count = 0usize;
    let mut result = Vec::with_capacity(dims.len());

    for dim in dims {
        match dim {
            Dim::Static(value) => result.push(value as i64),
            Dim::Dynamic => {
                dynamic_count += 1;
                if dynamic_count > 1 {
                    return None;
                }
                result.push(-1);
            }
            Dim::Symbolic(_) => unreachable!(), // Already checked in caller
        }
    }

    Some(result)
}

/// Evaluates shape tensors by walking the computation graph.
///
/// Handles common shape computation patterns like Shape → Gather → Concat
/// that appear in models with dynamic shapes.
struct ShapeEvaluator<'a> {
    builder: &'a GraphBuilder,
    reshape_input_shape: &'a Shape,
    cache: std::collections::HashMap<NodeIndex, Option<Vec<Dim>>>,
    stack: Vec<NodeIndex>,
}

impl<'a> ShapeEvaluator<'a> {
    fn new(builder: &'a GraphBuilder, reshape_input_shape: &'a Shape) -> Self {
        Self {
            builder,
            reshape_input_shape,
            cache: std::collections::HashMap::new(),
            stack: Vec::new(),
        }
    }

    /// Evaluate a shape tensor node and return its dimension values.
    fn eval(&mut self, idx: NodeIndex) -> Option<Vec<Dim>> {
        // Check cache first
        if let Some(cached) = self.cache.get(&idx) {
            return cached.clone();
        }

        // Detect cycles
        if self.stack.contains(&idx) {
            return None;
        }
        self.stack.push(idx);

        let result = self.eval_node(idx);

        self.stack.pop();
        self.cache.insert(idx, result.clone());
        result
    }

    /// Dispatch evaluation based on node operation type.
    fn eval_node(&mut self, idx: NodeIndex) -> Option<Vec<Dim>> {
        let graph = self.builder.graph();
        let op = graph.node(idx)?.op.op.clone();

        match op {
            NodeOp::Constant { data } => self.eval_constant(&data),
            NodeOp::Shape { start, end } => self.eval_shape(idx, start, end),
            NodeOp::Gather { axis } => self.eval_gather(idx, axis),
            NodeOp::Concat { axis } => self.eval_concat(idx, axis),
            NodeOp::Unsqueeze { .. } | NodeOp::Squeeze { .. } | NodeOp::Cast { .. } => {
                self.eval_passthrough(idx)
            }
            _ => None,
        }
    }

    /// Evaluate a constant node.
    fn eval_constant(&self, data: &ConstantData) -> Option<Vec<Dim>> {
        const_dims_from_data(data, self.reshape_input_shape)
    }

    /// Evaluate a Shape operation.
    fn eval_shape(&self, idx: NodeIndex, start: i64, end: i64) -> Option<Vec<Dim>> {
        let graph = self.builder.graph();
        let mut inputs: Vec<NodeIndex> = graph.predecessors(idx).collect();

        if inputs.len() != 1 {
            return None;
        }

        let input_node = graph.node(inputs.remove(0))?;
        let rank = input_node.op.shape.rank() as i64;

        // Normalize start/end indices
        let start_norm = if start < 0 { rank + start } else { start };
        let end_norm = if end < 0 { rank } else { end };

        if start_norm < 0 || end_norm < start_norm {
            return None;
        }

        let start_idx = start_norm as usize;
        let end_idx = end_norm as usize;

        if end_idx > input_node.op.shape.rank() {
            return None;
        }

        Some(input_node.op.shape.dims[start_idx..end_idx].to_vec())
    }

    /// Evaluate a Gather operation (axis=0 only).
    fn eval_gather(&mut self, idx: NodeIndex, axis: i32) -> Option<Vec<Dim>> {
        if axis != 0 {
            return None;
        }

        let inputs = self.builder.graph().predecessors_ordered(idx);
        if inputs.len() != 2 {
            return None;
        }

        let data_dims = self.eval(inputs[0])?;
        let indices = try_get_constant_i64_vec(self.builder.graph(), inputs[1])?;

        let mut result_dims = Vec::with_capacity(indices.len());
        let len = data_dims.len() as i64;

        for raw_idx in indices {
            let idx_norm = if raw_idx < 0 { len + raw_idx } else { raw_idx };
            if idx_norm < 0 || idx_norm as usize >= data_dims.len() {
                return None;
            }
            result_dims.push(data_dims[idx_norm as usize].clone());
        }

        Some(result_dims)
    }

    /// Evaluate a Concat operation (axis=0 only).
    fn eval_concat(&mut self, idx: NodeIndex, axis: i32) -> Option<Vec<Dim>> {
        if axis != 0 {
            return None;
        }

        let inputs = self.builder.graph().predecessors_ordered(idx);
        if inputs.is_empty() {
            return None;
        }

        let mut result_dims = Vec::new();
        for input in inputs {
            let dims = self.eval(input)?;
            result_dims.extend(dims);
        }

        Some(result_dims)
    }

    /// Evaluate pass-through operations (Unsqueeze, Squeeze, Cast).
    fn eval_passthrough(&mut self, idx: NodeIndex) -> Option<Vec<Dim>> {
        let mut inputs: Vec<NodeIndex> = self.builder.graph().predecessors(idx).collect();
        if inputs.len() != 1 {
            return None;
        }
        self.eval(inputs.remove(0))
    }
}

fn const_dims_from_data(data: &ConstantData, reshape_input_shape: &Shape) -> Option<Vec<Dim>> {
    let values: Vec<i64> = match data {
        ConstantData::I64(v) => v.clone(),
        ConstantData::I32(v) => v.iter().map(|&x| x as i64).collect(),
        _ => return None,
    };
    let mut dims = Vec::with_capacity(values.len());
    for (i, val) in values.into_iter().enumerate() {
        if val == -1 {
            dims.push(Dim::Dynamic);
        } else if val == 0 {
            if let Some(dim) = reshape_input_shape.dims.get(i) {
                dims.push(dim.clone());
            } else {
                return None;
            }
        } else if val > 0 {
            dims.push(Dim::Static(val as usize));
        } else {
            return None;
        }
    }
    Some(dims)
}

fn try_get_constant_i64_vec(
    graph: &hologram::ir::OperationGraph,
    idx: NodeIndex,
) -> Option<Vec<i64>> {
    let node = graph.node(idx)?;
    if let NodeOp::Constant { data } = &node.op.op {
        match data {
            ConstantData::I64(vals) => Some(vals.clone()),
            ConstantData::I32(vals) => Some(vals.iter().map(|&v| v as i64).collect()),
            _ => None,
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "reshape_test".to_string(),
            op_type: "Reshape".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_allowzero(allowzero: i64) -> NodeProto {
        NodeProto {
            name: "reshape_test".to_string(),
            op_type: "Reshape".to_string(),
            attribute: vec![AttributeProto {
                name: "allowzero".to_string(),
                i: allowzero,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_reshape_static_shape() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3, 4]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![6, 4]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_reshape_flatten_to_2d() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3, 4, 5]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![2, 60]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_with_inference() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3, 4]), DType::F32);
        // Shape with -1 to infer dimension
        let shape = builder.constant(ConstantData::I64(vec![-1, 4]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_with_allowzero() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3, 4]), DType::F32);
        let shape = builder.constant(
            ConstantData::I64(vec![0, 3, 4]), // 0 should copy from input
            Shape::static_shape(&[3]),
        );

        let result =
            translator.translate(&make_node_with_allowzero(1), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_to_scalar() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![]), Shape::static_shape(&[0]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_from_scalar() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![1, 1]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_dynamic_shape() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3, 4]), DType::F32);
        // Non-constant shape input (for dynamic reshape)
        let shape = builder.input("shape", Shape::static_shape(&[2]), DType::I64);

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_shape_graph_static_resolution() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 7, 8, 64]), DType::F32);
        let shape = builder.shape(data, 0, -1).unwrap();
        let indices = builder.constant(ConstantData::I64(vec![0, 1]), Shape::static_shape(&[2]));
        let gathered = builder.gather(shape, indices, 0).unwrap();
        let hidden = builder.constant(ConstantData::I64(vec![512]), Shape::static_shape(&[1]));
        let concat = builder.concat(&[gathered, hidden], 0).unwrap();

        let result = translator.translate(&make_node(), &[data, concat], &mut builder);
        assert!(result.is_ok());

        let output = result.unwrap()[0];
        let output_node = builder.graph().node(output).unwrap();
        let dims: Vec<_> = output_node
            .op
            .shape
            .dims
            .iter()
            .filter_map(|d| d.static_value())
            .collect();
        assert_eq!(dims, vec![1, 7, 512]);
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_reshape_no_inputs() {
        let translator = ReshapeTranslator;
        let err = translator.input_requirement().validate(0, "Reshape");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount {
                expected: 2,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_reshape_one_input() {
        let translator = ReshapeTranslator;
        let err = translator.input_requirement().validate(1, "Reshape");
        assert!(err.is_err());
    }

    #[test]
    fn test_reshape_too_many_inputs() {
        let translator = ReshapeTranslator;
        let err = translator.input_requirement().validate(3, "Reshape");
        assert!(err.is_err());
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_op_type() {
        let translator = ReshapeTranslator;
        assert_eq!(translator.onnx_op_type(), "Reshape");
    }

    #[test]
    fn test_input_requirement() {
        let translator = ReshapeTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::Exact(2)));
        assert!(!req.accepts_zero());
    }

    // ===== Symbolic Dimension Tests =====

    #[test]
    fn test_reshape_with_symbolic_input_dim() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        // Input with symbolic batch dimension
        let shape = Shape::new(vec![
            Dim::Symbolic("batch".to_string()),
            Dim::Static(128),
            Dim::Static(512),
        ]);
        let data = builder.input("data", shape, DType::F32);
        // Reshape to [batch, 128*512] - uses dynamic path to preserve symbolic
        let new_shape = builder.constant(
            ConstantData::I64(vec![0, 65536]), // 0 = copy from input
            Shape::static_shape(&[2]),
        );

        let result = translator.translate(&make_node(), &[data, new_shape], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_reshape_shape_from_symbolic_input() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        // Input tensor with symbolic sequence length
        let input_shape = Shape::new(vec![
            Dim::Static(1),
            Dim::Symbolic("seq_len".to_string()),
            Dim::Static(512),
        ]);
        let data = builder.input("data", input_shape, DType::F32);

        // Shape computed from input (Shape op will include symbolic dim)
        let shape_op = builder.shape(data, 0, -1).unwrap();

        let result = translator.translate(&make_node(), &[data, shape_op], &mut builder);
        assert!(result.is_ok());
        // Falls through to dynamic path which preserves symbolic dims
    }
}
