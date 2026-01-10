//! ONNX logical operations.
//!
//! Note: hologram-ir doesn't have built-in comparison/logical ops.
//! We use Custom operations to represent these for now.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::core::{OnnxError, Result};
use rustc_hash::FxHashMap;

/// Translate ONNX Equal to IR using Custom operation.
pub fn translate_equal(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel("Equal requires 2 inputs".into()));
    }

    // Get input node for shape/dtype info
    let shape = {
        let input_node = builder.graph().node(inputs[0])
            .ok_or_else(|| OnnxError::InvalidModel("Invalid input node".into()))?;
        input_node.shape.clone()
    };

    // Use Custom operation for Equal
    let idx = builder.graph_mut().add_op(
        hologram::ir::NodeOp::Custom {
            name: "Equal".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram::ir::DType::Bool,
    );
    builder.graph_mut().connect(inputs[0], idx);
    builder.graph_mut().connect(inputs[1], idx);

    Ok(vec![idx])
}

/// Translate ONNX Greater to IR using Custom operation.
pub fn translate_greater(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel("Greater requires 2 inputs".into()));
    }

    let shape = {
        let input_node = builder.graph().node(inputs[0])
            .ok_or_else(|| OnnxError::InvalidModel("Invalid input node".into()))?;
        input_node.shape.clone()
    };

    let idx = builder.graph_mut().add_op(
        hologram::ir::NodeOp::Custom {
            name: "Greater".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram::ir::DType::Bool,
    );
    builder.graph_mut().connect(inputs[0], idx);
    builder.graph_mut().connect(inputs[1], idx);

    Ok(vec![idx])
}

/// Translate ONNX Less to IR using Custom operation.
pub fn translate_less(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel("Less requires 2 inputs".into()));
    }

    let shape = {
        let input_node = builder.graph().node(inputs[0])
            .ok_or_else(|| OnnxError::InvalidModel("Invalid input node".into()))?;
        input_node.shape.clone()
    };

    let idx = builder.graph_mut().add_op(
        hologram::ir::NodeOp::Custom {
            name: "Less".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram::ir::DType::Bool,
    );
    builder.graph_mut().connect(inputs[0], idx);
    builder.graph_mut().connect(inputs[1], idx);

    Ok(vec![idx])
}

/// Translate ONNX LessOrEqual to IR using Custom operation.
pub fn translate_less_or_equal(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel("LessOrEqual requires 2 inputs".into()));
    }

    let shape = {
        let input_node = builder.graph().node(inputs[0])
            .ok_or_else(|| OnnxError::InvalidModel("Invalid input node".into()))?;
        input_node.shape.clone()
    };

    let idx = builder.graph_mut().add_op(
        hologram::ir::NodeOp::Custom {
            name: "LessOrEqual".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram::ir::DType::Bool,
    );
    builder.graph_mut().connect(inputs[0], idx);
    builder.graph_mut().connect(inputs[1], idx);

    Ok(vec![idx])
}

/// Translate ONNX GreaterOrEqual to IR using Custom operation.
pub fn translate_greater_or_equal(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel("GreaterOrEqual requires 2 inputs".into()));
    }

    let shape = {
        let input_node = builder.graph().node(inputs[0])
            .ok_or_else(|| OnnxError::InvalidModel("Invalid input node".into()))?;
        input_node.shape.clone()
    };

    let idx = builder.graph_mut().add_op(
        hologram::ir::NodeOp::Custom {
            name: "GreaterOrEqual".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram::ir::DType::Bool,
    );
    builder.graph_mut().connect(inputs[0], idx);
    builder.graph_mut().connect(inputs[1], idx);

    Ok(vec![idx])
}

/// Translate ONNX And to IR using Custom operation.
pub fn translate_and(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel("And requires 2 inputs".into()));
    }

    let shape = {
        let input_node = builder.graph().node(inputs[0])
            .ok_or_else(|| OnnxError::InvalidModel("Invalid input node".into()))?;
        input_node.shape.clone()
    };

    let idx = builder.graph_mut().add_op(
        hologram::ir::NodeOp::Custom {
            name: "And".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram::ir::DType::Bool,
    );
    builder.graph_mut().connect(inputs[0], idx);
    builder.graph_mut().connect(inputs[1], idx);

    Ok(vec![idx])
}

/// Translate ONNX Or to IR using Custom operation.
pub fn translate_or(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel("Or requires 2 inputs".into()));
    }

    let shape = {
        let input_node = builder.graph().node(inputs[0])
            .ok_or_else(|| OnnxError::InvalidModel("Invalid input node".into()))?;
        input_node.shape.clone()
    };

    let idx = builder.graph_mut().add_op(
        hologram::ir::NodeOp::Custom {
            name: "Or".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram::ir::DType::Bool,
    );
    builder.graph_mut().connect(inputs[0], idx);
    builder.graph_mut().connect(inputs[1], idx);

    Ok(vec![idx])
}

/// Translate ONNX Not to IR using Custom operation.
pub fn translate_not(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Not requires 1 input".into()));
    }

    let shape = {
        let input_node = builder.graph().node(inputs[0])
            .ok_or_else(|| OnnxError::InvalidModel("Invalid input node".into()))?;
        input_node.shape.clone()
    };

    let idx = builder.graph_mut().add_op(
        hologram::ir::NodeOp::Custom {
            name: "Not".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram::ir::DType::Bool,
    );
    builder.graph_mut().connect(inputs[0], idx);

    Ok(vec![idx])
}

/// Translate ONNX Where to IR.
///
/// Where(condition, x, y) = condition ? x : y
pub fn translate_where(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 3 {
        return Err(OnnxError::InvalidModel("Where requires 3 inputs (condition, x, y)".into()));
    }

    // Where: condition ? x : y
    let result = builder.where_select(inputs[0], inputs[1], inputs[2])?;
    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    #[test]
    fn test_translate_equal() {
        let mut builder = GraphBuilder::new();
        let input1 = builder.input("input1", Shape::static_shape(&[2, 3]), DType::F32);
        let input2 = builder.input("input2", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_equal(&[input1, input2], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_equal_insufficient_inputs() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_equal(&[input], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_greater() {
        let mut builder = GraphBuilder::new();
        let input1 = builder.input("input1", Shape::static_shape(&[5, 5]), DType::F32);
        let input2 = builder.input("input2", Shape::static_shape(&[5, 5]), DType::F32);

        let result = translate_greater(&[input1, input2], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_less() {
        let mut builder = GraphBuilder::new();
        let input1 = builder.input("input1", Shape::static_shape(&[10]), DType::F32);
        let input2 = builder.input("input2", Shape::static_shape(&[10]), DType::F32);

        let result = translate_less(&[input1, input2], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_and() {
        let mut builder = GraphBuilder::new();
        let input1 = builder.input("input1", Shape::static_shape(&[4, 4]), DType::Bool);
        let input2 = builder.input("input2", Shape::static_shape(&[4, 4]), DType::Bool);

        let result = translate_and(&[input1, input2], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_or() {
        let mut builder = GraphBuilder::new();
        let input1 = builder.input("input1", Shape::static_shape(&[3, 3, 3]), DType::Bool);
        let input2 = builder.input("input2", Shape::static_shape(&[3, 3, 3]), DType::Bool);

        let result = translate_or(&[input1, input2], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_not() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[8, 8]), DType::Bool);

        let result = translate_not(&[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_not_no_inputs() {
        let mut builder = GraphBuilder::new();

        let result = translate_not(&[], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_where() {
        let mut builder = GraphBuilder::new();
        let condition = builder.input("condition", Shape::static_shape(&[2, 3]), DType::Bool);
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);
        let y = builder.input("y", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_where(&[condition, x, y], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_where_insufficient_inputs() {
        let mut builder = GraphBuilder::new();
        let condition = builder.input("condition", Shape::static_shape(&[2, 3]), DType::Bool);
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_where(&[condition, x], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_greater_insufficient_inputs() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_greater(&[input], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_less_insufficient_inputs() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_less(&[input], &mut builder);
        assert!(result.is_err());
    }
}
