//! ONNX logical operations.
//!
//! Note: hologram-ir doesn't have built-in comparison/logical ops.
//! We use Custom operations to represent these for now.

use hologram_ir::{GraphBuilder, NodeIndex};
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
        hologram_ir::NodeOp::Custom {
            name: "Equal".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram_ir::DType::Bool,
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
        hologram_ir::NodeOp::Custom {
            name: "Greater".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram_ir::DType::Bool,
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
        hologram_ir::NodeOp::Custom {
            name: "Less".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram_ir::DType::Bool,
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
        hologram_ir::NodeOp::Custom {
            name: "And".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram_ir::DType::Bool,
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
        hologram_ir::NodeOp::Custom {
            name: "Or".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram_ir::DType::Bool,
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
        hologram_ir::NodeOp::Custom {
            name: "Not".to_string(),
            attrs: FxHashMap::default(),
        },
        shape,
        hologram_ir::DType::Bool,
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
