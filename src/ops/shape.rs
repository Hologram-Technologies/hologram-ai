#![allow(missing_docs)]
//! ONNX operations - STUBBED VERSION

use hologram_ir::{GraphBuilder as IRBuilder, NodeIndex as NodeId};
use crate::core::{OnnxError, Result, SymbolicShape};
use crate::proto::AttributeProto;
use std::collections::HashMap;


pub fn translate_concat(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Concat not implemented in simplified version".into()))
}

pub fn translate_expand(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Expand not implemented in simplified version".into()))
}

pub fn translate_flatten(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Flatten not implemented in simplified version".into()))
}

pub fn translate_range(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Range not implemented in simplified version".into()))
}

pub fn translate_reshape(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Reshape not implemented in simplified version".into()))
}

pub fn translate_split(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Split not implemented in simplified version".into()))
}

pub fn translate_squeeze(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Squeeze not implemented in simplified version".into()))
}

pub fn translate_transpose(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Transpose not implemented in simplified version".into()))
}

pub fn translate_unsqueeze(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Unsqueeze not implemented in simplified version".into()))
}
