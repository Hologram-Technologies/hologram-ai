#![allow(missing_docs)]
//! ONNX operations - STUBBED VERSION

use hologram_ir::{GraphBuilder as IRBuilder, NodeIndex as NodeId};
use crate::core::{OnnxError, Result, SymbolicShape};
use crate::proto::AttributeProto;
use std::collections::HashMap;


pub fn translate_batch_normalization(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Batch Normalization not implemented in simplified version".into()))
}

pub fn translate_group_normalization(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Group Normalization not implemented in simplified version".into()))
}

pub fn translate_instance_normalization(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Instance Normalization not implemented in simplified version".into()))
}

pub fn translate_layer_normalization(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Layer Normalization not implemented in simplified version".into()))
}
