#![allow(missing_docs)]
//! ONNX operations - STUBBED VERSION

use hologram_ir::{GraphBuilder as IRBuilder, NodeIndex as NodeId};
use crate::core::{OnnxError, Result, SymbolicShape};
use crate::proto::AttributeProto;
use std::collections::HashMap;


pub fn translate_gather(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Gather not implemented in simplified version".into()))
}

pub fn translate_gather_elements(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Gather Elements not implemented in simplified version".into()))
}

pub fn translate_slice(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Slice not implemented in simplified version".into()))
}
