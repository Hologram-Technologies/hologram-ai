#![allow(missing_docs)]
//! ONNX operations - STUBBED VERSION

use hologram_ir::{GraphBuilder as IRBuilder, NodeIndex as NodeId};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;


pub fn translate_depth_to_space(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _builder: &mut IRBuilder,
) -> Result<Vec<NodeId>> {
    Err(OnnxError::InvalidModel("Depth To Space not implemented in simplified version".into()))
}

pub fn translate_resize(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _builder: &mut IRBuilder,
) -> Result<Vec<NodeId>> {
    Err(OnnxError::InvalidModel("Resize not implemented in simplified version".into()))
}

pub fn translate_space_to_depth(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _builder: &mut IRBuilder,
) -> Result<Vec<NodeId>> {
    Err(OnnxError::InvalidModel("Space To Depth not implemented in simplified version".into()))
}

pub fn translate_upsample(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _builder: &mut IRBuilder,
) -> Result<Vec<NodeId>> {
    Err(OnnxError::InvalidModel("Upsample not implemented in simplified version".into()))
}
