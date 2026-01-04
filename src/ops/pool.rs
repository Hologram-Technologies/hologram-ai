#![allow(missing_docs)]
//! ONNX pooling operations.

use hologram_ir::{GraphBuilder as IRBuilder, NodeIndex as NodeId};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;


pub fn translate_average_pool(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _builder: &mut IRBuilder,
) -> Result<Vec<NodeId>> {
    Err(OnnxError::unsupported_op("AveragePool", 11))
}

pub fn translate_global_average_pool(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _builder: &mut IRBuilder,
) -> Result<Vec<NodeId>> {
    Err(OnnxError::unsupported_op("GlobalAveragePool", 11))
}

pub fn translate_global_max_pool(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _builder: &mut IRBuilder,
) -> Result<Vec<NodeId>> {
    Err(OnnxError::unsupported_op("GlobalMaxPool", 11))
}

pub fn translate_max_pool(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _builder: &mut IRBuilder,
) -> Result<Vec<NodeId>> {
    Err(OnnxError::unsupported_op("MaxPool", 11))
}
