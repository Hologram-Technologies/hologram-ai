#![allow(missing_docs)]
//! ONNX operations - STUBBED VERSION

use hologram_ir::{GraphBuilder as IRBuilder, NodeIndex as NodeId};
use crate::core::{OnnxError, Result, SymbolicShape};
use crate::proto::AttributeProto;
use std::collections::HashMap;


pub fn translate_attention(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Attention not implemented in simplified version".into()))
}

pub fn translate_gru(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Gru not implemented in simplified version".into()))
}

pub fn translate_lstm(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Lstm not implemented in simplified version".into()))
}

pub fn translate_multi_head_attention(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Multi Head Attention not implemented in simplified version".into()))
}

pub fn translate_rnn(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    Err(OnnxError::InvalidModel("Rnn not implemented in simplified version".into()))
}
