#![allow(missing_docs)]
//! ONNX operations - STUBBED VERSION

use hologram_ir::{GraphBuilder as IRBuilder, NodeIndex as NodeId};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;


pub fn translate_conv(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _builder: &mut IRBuilder,
) -> Result<Vec<NodeId>> {
    Err(OnnxError::InvalidModel("Conv not implemented in simplified version".into()))
}

pub fn translate_conv_transpose(
    _inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _builder: &mut IRBuilder,
) -> Result<Vec<NodeId>> {
    Err(OnnxError::InvalidModel("Conv Transpose not implemented in simplified version".into()))
}
