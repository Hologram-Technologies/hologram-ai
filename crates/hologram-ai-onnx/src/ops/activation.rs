//! Activation functions - Relu, Sigmoid, Tanh, etc.

use anyhow::{Context, Result};
use hologram::compiler::OpKind;

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX Relu operation.
pub struct ReluOp;

impl OpTranslator for ReluOp {
    fn op_type(&self) -> &'static str {
        "Relu"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("Relu has no input")?;
        let input_node = ctx.get_node(input_name).context("Relu input not found")?;

        Ok(TranslateResult::runtime(
            OpKind::Relu,
            input_node.shape.clone(),
            input_node.dtype,
        ))
    }
}

/// ONNX Sigmoid operation.
pub struct SigmoidOp;

impl OpTranslator for SigmoidOp {
    fn op_type(&self) -> &'static str {
        "Sigmoid"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("Sigmoid has no input")?;
        let input_node = ctx
            .get_node(input_name)
            .context("Sigmoid input not found")?;

        Ok(TranslateResult::runtime(
            OpKind::Sigmoid,
            input_node.shape.clone(),
            input_node.dtype,
        ))
    }
}

/// ONNX Tanh operation.
pub struct TanhOp;

impl OpTranslator for TanhOp {
    fn op_type(&self) -> &'static str {
        "Tanh"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("Tanh has no input")?;
        let input_node = ctx.get_node(input_name).context("Tanh input not found")?;

        Ok(TranslateResult::runtime(
            OpKind::Tanh,
            input_node.shape.clone(),
            input_node.dtype,
        ))
    }
}

/// ONNX Gelu operation.
pub struct GeluOp;

impl OpTranslator for GeluOp {
    fn op_type(&self) -> &'static str {
        "Gelu"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("Gelu has no input")?;
        let input_node = ctx.get_node(input_name).context("Gelu input not found")?;

        Ok(TranslateResult::runtime(
            OpKind::Gelu,
            input_node.shape.clone(),
            input_node.dtype,
        ))
    }
}

/// ONNX Softmax operation.
pub struct SoftmaxOp;

impl OpTranslator for SoftmaxOp {
    fn op_type(&self) -> &'static str {
        "Softmax"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("Softmax has no input")?;
        let input_node = ctx
            .get_node(input_name)
            .context("Softmax input not found")?;

        // Extract axis attribute (default -1 for last dimension)
        let axis = node
            .attribute
            .iter()
            .find(|a| a.name == "axis")
            .map(|a| a.i as i32)
            .unwrap_or(-1);

        Ok(TranslateResult::runtime(
            OpKind::Softmax { axis },
            input_node.shape.clone(),
            input_node.dtype,
        ))
    }
}
