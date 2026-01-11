//! Boolean operation translators.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{DType, GraphBuilder, NodeIndex, NodeOp};
use rustc_hash::FxHashMap;

/// Translator for ONNX And operation.
///
/// And(A, B) returns element-wise boolean A AND B
#[derive(Debug, Default)]
pub struct AndTranslator;

impl OnnxTranslator for AndTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "And"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(2)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let shape = {
            let input_node = builder
                .graph()
                .node(inputs[0])
                .ok_or_else(|| TranslationError::IrBuilder("Invalid input node".to_string()))?;
            input_node.op.shape.clone()
        };

        let idx = builder.graph_mut().add_op(
            NodeOp::Custom {
                name: "And".to_string(),
                attrs: FxHashMap::default(),
            },
            shape,
            DType::Bool,
        );
        builder.graph_mut().connect(inputs[0], idx);
        builder.graph_mut().connect(inputs[1], idx);

        Ok(vec![idx])
    }
}

/// Translator for ONNX Or operation.
///
/// Or(A, B) returns element-wise boolean A OR B
#[derive(Debug, Default)]
pub struct OrTranslator;

impl OnnxTranslator for OrTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Or"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(2)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let shape = {
            let input_node = builder
                .graph()
                .node(inputs[0])
                .ok_or_else(|| TranslationError::IrBuilder("Invalid input node".to_string()))?;
            input_node.op.shape.clone()
        };

        let idx = builder.graph_mut().add_op(
            NodeOp::Custom {
                name: "Or".to_string(),
                attrs: FxHashMap::default(),
            },
            shape,
            DType::Bool,
        );
        builder.graph_mut().connect(inputs[0], idx);
        builder.graph_mut().connect(inputs[1], idx);

        Ok(vec![idx])
    }
}

/// Translator for ONNX Not operation.
///
/// Not(A) returns element-wise boolean NOT A
#[derive(Debug, Default)]
pub struct NotTranslator;

impl OnnxTranslator for NotTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Not"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(1)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let shape = {
            let input_node = builder
                .graph()
                .node(inputs[0])
                .ok_or_else(|| TranslationError::IrBuilder("Invalid input node".to_string()))?;
            input_node.op.shape.clone()
        };

        let idx = builder.graph_mut().add_op(
            NodeOp::Custom {
                name: "Not".to_string(),
                attrs: FxHashMap::default(),
            },
            shape,
            DType::Bool,
        );
        builder.graph_mut().connect(inputs[0], idx);

        Ok(vec![idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::Shape;

    fn make_node(op_type: &str) -> NodeProto {
        NodeProto {
            name: format!("{}_test", op_type.to_lowercase()),
            op_type: op_type.to_string(),
            ..Default::default()
        }
    }

    // And tests

    #[test]
    fn test_and_basic() {
        let translator = AndTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[4, 4]), DType::Bool);
        let b = builder.input("b", Shape::static_shape(&[4, 4]), DType::Bool);

        let result = translator.translate(&make_node("And"), &[a, b], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_and_input_validation() {
        let translator = AndTranslator;

        let err = translator.input_requirement().validate(1, "And");
        assert!(err.is_err());

        assert!(translator.input_requirement().validate(2, "And").is_ok());

        let err = translator.input_requirement().validate(3, "And");
        assert!(err.is_err());
    }

    // Or tests

    #[test]
    fn test_or_basic() {
        let translator = OrTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[3, 3, 3]), DType::Bool);
        let b = builder.input("b", Shape::static_shape(&[3, 3, 3]), DType::Bool);

        let result = translator.translate(&make_node("Or"), &[a, b], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_or_input_validation() {
        let translator = OrTranslator;

        let err = translator.input_requirement().validate(1, "Or");
        assert!(err.is_err());

        assert!(translator.input_requirement().validate(2, "Or").is_ok());
    }

    // Not tests

    #[test]
    fn test_not_basic() {
        let translator = NotTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[8, 8]), DType::Bool);

        let result = translator.translate(&make_node("Not"), &[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_not_input_validation() {
        let translator = NotTranslator;

        let err = translator.input_requirement().validate(0, "Not");
        assert!(err.is_err());

        assert!(translator.input_requirement().validate(1, "Not").is_ok());

        let err = translator.input_requirement().validate(2, "Not");
        assert!(err.is_err());
    }
}
