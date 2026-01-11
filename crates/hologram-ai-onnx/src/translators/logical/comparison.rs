//! Comparison operation translators.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{DType, GraphBuilder, NodeIndex, NodeOp};
use rustc_hash::FxHashMap;

/// Translator for ONNX Equal operation.
///
/// Equal(A, B) returns element-wise boolean A == B
#[derive(Debug, Default)]
pub struct EqualTranslator;

impl OnnxTranslator for EqualTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Equal"
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
                name: "Equal".to_string(),
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

/// Translator for ONNX Greater operation.
///
/// Greater(A, B) returns element-wise boolean A > B
#[derive(Debug, Default)]
pub struct GreaterTranslator;

impl OnnxTranslator for GreaterTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Greater"
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
                name: "Greater".to_string(),
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

/// Translator for ONNX Less operation.
///
/// Less(A, B) returns element-wise boolean A < B
#[derive(Debug, Default)]
pub struct LessTranslator;

impl OnnxTranslator for LessTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Less"
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
                name: "Less".to_string(),
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

/// Translator for ONNX GreaterOrEqual operation.
///
/// GreaterOrEqual(A, B) returns element-wise boolean A >= B
#[derive(Debug, Default)]
pub struct GreaterOrEqualTranslator;

impl OnnxTranslator for GreaterOrEqualTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "GreaterOrEqual"
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
                name: "GreaterOrEqual".to_string(),
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

/// Translator for ONNX LessOrEqual operation.
///
/// LessOrEqual(A, B) returns element-wise boolean A <= B
#[derive(Debug, Default)]
pub struct LessOrEqualTranslator;

impl OnnxTranslator for LessOrEqualTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "LessOrEqual"
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
                name: "LessOrEqual".to_string(),
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

    // Equal tests

    #[test]
    fn test_equal_basic() {
        let translator = EqualTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node("Equal"), &[a, b], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_equal_input_validation() {
        let translator = EqualTranslator;
        let err = translator.input_requirement().validate(1, "Equal");
        assert!(err.is_err());

        assert!(translator.input_requirement().validate(2, "Equal").is_ok());

        let err = translator.input_requirement().validate(3, "Equal");
        assert!(err.is_err());
    }

    // Greater tests

    #[test]
    fn test_greater_basic() {
        let translator = GreaterTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[5, 5]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[5, 5]), DType::F32);

        let result = translator.translate(&make_node("Greater"), &[a, b], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_greater_input_validation() {
        let translator = GreaterTranslator;
        let err = translator.input_requirement().validate(1, "Greater");
        assert!(err.is_err());
    }

    // Less tests

    #[test]
    fn test_less_basic() {
        let translator = LessTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[10]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[10]), DType::F32);

        let result = translator.translate(&make_node("Less"), &[a, b], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_less_input_validation() {
        let translator = LessTranslator;
        let err = translator.input_requirement().validate(1, "Less");
        assert!(err.is_err());
    }

    // GreaterOrEqual tests

    #[test]
    fn test_greater_or_equal_basic() {
        let translator = GreaterOrEqualTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[4, 4]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[4, 4]), DType::F32);

        let result = translator.translate(&make_node("GreaterOrEqual"), &[a, b], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_greater_or_equal_input_validation() {
        let translator = GreaterOrEqualTranslator;
        let err = translator.input_requirement().validate(1, "GreaterOrEqual");
        assert!(err.is_err());
    }

    // LessOrEqual tests

    #[test]
    fn test_less_or_equal_basic() {
        let translator = LessOrEqualTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[3, 3, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 3, 3]), DType::F32);

        let result = translator.translate(&make_node("LessOrEqual"), &[a, b], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_less_or_equal_input_validation() {
        let translator = LessOrEqualTranslator;
        let err = translator.input_requirement().validate(1, "LessOrEqual");
        assert!(err.is_err());
    }
}
