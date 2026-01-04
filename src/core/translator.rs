//! ONNX to hologram IR lowering types.
//!
//! This module provides types for lowering IR functions to OperationGraph format,
//! which is the final serializable representation for .holo files.
//!
//! # Architecture
//!
//! ```text
//! hologram-onnx (top-level)   ←── Uses real translator
//!   ↓ IRFunction
//! hologram-onnx-core (this crate)
//!   ↓ lower_to_operation_graph()
//! OperationGraph
//!   ↓ to_bytes()
//! .holo file
//! ```
//!
//! **Note**: Full ONNX → IR translation lives in the top-level `hologram-onnx` crate
//! because it requires both `hologram-onnx-core` (shapes, parsing) and `hologram-onnx-ops`
//! (operation translators). Due to the dependency structure (ops → core), putting the
//! translator in core would create a cyclic dependency.
//!
//! # Usage
//!
//! For full ONNX → .holo compilation, use the top-level crate:
//! ```ignore
//! use hologram_onnx::{compile_onnx, OnnxCompiler};
//!
//! // Simple usage
//! let (holo, weights) = compile_onnx(&onnx_bytes)?;
//!
//! // With config
//! let compiler = OnnxCompiler::with_config(config);
//! let (holo, weights) = compiler.compile(&onnx_bytes)?;
//! ```
//!
//! For parsing and validation only (this crate):
//! ```ignore
//! use crate::core::{parse_model, validate_model};
//! let model = parse_model(&onnx_bytes)?;
//! validate_model(&model)?;
//! ```

use hologram_ir::OperationGraph as IRFunction;

use crate::Result;

/// Result of lowering to OperationGraph.
///
/// This wraps the IR function with serialization capabilities for .holo format.
/// The OperationGraph is the final representation before writing to disk.
#[derive(Debug, Clone)]
pub struct OperationGraph {
    ir_func: IRFunction,
}

impl OperationGraph {
    /// Create from IR function.
    pub fn from_ir(ir_func: IRFunction) -> Self {
        Self { ir_func }
    }

    /// Get node count - STUBBED
    pub fn node_count(&self) -> usize {
        0 // Stubbed - body field doesn't exist in new IR
    }

    /// Get the underlying IR function reference.
    pub fn ir_function(&self) -> &IRFunction {
        &self.ir_func
    }

    /// Serialize to .holo format bytes - STUBBED
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        use crate::OnnxError;
        Err(OnnxError::InvalidModel("Serialization not implemented in simplified version".into()))
    }
}

/// Lower IR function to OperationGraph.
///
/// Wraps the IR function for serialization to .holo format.
///
/// # Arguments
///
/// * `ir_func` - Decomposed IR function from the translation pipeline
///
/// # Returns
///
/// OperationGraph ready for serialization via `to_bytes()`.
///
/// # Example
///
/// ```ignore
/// use crate::core::lower_to_operation_graph;
///
/// let ir_func = translate_graph_to_ir(&graph, opset)?;
/// let ir_func = apply_ir_decomposition(ir_func, &config)?;
/// let op_graph = lower_to_operation_graph(ir_func)?;
/// let bytes = op_graph.to_bytes()?;
/// ```
pub fn lower_to_operation_graph(ir_func: IRFunction) -> Result<OperationGraph> {
    Ok(OperationGraph::from_ir(ir_func))
}

/// Translate ONNX GraphProto to hologram IR (stub).
///
/// This is a compatibility stub - full translation happens in the ops module.
/// For now, this returns an empty IR function.
pub fn translate_graph_to_ir(_graph: &crate::proto::GraphProto) -> Result<IRFunction> {
    use hologram_ir::GraphBuilder;

    // Create minimal IR function
    let builder = GraphBuilder::new();
    Ok(builder.build())
}

/// Apply IR-level decompositions (stub).
///
/// This is a compatibility stub - decompositions are now handled by hologram-ir.
/// Returns the IR function unchanged.
pub fn apply_ir_decomposition(ir_func: IRFunction, _config: &crate::OnnxConfig) -> Result<IRFunction> {
    Ok(ir_func)
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Rewrite tests to use new hologram-ir API
    #[test]
    #[ignore]
    fn test_operation_graph_serialization_format() {
        // Create a minimal IR function for testing
        let mut builder = GraphBuilder::new("test");
        let input_type = Type::tensor(ScalarType::F32, Shape::new(vec![Dim::Static(1)]));
        let input = builder.add_input("x", input_type);
        builder.set_output(input);
        let ir_func = builder.build();

        let op_graph = OperationGraph::from_ir(ir_func);
        let bytes = op_graph.to_bytes().unwrap();

        // Verify magic header
        assert_eq!(&bytes[0..4], b"HOLO");

        // Verify version
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(version, 1);
    }

    #[test]
    #[ignore]
    fn test_operation_graph_node_count() {
        let mut builder = GraphBuilder::new("multi_node");
        let input_type = Type::tensor(ScalarType::F32, Shape::new(vec![Dim::Static(1)]));
        let input = builder.add_input("x", input_type);
        builder.set_output(input);
        let ir_func = builder.build();

        let expected_len = ir_func.body.len();
        let op_graph = OperationGraph::from_ir(ir_func);
        assert_eq!(op_graph.node_count(), expected_len);
    }

    #[test]
    #[ignore]
    fn test_lower_to_operation_graph() {
        let mut builder = GraphBuilder::new("test_lower");
        let input_type = Type::tensor(ScalarType::F32, Shape::new(vec![Dim::Static(10)]));
        let input = builder.add_input("input", input_type);
        builder.set_output(input);
        let ir_func = builder.build();

        let result = lower_to_operation_graph(ir_func);
        assert!(result.is_ok());

        let op_graph = result.unwrap();
        assert!(op_graph.node_count() > 0);
    }
}
