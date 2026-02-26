//! ONNX operation translators.
//!
//! Each ONNX operation has its own module implementing [`OpTranslator`].
//! Use `register_ops!` macro to add new operations to the registry.
//!
//! # Adding a New Operation
//!
//! 1. Create a new module (e.g., `myop.rs`)
//! 2. Implement a struct with [`OpTranslator`] trait
//! 3. Add to `register_ops!` macro below
//!
//! ```ignore
//! // In myop.rs:
//! pub struct MyOp;
//! impl OpTranslator for MyOp { ... }
//!
//! // In mod.rs, add to register_ops!:
//! register_ops! {
//!     ...
//!     "MyOp" => myop::MyOp,
//! }
//! ```

mod traits;

// Operation implementations
mod activation;
mod arithmetic;
mod cast;
mod comparison;
mod concat;
mod constant;
mod conv;
mod expand;
mod gather;
mod gemm;
mod matmul;
mod range;
mod reduce;
mod reshape;
mod shape;
mod tile;
mod transpose;
mod unsqueeze;

// Re-exports
pub use constant::extract_constant_data;
pub use traits::{BroadcastInfo, ConstantScalar, OpTranslator, TranslateContext, TranslateResult};

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use hologram::compiler::{DType, OpKind, OperationGraph};

use crate::proto;

/// Macro to register ONNX operations.
///
/// This generates the registry HashMap at initialization time.
/// Each entry maps an ONNX op_type string to its translator.
macro_rules! register_ops {
    ( $( $name:literal => $translator:expr ),* $(,)? ) => {
        fn build_registry() -> HashMap<&'static str, Box<dyn OpTranslator>> {
            let mut map: HashMap<&'static str, Box<dyn OpTranslator>> = HashMap::new();
            $(
                map.insert($name, Box::new($translator));
            )*
            map
        }
    };
}

// Register all supported ONNX operations
register_ops! {
    // Shape operations
    "Shape"           => shape::ShapeOp,
    "ConstantOfShape" => shape::ConstantOfShapeOp,
    "Gather"          => gather::GatherOp,
    "Range"           => range::RangeOp,
    "Unsqueeze"       => unsqueeze::UnsqueezeOp,
    "Squeeze"         => unsqueeze::SqueezeOp,

    // Arithmetic
    "Add"  => arithmetic::AddOp,
    "Sub"  => arithmetic::SubOp,
    "Mul"  => arithmetic::MulOp,
    "Div"  => arithmetic::DivOp,
    "Pow"  => arithmetic::PowOp,
    "Sqrt" => arithmetic::SqrtOp,
    "Log"  => arithmetic::LogOp,
    "Exp"  => arithmetic::ExpOp,
    "Abs"  => arithmetic::AbsOp,
    "Neg"  => arithmetic::NegOp,
    "Min"  => arithmetic::MinOp,
    "Max"  => arithmetic::MaxOp,

    // Matrix
    "MatMul" => matmul::MatMulOp,
    "Gemm"   => gemm::GemmOp,

    // Activations
    "Relu"    => activation::ReluOp,
    "Sigmoid" => activation::SigmoidOp,
    "Tanh"    => activation::TanhOp,
    "Gelu"    => activation::GeluOp,
    "Softmax" => activation::SoftmaxOp,

    // Reductions
    "ReduceMean" => reduce::ReduceMeanOp,
    "ReduceSum"  => reduce::ReduceSumOp,
    "ReduceMax"  => reduce::ReduceMaxOp,

    // Shape manipulation
    "Reshape"   => reshape::ReshapeOp,
    "Transpose" => transpose::TransposeOp,
    "Concat"    => concat::ConcatOp,
    "Cast"      => cast::CastOp,
    "Expand"    => expand::ExpandOp,
    "Tile"      => tile::TileOp,

    // Constants
    "Constant" => constant::ConstantOp,

    // Comparison
    "Greater"        => comparison::GreaterOp,
    "Less"           => comparison::LessOp,
    "LessOrEqual"    => comparison::LessOrEqualOp,
    "GreaterOrEqual" => comparison::GreaterOrEqualOp,
    "Equal"          => comparison::EqualOp,
    "Where"          => comparison::WhereOp,

    // CNN operations
    "Conv"                => conv::ConvOp,
    "MaxPool"             => conv::MaxPoolOp,
    "GlobalAveragePool"   => conv::GlobalAveragePoolOp,
    "AveragePool"         => conv::AveragePoolOp,
    "BatchNormalization"  => conv::BatchNormalizationOp,
    "Flatten"             => conv::FlattenOp,
}

/// Registry singleton.
static REGISTRY: OnceLock<HashMap<&'static str, Box<dyn OpTranslator>>> = OnceLock::new();

/// Get the operation registry.
pub fn registry() -> &'static HashMap<&'static str, Box<dyn OpTranslator>> {
    REGISTRY.get_or_init(build_registry)
}

/// Get a translator by ONNX op type.
pub fn get_translator(op_type: &str) -> Option<&'static dyn OpTranslator> {
    registry().get(op_type).map(|b| b.as_ref())
}

/// Check if an operation is supported.
#[allow(dead_code)]
pub fn is_supported(op_type: &str) -> bool {
    registry().contains_key(op_type)
}

/// List all supported operations.
#[allow(dead_code)]
pub fn supported_ops() -> Vec<&'static str> {
    registry().keys().copied().collect()
}

/// Check if an operation requires expansion (creates multiple nodes).
pub fn requires_expansion(op_type: &str) -> bool {
    get_translator(op_type)
        .map(|t| t.requires_expansion())
        .unwrap_or(false)
}

/// Translate an ONNX node to hologram OpKind with shape inference.
///
/// Tries constant folding first, then falls back to runtime translation.
pub fn translate_node(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let result = translate_node_full(node, value_to_node, graph)?;
    Ok((result.op_kind, result.shape, result.dtype))
}

/// Translate with full result including constant data.
///
/// This is the preferred API - includes folded constant data.
pub fn translate_node_full(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<TranslateResult> {
    let op_type = node.op_type.as_str();

    let translator = get_translator(op_type)
        .with_context(|| format!("Unsupported ONNX operation: {}", op_type))?;

    let ctx = TranslateContext::new(graph, value_to_node, &graph.constants);

    // Try constant folding first
    if let Some(result) = translator.try_fold(node, &ctx) {
        return Ok(result);
    }

    translator.translate(node, &ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_contains_ops() {
        let ops = supported_ops();
        assert!(ops.contains(&"Add"));
        assert!(ops.contains(&"MatMul"));
        assert!(ops.contains(&"Relu"));
        assert!(ops.contains(&"Reshape"));
    }

    #[test]
    fn test_get_translator() {
        assert!(get_translator("Add").is_some());
        assert!(get_translator("NonExistent").is_none());
    }

    #[test]
    fn test_requires_expansion() {
        assert!(requires_expansion("Gemm"));
        assert!(!requires_expansion("Add"));
        assert!(!requires_expansion("MatMul"));
    }
}
