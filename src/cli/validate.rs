//! Validate ONNX models.
//!
//! This module provides functionality to validate ONNX models, checking for:
//! - Valid protobuf structure
//! - Supported operations
//! - Symbolic shape compatibility
//! - Graph integrity

use anyhow::{Context, Result};
use crate::core::{extract_opset_version, parse_model, validate_model};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use tracing::{info, warn};

/// List of supported ONNX operations
const SUPPORTED_OPS: &[&str] = &[
    // Core operations
    "MatMul",
    "Gemm",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Cast",
    // Activation functions
    "Relu",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Gelu",
    "Swish",
    "Elu",
    "Selu",
    "Clip",
    "LeakyRelu",
    "PRelu",
    // Shape operations
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Concat",
    "Split",
    "Flatten",
    // Convolution operations
    "Conv",
    "ConvTranspose",
    // Normalization operations
    "BatchNormalization",
    "LayerNormalization",
    "InstanceNormalization",
    "GroupNormalization",
    // Pooling operations
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    // Reduction operations
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    // Advanced operations
    "Attention",
    "MultiHeadAttention",
    "LSTM",
    "GRU",
    "RNN",
    // Unary operations
    "Sqrt",
    "Exp",
    "Log",
    "Neg",
    "Abs",
    "Reciprocal",
    // Indexing operations
    "Gather",
    "Slice",
    "GatherElements",
    // Resize operations
    "Resize",
    "Upsample",
    "DepthToSpace",
    "SpaceToDepth",
    // Logical and comparison operations
    "Where",
    "Equal",
    "Less",
    "Greater",
    "LessOrEqual",
    "GreaterOrEqual",
    "Not",
    "And",
    "Or",
    // Constant and identity operations
    "Constant",
    "Identity",
    "ConstantOfShape",
    "Shape",
    // Padding operations
    "Pad",
];

/// Validate an ONNX model.
///
/// # Arguments
///
/// * `model_path` - Path to ONNX model file
/// * `check_ops` - Check for unsupported operations
///
/// # Returns
///
/// Returns Ok(()) if validation succeeds, or an error describing the validation failure.
pub fn validate_command(model_path: &Path, check_ops: bool) -> Result<()> {
    info!("Validating ONNX model: {}", model_path.display());

    // Read model file
    let onnx_bytes = fs::read(model_path)
        .with_context(|| format!("Failed to read ONNX model from {}", model_path.display()))?;

    info!("Model file size: {} bytes", onnx_bytes.len());

    // Parse model
    info!("Parsing ONNX protobuf...");
    let model =
        parse_model(&onnx_bytes).context("Failed to parse ONNX model - invalid protobuf format")?;

    println!("✓ Protobuf structure is valid");

    // Validate model structure
    info!("Validating model structure...");
    validate_model(&model).context("Model validation failed")?;

    println!("✓ Model structure is valid");

    // Get opset version
    let opset_version = extract_opset_version(&model);
    println!("✓ Opset version: {}", opset_version);

    // Check graph
    let graph = model.graph.as_ref().context("Model has no graph")?;

    println!("✓ Graph: {} ({} nodes)", graph.name, graph.node.len());

    // Validate inputs and outputs
    if graph.input.is_empty() {
        warn!("Graph has no inputs");
    } else {
        println!("✓ Inputs: {}", graph.input.len());
    }

    if graph.output.is_empty() {
        warn!("Graph has no outputs");
    } else {
        println!("✓ Outputs: {}", graph.output.len());
    }

    // Check for unsupported operations if requested
    if check_ops {
        info!("Checking for unsupported operations...");
        let supported: HashSet<&str> = SUPPORTED_OPS.iter().copied().collect();
        let mut unsupported_ops: HashSet<String> = HashSet::new();

        for node in &graph.node {
            if !supported.contains(node.op_type.as_str()) {
                unsupported_ops.insert(node.op_type.clone());
            }
        }

        if !unsupported_ops.is_empty() {
            println!("\n⚠️  Unsupported operations found:");
            let mut ops: Vec<_> = unsupported_ops.iter().collect();
            ops.sort();
            for op in ops {
                println!("  - {}", op);
            }
            println!(
                "\n✗ Model contains {} unsupported operation type(s)",
                unsupported_ops.len()
            );
            println!("  Compilation may fail or produce incomplete results.");
            anyhow::bail!("Model contains unsupported operations");
        } else {
            println!(
                "✓ All operations are supported ({} unique types)",
                graph
                    .node
                    .iter()
                    .map(|n| &n.op_type)
                    .collect::<HashSet<_>>()
                    .len()
            );
        }
    }

    // Summary
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║              Validation Summary                            ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("  Status: ✓ VALID");
    println!("  File: {}", model_path.display());
    println!("  Opset: {}", opset_version);
    println!("  Nodes: {}", graph.node.len());
    if check_ops {
        println!("  All operations supported: YES");
    }
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_validate_command_missing_file() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("missing.onnx");

        let result = validate_command(&model_path, false);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to read ONNX model")
        );
    }

    #[test]
    fn test_supported_ops_list() {
        // Verify supported ops list is not empty
        assert!(!SUPPORTED_OPS.is_empty());

        // Verify no duplicates
        let mut unique = HashSet::new();
        for op in SUPPORTED_OPS {
            assert!(unique.insert(op), "Duplicate operation: {}", op);
        }
    }

    #[test]
    fn test_supported_ops_count() {
        // We have 72 operations implemented
        assert_eq!(SUPPORTED_OPS.len(), 72);
    }
}
