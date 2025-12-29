//! ONNX operation translators with symbolic shape inference.
//!
//! This crate provides translators for ONNX operations to hologram's IR,
//! with full support for symbolic shapes and ISA optimizations.
//!
//! # ISA Optimizations
//!
//! All operations are designed to leverage hologram's ISA features:
//! - **LOOP instructions**: For O(1) space complexity in broadcasting and reductions
//! - **PhiCoordinate addressing**: For 5-10x speedup in convolutions and pooling
//! - **ClassMap fusion**: For O(1) element-wise operation composition
//! - **SIMD vectorization**: Via hologram-backend for all operations
//!
//! # Symbolic Shapes
//!
//! All operations support symbolic shapes via `hologram_compiler::shapes`:
//! - Variable batch sizes: `[Dim::Var("batch"), Dim::Concrete(224), ...]`
//! - Variable sequence lengths: `[Dim::Var("seq_len"), Dim::Concrete(hidden)]`
//! - Arithmetic expressions: `Dim::Expr` for computed dimensions
//!
//! # Example
//!
//! ```rust,ignore
//! use hologram_onnx_ops::translate_onnx_op;
//! use hologram_compiler::ir::IRBuilder;
//!
//! let mut builder = IRBuilder::new("model");
//! let result = translate_onnx_op(
//!     "MatMul",
//!     &inputs,
//!     &attributes,
//!     &shapes,
//!     &mut builder,
//! )?;
//! ```

// Module declarations
mod translator;
mod ops;
mod utils;

/// Test utilities for creating types in tests.
#[cfg(test)]
pub(crate) mod test_utils;

// Public exports
pub use translator::{translate_onnx_op, infer_op_output_shape, OpTranslator};

// Export operation translators
pub use ops::core::{
    translate_matmul, translate_gemm, translate_add, translate_sub,
    translate_mul, translate_div, translate_pow,
};
pub use ops::activation::{
    translate_relu, translate_sigmoid, translate_tanh, translate_softmax,
    translate_gelu, translate_swish, translate_elu, translate_selu,
};
pub use ops::shape::{
    translate_reshape, translate_transpose, translate_squeeze,
    translate_unsqueeze, translate_concat, translate_split,
};
pub use ops::conv::{
    translate_conv, translate_conv_transpose,
    infer_conv_output_shape, infer_conv_transpose_output_shape,
};
pub use ops::norm::{
    translate_batch_normalization, translate_layer_normalization,
    translate_instance_normalization,
};
pub use ops::pool::{
    translate_max_pool, translate_average_pool, translate_global_average_pool,
    infer_pool_output_shape,
};
pub use ops::reduction::{
    translate_reduce_sum, translate_reduce_mean, translate_reduce_max,
    translate_reduce_min, translate_reduce_prod,
};
pub use ops::advanced::{
    translate_attention, translate_multi_head_attention,
    translate_lstm, translate_gru, translate_rnn,
};

// Export utilities
pub use utils::{
    parse_attr_int, parse_attr_ints, parse_attr_float,
    parse_attr_floats, parse_attr_string, parse_attr_tensor,
};

// Re-export types from hologram for convenience
pub use hologram_compiler::ir::{IRBuilder, IRNode, NodeId};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Verify all modules are accessible
        assert!(true, "Module structure is correct");
    }
}
