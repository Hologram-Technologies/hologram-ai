//! Trait-based lowering strategies for resolving op parameters.
//!
//! When an `AiOp` dispatches to `FloatNeedsShape`, the builder consults a
//! `LoweringStrategy` chain to resolve the op's parameters. Two built-in
//! strategies handle the common cases:
//!
//! - **`ConcreteStrategy`**: All dims must be concrete. Passes to next strategy
//!   when any dim is symbolic.
//! - **`DeferredStrategy`**: Emits ops with concrete dims where known, uses
//!   0-sentinels for symbolic dims and records `ParamRecipe` entries for the
//!   runtime to patch.
//!
//! Both strategies share a single unified resolver (`resolve_op`) that converts
//! `AiOp` → `(FloatOp, Vec<ParamRecipe>)`. The strategies differ only in what
//! they accept: ConcreteStrategy rejects any deferred recipes; DeferredStrategy
//! accepts them and embeds them in the archive.

use super::op_resolver::{
    dim_recipe, is_deferred, last_dim_expr, resolve_op, resolve_or_zero, second_last_dim_expr,
};
use crate::exec_context::{NodeShapeRecipe, ParamRecipe};
use crate::ir::{AiOp, DimVarId, TensorId, TensorInfo};
use anyhow::Result;
use hologram::{FloatOp, GraphOp};
use std::collections::HashMap;

/// Result of lowering an op with a strategy.
pub struct SymbolicLowering {
    /// The graph op to emit (may contain 0-sentinels for deferred dims).
    pub graph_op: GraphOp,
    /// Recipe for the runtime to patch deferred params. `None` if all concrete.
    pub recipe: Option<NodeShapeRecipe>,
}

/// Strategy for resolving `FloatNeedsShape` ops during lowering.
pub trait LoweringStrategy: Send + Sync {
    fn name(&self) -> &str;

    /// Attempt to lower an op. Returns `Ok(Some(...))` if this strategy can
    /// handle it, `Ok(None)` if it should be deferred to the next strategy,
    /// or `Err` if the op is fundamentally unlowerable.
    fn lower(
        &self,
        op: &AiOp,
        inputs: &[TensorId],
        tensor_info: &HashMap<TensorId, TensorInfo>,
        dim_var_names: &HashMap<DimVarId, u32>,
    ) -> Result<Option<SymbolicLowering>>;
}

// ── Strategy implementations ────────────────────────────────────────────────

/// Resolves ops only when all required dimensions are concrete.
/// Returns `Ok(None)` (pass to next strategy) when any dim is symbolic.
pub struct ConcreteStrategy;

impl LoweringStrategy for ConcreteStrategy {
    fn name(&self) -> &str {
        "concrete"
    }

    fn lower(
        &self,
        op: &AiOp,
        inputs: &[TensorId],
        tensor_info: &HashMap<TensorId, TensorInfo>,
        dim_var_names: &HashMap<DimVarId, u32>,
    ) -> Result<Option<SymbolicLowering>> {
        match resolve_op(op, inputs, tensor_info, dim_var_names)? {
            Some((float_op, recipes)) => {
                if recipes.iter().any(is_deferred) {
                    Ok(None)
                } else {
                    let graph_op = wrap_graph_op(op, float_op);
                    Ok(Some(SymbolicLowering {
                        graph_op,
                        recipe: None,
                    }))
                }
            }
            None => Ok(None),
        }
    }
}

/// Resolves ops with a mix of concrete and symbolic dimensions.
/// Concrete dims are baked in; symbolic dims get 0-sentinels plus a recipe.
pub struct DeferredStrategy;

impl LoweringStrategy for DeferredStrategy {
    fn name(&self) -> &str {
        "deferred"
    }

    fn lower(
        &self,
        op: &AiOp,
        inputs: &[TensorId],
        tensor_info: &HashMap<TensorId, TensorInfo>,
        dim_var_names: &HashMap<DimVarId, u32>,
    ) -> Result<Option<SymbolicLowering>> {
        match resolve_op(op, inputs, tensor_info, dim_var_names)? {
            Some((float_op, recipes)) => {
                let recipe = if recipes.iter().any(is_deferred) {
                    Some(NodeShapeRecipe {
                        node_index: 0, // Caller patches with actual index
                        params: recipes,
                    })
                } else {
                    None
                };
                let graph_op = wrap_graph_op(op, float_op);
                Ok(Some(SymbolicLowering { graph_op, recipe }))
            }
            None => Ok(None),
        }
    }
}

/// Extract m/k/n recipes for MatMul-family ops.
/// Missing shape dims use `Concrete(1)` as a hint — the runtime's
/// `infer_matmul_k` will override from actual buffer sizes.
pub(super) fn matmul_recipes(
    inputs: &[TensorId],
    tensor_info: &HashMap<TensorId, TensorInfo>,
    dim_var_names: &HashMap<DimVarId, u32>,
) -> Option<(u32, u32, u32, Vec<ParamRecipe>)> {
    // Use Concrete(1) as fallback for missing dims — the runtime infers
    // actual dimensions from buffer sizes when compiled hints don't match.
    let fallback = ParamRecipe::Concrete(1);
    let k_recipe = dim_recipe(last_dim_expr(inputs.first(), tensor_info), dim_var_names)
        .or_else(|| {
            dim_recipe(
                second_last_dim_expr(inputs.get(1), tensor_info),
                dim_var_names,
            )
        })
        .unwrap_or(fallback.clone());
    let n_recipe = dim_recipe(last_dim_expr(inputs.get(1), tensor_info), dim_var_names)
        .unwrap_or(fallback.clone());
    let m_recipe = dim_recipe(
        second_last_dim_expr(inputs.first(), tensor_info),
        dim_var_names,
    )
    .unwrap_or(fallback);

    let m = resolve_or_zero(&m_recipe) as u32;
    let k = resolve_or_zero(&k_recipe) as u32;
    let n = resolve_or_zero(&n_recipe) as u32;

    let any_deferred = is_deferred(&m_recipe) || is_deferred(&k_recipe) || is_deferred(&n_recipe);
    let recipes = if any_deferred {
        vec![m_recipe, k_recipe, n_recipe]
    } else {
        vec![]
    };

    Some((m, k, n, recipes))
}

/// Extract m/k/n recipes for Gemm with trans_b=true.
/// Weight B is stored as [n, k], so:
///   k = last_dim(B)   (not second_last)
///   n = second_last_dim(B)   (not last)
pub(super) fn gemm_trans_b_recipes(
    inputs: &[TensorId],
    tensor_info: &HashMap<TensorId, TensorInfo>,
    dim_var_names: &HashMap<DimVarId, u32>,
) -> Option<(u32, u32, u32, Vec<ParamRecipe>)> {
    let fallback = ParamRecipe::Concrete(1);
    // k = last_dim(input[0]) = last_dim(B) (both should agree)
    let k_recipe = dim_recipe(last_dim_expr(inputs.first(), tensor_info), dim_var_names)
        .or_else(|| dim_recipe(last_dim_expr(inputs.get(1), tensor_info), dim_var_names))
        .unwrap_or(fallback.clone());
    // n = second_last_dim(B) when trans_b (B is [n, k])
    let n_recipe = dim_recipe(
        second_last_dim_expr(inputs.get(1), tensor_info),
        dim_var_names,
    )
    .unwrap_or(fallback.clone());
    let m_recipe = dim_recipe(
        second_last_dim_expr(inputs.first(), tensor_info),
        dim_var_names,
    )
    .unwrap_or(fallback);

    let m = resolve_or_zero(&m_recipe) as u32;
    let k = resolve_or_zero(&k_recipe) as u32;
    let n = resolve_or_zero(&n_recipe) as u32;

    let any_deferred = is_deferred(&m_recipe) || is_deferred(&k_recipe) || is_deferred(&n_recipe);
    let recipes = if any_deferred {
        vec![m_recipe, k_recipe, n_recipe]
    } else {
        vec![]
    };

    Some((m, k, n, recipes))
}

/// Wrap a resolved `FloatOp` in the appropriate `GraphOp`.
///
/// For fused AiOp variants (MatMulRelu/Gelu/Silu), wraps in
/// `GraphOp::FusedMatMulActivation` so the tape builder maps to the
/// fused kernel (InlineMatMulActivation) which applies the activation
/// in-register after matmul writeback — eliminating the intermediate buffer.
///
/// For all other ops, wraps in `GraphOp::Float`.
fn wrap_graph_op(ai_op: &AiOp, float_op: FloatOp) -> GraphOp {
    match ai_op {
        AiOp::MatMulRelu => match float_op {
            FloatOp::MatMul { m, k, n } => GraphOp::FusedMatMulActivation {
                m,
                k,
                n,
                activation: FloatOp::Relu,
            },
            _ => GraphOp::Float(float_op),
        },
        AiOp::MatMulGelu => match float_op {
            FloatOp::MatMul { m, k, n } => GraphOp::FusedMatMulActivation {
                m,
                k,
                n,
                activation: FloatOp::Gelu,
            },
            _ => GraphOp::Float(float_op),
        },
        AiOp::MatMulSilu => match float_op {
            FloatOp::MatMul { m, k, n } => GraphOp::FusedMatMulActivation {
                m,
                k,
                n,
                activation: FloatOp::Silu,
            },
            _ => GraphOp::Float(float_op),
        },
        _ => GraphOp::Float(float_op),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;
    use crate::ir::{shape::shape_from_concrete, DimExpr, DimVarId, TensorInfo};

    fn make_tensor_info(shape: &[DimExpr], dtype: DType) -> TensorInfo {
        TensorInfo::new(dtype, shape.iter().cloned().collect())
    }

    #[test]
    fn concrete_strategy_resolves_matmul() {
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 128, 2048])),
        );
        ti.insert(
            1u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[2048, 4096])),
        );

        let strategy = ConcreteStrategy;
        let result = strategy
            .lower(&AiOp::MatMul, &[0, 1], &ti, &HashMap::new())
            .unwrap();

        assert!(result.is_some());
        let lowering = result.unwrap();
        assert!(lowering.recipe.is_none());
        match lowering.graph_op {
            GraphOp::Float(FloatOp::MatMul { m, k, n }) => {
                assert_eq!(m, 128);
                assert_eq!(k, 2048);
                assert_eq!(n, 4096);
            }
            _ => panic!("expected MatMul"),
        }
    }

    #[test]
    fn concrete_strategy_defers_symbolic_matmul() {
        let seq_var = DimVarId(0);
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            make_tensor_info(
                &[
                    DimExpr::Concrete(1),
                    DimExpr::Var(seq_var),
                    DimExpr::Concrete(2048),
                ],
                DType::F32,
            ),
        );
        ti.insert(
            1u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[2048, 4096])),
        );

        let strategy = ConcreteStrategy;
        let result = strategy
            .lower(&AiOp::MatMul, &[0, 1], &ti, &HashMap::new())
            .unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn deferred_strategy_handles_symbolic_matmul() {
        let seq_var = DimVarId(0);
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            make_tensor_info(
                &[
                    DimExpr::Concrete(1),
                    DimExpr::Var(seq_var),
                    DimExpr::Concrete(2048),
                ],
                DType::F32,
            ),
        );
        ti.insert(
            1u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[2048, 4096])),
        );

        let mut dim_var_names = HashMap::new();
        dim_var_names.insert(seq_var, 1u32);

        let strategy = DeferredStrategy;
        let result = strategy
            .lower(&AiOp::MatMul, &[0, 1], &ti, &dim_var_names)
            .unwrap();

        assert!(result.is_some());
        let lowering = result.unwrap();

        match lowering.graph_op {
            GraphOp::Float(FloatOp::MatMul { m, k, n }) => {
                assert_eq!(m, 0); // deferred
                assert_eq!(k, 2048);
                assert_eq!(n, 4096);
            }
            _ => panic!("expected MatMul"),
        }

        let recipe = lowering.recipe.unwrap();
        assert_eq!(recipe.params.len(), 3);
        assert_eq!(recipe.params[0], ParamRecipe::DimVar(1)); // m = seq_len
        assert_eq!(recipe.params[1], ParamRecipe::Concrete(2048)); // k
        assert_eq!(recipe.params[2], ParamRecipe::Concrete(4096)); // n
    }

    #[test]
    fn deferred_strategy_rmsnorm_concrete() {
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 128, 2048])),
        );

        let strategy = DeferredStrategy;
        let result = strategy
            .lower(&AiOp::RmsNorm { epsilon: 1e-5 }, &[0], &ti, &HashMap::new())
            .unwrap();

        assert!(result.is_some());
        let lowering = result.unwrap();
        assert!(lowering.recipe.is_none());
        match lowering.graph_op {
            GraphOp::Float(FloatOp::RmsNorm { size, .. }) => assert_eq!(size, 2048),
            _ => panic!("expected RmsNorm"),
        }
    }

    #[test]
    fn size_op_with_symbolic_dim() {
        let seq_var = DimVarId(0);
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            make_tensor_info(&[DimExpr::Concrete(1), DimExpr::Var(seq_var)], DType::F32),
        );

        let mut dim_var_names = HashMap::new();
        dim_var_names.insert(seq_var, 0u32);

        // ConcreteStrategy: size_op no longer emits a recipe, so symbolic dims
        // produce size=0 with no recipe — ConcreteStrategy now accepts this.
        let concrete = ConcreteStrategy;
        let result = concrete
            .lower(&AiOp::Softmax { axis: -1 }, &[0], &ti, &dim_var_names)
            .unwrap();
        // size=0 (sentinel), no deferred recipes → ConcreteStrategy accepts it.
        assert!(result.is_some());
        let lowering = result.unwrap();
        match lowering.graph_op {
            GraphOp::Float(FloatOp::Softmax { size }) => assert_eq!(size, 0),
            _ => panic!("expected Softmax"),
        }
        assert!(lowering.recipe.is_none());

        // DeferredStrategy: same — no recipe needed; resolve_dynamic_sizes() handles size=0.
        let deferred = DeferredStrategy;
        let result = deferred
            .lower(&AiOp::Softmax { axis: -1 }, &[0], &ti, &dim_var_names)
            .unwrap();
        assert!(result.is_some());
        let lowering = result.unwrap();
        match lowering.graph_op {
            GraphOp::Float(FloatOp::Softmax { size }) => assert_eq!(size, 0),
            _ => panic!("expected Softmax"),
        }
        assert!(lowering.recipe.is_none());
    }
}
