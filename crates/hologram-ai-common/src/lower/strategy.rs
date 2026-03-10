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

use crate::exec_context::{NodeShapeRecipe, ParamRecipe};
use crate::ir::{AiOp, DimExpr, DimVarId, TensorId, TensorInfo};
use anyhow::Result;
use hologram::{f32_to_bits, FloatDType, FloatOp, GraphOp};
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
                    Ok(None) // Has symbolic dims — pass to next strategy
                } else {
                    Ok(Some(SymbolicLowering {
                        graph_op: GraphOp::Float(float_op),
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
                Ok(Some(SymbolicLowering {
                    graph_op: GraphOp::Float(float_op),
                    recipe,
                }))
            }
            None => Ok(None),
        }
    }
}

// ── Unified resolver ────────────────────────────────────────────────────────
//
// Single match block that maps AiOp → (FloatOp, Vec<ParamRecipe>).
// Symbolic dims become 0-sentinels in the FloatOp and DimVar/Product recipes.
// Strategy implementations decide whether to accept or reject deferred recipes.

/// Resolve a single-size op: extract `last_dim` from input 0 as a recipe,
/// then call $make_op(size_u32) to build the FloatOp.
macro_rules! size_op {
    ($inputs:expr, $ti:expr, $dvn:expr, |$size:ident| $make_op:expr) => {{
        let recipe = dim_recipe(last_dim_expr($inputs.first(), $ti), $dvn);
        let $size = recipe.as_ref().map(resolve_or_zero).unwrap_or(0) as u32;
        let recipes = match recipe {
            Some(r) => vec![r],
            None => vec![],
        };
        ($make_op, recipes)
    }};
}

/// Extract m/k/n recipes for MatMul-family ops.
/// Missing shape dims use `Concrete(1)` as a hint — the runtime's
/// `infer_matmul_k` will override from actual buffer sizes.
fn matmul_recipes(
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

fn resolve_op(
    op: &AiOp,
    inputs: &[TensorId],
    tensor_info: &HashMap<TensorId, TensorInfo>,
    dim_var_names: &HashMap<DimVarId, u32>,
) -> Result<Option<(FloatOp, Vec<ParamRecipe>)>> {
    let result = match op {
        // ── MatMul family ───────────────────────────────────────────────
        AiOp::MatMul | AiOp::BatchMatMul => {
            let (m, k, n, recipes) = match matmul_recipes(inputs, tensor_info, dim_var_names) {
                Some(v) => v,
                None => return Ok(None),
            };
            (FloatOp::MatMul { m, k, n }, recipes)
        }
        AiOp::Gemm {
            alpha,
            beta,
            trans_a,
            trans_b,
        } => {
            let (m, k, n, recipes) = match matmul_recipes(inputs, tensor_info, dim_var_names) {
                Some(v) => v,
                None => return Ok(None),
            };
            (
                FloatOp::Gemm {
                    m,
                    k,
                    n,
                    alpha: f32_to_bits(*alpha),
                    beta: f32_to_bits(*beta),
                    trans_a: *trans_a,
                    trans_b: *trans_b,
                },
                recipes,
            )
        }

        // ── Single-size ops (macro-generated) ───────────────────────────
        AiOp::Softmax { .. } => {
            size_op!(inputs, tensor_info, dim_var_names, |size| {
                FloatOp::Softmax { size }
            })
        }
        AiOp::LogSoftmax { .. } => {
            size_op!(inputs, tensor_info, dim_var_names, |size| {
                FloatOp::LogSoftmax { size }
            })
        }
        AiOp::RmsNorm { epsilon } => {
            size_op!(inputs, tensor_info, dim_var_names, |size| {
                FloatOp::RmsNorm {
                    size,
                    epsilon: f32_to_bits(*epsilon),
                }
            })
        }
        AiOp::LayerNorm { epsilon, .. } => {
            size_op!(inputs, tensor_info, dim_var_names, |size| {
                FloatOp::LayerNorm {
                    size,
                    epsilon: f32_to_bits(*epsilon),
                }
            })
        }
        AiOp::ReduceSum { .. } => {
            size_op!(inputs, tensor_info, dim_var_names, |size| {
                FloatOp::ReduceSum { size }
            })
        }
        AiOp::ReduceMean { .. } => {
            size_op!(inputs, tensor_info, dim_var_names, |size| {
                FloatOp::ReduceMean { size }
            })
        }
        AiOp::ReduceMax { .. } => {
            size_op!(inputs, tensor_info, dim_var_names, |size| {
                FloatOp::ReduceMax { size }
            })
        }
        AiOp::ReduceMin { .. } => {
            size_op!(inputs, tensor_info, dim_var_names, |size| {
                FloatOp::ReduceMin { size }
            })
        }

        // ── Ops with non-shape params ───────────────────────────────────
        AiOp::Gather { axis } | AiOp::GatherElements { axis } => {
            // dim = product of dims AFTER the gather axis (row width in hologram's
            // table-based Gather). For axis=0 on [N], dim=1. For axis=0 on
            // [N, D], dim=D. For axis=-1 on [A, B, C], dim=1.
            let dim = gather_row_width(inputs.first(), *axis, tensor_info).unwrap_or(1) as u32;
            let dtype = input_float_dtype(inputs.first(), tensor_info);
            (FloatOp::Gather { dim, dtype }, vec![])
        }
        AiOp::Concat { axis } => {
            let size_a =
                concrete_concat_row_size(inputs.first(), *axis, tensor_info).unwrap_or(1) as u32;
            let size_b =
                concrete_concat_row_size(inputs.get(1), *axis, tensor_info).unwrap_or(1) as u32;
            let dtype = input_float_dtype(inputs.first(), tensor_info);
            (
                FloatOp::Concat {
                    size_a,
                    size_b,
                    dtype,
                },
                vec![],
            )
        }
        AiOp::Embed => {
            let dim = concrete_last_dim(inputs.get(1), tensor_info).unwrap_or(1) as u32;
            (FloatOp::Embed { dim }, vec![])
        }

        // ── Attention ops (params from AiOp fields, always concrete) ────
        AiOp::MultiHeadAttention {
            num_heads,
            head_dim,
            scale,
            causal,
        } => {
            let s = scale.unwrap_or((*head_dim as f32).sqrt().recip());
            (
                FloatOp::Attention {
                    head_dim: *head_dim,
                    num_q_heads: *num_heads,
                    num_kv_heads: *num_heads,
                    scale: f32_to_bits(s),
                    causal: *causal,
                },
                vec![],
            )
        }
        AiOp::GroupedQueryAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            causal,
        } => {
            let s = scale.unwrap_or((*head_dim as f32).sqrt().recip());
            (
                FloatOp::Attention {
                    head_dim: *head_dim,
                    num_q_heads: *num_heads,
                    num_kv_heads: *num_kv_heads,
                    scale: f32_to_bits(s),
                    causal: *causal,
                },
                vec![],
            )
        }
        AiOp::FlashAttentionHint => (
            FloatOp::Attention {
                head_dim: 64,
                num_q_heads: 1,
                num_kv_heads: 1,
                scale: f32_to_bits(0.125),
                causal: true,
            },
            vec![],
        ),

        // ── Type/shape ops (no dims needed) ─────────────────────────────
        AiOp::Cast { to } => {
            let from = input_float_dtype(inputs.first(), tensor_info);
            (
                FloatOp::Cast {
                    from,
                    to: ai_dtype_to_float_dtype(to),
                },
                vec![],
            )
        }
        AiOp::Shape { .. } => {
            let dtype = input_float_dtype(inputs.first(), tensor_info);
            (FloatOp::Shape { dtype }, vec![])
        }

        AiOp::Slice {
            axes,
            starts,
            ends,
            steps,
        } => {
            // Handle single-axis contiguous slices.
            if axes.len() != 1 || starts.len() != 1 || ends.len() != 1 {
                return Ok(None);
            }
            // Only handle step=1.
            if steps.first().copied().unwrap_or(1) != 1 {
                return Ok(None);
            }
            let axis = axes[0];
            let start = starts[0];
            let end = ends[0];
            // Determine the input shape to resolve negative indices.
            let in_shape = inputs
                .first()
                .and_then(|tid| tensor_info.get(tid))
                .map(|info| &info.shape);
            let ndim = in_shape.map(|s| s.len() as i64).unwrap_or(0);
            // Normalize axis.
            let norm_axis = if axis < 0 { ndim + axis } else { axis };
            if norm_axis < 0 || norm_axis >= ndim {
                return Ok(None);
            }
            let axis_from_end = (ndim - norm_axis) as u8;
            // Resolve axis size from shape.
            let axis_size = in_shape
                .and_then(|s| s.get(norm_axis as usize))
                .and_then(|d| d.as_concrete())
                .unwrap_or(0) as i64;
            // Normalize start/end with respect to axis size.
            let norm_start = if start < 0 {
                (axis_size + start).max(0) as u32
            } else {
                start.min(axis_size) as u32
            };
            let norm_end = if end < 0 {
                (axis_size + end).max(0) as u32
            } else if end > axis_size {
                axis_size as u32
            } else {
                end as u32
            };
            (
                FloatOp::Slice {
                    axis_from_end,
                    start: norm_start,
                    end: norm_end,
                },
                vec![],
            )
        }

        _ => return Ok(None),
    };

    Ok(Some(result))
}

// ── Dim expression helpers ──────────────────────────────────────────────────

fn last_dim_expr(
    tid: Option<&TensorId>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> Option<DimExpr> {
    tid.and_then(|t| tensor_info.get(t))
        .and_then(|info| info.shape.last())
        .cloned()
}

fn second_last_dim_expr(
    tid: Option<&TensorId>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> Option<DimExpr> {
    tid.and_then(|t| tensor_info.get(t)).and_then(|info| {
        let n = info.shape.len();
        if n >= 2 {
            info.shape.get(n - 2).cloned()
        } else {
            None
        }
    })
}

/// Convert a `DimExpr` to a `ParamRecipe`.
/// Returns None if the expression can't be mapped.
fn dim_recipe(
    expr: Option<DimExpr>,
    dim_var_names: &HashMap<DimVarId, u32>,
) -> Option<ParamRecipe> {
    let expr = expr?;
    match &expr {
        DimExpr::Concrete(v) => Some(ParamRecipe::Concrete(*v)),
        DimExpr::Var(id) => Some(
            dim_var_names
                .get(id)
                .map(|&idx| ParamRecipe::DimVar(idx))
                .unwrap_or(ParamRecipe::RuntimeInferred),
        ),
        DimExpr::Dynamic => Some(ParamRecipe::RuntimeInferred),
        DimExpr::Mul(a, b) => match (a.as_ref(), b.as_ref()) {
            (DimExpr::Var(id), DimExpr::Concrete(v)) | (DimExpr::Concrete(v), DimExpr::Var(id)) => {
                dim_var_names
                    .get(id)
                    .map(|&idx| ParamRecipe::Product(idx, *v))
            }
            _ => expr.evaluate().map(ParamRecipe::Concrete),
        },
        _ => expr.evaluate().map(ParamRecipe::Concrete),
    }
}

fn resolve_or_zero(recipe: &ParamRecipe) -> u64 {
    match recipe {
        ParamRecipe::Concrete(v) => *v,
        _ => 0,
    }
}

fn is_deferred(recipe: &ParamRecipe) -> bool {
    !matches!(recipe, ParamRecipe::Concrete(_))
}

// ── Concrete dim helpers ────────────────────────────────────────────────────

fn concrete_last_dim(
    tid: Option<&TensorId>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> Option<u64> {
    tid.and_then(|t| tensor_info.get(t))
        .and_then(|info| info.shape.last())
        .and_then(|dim| dim.as_concrete())
}

/// Product of dims after the gather axis. This is the "row width" in
/// hologram's table-based Gather: each row is this many elements.
fn gather_row_width(
    tid: Option<&TensorId>,
    axis: i64,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> Option<u64> {
    let info = tid.and_then(|t| tensor_info.get(t))?;
    if info.shape.is_empty() {
        return Some(1);
    }
    let ndim = info.shape.len();
    let ax = if axis < 0 {
        (ndim as i64 + axis).max(0) as usize
    } else {
        (axis as usize).min(ndim.saturating_sub(1))
    };
    let mut product = 1u64;
    for dim in info.shape.iter().skip(ax + 1) {
        product = product.saturating_mul(dim.as_concrete()?);
    }
    Some(product.max(1))
}

fn concrete_concat_row_size(
    tid: Option<&TensorId>,
    axis: i64,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> Option<usize> {
    let info = tid.and_then(|t| tensor_info.get(t))?;
    if info.shape.is_empty() {
        return None;
    }
    let ndim = info.shape.len();
    let ax = if axis < 0 {
        (ndim as i64 + axis).max(0) as usize
    } else {
        (axis as usize).min(ndim.saturating_sub(1))
    };
    let mut product = 1usize;
    for dim in info.shape.iter().skip(ax + 1) {
        product = product.saturating_mul(dim.as_concrete()? as usize);
    }
    Some(product.max(1))
}

/// Look up the logical dtype of a tensor, defaulting to F32.
pub(crate) fn input_float_dtype(
    tid: Option<&TensorId>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> FloatDType {
    tid.and_then(|t| tensor_info.get(t))
        .map(|info| ai_dtype_to_float_dtype(&info.logical_dtype))
        .unwrap_or(FloatDType::F32)
}

/// Convert hologram-ai `DType` to hologram base crate `FloatDType`.
pub(crate) fn ai_dtype_to_float_dtype(dtype: &crate::ir::DType) -> FloatDType {
    use crate::ir::DType;
    match dtype {
        DType::F32 => FloatDType::F32,
        DType::F16 => FloatDType::F16,
        DType::BF16 => FloatDType::BF16,
        DType::INT8 => FloatDType::I8,
        DType::INT4 => FloatDType::I8,
        DType::U8 => FloatDType::U8,
        DType::INT32 => FloatDType::I32,
        DType::INT64 => FloatDType::I64,
        DType::BOOL => FloatDType::Bool,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;
    use crate::ir::{shape::shape_from_concrete, DimVarId, TensorInfo};

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

        // ConcreteStrategy should reject
        let concrete = ConcreteStrategy;
        let result = concrete
            .lower(&AiOp::Softmax { axis: -1 }, &[0], &ti, &dim_var_names)
            .unwrap();
        assert!(result.is_none());

        // DeferredStrategy should produce recipe
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
        let recipe = lowering.recipe.unwrap();
        assert_eq!(recipe.params, vec![ParamRecipe::DimVar(0)]);
    }
}
