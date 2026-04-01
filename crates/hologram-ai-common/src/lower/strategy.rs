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
                    Ok(None)
                } else {
                    let graph_op = wrap_graph_op(op, float_op);
                    Ok(Some(SymbolicLowering { graph_op, recipe: None }))
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

// ── Unified resolver ────────────────────────────────────────────────────────
//
// Single match block that maps AiOp → (FloatOp, Vec<ParamRecipe>).
// Symbolic dims become 0-sentinels in the FloatOp and DimVar/Product recipes.
// Strategy implementations decide whether to accept or reject deferred recipes.

/// Resolve a single-size op: extract the concrete last-dim from input 0,
/// or emit 0 when symbolic. `resolve_dynamic_sizes()` in hologram-exec
/// patches the 0-sentinel at runtime from the actual input shape, so no
/// `ParamRecipe` is needed here.
macro_rules! size_op {
    ($inputs:expr, $ti:expr, $_dvn:expr, |$size:ident| $make_op:expr) => {{
        let $size = concrete_last_dim($inputs.first(), $ti).unwrap_or(0) as u32;
        ($make_op, vec![])
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

/// Extract m/k/n recipes for Gemm with trans_b=true.
/// Weight B is stored as [n, k], so:
///   k = last_dim(B)   (not second_last)
///   n = second_last_dim(B)   (not last)
fn gemm_trans_b_recipes(
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
    let n_recipe = dim_recipe(second_last_dim_expr(inputs.get(1), tensor_info), dim_var_names)
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
            FloatOp::MatMul { m, k, n } => GraphOp::FusedMatMulActivation { m, k, n, activation: FloatOp::Relu },
            _ => GraphOp::Float(float_op),
        },
        AiOp::MatMulGelu => match float_op {
            FloatOp::MatMul { m, k, n } => GraphOp::FusedMatMulActivation { m, k, n, activation: FloatOp::Gelu },
            _ => GraphOp::Float(float_op),
        },
        AiOp::MatMulSilu => match float_op {
            FloatOp::MatMul { m, k, n } => GraphOp::FusedMatMulActivation { m, k, n, activation: FloatOp::Silu },
            _ => GraphOp::Float(float_op),
        },
        _ => GraphOp::Float(float_op),
    }
}

fn resolve_op(
    op: &AiOp,
    inputs: &[TensorId],
    tensor_info: &HashMap<TensorId, TensorInfo>,
    dim_var_names: &HashMap<DimVarId, u32>,
) -> Result<Option<(FloatOp, Vec<ParamRecipe>)>> {
    let result = match op {
        // ── MatMul family ───────────────────────────────────────────────
        AiOp::MatMul | AiOp::BatchMatMul
        | AiOp::MatMulRelu | AiOp::MatMulGelu | AiOp::MatMulSilu
        | AiOp::ConcatMatMul { .. } => {
            // TODO: lower MatMulRelu/Gelu/Silu to fused FloatOp variants once
            // hologram base adds FloatOp::MatMulRelu etc. For now, lower as
            // plain MatMul (activation is lost — the fusion pass should not be
            // registered in the pipeline until fused kernels exist).
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
            let (m, k, n, recipes) = if *trans_b {
                match gemm_trans_b_recipes(inputs, tensor_info, dim_var_names) {
                    Some(v) => v,
                    None => return Ok(None),
                }
            } else {
                match matmul_recipes(inputs, tensor_info, dim_var_names) {
                    Some(v) => v,
                    None => return Ok(None),
                }
            };
            // Validate k against weight param size when available.
            if let Some(weight_info) = inputs.get(1).and_then(|tid| tensor_info.get(tid)) {
                let weight_elems: u64 = weight_info.shape.iter()
                    .filter_map(|d| d.as_concrete())
                    .product();
                if k > 0 && n > 0 && weight_elems > 0 && (k as u64 * n as u64) != weight_elems {
                    let a_shape: Vec<_> = inputs.first()
                        .and_then(|tid| tensor_info.get(tid))
                        .map(|i| i.shape.iter().filter_map(|d| d.as_concrete()).collect::<Vec<_>>())
                        .unwrap_or_default();
                    let w_shape: Vec<_> = weight_info.shape.iter()
                        .filter_map(|d| d.as_concrete())
                        .collect();
                    tracing::warn!(
                        m, k, n,
                        weight_elems,
                        ?a_shape,
                        ?w_shape,
                        trans_b,
                        "Gemm k*n doesn't match weight element count"
                    );
                }
            }
            let qb = quant_code(inputs.get(1), tensor_info);
            (
                FloatOp::Gemm {
                    m,
                    k,
                    n,
                    alpha: f32_to_bits(*alpha),
                    beta: f32_to_bits(*beta),
                    trans_a: *trans_a,
                    trans_b: *trans_b,
                    quant_b: qb,
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
        AiOp::FusedLayerNormResidual { epsilon } => {
            size_op!(inputs, tensor_info, dim_var_names, |size| {
                FloatOp::AddRmsNorm {
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
            let quant = quant_code(inputs.get(1), tensor_info);
            (FloatOp::Embed { dim, quant }, vec![])
        }

        // ── RoPE: needs hidden_dim from input shape to compute n_heads ──
        AiOp::RotaryEmbedding { base, dim } => {
            let hidden_dim = concrete_last_dim(inputs.first(), tensor_info).unwrap_or(*dim as u64);
            let n_heads = (hidden_dim / (*dim as u64).max(1)) as u32;
            (
                FloatOp::RotaryEmbedding {
                    dim: *dim,
                    base: f32_to_bits(*base),
                    n_heads,
                },
                vec![],
            )
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
                    heads_first: false,
                    qk_norm: false,
                    rope: false,
                    rope_base: 0,
                    sparse_v: true,
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
            heads_first,
            qk_norm,
            rope,
            rope_base,
        } => {
            let s = scale.unwrap_or((*head_dim as f32).sqrt().recip());
            (
                FloatOp::Attention {
                    head_dim: *head_dim,
                    num_q_heads: *num_heads,
                    num_kv_heads: *num_kv_heads,
                    scale: f32_to_bits(s),
                    causal: *causal,
                    heads_first: *heads_first,
                    qk_norm: *qk_norm,
                    rope: *rope,
                    rope_base: f32_to_bits(*rope_base),
                    sparse_v: true,
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
                heads_first: false,
                qk_norm: false,
                rope: false,
                rope_base: 0,
                sparse_v: true,
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
        AiOp::Shape { start, end } => {
            let dtype = input_float_dtype(inputs.first(), tensor_info);
            (
                FloatOp::Shape {
                    dtype,
                    start: start.unwrap_or(0),
                    end: end.unwrap_or(i64::MAX),
                },
                vec![],
            )
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
            // When axis_size=0 (dynamic/unknown sentinel), preserve positive indices
            // as-is rather than clamping to 0 — they'll be validated at runtime.
            let norm_start = if start < 0 {
                if axis_size > 0 { (axis_size + start).max(0) as u32 } else { 0 }
            } else if axis_size > 0 {
                start.min(axis_size) as u32
            } else {
                start as u32
            };
            let norm_end = if end < 0 {
                if axis_size > 0 { (axis_size + end).max(0) as u32 } else { 0 }
            } else if end > axis_size && axis_size > 0 {
                axis_size as u32
            } else {
                end as u32
            };
            (
                FloatOp::Slice {
                    axis_from_end,
                    start: norm_start,
                    end: norm_end,
                    axis_size: axis_size as u32,
                },
                vec![],
            )
        }

        // ── Vision ops ───────────────────────────────────────────────────
        AiOp::Conv {
            kernel_shape,
            strides,
            pads,
            dilations,
            group,
            ..
        } => {
            let (kh, kw) = get_hw(kernel_shape, 1);
            let (sh, sw) = get_hw(strides, 1);
            let (ph, pw) = get_pads_hw(pads);
            let (dh, dw) = get_hw(dilations, 1);
            // Input is [N, C, H, W] — extract H (idx 2) and W (idx 3).
            let ih = dim_at_expr(inputs.first(), tensor_info, 2)
                .and_then(|e| e.evaluate())
                .unwrap_or(0) as u32;
            let iw = dim_at_expr(inputs.first(), tensor_info, 3)
                .and_then(|e| e.evaluate())
                .unwrap_or(0) as u32;
            (
                FloatOp::Conv2d {
                    kernel_h: kh as u32, kernel_w: kw as u32,
                    stride_h: sh as u32, stride_w: sw as u32,
                    pad_h: ph as u32, pad_w: pw as u32,
                    dilation_h: dh as u32, dilation_w: dw as u32,
                    group: *group as u32,
                    input_h: ih, input_w: iw,
                },
                vec![],
            )
        }
        AiOp::ConvTranspose {
            kernel_shape,
            strides,
            pads,
            output_padding,
            dilations,
            group,
            ..
        } => {
            let (kh, kw) = get_hw(kernel_shape, 1);
            let (sh, sw) = get_hw(strides, 1);
            let (ph, pw) = get_pads_hw(pads);
            let (dh, dw) = get_hw(dilations, 1);
            let (oph, opw) = get_hw(output_padding, 0);
            let ih = dim_at_expr(inputs.first(), tensor_info, 2)
                .and_then(|e| e.evaluate())
                .unwrap_or(0) as u32;
            let iw = dim_at_expr(inputs.first(), tensor_info, 3)
                .and_then(|e| e.evaluate())
                .unwrap_or(0) as u32;
            (
                FloatOp::ConvTranspose {
                    kernel_h: kh as u32, kernel_w: kw as u32,
                    stride_h: sh as u32, stride_w: sw as u32,
                    pad_h: ph as u32, pad_w: pw as u32,
                    dilation_h: dh as u32, dilation_w: dw as u32,
                    group: *group as u32,
                    output_pad_h: oph as u32, output_pad_w: opw as u32,
                    input_h: ih, input_w: iw,
                },
                vec![],
            )
        }
        AiOp::MaxPool {
            kernel_shape,
            strides,
            pads,
            ..
        } => {
            let (kh, kw) = get_hw(kernel_shape, 1);
            let (sh, sw) = get_hw(strides, 1);
            let (ph, pw) = get_pads_hw(pads);
            (
                FloatOp::MaxPool2d {
                    kernel_h: kh as u32, kernel_w: kw as u32,
                    stride_h: sh as u32, stride_w: sw as u32,
                    pad_h: ph as u32, pad_w: pw as u32,
                },
                vec![],
            )
        }
        AiOp::AveragePool {
            kernel_shape,
            strides,
            pads,
            ..
        } => {
            let (kh, kw) = get_hw(kernel_shape, 1);
            let (sh, sw) = get_hw(strides, 1);
            let (ph, pw) = get_pads_hw(pads);
            (
                FloatOp::AvgPool2d {
                    kernel_h: kh as u32, kernel_w: kw as u32,
                    stride_h: sh as u32, stride_w: sw as u32,
                    pad_h: ph as u32, pad_w: pw as u32,
                },
                vec![],
            )
        }
        AiOp::GlobalAveragePool => {
            let channels = dim_at_expr(inputs.first(), tensor_info, 1)
                .and_then(|e| e.evaluate())
                .unwrap_or(0) as u32;
            let spatial_h = dim_at_expr(inputs.first(), tensor_info, 2)
                .and_then(|e| e.evaluate())
                .unwrap_or(0) as u32;
            let spatial_w = dim_at_expr(inputs.first(), tensor_info, 3)
                .and_then(|e| e.evaluate())
                .unwrap_or(0) as u32;
            (FloatOp::GlobalAvgPool { channels, spatial_h, spatial_w }, vec![])
        }
        AiOp::Resize { mode, .. } => {
            let mode_u8 = match mode.as_str() {
                "nearest" => 0,
                "linear" | "bilinear" => 1,
                "cubic" | "bicubic" => 2,
                _ => 0,
            };
            (FloatOp::Resize { mode: mode_u8 }, vec![])
        }
        AiOp::Pad { mode } => {
            let mode_u8 = match mode.as_str() {
                "constant" => 0,
                "reflect" => 1,
                "edge" => 2,
                _ => 0,
            };
            (FloatOp::PadOp { mode: mode_u8 }, vec![])
        }
        AiOp::InstanceNorm { epsilon } => {
            // InstanceNorm normalizes across ALL spatial dims (H×W), not just the
            // last dim. Compute size = product of dims[2..] for NCHW tensors.
            let size = inputs
                .first()
                .and_then(|tid| tensor_info.get(tid))
                .map(|info| {
                    if info.shape.len() >= 3 {
                        info.shape[2..]
                            .iter()
                            .filter_map(|d| d.as_concrete())
                            .product::<u64>() as u32
                    } else {
                        info.shape
                            .last()
                            .and_then(|d| d.as_concrete())
                            .unwrap_or(0) as u32
                    }
                })
                .unwrap_or(0);
            (
                FloatOp::InstanceNorm {
                    size,
                    epsilon: f32_to_bits(*epsilon),
                },
                vec![],
            )
        }
        AiOp::GroupNorm { num_groups, epsilon } => {
            (FloatOp::GroupNorm { num_groups: *num_groups, epsilon: f32_to_bits(*epsilon) }, vec![])
        }
        AiOp::LRN { alpha, beta, bias, size } => (
            FloatOp::LRN {
                size: *size as u32,
                alpha: f32_to_bits(*alpha),
                beta: f32_to_bits(*beta),
                bias: f32_to_bits(*bias),
            },
            vec![],
        ),

        // ── Utility ops ─────────────────────────────────────────────────
        AiOp::ReduceProd { .. } => {
            size_op!(inputs, tensor_info, dim_var_names, |size| {
                FloatOp::ReduceProd { size }
            })
        }
        AiOp::TopK { axis, largest, .. } => {
            let norm_axis = normalize_axis(*axis, inputs.first(), tensor_info);
            (
                FloatOp::TopK {
                    axis: norm_axis as u32,
                    largest: *largest,
                },
                vec![],
            )
        }
        AiOp::ScatterND { .. } => (FloatOp::ScatterND, vec![]),
        AiOp::CumSum { .. } => {
            // CumSum axis is typically 0 or from a 1-element input tensor.
            // Default to axis 0.
            (FloatOp::CumSum { axis: 0 }, vec![])
        }
        AiOp::NonZero => (FloatOp::NonZero, vec![]),
        AiOp::Compress { axis } => {
            let norm_axis = axis
                .map(|a| normalize_axis(a, inputs.first(), tensor_info))
                .unwrap_or(0);
            (FloatOp::Compress { axis: norm_axis as u32 }, vec![])
        }
        AiOp::ReverseSequence { batch_axis, time_axis } => {
            let ba = normalize_axis(*batch_axis, inputs.first(), tensor_info);
            let ta = normalize_axis(*time_axis, inputs.first(), tensor_info);
            (
                FloatOp::ReverseSequence {
                    batch_axis: ba as u32,
                    time_axis: ta as u32,
                },
                vec![],
            )
        }

        // ── KV cache ops ─────────────────────────────────────────────────
        AiOp::KvSlotWrite { layer, is_key, n_kv_heads, head_dim, layout } => {
            (
                FloatOp::KvWrite {
                    layer: *layer as u32,
                    n_kv_heads: *n_kv_heads,
                    head_dim: *head_dim,
                    is_key: *is_key,
                    heads_first: matches!(layout, crate::ir::KvLayout::HeadsFirst),
                },
                vec![],
            )
        }
        AiOp::KvSlotRead { layer, n_kv_heads, head_dim, layout } => (
            FloatOp::KvRead {
                layer: *layer as u32,
                n_kv_heads: *n_kv_heads,
                head_dim: *head_dim,
                heads_first: matches!(layout, crate::ir::KvLayout::HeadsFirst),
            },
            vec![],
        ),

        _ => return Ok(None),
    };

    Ok(Some(result))
}

// ── Spatial param helpers ────────────────────────────────────────────────────

/// Extract (H, W) from a spatial param vec, using `default` if missing.
fn get_hw(v: &[u64], default: u64) -> (u64, u64) {
    match v.len() {
        0 => (default, default),
        1 => (v[0], v[0]),
        _ => (v[0], v[1]),
    }
}

/// Extract (pad_h, pad_w) from ONNX-style pads [h_begin, w_begin, h_end, w_end].
/// Returns (h_begin, w_begin) — symmetric padding assumed for now.
fn get_pads_hw(pads: &[u64]) -> (u64, u64) {
    match pads.len() {
        0 => (0, 0),
        2 => (pads[0], pads[1]),
        4 => (pads[0], pads[1]), // [h_begin, w_begin, h_end, w_end]
        _ => (pads.first().copied().unwrap_or(0), pads.get(1).copied().unwrap_or(0)),
    }
}

/// Normalize a potentially-negative axis to a positive index.
fn normalize_axis(
    axis: i64,
    tid: Option<&TensorId>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> i64 {
    if axis >= 0 {
        return axis;
    }
    let ndim = tid
        .and_then(|t| tensor_info.get(t))
        .map(|info| info.shape.len() as i64)
        .unwrap_or(0);
    (ndim + axis).max(0)
}

// ── Dim expression helpers ──────────────────────────────────────────────────

fn dim_at_expr(
    tid: Option<&TensorId>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
    idx: usize,
) -> Option<DimExpr> {
    tid.and_then(|t| tensor_info.get(t))
        .and_then(|info| info.shape.get(idx))
        .cloned()
}

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
    // Return 0 if any dim is a 0-sentinel (dynamic); resolve_dynamic_sizes handles it.
    Some(product)
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
    // Row size = dim[ax] * product(dims after ax).
    // dispatch_concat uses size_a/size_b to interleave rows: for each "outer"
    // position (product of dims before ax), it copies size_a elements from A
    // then size_b from B. Including dim[ax] ensures mid-axis and last-axis
    // concats produce the correct stride, not just axis=0 (simple append).
    let ax_dim = info.shape.get(ax)?.as_concrete()? as usize;
    let mut product = ax_dim;
    for dim in info.shape.iter().skip(ax + 1) {
        product = product.saturating_mul(dim.as_concrete()? as usize);
    }
    // Return 0 if any dim is a 0-sentinel (dynamic); resolve_dynamic_sizes handles it.
    Some(product)
}

/// Get the quantization code for a tensor: 0=none, 1=Q4_0, 2=Q8_0.
fn quant_code(
    tid: Option<&TensorId>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> u8 {
    use hologram_ai_quant::QuantScheme;
    tid.and_then(|t| tensor_info.get(t))
        .map(|info| match info.quant.scheme {
            QuantScheme::Q4_0 => 1,
            QuantScheme::Q8_0 => 2,
            QuantScheme::Q6K => 3,
            _ => 0,
        })
        .unwrap_or(0)
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
        DType::F64 => FloatDType::F32, // F64 → F32 at lowering (hologram base has no f64 kernels)
        DType::F16 => FloatDType::F16,
        DType::BF16 => FloatDType::BF16,
        DType::INT8 => FloatDType::I8,
        DType::INT4 => FloatDType::I8,
        DType::U8 => FloatDType::U8,
        DType::INT16 => FloatDType::I32, // INT16 widened to I32 at lowering
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
