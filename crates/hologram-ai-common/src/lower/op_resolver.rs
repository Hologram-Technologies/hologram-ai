//! Op resolution: AiOp → (FloatOp, Vec<ParamRecipe>).

use crate::exec_context::ParamRecipe;
use crate::ir::{AiOp, DimExpr, DimVarId, TensorId, TensorInfo};
use anyhow::Result;
use hologram::hologram_core::op::FloatOp;
use hologram::{f32_to_bits, FloatDType};
use std::collections::HashMap;

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

pub(crate) fn resolve_op(
    op: &AiOp,
    inputs: &[TensorId],
    tensor_info: &HashMap<TensorId, TensorInfo>,
    dim_var_names: &HashMap<DimVarId, u32>,
) -> Result<Option<(FloatOp, Vec<ParamRecipe>)>> {
    let result = match op {
        // ── MatMul family ───────────────────────────────────────────────
        AiOp::MatMul
        | AiOp::BatchMatMul
        | AiOp::MatMulRelu
        | AiOp::MatMulGelu
        | AiOp::MatMulSilu
        | AiOp::ConcatMatMul { .. } => {
            // TODO: lower MatMulRelu/Gelu/Silu to fused FloatOp variants once
            // hologram base adds FloatOp::MatMulRelu etc. For now, lower as
            // plain MatMul (activation is lost — the fusion pass should not be
            // registered in the pipeline until fused kernels exist).
            let (m, k, n, recipes) =
                match super::strategy::matmul_recipes(inputs, tensor_info, dim_var_names) {
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
                match super::strategy::gemm_trans_b_recipes(inputs, tensor_info, dim_var_names) {
                    Some(v) => v,
                    None => return Ok(None),
                }
            } else {
                match super::strategy::matmul_recipes(inputs, tensor_info, dim_var_names) {
                    Some(v) => v,
                    None => return Ok(None),
                }
            };
            // Validate k against weight param size when available.
            if let Some(weight_info) = inputs.get(1).and_then(|tid| tensor_info.get(tid)) {
                let weight_elems: u64 = weight_info
                    .shape
                    .iter()
                    .filter_map(|d| d.as_concrete())
                    .product();
                if k > 0 && n > 0 && weight_elems > 0 && (k as u64 * n as u64) != weight_elems {
                    let a_shape: Vec<_> = inputs
                        .first()
                        .and_then(|tid| tensor_info.get(tid))
                        .map(|i| {
                            i.shape
                                .iter()
                                .filter_map(|d| d.as_concrete())
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let w_shape: Vec<_> = weight_info
                        .shape
                        .iter()
                        .filter_map(|d| d.as_concrete())
                        .collect();
                    tracing::warn!(
                        m,
                        k,
                        n,
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
            let dim = concrete_last_dim(inputs.get(1), tensor_info).unwrap_or_else(|| {
                tracing::warn!(
                    table_tid = ?inputs.get(1),
                    has_info = inputs.get(1).is_some_and(|t| tensor_info.contains_key(t)),
                    shape = ?inputs.get(1).and_then(|t| tensor_info.get(t)).map(|i| &i.shape),
                    "Embed: concrete_last_dim failed, falling back to dim=1"
                );
                1
            }) as u32;
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
            // For empty-param Slices (dynamic params not resolved at import),
            // try resolving from DataPropagation's known_i64_values.
            let (axes, starts, ends, steps) = if axes.is_empty() && starts.is_empty() {
                // Read starts/ends/axes/steps from known_i64_values of inputs[1-4].
                let resolve_input = |idx: usize| -> Option<Vec<i64>> {
                    inputs
                        .get(idx)
                        .and_then(|tid| tensor_info.get(tid))
                        .and_then(|info| info.known_i64_values.as_ref())
                        .map(|vals| vals.iter().filter_map(|v| *v).collect())
                };
                let s = resolve_input(1).unwrap_or_default();
                let e = resolve_input(2).unwrap_or_default();
                let a = resolve_input(3).unwrap_or_else(|| (0..s.len() as i64).collect());
                let st = resolve_input(4).unwrap_or_else(|| vec![1; s.len()]);
                if s.is_empty() || e.is_empty() {
                    return Ok(None);
                }
                (a, s, e, st)
            } else {
                (axes.clone(), starts.clone(), ends.clone(), steps.clone())
            };

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
                if axis_size > 0 {
                    (axis_size + start).max(0) as u32
                } else {
                    0
                }
            } else if axis_size > 0 {
                start.min(axis_size) as u32
            } else {
                start as u32
            };
            let norm_end = if end < 0 {
                if axis_size > 0 {
                    (axis_size + end).max(0) as u32
                } else {
                    0
                }
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
                    kernel_h: kh as u32,
                    kernel_w: kw as u32,
                    stride_h: sh as u32,
                    stride_w: sw as u32,
                    pad_h: ph as u32,
                    pad_w: pw as u32,
                    dilation_h: dh as u32,
                    dilation_w: dw as u32,
                    group: *group as u32,
                    input_h: ih,
                    input_w: iw,
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
                    kernel_h: kh as u32,
                    kernel_w: kw as u32,
                    stride_h: sh as u32,
                    stride_w: sw as u32,
                    pad_h: ph as u32,
                    pad_w: pw as u32,
                    dilation_h: dh as u32,
                    dilation_w: dw as u32,
                    group: *group as u32,
                    output_pad_h: oph as u32,
                    output_pad_w: opw as u32,
                    input_h: ih,
                    input_w: iw,
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
                    kernel_h: kh as u32,
                    kernel_w: kw as u32,
                    stride_h: sh as u32,
                    stride_w: sw as u32,
                    pad_h: ph as u32,
                    pad_w: pw as u32,
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
                    kernel_h: kh as u32,
                    kernel_w: kw as u32,
                    stride_h: sh as u32,
                    stride_w: sw as u32,
                    pad_h: ph as u32,
                    pad_w: pw as u32,
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
            (
                FloatOp::GlobalAvgPool {
                    channels,
                    spatial_h,
                    spatial_w,
                },
                vec![],
            )
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
                        info.shape.last().and_then(|d| d.as_concrete()).unwrap_or(0) as u32
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
        AiOp::GroupNorm {
            num_groups,
            epsilon,
        } => (
            FloatOp::GroupNorm {
                num_groups: *num_groups,
                epsilon: f32_to_bits(*epsilon),
            },
            vec![],
        ),
        AiOp::LRN {
            alpha,
            beta,
            bias,
            size,
        } => (
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
            (
                FloatOp::Compress {
                    axis: norm_axis as u32,
                },
                vec![],
            )
        }
        AiOp::ReverseSequence {
            batch_axis,
            time_axis,
        } => {
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
        AiOp::KvSlotWrite {
            layer,
            is_key,
            n_kv_heads,
            head_dim,
            layout,
        } => (
            FloatOp::KvWrite {
                layer: *layer as u32,
                n_kv_heads: *n_kv_heads,
                head_dim: *head_dim,
                is_key: *is_key,
                heads_first: matches!(layout, crate::ir::KvLayout::HeadsFirst),
            },
            vec![],
        ),
        AiOp::KvSlotRead {
            layer,
            n_kv_heads,
            head_dim,
            layout,
        } => (
            FloatOp::KvRead {
                layer: *layer as u32,
                n_kv_heads: *n_kv_heads,
                head_dim: *head_dim,
                heads_first: matches!(layout, crate::ir::KvLayout::HeadsFirst),
            },
            vec![],
        ),

        // ── Deep decode fusions (Plan 054) ─────────────────────────────
        AiOp::FusedNormProjection {
            epsilon,
            split_sizes,
            has_residual_add,
        } => {
            // Input layout depends on has_residual_add:
            //   false: [x, norm_weight, proj_weight]
            //   true:  [x, residual, norm_weight, proj_weight]
            let norm_size = concrete_last_dim(inputs.first(), tensor_info).unwrap_or(0) as u32;
            let n_total: u32 = split_sizes.iter().sum::<usize>() as u32;
            // k = norm_size (hidden dim = projection input dim)
            let k = norm_size;
            let eps_bits = f32_to_bits(*epsilon as f32);
            if *has_residual_add {
                (
                    FloatOp::AddNormProjectionGemv {
                        norm_size,
                        epsilon: eps_bits,
                        k,
                        n_total,
                    },
                    vec![],
                )
            } else {
                (
                    FloatOp::NormProjectionGemv {
                        norm_size,
                        epsilon: eps_bits,
                        k,
                        n_total,
                    },
                    vec![],
                )
            }
        }
        AiOp::FusedSwiGluProjection => {
            // Inputs: [gate, up, W_down]
            // k = last_dim(gate) = FFN intermediate dim
            // n = last_dim(W_down) = hidden dim
            let k = concrete_last_dim(inputs.first(), tensor_info).unwrap_or(0) as u32;
            let n = concrete_last_dim(inputs.get(2), tensor_info).unwrap_or(0) as u32;
            (FloatOp::SwiGluProjectionGemv { k, n }, vec![])
        }

        AiOp::Expand => {
            // Expand physically broadcasts data. The second input is a shape tensor
            // whose known_i64_values give us the target shape. If the output shape
            // is already propagated in tensor_info, we can also read it from there.
            //
            // We try: (1) second input's known_i64_values, (2) output tensor shape.
            let target_dims: Option<Vec<u32>> = inputs
                .get(1)
                .and_then(|tid| tensor_info.get(tid))
                .and_then(|info| info.known_i64_values.as_ref())
                .and_then(|vals| {
                    let concrete: Vec<u32> =
                        vals.iter().filter_map(|v| v.map(|x| x as u32)).collect();
                    if concrete.len() == vals.len() {
                        Some(concrete)
                    } else {
                        None
                    }
                });

            let target_dims = match target_dims {
                Some(d) => d,
                None => {
                    // Fall back: try the input data shape (when Expand is identity).
                    let in_shape = inputs
                        .first()
                        .and_then(|tid| tensor_info.get(tid))
                        .map(|info| &info.shape);
                    match in_shape {
                        Some(s) => {
                            let concrete: Vec<u32> = s
                                .iter()
                                .filter_map(|d| d.as_concrete().map(|c| c as u32))
                                .collect();
                            if concrete.len() == s.len() {
                                concrete
                            } else {
                                return Ok(None);
                            }
                        }
                        None => return Ok(None),
                    }
                }
            };

            let ndim = target_dims.len().min(8) as u8;
            let mut target_shape = [0u32; 8];
            for (i, &d) in target_dims.iter().take(8).enumerate() {
                target_shape[i] = d;
            }

            (FloatOp::Expand { ndim, target_shape }, vec![])
        }

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
        _ => (
            pads.first().copied().unwrap_or(0),
            pads.get(1).copied().unwrap_or(0),
        ),
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

pub(super) fn last_dim_expr(
    tid: Option<&TensorId>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> Option<DimExpr> {
    tid.and_then(|t| tensor_info.get(t))
        .and_then(|info| info.shape.last())
        .cloned()
}

pub(super) fn second_last_dim_expr(
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
pub(super) fn dim_recipe(
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

pub(super) fn resolve_or_zero(recipe: &ParamRecipe) -> u64 {
    match recipe {
        ParamRecipe::Concrete(v) => *v,
        _ => 0,
    }
}

pub(crate) fn is_deferred(recipe: &ParamRecipe) -> bool {
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
fn quant_code(tid: Option<&TensorId>, tensor_info: &HashMap<TensorId, TensorInfo>) -> u8 {
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
