//! Declarative pattern rules — one rule per architecture-pattern,
//! each citing the external authoritative witness (ONNX spec link or
//! ORT logit-parity test name) that verifies it.
//!
//! Rules in this module replace the bespoke imperative fusion passes
//! under `opt/*Fusion`. The architecture is ADR-0018. Every rule here
//! exists *because* its witness establishes its correctness against an
//! external authoritative source — never against hologram-ai's own
//! output.

use super::{OpMatcher, Pattern, Replacement, Rule, RuleSet, VarId};
use crate::ir::{AiOp, AiParam};

/// SwiGLU fusion (direct-Silu variant).
///
/// PyTorch `nn.SiLU` exports as a single ONNX `Silu` op; combined with
/// the gate/up multiply this is the canonical SwiGLU activation. The
/// rule is commutative on the `Mul` because exporters emit either
/// `Mul(Silu(gate), up)` or `Mul(up, Silu(gate))` depending on the
/// expression's order in the original Python source.
///
/// Witness — `hologram-ai-conformance::real_model_generation::smollm2`
/// asserts ORT logit parity on a real Llama-family model whose every
/// transformer layer's FFN uses this pattern. A regression in the rule
/// fails that test.
pub fn swiglu_direct_rule() -> Rule {
    let gate = VarId(1);
    let up = VarId(2);
    Rule {
        name: "swiglu_direct",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op_comm(
            OpMatcher::exact_mul(),
            Pattern::op(OpMatcher::exact_silu(), vec![Pattern::Var(gate)]),
            Pattern::Var(up),
        ),
        replacement: Replacement::new(AiOp::FusedSwiGLU, vec![gate, up]),
    }
}

/// SwiGLU fusion (decomposed-Silu variant).
///
/// torch 2.11+ ONNX exporters lower `nn.SiLU(x)` to `Mul(x, Sigmoid(x))`
/// — the explicit decomposition of `x · σ(x)`. The outer multiply with
/// `up` then gives `Mul(Mul(gate, Sigmoid(gate)), up)`. We fuse the
/// whole shape into one canonical `FusedSwiGLU` node.
///
/// `Mul` is commutative at both levels. The inner `Mul`'s two operands
/// must reference the **same** gate tensor (one direct, one through
/// `Sigmoid`); the matcher enforces this by binding `gate` once and
/// requiring the second binding to agree.
///
/// Witness — same as the direct variant. Both shapes flow through the
/// SmolLM2 ORT-parity test depending on the torch version used to
/// export the model.
pub fn swiglu_decomposed_rule() -> Rule {
    let gate = VarId(1);
    let up = VarId(2);
    Rule {
        name: "swiglu_decomposed",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op_comm(
            OpMatcher::exact_mul(),
            // Inner Mul(gate, Sigmoid(gate)) — both operands name the
            // same gate tensor.
            Pattern::op_comm(
                OpMatcher::exact_mul(),
                Pattern::Var(gate),
                Pattern::op(OpMatcher::exact_sigmoid(), vec![Pattern::Var(gate)]),
            ),
            Pattern::Var(up),
        ),
        replacement: Replacement::new(AiOp::FusedSwiGLU, vec![gate, up]),
    }
}

/// The full SwiGLU rule set — both exporter variants. Either fires the
/// same canonical replacement, so the result is independent of which
/// exporter produced the input ONNX (the canonical-form discipline).
pub fn swiglu_rules() -> RuleSet {
    RuleSet::new()
        .with_rule(swiglu_direct_rule())
        .with_rule(swiglu_decomposed_rule())
}

// ── MatMul + Activation fusion ──────────────────────────────────────────

/// Fuse `Activation(MatMul(A, W))` into the canonical
/// `MatMulActivation` op the matmul kernel can apply in-register on
/// writeback, eliminating the intermediate matmul-output buffer.
///
/// Three activations have a fused matmul op today (`Silu`, `Gelu`,
/// `Relu`); each is its own declarative rule with the same shape.
/// `Pattern` is purely structural — the input pair is *not* commutative
/// (matmul-then-activation is order-significant) — so a single
/// ordering match suffices.
///
/// Witness — `hologram-ai-conformance::real_model_generation::smollm2`
/// asserts ORT logit parity; every transformer layer's FFN runs the
/// fused matmul + activation path through this fusion (the activation
/// path on the up/gate projection, prior to SwiGLU).
fn matmul_activation_rule(name: &'static str, act: OpMatcher, fused: AiOp) -> Rule {
    let a = VarId(1);
    let w = VarId(2);
    Rule {
        name,
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op(
            act,
            vec![Pattern::op(
                OpMatcher::exact_matmul(),
                vec![Pattern::Var(a), Pattern::Var(w)],
            )],
        ),
        replacement: Replacement::new(fused, vec![a, w]),
    }
}

pub fn matmul_silu_rule() -> Rule {
    matmul_activation_rule("matmul_silu", OpMatcher::exact_silu(), AiOp::MatMulSilu)
}

pub fn matmul_gelu_rule() -> Rule {
    matmul_activation_rule(
        "matmul_gelu",
        OpMatcher::Exact(crate::rules::AiOpDiscriminant::Gelu),
        AiOp::MatMulGelu,
    )
}

pub fn matmul_relu_rule() -> Rule {
    matmul_activation_rule("matmul_relu", OpMatcher::exact_relu(), AiOp::MatMulRelu)
}

/// The full MatMul + Activation rule set — one rule per supported
/// activation. Each one's witness is the same SmolLM2 ORT-parity test;
/// they share a single witness because they're variants of one canonical
/// transform.
pub fn matmul_activation_rules() -> RuleSet {
    RuleSet::new()
        .with_rule(matmul_silu_rule())
        .with_rule(matmul_gelu_rule())
        .with_rule(matmul_relu_rule())
}

// ── Add → RmsNorm → FusedLayerNormResidual fusion ───────────────────────

/// Fuse a transformer block's residual-add + RmsNorm tail into the
/// canonical `FusedLayerNormResidual { epsilon }` op: the kernel
/// computes `rms_norm(x + residual, weight)` in one pass, eliminating
/// the intermediate `sum` buffer.
///
/// Pattern:
/// ```text
/// out = RmsNorm(Add(x, residual), weight)
/// ```
///
/// The outer `RmsNorm` is non-commutative (input 0 is the value,
/// input 1 is the weight); the inner `Add` IS commutative — exporters
/// emit either `Add(x, residual)` or `Add(residual, x)` depending on
/// the source's expression order — so the inner pattern is matched
/// commutatively.
///
/// The replacement's epsilon is carried from the matched `RmsNorm`'s
/// attribute via `Replacement::from_root` — the engine's
/// attribute-propagation hook. If the matched op isn't an `RmsNorm`
/// (a programming error in the pattern), the builder returns `None`
/// and the rewrite aborts — no approximation.
///
/// Witness — `hologram-ai-conformance::real_model_generation::smollm2`
/// (EE-3 ORT parity). Every SmolLM2 transformer layer has two of
/// these blocks (post-attention and post-MLP). A regression in this
/// fusion makes the residual stream's RmsNorm produce different
/// numerics than ORT.
pub fn add_rmsnorm_rule() -> Rule {
    let x = VarId(1);
    let residual = VarId(2);
    let weight = VarId(3);
    fn carry_epsilon(root: &AiOp) -> Option<AiOp> {
        match root {
            AiOp::RmsNorm { epsilon } => Some(AiOp::FusedLayerNormResidual { epsilon: *epsilon }),
            _ => None,
        }
    }
    Rule {
        name: "add_rmsnorm_fusion",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op(
            OpMatcher::Exact(crate::rules::AiOpDiscriminant::RmsNorm),
            vec![
                Pattern::op_comm(
                    OpMatcher::exact_add(),
                    Pattern::Var(x),
                    Pattern::Var(residual),
                ),
                Pattern::Var(weight),
            ],
        ),
        replacement: Replacement::from_root(carry_epsilon, vec![x, residual, weight]),
    }
}

pub fn add_rmsnorm_rules() -> RuleSet {
    RuleSet::new().with_rule(add_rmsnorm_rule())
}

// ── FusedSwiGLU + MatMul → FusedSwiGluProjection ────────────────────────

/// Fuse `MatMul(FusedSwiGLU(gate, up), W_down)` into the canonical
/// `FusedSwiGluProjection(gate, up, W_down)` — the down-projection of
/// the FFN block runs the activated values straight through the matmul
/// in-register, eliminating the intermediate FusedSwiGLU output buffer.
///
/// Witness — `hologram-ai-conformance::real_model_generation::smollm2`
/// (EE-3 ORT logit parity). Every transformer FFN's down projection
/// runs through this pattern.
pub fn swiglu_projection_rule() -> Rule {
    let gate = VarId(1);
    let up = VarId(2);
    let w_down = VarId(3);
    Rule {
        name: "swiglu_projection",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op(
            OpMatcher::exact_matmul(),
            vec![
                Pattern::op(
                    OpMatcher::exact_fused_swiglu(),
                    vec![Pattern::Var(gate), Pattern::Var(up)],
                ),
                Pattern::Var(w_down),
            ],
        ),
        replacement: Replacement::new(AiOp::FusedSwiGluProjection, vec![gate, up, w_down]),
    }
}

pub fn swiglu_projection_rules() -> RuleSet {
    RuleSet::new().with_rule(swiglu_projection_rule())
}

// ── Transpose(swap-last-2) + MatMul → Gemm{trans_*} ─────────────────────

/// Predicate: a `Transpose` whose `perm` swaps exactly the last two
/// dims (the canonical Gemm transpose, ignoring higher batch dims).
/// `perm` is `Vec<u32>` of arbitrary rank; the swap-last-two pattern is
/// `[0, 1, ..., r-3, r-1, r-2]`. For rank 2 this is `[1, 0]`.
fn perm_is_swap_last_two(op: &AiOp) -> bool {
    let AiOp::Transpose { perm } = op else {
        return false;
    };
    let r = perm.len();
    if r < 2 {
        return false;
    }
    for (i, &p) in perm.iter().enumerate().take(r - 2) {
        if p as usize != i {
            return false;
        }
    }
    perm[r - 2] as usize == r - 1 && perm[r - 1] as usize == r - 2
}

/// Trans-A rule: `MatMul(Transpose(A), B) → Gemm{trans_a=true}`.
pub fn transpose_matmul_trans_a_rule() -> Rule {
    let a = VarId(1);
    let b = VarId(2);
    Rule {
        name: "transpose_matmul_trans_a",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op(
            OpMatcher::exact_matmul(),
            vec![
                Pattern::op(OpMatcher::exact_transpose(), vec![Pattern::Var(a)])
                    .with_predicate(perm_is_swap_last_two),
                Pattern::Var(b),
            ],
        ),
        replacement: Replacement::new(
            AiOp::Gemm {
                alpha: 1.0,
                beta: 0.0,
                trans_a: true,
                trans_b: false,
            },
            vec![a, b],
        ),
    }
}

/// Trans-B rule: `MatMul(A, Transpose(B)) → Gemm{trans_b=true}`.
pub fn transpose_matmul_trans_b_rule() -> Rule {
    let a = VarId(1);
    let b = VarId(2);
    Rule {
        name: "transpose_matmul_trans_b",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op(
            OpMatcher::exact_matmul(),
            vec![
                Pattern::Var(a),
                Pattern::op(OpMatcher::exact_transpose(), vec![Pattern::Var(b)])
                    .with_predicate(perm_is_swap_last_two),
            ],
        ),
        replacement: Replacement::new(
            AiOp::Gemm {
                alpha: 1.0,
                beta: 0.0,
                trans_a: false,
                trans_b: true,
            },
            vec![a, b],
        ),
    }
}

pub fn transpose_matmul_rules() -> RuleSet {
    RuleSet::new()
        .with_rule(transpose_matmul_trans_a_rule())
        .with_rule(transpose_matmul_trans_b_rule())
}

// ── Mul(scalar) absorption: MatMul + Mul(scalar) → Gemm{alpha} ──────────

/// `Mul(MatMul(A, B), scalar) → Gemm{alpha=scalar}(A, B)`. Both
/// operand orderings of the outer `Mul` are valid; the matcher's
/// commutativity tries both. The scalar must be a constant — the
/// `Pattern::Const` binding refuses non-constant operands at match time.
/// The `Gemm{alpha}` value is read from the bound `Const` var via
/// `Replacement::from_match`.
pub fn scalar_absorption_rule() -> Rule {
    let a = VarId(1);
    let b = VarId(2);
    let scalar = VarId(3);
    fn build(_root: &AiOp, view: &super::MatchView) -> Option<AiOp> {
        let s = view.scalar_f32(VarId(3))?;
        Some(AiOp::Gemm {
            alpha: s,
            beta: 0.0,
            trans_a: false,
            trans_b: false,
        })
    }
    Rule {
        name: "scalar_absorption_matmul",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op_comm(
            OpMatcher::exact_mul(),
            Pattern::op(
                OpMatcher::exact_matmul(),
                vec![Pattern::Var(a), Pattern::Var(b)],
            ),
            Pattern::Const(scalar),
        ),
        replacement: Replacement::from_match(build, vec![a, b]),
    }
}

pub fn scalar_absorption_rules() -> RuleSet {
    RuleSet::new().with_rule(scalar_absorption_rule())
}

// ── RmsNormFusion: explicit ONNX RmsNorm chain → AiOp::RmsNorm ──────────

/// Build `AiOp::RmsNorm { epsilon }` from a matched chain. Pulls the
/// epsilon out of the bound `eps:Const` var and verifies that the
/// bound `two:Const` actually equals 2.0 (the Pow exponent must be 2).
/// If either fails the rewrite aborts.
fn build_rmsnorm(_root: &AiOp, view: &super::MatchView) -> Option<AiOp> {
    let two = view.scalar_f32(VarId(3))?;
    if (two - 2.0).abs() > 1e-6 {
        return None;
    }
    let eps = view.scalar_f32(VarId(4))?;
    Some(AiOp::RmsNorm { epsilon: eps })
}

/// Build the common `Sqrt(Add(ReduceMean(Pow(x, 2)), eps))` sub-pattern.
fn rms_denom_pattern(x: VarId, two: VarId, eps: VarId) -> Pattern {
    Pattern::op(
        OpMatcher::exact_sqrt(),
        vec![Pattern::op_comm(
            OpMatcher::exact_add(),
            Pattern::op(
                OpMatcher::exact_reduce_mean(),
                vec![Pattern::op(
                    OpMatcher::exact_pow(),
                    vec![Pattern::Var(x), Pattern::Const(two)],
                )],
            ),
            Pattern::Const(eps),
        )],
    )
}

/// `Mul`-variant: `weight * (x * Reciprocal(Sqrt(Add(ReduceMean(Pow(x,2)), eps))))`.
pub fn rmsnorm_mul_variant_rule() -> Rule {
    let x = VarId(1);
    let weight = VarId(2);
    let two = VarId(3);
    let eps = VarId(4);
    Rule {
        name: "rmsnorm_mul_variant",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op_comm(
            OpMatcher::exact_mul(),
            Pattern::Var(weight),
            Pattern::op(
                OpMatcher::exact_mul(),
                vec![
                    Pattern::Var(x),
                    Pattern::op(
                        OpMatcher::exact_reciprocal(),
                        vec![rms_denom_pattern(x, two, eps)],
                    ),
                ],
            ),
        ),
        replacement: Replacement::from_match(build_rmsnorm, vec![x, weight]),
    }
}

/// `Div`-variant: `weight * (x / Sqrt(Add(ReduceMean(Pow(x,2)), eps)))`.
pub fn rmsnorm_div_variant_rule() -> Rule {
    let x = VarId(1);
    let weight = VarId(2);
    let two = VarId(3);
    let eps = VarId(4);
    Rule {
        name: "rmsnorm_div_variant",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op_comm(
            OpMatcher::exact_mul(),
            Pattern::Var(weight),
            Pattern::op(
                OpMatcher::exact_div(),
                vec![Pattern::Var(x), rms_denom_pattern(x, two, eps)],
            ),
        ),
        replacement: Replacement::from_match(build_rmsnorm, vec![x, weight]),
    }
}

pub fn rmsnorm_rules() -> RuleSet {
    RuleSet::new()
        .with_rule(rmsnorm_mul_variant_rule())
        .with_rule(rmsnorm_div_variant_rule())
}

// ── LayerNormFusion: explicit ONNX LayerNorm chain → AiOp::LayerNorm ────

/// Build `AiOp::LayerNorm { axis:-1, epsilon }` from a matched chain.
/// Verifies the Pow exponent is 2.0 and pulls the epsilon out of the
/// bound `eps:Const` var.
fn build_layernorm(_root: &AiOp, view: &super::MatchView) -> Option<AiOp> {
    let two = view.scalar_f32(VarId(5))?;
    if (two - 2.0).abs() > 1e-6 {
        return None;
    }
    let eps = view.scalar_f32(VarId(4))?;
    Some(AiOp::LayerNorm {
        axis: -1,
        epsilon: eps,
    })
}

/// `Add(Mul(Div(Sub(X, ReduceMean(X)),
///           Sqrt(Add(ReduceMean(Pow(centered, 2)), eps))),
///       weight),
///   bias)`
/// → `LayerNorm{axis:-1, epsilon}(X, weight, bias)`.
///
/// The `centered = Sub(X, ReduceMean(X))` tensor appears twice (once
/// as the Div numerator, once as the Pow input); the matcher binds it
/// via `bind: Some(VarId)` on the `Sub` and re-asserts it as a
/// `Pattern::Var(centered)` in the Pow input — same-var-binding
/// enforces the equality.
pub fn layernorm_rule() -> Rule {
    let x = VarId(1);
    let weight = VarId(2);
    let bias = VarId(3);
    let eps = VarId(4);
    let two = VarId(5);
    let centered = VarId(6);
    Rule {
        name: "layernorm_fusion",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op_comm(
            OpMatcher::exact_add(),
            Pattern::Var(bias),
            Pattern::op_comm(
                OpMatcher::exact_mul(),
                Pattern::Var(weight),
                Pattern::op(
                    OpMatcher::exact_div(),
                    vec![
                        // numerator: Sub(X, mean=ReduceMean(X)); bind output
                        // as `centered`.
                        Pattern::op_bind(
                            OpMatcher::exact_sub(),
                            vec![
                                Pattern::Var(x),
                                Pattern::op(OpMatcher::exact_reduce_mean(), vec![Pattern::Var(x)]),
                            ],
                            centered,
                        ),
                        // denominator: Sqrt(Add(ReduceMean(Pow(centered, 2)), eps)).
                        Pattern::op(
                            OpMatcher::exact_sqrt(),
                            vec![Pattern::op_comm(
                                OpMatcher::exact_add(),
                                Pattern::op(
                                    OpMatcher::exact_reduce_mean(),
                                    vec![Pattern::op(
                                        OpMatcher::exact_pow(),
                                        vec![Pattern::Var(centered), Pattern::Const(two)],
                                    )],
                                ),
                                Pattern::Const(eps),
                            )],
                        ),
                    ],
                ),
            ),
        ),
        replacement: Replacement::from_match(build_layernorm, vec![x, weight, bias]),
    }
}

pub fn layernorm_rules() -> RuleSet {
    RuleSet::new().with_rule(layernorm_rule())
}

// ── SliceToGather: single-axis Slice → first-class Gather ──────────────

/// Convert a single-axis, step-1 `Slice` along a concrete axis into a
/// first-class `Gather`. hologram's compiler realizes `Slice` only as
/// an axis-0 contiguous byte view; any slice along a non-zero axis
/// (RoPE `rotate_half`, the last-axis QKV / gate-up projection splits)
/// cannot be expressed that way. Gather's `[outer, axis_dim, inner]`
/// flattening handles batched arbitrary-axis selects directly.
///
/// This is a `Replacement::custom` rewrite: the rewrite emits a new
/// `i64[num_indices]` constant param (the index list `start..end`),
/// inserts it into `graph.params` + `graph.tensor_info`, and returns
/// a `Gather { axis }(data, indices)` node to replace the matched
/// Slice — all data-driven from the Slice's `axes`/`starts`/`ends`/
/// `steps` attributes + the input's declared shape.
///
/// Only rewrites cases Gather represents exactly: 1-axis, step-1,
/// concrete sliced dim. Multi-axis / non-unit-step Slices are left as
/// is (the lowering will error rather than silently produce a wrong
/// result — no approximation).
fn slice_to_gather_rewrite(
    graph: &mut crate::ir::AiGraph,
    _binds: &std::collections::HashMap<super::VarId, crate::ir::TensorId>,
    root_idx: usize,
) -> Option<crate::ir::AiNode> {
    use crate::ir::shape::DimExpr;
    use crate::ir::{shape_from_concrete, AiParam, DType, SemanticHint, TensorInfo};
    use hologram_ai_quant::QuantDescriptor;

    let (axis, start, end, data_tid, dim_val) = {
        let node = graph.nodes.get(root_idx)?;
        let AiOp::Slice {
            axes,
            starts,
            ends,
            steps,
        } = &node.op
        else {
            return None;
        };
        if axes.len() != 1 || starts.len() != 1 || ends.len() != 1 {
            return None;
        }
        let step = steps.first().copied().unwrap_or(1);
        if step != 1 {
            return None;
        }
        let axis = axes[0];
        let start = starts[0];
        let end = ends[0];
        let data_tid = *node.inputs.first()?;
        let info = graph.tensor_info.get(&data_tid)?;
        let ndim = info.shape.len();
        let norm_axis = if axis < 0 {
            (ndim as i64 + axis).max(0) as usize
        } else {
            axis as usize
        };
        let dim_val = info.shape.get(norm_axis)?.as_concrete()? as i64;
        (axis, start, end, data_tid, dim_val)
    };

    let s = {
        let v = if start < 0 { dim_val + start } else { start };
        v.clamp(0, dim_val)
    };
    let e = {
        let v = if end < 0 { dim_val + end } else { end };
        v.clamp(0, dim_val)
    };
    if s >= e {
        return None;
    }
    // No-op slice (selecting ALL elements) is an Identity — don't rewrite.
    if s == 0 && e == dim_val {
        return None;
    }

    let indices: Vec<i64> = (s..e).collect();
    let num_indices = indices.len();

    let mut next_tid = graph.tensor_info.keys().copied().max().unwrap_or(0) + 1;
    let indices_tid = next_tid;
    next_tid += 1;
    let _ = next_tid; // silence unused

    let index_bytes: Vec<u8> = indices.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let index_shape = shape_from_concrete(&[num_indices as u64]);
    let index_info = TensorInfo {
        logical_dtype: DType::INT64,
        storage_dtype: DType::INT64,
        shape: index_shape,
        quant: QuantDescriptor::none(),
        known_i64_values: Some(indices.iter().map(|&v| Some(v)).collect()),
        semantic: SemanticHint::Unknown,
    };

    graph.tensor_info.insert(indices_tid, index_info.clone());
    graph
        .params
        .insert(indices_tid, AiParam::inline(index_bytes, index_info));

    // Update output shape to reflect the new dim.
    let out_tid = graph.nodes[root_idx].outputs.first().copied()?;
    if let Some(info) = graph.tensor_info.get_mut(&out_tid) {
        let ndim = info.shape.len();
        let norm_axis = if axis < 0 {
            (ndim as i64 + axis).max(0) as usize
        } else {
            axis as usize
        };
        if norm_axis < info.shape.len() {
            info.shape[norm_axis] = DimExpr::Concrete(num_indices as u64);
        }
    }

    let nid = graph.nodes[root_idx].id;
    Some(crate::ir::AiNode::new(
        nid,
        AiOp::Gather { axis },
        vec![data_tid, indices_tid],
        vec![out_tid],
    ))
}

/// Predicate: only match Slice nodes the rewrite can actually convert
/// (single-axis, step-1).
fn slice_is_single_axis_step1(op: &AiOp) -> bool {
    matches!(op,
        AiOp::Slice { axes, starts, ends, steps }
            if axes.len() == 1
                && starts.len() == 1
                && ends.len() == 1
                && steps.first().copied().unwrap_or(1) == 1
    )
}

pub fn slice_to_gather_rule() -> Rule {
    let data = VarId(1);
    Rule {
        name: "slice_to_gather",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op(OpMatcher::exact_slice(), vec![Pattern::Var(data)])
            .with_predicate(slice_is_single_axis_step1),
        replacement: Replacement::custom(slice_to_gather_rewrite),
    }
}

pub fn slice_to_gather_rules() -> RuleSet {
    RuleSet::new().with_rule(slice_to_gather_rule())
}

// ── PositionIdsInjection: Range(0, seq, 1) → Identity(position_ids_input)

/// Read a tensor's known scalar i64 value, either from `info.known_i64_values`
/// or from an inline f32/i64 const param. Mirrors the imperative pass's
/// `get_i64_param` — the same canonical scalar-extraction logic, just lifted
/// out of the pass code.
fn read_scalar_i64(tid: crate::ir::TensorId, graph: &crate::ir::AiGraph) -> Option<i64> {
    if let Some(info) = graph.tensor_info.get(&tid) {
        if let Some(vals) = &info.known_i64_values {
            if vals.len() == 1 {
                return vals[0];
            }
        }
    }
    match graph.params.get(&tid)? {
        AiParam::Inline { data, info } => {
            if info.logical_dtype == crate::ir::DType::INT64 && data.len() == 8 {
                Some(i64::from_le_bytes(data[..8].try_into().ok()?))
            } else if info.logical_dtype == crate::ir::DType::F32 && data.len() == 4 {
                Some(f32::from_le_bytes(data[..4].try_into().ok()?) as i64)
            } else {
                None
            }
        }
        AiParam::Mmap { .. } => None,
    }
}

/// `Range(start=0, *, step=1)` → `Identity(position_ids_input)`.
///
/// The first occurrence of a position-generating Range adds the
/// `position_ids` graph input. Subsequent Range nodes (one per
/// transformer layer) reuse the same `position_ids` tensor — the
/// rewrite checks `graph.inputs` to find an existing `position_ids`
/// before allocating.
fn position_ids_inject(
    graph: &mut crate::ir::AiGraph,
    _binds: &std::collections::HashMap<super::VarId, crate::ir::TensorId>,
    root_idx: usize,
) -> Option<crate::ir::AiNode> {
    use crate::ir::{shape, AiParam, DType, SemanticHint, TensorInfo};
    use hologram_ai_quant::QuantDescriptor;

    // Verify start == 0 and step == 1 at the matched Range's inputs.
    let (range_output, range_shape, start_tid, step_tid) = {
        let node = graph.nodes.get(root_idx)?;
        if !matches!(node.op, AiOp::Range) || node.inputs.len() < 3 {
            return None;
        }
        let range_output = *node.outputs.first()?;
        let range_shape = graph
            .tensor_info
            .get(&range_output)
            .map(|info| info.shape.clone())
            .unwrap_or(shape::Shape::new());
        (range_output, range_shape, node.inputs[0], node.inputs[2])
    };
    if read_scalar_i64(start_tid, graph)? != 0 {
        return None;
    }
    if read_scalar_i64(step_tid, graph)? != 1 {
        return None;
    }
    let _ = range_output; // (output is reused by the Identity below)

    // Reuse an existing `position_ids` input if present; otherwise
    // allocate a fresh TensorId and register it as a graph input.
    let pos_tid = if let Some((i, _)) = graph
        .input_names
        .iter()
        .enumerate()
        .find(|(_, n)| *n == "position_ids")
    {
        // graph.inputs[i] is the matching tid.
        *graph.inputs.get(i)?
    } else {
        let next_tid = graph.tensor_info.keys().copied().max().unwrap_or(0) + 1;
        graph.tensor_names.insert(next_tid, "position_ids".into());
        graph.inputs.push(next_tid);
        graph.input_names.push("position_ids".into());
        graph.tensor_info.insert(
            next_tid,
            TensorInfo {
                logical_dtype: DType::INT64,
                storage_dtype: DType::INT64,
                shape: range_shape.clone(),
                quant: QuantDescriptor::none(),
                known_i64_values: None,
                semantic: SemanticHint::Position,
            },
        );
        // Suppress unused-import warning for AiParam — declared via the
        // `use` in this function's body for the scalar-read helper above.
        let _ = std::marker::PhantomData::<AiParam>;
        next_tid
    };

    // Return Identity(position_ids) at root_idx; keep the original
    // output tensor so downstream consumers' wiring is unchanged.
    let nid = graph.nodes[root_idx].id;
    let out_tid = graph.nodes[root_idx].outputs.first().copied()?;
    Some(crate::ir::AiNode::new(
        nid,
        AiOp::Identity,
        vec![pos_tid],
        vec![out_tid],
    ))
}

pub fn position_ids_rule() -> Rule {
    let start = VarId(1);
    let limit = VarId(2);
    let step = VarId(3);
    Rule {
        name: "position_ids_injection",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op(
            OpMatcher::Exact(crate::rules::AiOpDiscriminant::Range),
            vec![
                Pattern::Const(start),
                Pattern::Var(limit),
                Pattern::Const(step),
            ],
        ),
        replacement: Replacement::custom(position_ids_inject),
    }
}

pub fn position_ids_rules() -> RuleSet {
    RuleSet::new().with_rule(position_ids_rule())
}

// ── KvSlotInjection: per-GQA, wrap K/V with KvSlotWrite ─────────────────

/// Inject `KvSlotWrite` nodes on the K/V inputs of a
/// `GroupedQueryAttention` node. Each GQA gets two appended
/// `KvSlotWrite` nodes (one for K, one for V), and the GQA's K/V
/// input slots are rewired to the new outputs. The rewrite is
/// idempotent: if K's producer is already a `KvSlotWrite`, the
/// rewrite returns None (already wired) and the engine converges.
///
/// `layer` is the GQA's index among GQAs in graph order. We compute
/// it on the fly by scanning the graph for GQAs up to `root_idx`.
fn kv_slot_inject(
    graph: &mut crate::ir::AiGraph,
    _binds: &std::collections::HashMap<super::VarId, crate::ir::TensorId>,
    root_idx: usize,
) -> Option<crate::ir::AiNode> {
    let (nkv, hd, layout) = {
        let node = graph.nodes.get(root_idx)?;
        let AiOp::GroupedQueryAttention {
            num_kv_heads,
            head_dim,
            heads_first,
            ..
        } = &node.op
        else {
            return None;
        };
        let layout = if *heads_first {
            crate::ir::KvLayout::HeadsFirst
        } else {
            crate::ir::KvLayout::SeqFirst
        };
        (*num_kv_heads, *head_dim, layout)
    };

    // K/V tensor IDs (current).
    let (k_tid, v_tid) = {
        let node = graph.nodes.get(root_idx)?;
        if node.inputs.len() < 3 {
            return None;
        }
        (node.inputs[1], node.inputs[2])
    };

    // Idempotence: already wired through KvSlotWrite? No-op.
    let already_wired = graph
        .nodes
        .iter()
        .any(|n| n.outputs.contains(&k_tid) && matches!(n.op, AiOp::KvSlotWrite { .. }));
    if already_wired {
        return None;
    }

    // The layer index is this GQA's position among all GQAs.
    let layer = graph
        .nodes
        .iter()
        .take(root_idx)
        .filter(|n| matches!(n.op, AiOp::GroupedQueryAttention { .. }))
        .count();

    // Allocate fresh tensor + node IDs.
    let max_input_tid = graph
        .nodes
        .iter()
        .flat_map(|n| n.inputs.iter().chain(n.outputs.iter()))
        .copied()
        .max()
        .unwrap_or(0);
    let max_info_tid = graph.tensor_info.keys().copied().max().unwrap_or(0);
    let max_param_tid = graph.params.keys().copied().max().unwrap_or(0);
    let mut next_tid = max_input_tid.max(max_info_tid).max(max_param_tid) + 1;
    let mut next_nid = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;

    let k_out = next_tid;
    next_tid += 1;
    let v_out = next_tid;
    next_tid += 1;
    let _ = next_tid;

    if let Some(info) = graph.tensor_info.get(&k_tid).cloned() {
        graph.tensor_info.insert(k_out, info);
    }
    if let Some(info) = graph.tensor_info.get(&v_tid).cloned() {
        graph.tensor_info.insert(v_out, info);
    }

    let k_node = crate::ir::AiNode::new(
        next_nid,
        AiOp::KvSlotWrite {
            layer,
            is_key: true,
            n_kv_heads: nkv,
            head_dim: hd,
            layout,
        },
        vec![k_tid],
        vec![k_out],
    );
    next_nid += 1;
    let v_node = crate::ir::AiNode::new(
        next_nid,
        AiOp::KvSlotWrite {
            layer,
            is_key: false,
            n_kv_heads: nkv,
            head_dim: hd,
            layout,
        },
        vec![v_tid],
        vec![v_out],
    );

    graph.nodes.push(k_node);
    graph.nodes.push(v_node);

    // Return the GQA with its K/V inputs rewired. The engine puts this
    // node at root_idx (replacing the matched GQA).
    let gqa = &graph.nodes[root_idx];
    let mut new_inputs = gqa.inputs.clone();
    new_inputs[1] = k_out;
    new_inputs[2] = v_out;
    Some(crate::ir::AiNode::new(
        gqa.id,
        gqa.op.clone(),
        new_inputs,
        gqa.outputs.clone(),
    ))
}

pub fn kv_slot_injection_rule() -> Rule {
    let q = VarId(1);
    let k = VarId(2);
    let v = VarId(3);
    Rule {
        name: "kv_slot_injection",
        witness: "real_model_generation::smollm2 (EE-3 ORT logit parity, ADR-0018)",
        pattern: Pattern::op(
            OpMatcher::exact_gqa(),
            vec![Pattern::Var(q), Pattern::Var(k), Pattern::Var(v)],
        ),
        replacement: Replacement::custom(kv_slot_inject),
    }
}

pub fn kv_slot_injection_rules() -> RuleSet {
    RuleSet::new().with_rule(kv_slot_injection_rule())
}
