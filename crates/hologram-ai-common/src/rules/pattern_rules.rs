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
use crate::ir::AiOp;

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
