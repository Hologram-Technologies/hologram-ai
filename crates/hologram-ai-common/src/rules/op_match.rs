//! `OpMatcher` — compare an `AiOp` value against a declarative selector
//! (discriminant-only, ignoring per-op fields), so a pattern can name
//! "any `Relu`" or "any `MatMul`" without committing to specific
//! attribute values. Per-op attributes that *do* matter (a `Concat`'s
//! `axis`, a `Softmax`'s `axis`, a `RotaryEmbedding`'s `base`, …) are
//! declared as their own constraints on bound variables — kept out of
//! the discriminant matcher so the pattern shape stays small.

use crate::ir::AiOp;

/// Selector over the closed `AiOp` catalog used by [`super::Pattern::Op`].
///
/// `Exact` matches a specific discriminant ignoring fields; `AnyOf`
/// matches any of a small set. The matcher is intentionally narrow —
/// per-op field constraints belong on bound variables, not on the
/// discriminant test.
#[derive(Debug, Clone)]
pub enum OpMatcher {
    Exact(AiOpDiscriminant),
    AnyOf(Vec<AiOpDiscriminant>),
}

impl OpMatcher {
    pub fn matches(&self, op: &AiOp) -> bool {
        let d = AiOpDiscriminant::of(op);
        match self {
            OpMatcher::Exact(want) => d == *want,
            OpMatcher::AnyOf(set) => set.contains(&d),
        }
    }

    pub fn exact_relu() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Relu)
    }
    pub fn exact_sigmoid() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Sigmoid)
    }
    pub fn exact_silu() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Silu)
    }
    pub fn exact_tanh() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Tanh)
    }
    pub fn exact_matmul() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::MatMul)
    }
    pub fn exact_add() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Add)
    }
    pub fn exact_mul() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Mul)
    }
    pub fn exact_softmax() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Softmax)
    }
    pub fn exact_transpose() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Transpose)
    }
    pub fn exact_reshape() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Reshape)
    }
    pub fn exact_fused_swiglu() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::FusedSwiGLU)
    }
    pub fn exact_div() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Div)
    }
    pub fn exact_sub() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Sub)
    }
    pub fn exact_pow() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Pow)
    }
    pub fn exact_sqrt() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Sqrt)
    }
    pub fn exact_reciprocal() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Reciprocal)
    }
    pub fn exact_reduce_mean() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::ReduceMean)
    }
    pub fn exact_concat() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Concat)
    }
    pub fn exact_slice() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Slice)
    }
    pub fn exact_gather() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::Gather)
    }
    pub fn exact_gqa() -> Self {
        OpMatcher::Exact(AiOpDiscriminant::GroupedQueryAttention)
    }
}

/// Discriminant identity of an `AiOp` value — strips per-variant
/// fields so the matcher compares shape, not attribute values.
///
/// Only the variants used by the current rule set are enumerated; the
/// `Other` arm absorbs the rest so `AiOpDiscriminant::of(op)` is total.
/// Add a variant here when a new rule needs to discriminate on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AiOpDiscriminant {
    Relu,
    Gelu,
    Silu,
    Tanh,
    Sigmoid,
    Softmax,
    LogSoftmax,
    LayerNorm,
    RmsNorm,
    MatMul,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Sqrt,
    Reciprocal,
    ReduceMean,
    ReduceSum,
    Transpose,
    Reshape,
    Concat,
    Slice,
    Gather,
    Cast,
    Dequantize,
    Range,
    Equal,
    Where,
    FusedSwiGLU,
    GroupedQueryAttention,
    Identity,
    Other,
}

impl AiOpDiscriminant {
    pub fn of(op: &AiOp) -> Self {
        use AiOp::*;
        match op {
            Relu => Self::Relu,
            Gelu | GeluApprox => Self::Gelu,
            Silu => Self::Silu,
            Tanh => Self::Tanh,
            Sigmoid => Self::Sigmoid,
            Softmax { .. } => Self::Softmax,
            LogSoftmax { .. } => Self::LogSoftmax,
            LayerNorm { .. } => Self::LayerNorm,
            RmsNorm { .. } => Self::RmsNorm,
            MatMul => Self::MatMul,
            Add => Self::Add,
            Sub => Self::Sub,
            Mul => Self::Mul,
            Div => Self::Div,
            Transpose { .. } => Self::Transpose,
            Reshape { .. } => Self::Reshape,
            Concat { .. } => Self::Concat,
            Slice { .. } => Self::Slice,
            Gather { .. } => Self::Gather,
            Cast { .. } => Self::Cast,
            Dequantize { .. } => Self::Dequantize,
            Pow => Self::Pow,
            Sqrt => Self::Sqrt,
            Reciprocal => Self::Reciprocal,
            ReduceMean { .. } => Self::ReduceMean,
            ReduceSum { .. } => Self::ReduceSum,
            Range => Self::Range,
            Equal => Self::Equal,
            Where => Self::Where,
            FusedSwiGLU => Self::FusedSwiGLU,
            GroupedQueryAttention { .. } => Self::GroupedQueryAttention,
            Identity => Self::Identity,
            _ => Self::Other,
        }
    }
}
