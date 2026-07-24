# ADR-0015: Symbolic Shapes and Dimensions

- Status: Accepted
- Date: 2026-03-07
- Owners: Architecture

---

## Context

The `hologram-ai` IR represents tensor shapes as `SmallVec<[Dim; 6]>` where `Dim` is a
flat enum with three variants: `Concrete(u64)`, `Symbolic(String)`, and `Dynamic`. This
representation has several limitations:

1. **No dimension algebra.** Common ML shape relationships (e.g., `head_dim = hidden_dim /
   num_heads`, padding via `ceil(seq_len / block_size)`) cannot be expressed. Dimensions
   are either known constants or opaque strings.

2. **No variable registry.** Symbolic dimension names are bare strings scattered across
   `TensorInfo` entries. There is no central place to track bounds (e.g.,
   `1 <= seq_len <= 2048`), resolve variables, or detect naming conflicts.

3. **No shape inference with symbolic dims.** The `ShapePropagation` pass cannot propagate
   shapes through ops when inputs have symbolic dimensions. It can only forward-copy known
   shapes and leave unknowns as `Dynamic`.

4. **No constraint validation.** Shape compatibility (e.g., MatMul inner dimensions must
   match, Reshape product equality) is not checked until lowering, where errors are
   harder to diagnose.

5. **Single lowering strategy.** The MVP approach is to concretize all symbolic dims to
   their upper bound and rebuild the `hologram::Graph` when dimensions change. There is
   no support for bucketed compilation or shape profiles.

---

## Decision

### 1. Replace `Dim` with `DimExpr` — a symbolic dimension expression type

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct DimVarId(u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimExpr {
    Concrete(u64),
    Var(DimVarId),
    Add(Box<DimExpr>, Box<DimExpr>),
    Sub(Box<DimExpr>, Box<DimExpr>),
    Mul(Box<DimExpr>, Box<DimExpr>),
    Div(Box<DimExpr>, Box<DimExpr>),
    Mod(Box<DimExpr>, Box<DimExpr>),
    CeilDiv(Box<DimExpr>, Box<DimExpr>),
    Max(Box<DimExpr>, Box<DimExpr>),
    Min(Box<DimExpr>, Box<DimExpr>),
    Dynamic,
}
```

`DimExpr` covers the operations needed for ML shape rules: arithmetic for Reshape product
constraints, `CeilDiv` for padding/tiling, `Max` for broadcast, `Min` for clamp/bound.
This is deliberately not a general computer algebra system.

`DimVarId(u32)` is an interned index into a `DimVarTable`, following the same pattern as
`NodeId` and `TensorId` elsewhere in the IR.

### 2. Add `DimVarTable` — a dimension variable registry with bounds

```rust
pub struct DimVarTable {
    entries: Vec<DimVarEntry>,
    name_to_id: HashMap<String, DimVarId>,
}

pub struct DimVarEntry {
    pub name: String,
    pub lower: Option<u64>,
    pub upper: Option<u64>,
    pub fixed: Option<u64>,
    pub source: DimVarSource,
}
```

`DimVarTable` lives on `AiGraph` as a first-class field. All importers intern variable
names into this table. Canonical names (`batch`, `seq_len`, `vocab_size`, `hidden_dim`,
`num_heads`, `num_kv_heads`, `head_dim`, `ffn_dim`) are standardized; importers normalize
format-specific names.

Bounds are tightened via intersection when the same variable is encountered from multiple
sources (e.g., GGUF metadata provides an upper bound, user config overrides it).

### 3. Add `ConstraintStore` — shape constraint collection and validation

```rust
pub enum ShapeConstraint {
    DimEqual { node_id: NodeId, lhs: DimExpr, rhs: DimExpr, context: String },
    BroadcastCompatible(DimExpr, DimExpr),
    ProductEqual { node_id: NodeId, input_dims: Vec<DimExpr>, output_dims: Vec<DimExpr> },
    Positive { node_id: NodeId, dim: DimExpr, context: String },
    Divisible { node_id: NodeId, dividend: DimExpr, divisor: DimExpr, context: String },
}
```

`ConstraintStore` lives on `AiGraph`. Constraints are collected during shape propagation
and validated in two phases:

- **Eager:** Constraints where both sides are concrete are validated immediately. Mismatches
  are errors.
- **Lazy:** Constraints involving symbolic expressions are deferred and validated when
  variables are concretized (at lowering time or via explicit `concretize_to_upper()`).

### 4. Implement symbolic shape inference per `AiOp`

The `ShapePropagation` pass calls `infer_shapes(op, inputs, dim_vars)` for each node in
topological order. Each op has inference rules that produce output shapes and constraints:

- **MatMul:** inner-dim equality constraint
- **Broadcast ops:** `Max(a, b)` with broadcast-compatibility constraint
- **Reshape:** product equality, `-1` inference via `Div`
- **Concat:** `Add` along concat axis
- **Attention ops:** decompose via `num_heads`, `head_dim`

### 5. Introduce lowering shape strategies

```rust
pub enum ShapeStrategy {
    FixToMax,
    Bucketed(BucketConfig),
    Profiles(Vec<ShapeProfile>),
    PaddedMax,
}
```

All strategies concretize symbolic dims *before* calling `lower()`. The `hologram::Graph`
always receives fully concrete dimensions. Strategies differ in how many concrete variants
are produced and how the runtime selects between them.

`FixToMax` is the current MVP behavior, preserved as the default. Other strategies are
additive and introduced in later phases.

### 6. `Shape` type alias changes

```rust
// Before:
pub type Shape = SmallVec<[Dim; 6]>;

// After:
pub type Shape = SmallVec<[DimExpr; 4]>;
```

Inline capacity reduced from 6 to 4 because `DimExpr` is larger than the old `Dim`. Most
ML tensors have 2–4 dimensions; 5+ dimensions (rare) will heap-allocate.

---

## Consequences

**Positive:**

- Shape relationships (e.g., `head_dim = hidden_dim / num_heads`) are expressible and
  propagated through the graph
- Shape errors are caught at compile time with clear diagnostics (which op, which tensors,
  what constraint violated)
- Variable bounds enable early validation (e.g., reject `seq_len = 0` before lowering)
- Bucketed and profiled compilation reduce recompilation for common shape changes
- Foundation for true dynamic shape dispatch in the future

**Negative:**

- Type migration from `Dim` to `DimExpr` touches every file that constructs or matches
  on shapes (importers, passes, lowering, tests)
- `DimExpr` is larger per-dimension than the old `Dim`, increasing `TensorInfo` size
- Expression simplification is deliberately limited — no algebraic canonicalization, no
  equation solving beyond constant folding

**Neutral:**

- `Dynamic` variant is preserved for truly data-dependent dimensions (output of `NonZero`,
  `Compress`, etc.)
- MVP behavior (FixToMax) is unchanged — the new system is strictly additive
- `hologram::Graph` API is not affected — concretization always happens before lowering

---

## Alternatives Considered

**Keep `Dim::Symbolic(String)` and add bounds separately**
Rejected. Without expression algebra, the system cannot express derived dimensions
(`head_dim = hidden_dim / num_heads`), cannot perform symbolic shape inference (MatMul
inner-dim matching with symbolic dims), and cannot validate Reshape product equality.
The bare string approach would require a separate, parallel system for shape validation
that duplicates most of the work `DimExpr` handles naturally.

**Full symbolic execution / computer algebra system**
Rejected. ML shape inference needs only a small set of operations (add, mul, div, ceildiv,
max, min). General-purpose CAS features (polynomial canonicalization, equation solving,
substitution chains) add complexity without benefit. The three-tier resolution strategy
(immediate error, immediate fix, deferred constraint) handles the practical cases.

**Intern `DimExpr` via hash-consing**
Rejected for now. Expression trees in ML shape inference are shallow (depth 1-3) and there
are few of them per graph (hundreds, not millions). Structural equality via derived `Eq`
is sufficient. Interning adds thread-safety and lifecycle complexity. Can be added behind
the same API later if profiling shows a need.

**Use `Arc<DimExpr>` for shared sub-expressions**
Rejected. `Box<DimExpr>` is simpler and expressions are not shared across threads.
Shallow trees make clone cheap. `Arc` would add atomic reference-counting overhead
without benefit.
