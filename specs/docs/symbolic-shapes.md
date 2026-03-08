# hologram-ai: Symbolic Shapes and Dimensions

---

## 1. Overview

The symbolic shape system replaces the flat `Dim` enum with a proper expression type
(`DimExpr`), a variable registry with bounds (`DimVarTable`), per-op shape inference rules,
a constraint collection and validation system, and multiple lowering strategies for
converting symbolic shapes to the concrete dimensions that `hologram::Graph` requires.

See [ADR-0015](../../adrs/0015-hologram-ai-symbolic-shapes.md) for the decision record.

---

## 2. Core Types

### `DimVarId` — Interned Variable Identifier

```rust
/// Compact interned identifier for a dimension variable.
/// Points into the `DimVarTable` for name and bounds resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct DimVarId(u32);
```

### `DimExpr` — Symbolic Dimension Expression

```rust
/// A dimension expression supporting the algebra needed for ML shape inference.
///
/// Deliberately limited to the operations needed for ML shape rules:
/// arithmetic for Reshape product constraints, CeilDiv for padding/tiling,
/// Max for broadcast, Min for clamp/bound. Not a general CAS.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimExpr {
    /// A known constant dimension.
    Concrete(u64),

    /// A symbolic variable (batch_size, seq_len, etc.).
    /// Resolved via DimVarTable for name and bounds.
    Var(DimVarId),

    /// Arithmetic operations.
    Add(Box<DimExpr>, Box<DimExpr>),
    Sub(Box<DimExpr>, Box<DimExpr>),
    Mul(Box<DimExpr>, Box<DimExpr>),
    Div(Box<DimExpr>, Box<DimExpr>),
    Mod(Box<DimExpr>, Box<DimExpr>),

    /// Ceiling division: ceil(a / b) = (a + b - 1) / b.
    /// Common in ML for block/tile sizing, padding calculations.
    CeilDiv(Box<DimExpr>, Box<DimExpr>),

    /// Maximum of two expressions. Used in broadcast rules.
    Max(Box<DimExpr>, Box<DimExpr>),

    /// Minimum of two expressions. Used in clamp/bound calculations.
    Min(Box<DimExpr>, Box<DimExpr>),

    /// Truly unknown dimension — cannot be expressed symbolically.
    /// Escape hatch for data-dependent dimensions (NonZero, Compress, etc.).
    Dynamic,
}
```

### `Shape` — Updated Type Alias

```rust
/// Compact tensor shape. Up to 4 dims inline before heap allocation.
/// Reduced from SmallVec<[Dim; 6]> because DimExpr is larger per element.
/// Covers the vast majority of ML tensors (scalars through 4D batched tensors).
pub type Shape = SmallVec<[DimExpr; 4]>;
```

### `DimExpr` Key Methods

```rust
impl DimExpr {
    /// Convenience constructors.
    pub fn concrete(n: u64) -> Self;
    pub fn var(id: DimVarId) -> Self;

    /// Returns the concrete value if this is a Concrete variant.
    pub fn as_concrete(&self) -> Option<u64>;

    /// Returns true if this expression contains no Var or Dynamic nodes.
    pub fn is_concrete(&self) -> bool;

    /// Attempt to evaluate to a concrete u64.
    /// Returns None if any Var or Dynamic is encountered.
    /// Division by zero returns None.
    pub fn evaluate(&self) -> Option<u64>;

    /// Substitute all occurrences of `var` with `value`.
    pub fn substitute(&self, var: DimVarId, value: &DimExpr) -> DimExpr;

    /// Simplify constant sub-expressions.
    /// E.g., Add(Concrete(3), Concrete(5)) => Concrete(8).
    /// Does NOT perform algebraic simplification (no x+0 => x, no reordering).
    pub fn simplify(&self) -> DimExpr;

    /// Collect all DimVarIds referenced in this expression.
    pub fn free_vars(&self) -> HashSet<DimVarId>;
}
```

---

## 3. DimVarTable — Variable Registry

```rust
/// Where a dimension variable was introduced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimVarSource {
    /// Imported from ONNX dim_param or GGUF metadata.
    Import,
    /// Inferred by shape propagation.
    Inferred,
    /// Specified by user configuration (e.g., --max-seq-len).
    UserConfig,
}

/// A named dimension variable with optional bounds.
#[derive(Debug, Clone)]
pub struct DimVarEntry {
    /// Human-readable name (e.g., "batch", "seq_len").
    pub name: String,
    /// Inclusive lower bound. None means unbounded below (treated as 0).
    pub lower: Option<u64>,
    /// Inclusive upper bound. None means unbounded above.
    pub upper: Option<u64>,
    /// If Some, this variable is fixed to a concrete value.
    pub fixed: Option<u64>,
    /// Where this variable was defined.
    pub source: DimVarSource,
}

/// Registry of all dimension variables in an AiGraph.
/// Variables are interned: each unique name maps to exactly one DimVarId.
#[derive(Debug, Clone, Default)]
pub struct DimVarTable {
    entries: Vec<DimVarEntry>,
    name_to_id: HashMap<String, DimVarId>,
}
```

### DimVarTable Operations

```rust
impl DimVarTable {
    /// Intern a variable name, returning its ID.
    /// If the name already exists, returns the existing ID.
    pub fn intern(&mut self, name: &str) -> DimVarId;

    /// Intern with bounds. If the variable exists, tightens bounds
    /// (max of lowers, min of uppers — intersection semantics).
    pub fn intern_with_bounds(
        &mut self,
        name: &str,
        lower: Option<u64>,
        upper: Option<u64>,
        source: DimVarSource,
    ) -> DimVarId;

    /// Fix a variable to a concrete value. Validates against bounds.
    pub fn fix(&mut self, id: DimVarId, value: u64) -> Result<(), ShapeError>;

    /// Look up a variable by ID.
    pub fn get(&self, id: DimVarId) -> &DimVarEntry;

    /// Look up a variable by name.
    pub fn lookup(&self, name: &str) -> Option<DimVarId>;

    /// Produce a substitution map for all fixed variables.
    pub fn fixed_substitutions(&self) -> HashMap<DimVarId, DimExpr>;

    /// Concretize all unfixed variables to their upper bound (MVP lowering).
    /// Fails if any variable has no upper bound.
    pub fn concretize_to_upper(&mut self) -> Result<(), ShapeError>;

    /// Iterate all variables.
    pub fn iter(&self) -> impl Iterator<Item = (DimVarId, &DimVarEntry)>;
}
```

### Canonical Variable Names

Importers normalize format-specific names to these canonical names:

```rust
pub mod canonical_vars {
    pub const BATCH: &str = "batch";
    pub const SEQ_LEN: &str = "seq_len";
    pub const VOCAB_SIZE: &str = "vocab_size";
    pub const HIDDEN_DIM: &str = "hidden_dim";
    pub const NUM_HEADS: &str = "num_heads";
    pub const NUM_KV_HEADS: &str = "num_kv_heads";
    pub const HEAD_DIM: &str = "head_dim";
    pub const FFN_DIM: &str = "ffn_dim";
}
```

**ONNX normalization:** `dim_param` values like `"batch_size"`, `"N"`, `"batch"` are mapped
to canonical names by the importer. Unknown names pass through as-is.

**GGUF normalization:** Architecture metadata provides concrete values for most dimensions.
`context_length` → `intern_with_bounds("seq_len", Some(1), Some(context_length))`.
`vocab_size` → `fix(id, vocab_size)`.

**User config:** CLI flags like `--max-seq-len 2048` call
`intern_with_bounds("seq_len", Some(1), Some(2048), UserConfig)`.

---

## 4. Shape Inference

### Architecture

Shape inference is dispatched per-op by the `ShapePropagation` pass:

```rust
/// Shape inference result for a single node.
pub struct InferredShapes {
    /// Output shapes — one per output tensor of the node.
    pub output_shapes: Vec<Shape>,
    /// Constraints that must hold (validated eagerly or deferred).
    pub constraints: Vec<ShapeConstraint>,
}

/// Dispatch shape inference for an op.
pub fn infer_shapes(
    op: &AiOp,
    inputs: &[&TensorInfo],
    dim_vars: &DimVarTable,
) -> Result<InferredShapes, ShapeError>
```

### Per-Op Rules

#### MatMul / BatchMatMul

```
MatMul([..., M, K1], [..., K2, N]) → [..., M, N]
Constraint: DimEqual(K1, K2, "MatMul inner dimension")
```

Batch dimensions are broadcast using the standard broadcast rules.

#### Elementwise Binary (Add, Sub, Mul, Div, etc.)

Shapes are broadcast element-wise following NumPy rules:

```rust
fn broadcast_dim(a: &DimExpr, b: &DimExpr) -> (DimExpr, Vec<ShapeConstraint>) {
    // Concrete(1) + anything => the other dim
    // Same concrete values => that value
    // Different concrete values => error
    // Symbolic + Concrete(1) => symbolic
    // Two symbolics => Max(a, b) + BroadcastCompatible constraint
}
```

#### Reshape

```
Reshape(input_shape, target_shape) → target_shape
Constraint: ProductEqual(input_dims, output_dims)
```

If exactly one target dimension is a placeholder (from ONNX `-1`), it is solved:
`inferred = Div(product(input_dims), product(known_target_dims))`.

A `Divisible` constraint is also emitted.

#### Concat

```
Concat([..., D1, ...], [..., D2, ...], ..., axis=a) → [..., D1 + D2 + ..., ...]
Constraint: DimEqual for all non-axis dimensions (pairwise)
```

The axis dimension becomes `Add(D1, Add(D2, ...))`.

#### Transpose

```
Transpose(input, perm=[p0, p1, ...]) → [input[p0], input[p1], ...]
```

No constraints — just reorders dimensions.

#### Reductions (ReduceSum, ReduceMean, ReduceMax, etc.)

```
keepdims=true:  dim[axis] → Concrete(1)
keepdims=false: dim[axis] removed from shape
```

#### Attention Ops (MultiHeadAttention, GroupedQueryAttention)

```
MHA(Q: [B, S, H*D], K: [B, S_kv, H*D], V: [B, S_kv, H*D]) → [B, S, H*D]
Constraints:
  - Q last dim == H * D (Divisible)
  - K last dim == H * D (or num_kv_heads * D for GQA)
  - K and V batch/kv_seq dims match
```

`H`, `D` come from the op's `num_heads`, `head_dim` attributes.

#### Softmax, LayerNorm, RmsNorm

Output shape equals input shape (shape-preserving ops).

#### Slice

```
Slice(input, axes, starts, ends, steps) → output
  output_dim[axis] = CeilDiv(Sub(end, start), step)
  Non-axis dims are preserved.
```

#### Split

```
Split(input, axis, sizes) → [output_1, output_2, ...]
  output_i_dim[axis] = sizes[i]
  Constraint: Sum(sizes) == input_dim[axis]
```

#### Gather

```
Gather(data: [D0, ..., D_{axis}, ..., D_n], indices: [I0, ..., I_m], axis) →
  [D0, ..., D_{axis-1}, I0, ..., I_m, D_{axis+1}, ..., D_n]
```

### Three-Tier Resolution

1. **Immediate error:** Both sides concrete and unequal → `ShapeError::DimMismatch`.
2. **Immediate fix:** One side concrete, other is bare `Var` → fix the variable in
   `DimVarTable` (or tighten bounds).
3. **Deferred constraint:** Both sides symbolic → record `ShapeConstraint`, validate
   at concretization time.

---

## 5. Shape Constraints

### Constraint Types

```rust
#[derive(Debug, Clone)]
pub enum ShapeConstraint {
    /// Two dimension expressions must be equal.
    DimEqual {
        node_id: NodeId,
        lhs: DimExpr,
        rhs: DimExpr,
        context: String,  // e.g., "MatMul inner dimension"
    },

    /// Two expressions must be broadcast-compatible
    /// (both equal, or at least one is 1).
    BroadcastCompatible(DimExpr, DimExpr),

    /// Product equality for reshape operations.
    ProductEqual {
        node_id: NodeId,
        input_dims: Vec<DimExpr>,
        output_dims: Vec<DimExpr>,
    },

    /// A dimension must be positive (> 0).
    Positive {
        node_id: NodeId,
        dim: DimExpr,
        context: String,
    },

    /// A dimension must divide evenly into another.
    /// E.g., hidden_dim must be divisible by num_heads.
    Divisible {
        node_id: NodeId,
        dividend: DimExpr,
        divisor: DimExpr,
        context: String,
    },
}
```

### ConstraintStore

```rust
/// Collects shape constraints during shape propagation.
/// Validates eagerly for concrete constraints, defers symbolic ones.
#[derive(Debug, Clone, Default)]
pub struct ConstraintStore {
    deferred: Vec<ShapeConstraint>,
    errors: Vec<ShapeError>,
}

impl ConstraintStore {
    /// Add a constraint. Concrete constraints are validated immediately;
    /// symbolic constraints are deferred.
    pub fn add(&mut self, constraint: ShapeConstraint);

    /// Validate all deferred constraints given current variable assignments.
    /// Returns all errors found.
    pub fn validate_all(&self, dim_vars: &DimVarTable) -> Vec<ShapeError>;

    /// Returns true if any errors have been recorded (eager or deferred).
    pub fn has_errors(&self) -> bool;

    /// Number of deferred constraints still pending.
    pub fn deferred_count(&self) -> usize;
}
```

---

## 6. Shape Errors

```rust
#[derive(Debug, Clone)]
pub enum ShapeError {
    /// Concrete dimension mismatch.
    DimMismatch {
        op: String,
        node_id: NodeId,
        expected: DimExpr,
        got: DimExpr,
    },

    /// Broadcast incompatibility between two concrete dims.
    BroadcastIncompatible { dim_a: u64, dim_b: u64 },

    /// Dimension variable violates its declared bounds.
    BoundsViolation {
        var: String,
        value: u64,
        lower: Option<u64>,
        upper: Option<u64>,
    },

    /// Unfixed variable with no upper bound at concretization time.
    UnboundedVariable { var: String },

    /// Rank mismatch (expected N dims, got M).
    RankMismatch {
        op: String,
        node_id: NodeId,
        expected_rank: usize,
        got_rank: usize,
    },

    /// Reshape product equality cannot be satisfied.
    ReshapeProductMismatch {
        node_id: NodeId,
        input_product: DimExpr,
        target_product: DimExpr,
    },

    /// Missing shape information for a tensor.
    MissingShape { tensor_id: TensorId },

    /// Division not exact (Divisible constraint violated).
    DivisibilityViolation {
        node_id: NodeId,
        dividend: u64,
        divisor: u64,
        context: String,
    },
}
```

---

## 7. Lowering Strategies

### Strategy Types

```rust
/// How symbolic dimensions are resolved at lowering time.
#[derive(Debug, Clone)]
pub enum ShapeStrategy {
    /// MVP: fix all symbolic dims to their upper bound.
    /// Single graph, single schedule. Rebuild when dims differ.
    FixToMax,

    /// Compile a set of bucket sizes for a key variable (usually seq_len).
    /// At runtime, select the smallest bucket >= actual value.
    Bucketed(BucketConfig),

    /// Compile for specific named shape assignments.
    /// Each profile produces a separate schedule.
    Profiles(Vec<ShapeProfile>),

    /// Fix all dims to max, but track actual lengths via metadata tensors.
    /// Attention masking handles the padding. Single graph, single schedule.
    PaddedMax,
}

/// Configuration for bucketed compilation.
#[derive(Debug, Clone)]
pub struct BucketConfig {
    /// The dimension variable to bucket (typically seq_len).
    pub variable: DimVarId,
    /// Bucket boundaries (sorted ascending). E.g., [128, 512, 1024, 2048, 4096].
    pub buckets: Vec<u64>,
}

/// A specific assignment of values to all symbolic dimension variables.
#[derive(Debug, Clone)]
pub struct ShapeProfile {
    pub name: String,
    pub assignments: HashMap<DimVarId, u64>,
}
```

### CompiledModel Changes

```rust
pub enum ShapeScheduleMap {
    /// Single schedule (FixToMax, PaddedMax).
    Single(Arc<hologram::ExecutionSchedule>),

    /// Multiple schedules keyed by profile name.
    Profiled(HashMap<String, Arc<hologram::ExecutionSchedule>>),

    /// Multiple schedules keyed by bucket index.
    Bucketed {
        config: BucketConfig,
        schedules: Vec<Arc<hologram::ExecutionSchedule>>,
    },
}
```

### Lowering Entry Point

```rust
/// Produce concrete lowering output(s) from a symbolic AiGraph.
pub fn lower_with_strategy(
    ai_graph: &AiGraph,
    kv_layout: &KvCacheLayout,
    opts: &LoweringOptions,
    strategy: &ShapeStrategy,
) -> Result<Vec<(String, LoweringOutput)>, ShapeError>
```

Each strategy:
1. Clones the `AiGraph`
2. Fixes variables in `DimVarTable` per the strategy
3. Calls `concretize_shapes()` to resolve all `DimExpr` → `DimExpr::Concrete`
4. Calls `lower()` which requires fully concrete shapes

`hologram::Graph` always receives concrete dimensions. No changes to the hologram API.

### KV-Cache Interaction

`KvCacheLayout.max_seq_len` is derived from `DimVarTable`:

```rust
let seq_len_id = graph.dim_vars.lookup(canonical_vars::SEQ_LEN)?;
let max_seq = graph.dim_vars.get(seq_len_id).upper
    .or(graph.dim_vars.get(seq_len_id).fixed)?;
```

For bucketed compilation, the KV-cache is always sized to the maximum bucket. The
`present_len` counter in `InferenceSession` ensures only the used portion is active.

---

## 8. AiGraph Changes

```rust
pub struct AiGraph {
    pub name: String,
    pub nodes: Vec<AiNode>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub params: HashMap<TensorId, AiParam>,
    pub tensor_info: HashMap<TensorId, TensorInfo>,
    pub metadata: HashMap<String, MetaValue>,
    pub warnings: Vec<ImportWarning>,
    pub dim_vars: DimVarTable,              // NEW — dimension variable registry
    pub shape_constraints: ConstraintStore,  // NEW — collected shape constraints
}
```

---

## 9. Module Layout

The shape module in `hologram-ai-common` expands:

```
crates/hologram-ai-common/src/ir/
  shape/
    mod.rs           — re-exports
    dim_expr.rs      — DimExpr, DimVarId
    dim_var.rs       — DimVarTable, DimVarEntry, DimVarSource, canonical_vars
    constraint.rs    — ShapeConstraint, ConstraintStore
    error.rs         — ShapeError
    infer.rs         — infer_shapes(), per-op rules
    compat.rs        — shape_from_concrete(), migration helpers
```

Public re-exports from `hologram-ai-common`:

```rust
pub use shape::{
    DimExpr, DimVarId, DimVarTable, DimVarEntry, DimVarSource,
    Shape, ShapeConstraint, ConstraintStore, ShapeError,
    canonical_vars, infer_shapes, InferredShapes,
    shape_from_concrete, concretize_shapes,
};
```

---

## 10. Phased Implementation

### Phase 0: Foundation (no behavior change)

- Define `DimExpr`, `DimVarId`, `DimVarTable`, `ShapeError` types
- Migrate `Shape` alias from `SmallVec<[Dim; 6]>` to `SmallVec<[DimExpr; 4]>`
- Add `dim_vars: DimVarTable` to `AiGraph`
- Update importers to intern into `DimVarTable` (ONNX `dim_param` → `Var(id)`, GGUF metadata → fixed vars)
- Update lowering `concrete_dim()` to use `DimExpr::as_concrete()`
- Preserve `FixToMax` behavior via `concretize_to_upper()`
- **Exit criterion:** All existing tests pass identically

### Phase 1: Shape Propagation

- Implement `infer_shapes()` for core ops: MatMul, Add/Mul (broadcast), Reshape, Concat, Transpose, Reduce, Softmax, LayerNorm, RmsNorm
- Implement `ShapePropagation` pass, add to `OptPipeline::mvp()`
- Define `ShapeConstraint` and `ConstraintStore`, collect constraints
- **Exit criterion:** ShapePropagation fills output shapes for TinyLlama GGUF graph

### Phase 2: Constraint Validation + Bucketed Compilation

- Implement `ConstraintStore::validate_all()` for deferred constraints
- Add `ShapeStrategy::Bucketed` for seq_len bucketing
- Implement `lower_with_strategy()`, update `CompiledModel` with `ShapeScheduleMap`
- Update `InferenceSession::run()` to select schedule by input shape
- Add remaining op inference rules: Gather, Scatter, Slice, Split, attention ops
- **Exit criterion:** Bucketed model selects correct schedule for varying input lengths

### Phase 3: Profiles + Advanced

- Implement `ShapeStrategy::Profiles` for multi-variable specialization
- Implement `ShapeStrategy::PaddedMax` with attention mask length tracking
- CLI: `hologram-ai inspect` shows variable table and constraints
- CLI: `hologram-ai compile --shape-strategy bucketed:128,512,2048`
- **Exit criterion:** CLI shape strategy works end-to-end
