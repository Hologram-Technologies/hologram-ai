# Prompt 09 — Symbolic Shapes and Dimensions

## Goal

Implement the symbolic shape system for `hologram-ai-common`: `DimExpr` expression type,
`DimVarTable` variable registry, `ShapePropagation` pass with per-op inference rules,
`ConstraintStore` validation, and shape concretization for lowering. This is Phase 0 + Phase 1
of the symbolic shapes plan (ADR-0015).

## Context

- ADR: `specs/adrs/0015-hologram-ai-symbolic-shapes.md`
- Project doc: `specs/projects/hologram-ai/symbolic-shapes.md`
- Architecture: `specs/projects/hologram-ai/architecture.md` (section 7)
- Lowering: `specs/projects/hologram-ai/lowering.md` (Shape section)

## Crate: `hologram-ai-common`

All changes are in `crates/hologram-ai-common/`. The shape module at `src/ir/shape.rs`
expands into a `src/ir/shape/` directory module.

---

## Phase 0: Foundation Types

### Step 1: Create `src/ir/shape/` module directory

Replace `src/ir/shape.rs` with:

```
src/ir/shape/
  mod.rs           — re-exports
  dim_expr.rs      — DimExpr, DimVarId
  dim_var.rs       — DimVarTable, DimVarEntry, DimVarSource, canonical_vars
  constraint.rs    — ShapeConstraint, ConstraintStore
  error.rs         — ShapeError
  infer.rs         — infer_shapes() (Phase 1)
  compat.rs        — shape_from_concrete(), migration helpers
```

### Step 2: Define `DimExpr` in `dim_expr.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct DimVarId(pub(crate) u32);

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

Implement:
- `concrete(n) -> Self`, `var(id) -> Self`
- `as_concrete() -> Option<u64>`
- `is_concrete() -> bool`
- `evaluate() -> Option<u64>` — returns None if any Var/Dynamic; div-by-zero → None
- `substitute(var: DimVarId, value: &DimExpr) -> DimExpr`
- `simplify() -> DimExpr` — fold constant sub-expressions only (no algebraic rules)
- `free_vars() -> HashSet<DimVarId>`
- `impl Display for DimExpr` — e.g., `"batch"`, `"seq_len + 1"`, `"ceil(hidden / 8)"`

Tests:
- `evaluate()` returns correct values for nested expressions
- `substitute()` replaces variables correctly
- `simplify()` folds `Add(Concrete(3), Concrete(5))` → `Concrete(8)`
- `simplify()` preserves `Add(Var(x), Concrete(5))` unchanged
- `free_vars()` collects all referenced variables
- `is_concrete()` is true only for fully concrete trees

### Step 3: Define `DimVarTable` in `dim_var.rs`

```rust
pub enum DimVarSource { Import, Inferred, UserConfig }

pub struct DimVarEntry {
    pub name: String,
    pub lower: Option<u64>,
    pub upper: Option<u64>,
    pub fixed: Option<u64>,
    pub source: DimVarSource,
}

pub struct DimVarTable {
    entries: Vec<DimVarEntry>,
    name_to_id: HashMap<String, DimVarId>,
}
```

Implement:
- `intern(name) -> DimVarId`
- `intern_with_bounds(name, lower, upper, source) -> DimVarId` — tighten via intersection
- `fix(id, value) -> Result<(), ShapeError>` — validate bounds
- `get(id) -> &DimVarEntry`
- `lookup(name) -> Option<DimVarId>`
- `fixed_substitutions() -> HashMap<DimVarId, DimExpr>`
- `concretize_to_upper() -> Result<(), ShapeError>` — fails if any var has no upper bound
- `iter() -> impl Iterator<Item = (DimVarId, &DimVarEntry)>`

Define canonical variable names:

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

Tests:
- `intern()` returns same ID for same name
- `intern_with_bounds()` tightens bounds on repeat calls
- `fix()` succeeds within bounds, fails outside
- `concretize_to_upper()` fixes all vars to upper bound
- `concretize_to_upper()` fails if any var has no upper bound

### Step 4: Define `ShapeError` in `error.rs`

```rust
pub enum ShapeError {
    DimMismatch { op: String, node_id: NodeId, expected: DimExpr, got: DimExpr },
    BroadcastIncompatible { dim_a: u64, dim_b: u64 },
    BoundsViolation { var: String, value: u64, lower: Option<u64>, upper: Option<u64> },
    UnboundedVariable { var: String },
    RankMismatch { op: String, node_id: NodeId, expected_rank: usize, got_rank: usize },
    ReshapeProductMismatch { node_id: NodeId, input_product: DimExpr, target_product: DimExpr },
    MissingShape { tensor_id: TensorId },
    DivisibilityViolation { node_id: NodeId, dividend: u64, divisor: u64, context: String },
}
```

Implement `Display` and `std::error::Error`.

### Step 5: Define `ShapeConstraint` and `ConstraintStore` in `constraint.rs`

```rust
pub enum ShapeConstraint {
    DimEqual { node_id: NodeId, lhs: DimExpr, rhs: DimExpr, context: String },
    BroadcastCompatible(DimExpr, DimExpr),
    ProductEqual { node_id: NodeId, input_dims: Vec<DimExpr>, output_dims: Vec<DimExpr> },
    Positive { node_id: NodeId, dim: DimExpr, context: String },
    Divisible { node_id: NodeId, dividend: DimExpr, divisor: DimExpr, context: String },
}

pub struct ConstraintStore {
    deferred: Vec<ShapeConstraint>,
    errors: Vec<ShapeError>,
}
```

Implement:
- `add(constraint)` — validate eagerly if concrete, defer if symbolic
- `validate_all(dim_vars) -> Vec<ShapeError>` — substitute fixed vars and check
- `has_errors() -> bool`
- `deferred_count() -> usize`

### Step 6: Migration helpers in `compat.rs`

```rust
/// Create a fully-concrete shape from a slice of u64 values.
pub fn shape_from_concrete(dims: &[u64]) -> Shape {
    dims.iter().map(|&d| DimExpr::Concrete(d)).collect()
}

/// Concretize all shapes in a graph using fixed variable assignments.
pub fn concretize_shapes(graph: &mut AiGraph) -> Result<(), ShapeError> {
    let subs = graph.dim_vars.fixed_substitutions();
    for info in graph.tensor_info.values_mut() {
        for dim in info.shape.iter_mut() {
            *dim = substitute_all(dim, &subs).simplify();
            if !dim.is_concrete() {
                return Err(ShapeError::UnboundedVariable {
                    var: format!("{}", dim),
                });
            }
        }
    }
    Ok(())
}

fn substitute_all(expr: &DimExpr, subs: &HashMap<DimVarId, DimExpr>) -> DimExpr {
    match expr {
        DimExpr::Var(id) => subs.get(id).cloned().unwrap_or_else(|| expr.clone()),
        // ... recurse for compound expressions
    }
}
```

### Step 7: Update `AiGraph`

Add two fields to `AiGraph` in `src/ir/graph.rs`:

```rust
pub dim_vars: DimVarTable,
pub shape_constraints: ConstraintStore,
```

Initialize both with `Default::default()` in constructors.

### Step 8: Update `Shape` type alias

Change `pub type Shape = SmallVec<[Dim; 6]>` to `pub type Shape = SmallVec<[DimExpr; 4]>`.
Remove the old `Dim` enum.

### Step 9: Update importers

**ONNX importer** (`hologram-ai-onnx`):
- In `shape_from_shape_proto()`: replace `Dim::Symbolic(p.clone())` with
  `DimExpr::Var(dim_vars.intern(normalize_onnx_dim_name(p)))`.
- Thread `&mut DimVarTable` through the graph builder.
- Replace `Dim::Concrete(n)` with `DimExpr::Concrete(n)`.
- Replace `Dim::Dynamic` with `DimExpr::Dynamic`.

**GGUF importer** (`hologram-ai-gguf`):
- After reading architecture metadata, intern canonical vars with bounds:
  ```rust
  let seq_id = dim_vars.intern_with_bounds("seq_len", Some(1), Some(context_length), Import);
  let vocab_id = dim_vars.intern("vocab_size");
  dim_vars.fix(vocab_id, vocab_size)?;
  // etc. for hidden_dim, num_heads, num_kv_heads, head_dim, ffn_dim
  ```
- Replace `Dim::Concrete(n)` with `DimExpr::Concrete(n)` in tensor shapes.

### Step 10: Update lowering

In `src/lower/builder.rs`, update `concrete_dim()` (or equivalent) to use
`DimExpr::as_concrete()`. Add a pre-lowering check:

```rust
for (tid, info) in &ai_graph.tensor_info {
    for (i, dim) in info.shape.iter().enumerate() {
        if !dim.is_concrete() {
            bail!("tensor {} dim {} not concrete at lowering: {}. \
                   Call concretize_shapes() first.", tid, i, dim);
        }
    }
}
```

### Phase 0 Exit Criterion

All existing tests pass identically. No new behavior; only new types underneath.

---

## Phase 1: Shape Propagation Pass

### Step 11: Implement `infer_shapes()` in `infer.rs`

Dispatch function that calls per-op shape inference:

```rust
pub fn infer_shapes(
    op: &AiOp,
    inputs: &[&TensorInfo],
    dim_vars: &DimVarTable,
) -> Result<InferredShapes, ShapeError>
```

Implement rules for these ops (priority order):
1. `MatMul`, `BatchMatMul` — inner-dim constraint
2. `Add`, `Sub`, `Mul`, `Div` — broadcast rules
3. `Reshape` — product equality, `-1` inference
4. `Concat` — axis-dim summation
5. `Transpose` — permutation
6. `Softmax`, `LayerNorm`, `RmsNorm` — shape-preserving
7. `ReduceSum`, `ReduceMean`, `ReduceMax` — axis removal/keepdims
8. `Unsqueeze`, `Squeeze` — axis add/remove
9. `Identity`, `Cast` — pass-through

For ops not yet covered, return the input shapes unchanged with a warning.

### Step 12: Implement `ShapePropagation` pass

```rust
pub struct ShapePropagation;

impl Pass for ShapePropagation {
    fn name(&self) -> &str { "shape_propagation" }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        // Walk nodes in topological order.
        // For each node, call infer_shapes().
        // Update output tensor shapes.
        // Collect constraints into graph.shape_constraints.
        // Log warnings for inference failures (don't fail the pass).
        Ok(graph)
    }
}
```

Add `ShapePropagation` to `OptPipeline::mvp()` after `ConstantFolding` and before
`DeadNodeElimination`.

### Step 13: Tests

Unit tests in `shape/infer.rs`:
- MatMul: `[2, 3] x [3, 4] → [2, 4]`
- MatMul: `[B, M, K] x [B, K, N] → [B, M, N]` with symbolic dims
- MatMul: mismatched inner dim → DimMismatch error
- Broadcast: `[B, 1, D] + [1, S, D] → [B, S, D]`
- Broadcast: symbolic dims → Max expression + constraint
- Reshape: `[2, 3, 4] → [6, 4]` with product equality
- Reshape: `[B, S, H*D] → [B, S, H, D]` with symbolic dims
- Concat: `[B, S1, D] ++ [B, S2, D] → [B, S1+S2, D]`
- Transpose: `[B, S, D] with perm [0, 2, 1] → [B, D, S]`
- Reduction: `[B, S, D] reduce axis=1 keepdims=false → [B, D]`

Integration test:
- Build a small graph (embed → matmul → add → softmax) with symbolic `batch` and `seq_len`
- Run `ShapePropagation`
- Verify all output shapes contain the expected symbolic expressions
- Verify constraints are collected (MatMul inner dim, etc.)

### Phase 1 Exit Criterion

`ShapePropagation` fills output shapes for a TinyLlama GGUF graph. The pass is in the
default pipeline. Constraint collection works but deferred validation is not yet wired
into lowering.

---

## Important Implementation Notes

### Do NOT:
- Implement algebraic simplification beyond constant folding (no `x + 0 → x`)
- Implement equation solving
- Change `hologram::Graph` or any hologram crate
- Implement `ShapeStrategy::Bucketed/Profiles/PaddedMax` (Phase 2)
- Add `ShapeScheduleMap` to `CompiledModel` (Phase 2)

### Do:
- Keep `FixToMax` as the only lowering strategy
- Make the old `Dim` enum a type alias during migration if needed for incremental porting
- Use `#[cfg(test)]` modules within each shape submodule for unit tests
- Add `impl From<u64> for DimExpr` for ergonomic construction
- Ensure `DimExpr` derives `Serialize, Deserialize` (via serde) for graph serialization
