# Plan: Preserve Symbolic Dimensions Through ONNX Translation

**Status: PHASE 2 COMPLETE - READY FOR TESTING**

## Implementation Progress

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ DONE | Use `from_value_info_preserve_symbolic` for graph inputs |
| Phase 2 | ✅ DONE | Fix translators that fail on symbolic dims (Split/Reshape/Transpose) |
| Phase 3 | ✅ DONE | Fix output metadata generation |
| Phase 4 | ⏳ PENDING | Clean up resolution logic (optional) |

## Changes Made

1. **translator.rs**: Changed lines 232, 312, 511 to use `from_value_info_preserve_symbolic`
2. **shapes.rs**: Updated `resolve_symbolic_dimension()` to preserve non-batch symbolic dims as `Dim::Symbolic`
3. **lib.rs**: Updated output metadata generation to use `from_value_info_preserve_symbolic`
4. **split.rs**: Fixed to handle symbolic axis dims (allows identity pass-through for symbolic splits)
5. **reshape.rs**: Falls through to `reshape_dynamic` when symbolic dims detected (preserves names)
6. **transpose.rs**: Added safety check to skip constant folding for symbolic dims (builder already preserves them)

## Problem Statement

ONNX graphs aren't mapping naturally to hologram despite hologram's symbolic shape support because the ONNX translator **prematurely resolves symbolic dimensions to concrete defaults**. For example:

* `batch` → `Static(1)` (intentional for inference)
* Other symbolic dims should be preserved as `Dim::Symbolic`

Hologram's backend already supports symbolic dimensions via `DimExpr` expressions that get resolved at runtime when actual input data arrives. The translator should preserve symbolic dims and let hologram handle them naturally.

## Root Cause Analysis

### Two Methods Exist for Shape Extraction

In `crates/hologram-ai-onnx/src/core/shapes.rs`:

1. **`from_value_info()`** (lines 198-247) - **Resolves** symbolic dims:
   * Calls `resolve_symbolic_dimension()` which converts "batch", "batch\_size", "N" → `Static(1)`
   * Other symbolic names are preserved as `Dim::Symbolic`

2. **`from_value_info_preserve_symbolic()`** (lines 254-293) - **Preserves** symbolic dims:
   * `DimParam(name)` → `Dim::Symbolic(name)` directly
   * No resolution to defaults

### Current Usage Pattern

| Location | Method Used | Effect |
|----------|-------------|--------|
| Graph inputs | `from_value_info()` | **LOSES** symbolic batch dims |
| Subgraph inputs | `from_value_info()` | **LOSES** symbolic batch dims |
| Graph outputs | `from_value_info_preserve_symbolic()` | Preserves symbolic dims ✓ |

### Operations That Fail on Symbolic Dims

| File | Lines | Issue |
|------|-------|-------|
| `split.rs` | 69-76 | Returns error if axis dim is symbolic |
| `reshape.rs` | 129-131 | Converts `Dim::Symbolic` to `-1` (loses semantic meaning) |
| `transpose.rs` | ~92 | Uses `unwrap_or(1)` defaulting symbolic to 1 |
| `lib.rs` | 809-823 | Output metadata converts ALL symbolic dims to concrete defaults |

***

## Implementation Plan

### Phase 1: Use Preserve Method for Inputs

**File: `crates/hologram-ai-onnx/src/core/translator.rs`**

#### Change 1: Subgraph inputs (line 234)

```rust
// BEFORE:
let shape = crate::core::SymbolicShape::from_value_info(input)?;

// AFTER:
let shape = crate::core::SymbolicShape::from_value_info_preserve_symbolic(input)?;
```

#### Change 2: Main graph inputs (line 315)

```rust
// BEFORE:
let shape = crate::core::SymbolicShape::from_value_info(input)?;

// AFTER:
let shape = crate::core::SymbolicShape::from_value_info_preserve_symbolic(input)?;
```

#### Change 3: `translate_graph_to_ir_with_groups` inputs (line 702)

```rust
// BEFORE:
let shape = crate::core::SymbolicShape::from_value_info(input)?;

// AFTER:
let shape = crate::core::SymbolicShape::from_value_info_preserve_symbolic(input)?;
```

***

### Phase 2: Fix Translators That Fail on Symbolic Dims

#### Change 4: Split Operation (`translators/shape/split.rs` lines 69-76)

```rust
// BEFORE:
let axis_size = match axis_dim {
    Dim::Static(s) => *s,
    _ => {
        return Err(TranslationError::ShapeInference(
            "Split: cannot split along dynamic dimension".to_string(),
        ));
    }
};

// AFTER:
// Pass symbolic dimensions through to hologram IR.
// Hologram will resolve the actual dimension at runtime.
let axis_size = match axis_dim {
    Dim::Static(s) => Some(*s),
    Dim::Symbolic(_) | Dim::Dynamic => None,
};

// Then update the split logic to handle None (symbolic) case:
// - If axis_size is None, create a dynamic split operation
// - Hologram's backend will compute splits at runtime
```

#### Change 5: Reshape Operation (`translators/shape/reshape.rs` lines 129-131)

```rust
// BEFORE:
match dim {
    Dim::Static(value) => result.push(value as i64),
    Dim::Dynamic | Dim::Symbolic(_) => {
        dynamic_count += 1;
        result.push(-1);  // LOSES semantic meaning
    }
}

// AFTER:
// Preserve symbolic dimension information
match dim {
    Dim::Static(value) => result.push(ShapeDim::Static(value)),
    Dim::Symbolic(name) => result.push(ShapeDim::Symbolic(name.clone())),
    Dim::Dynamic => result.push(ShapeDim::Dynamic),
}
// Create reshape with symbolic shape info preserved
```

#### Change 6: Transpose Operation (`translators/shape/transpose.rs` line ~92)

```rust
// BEFORE:
let in_dims: Vec<usize> = input_node
    .op
    .shape
    .dims
    .iter()
    .map(|d| d.static_value().unwrap_or(1))  // CONVERTS symbolic to 1!
    .collect();

// AFTER:
// Preserve symbolic dimensions through transpose
let out_dims: Vec<Dim> = perm
    .iter()
    .map(|&p| input_node.op.shape.dims[p].clone())
    .collect();
```

***

### Phase 3: Fix Output Metadata Generation

**File: `crates/hologram-ai-onnx/src/lib.rs` (lines 809-823)**

```rust
// BEFORE:
let dims: Vec<usize> = declared_shape
    .dims()
    .iter()
    .enumerate()
    .map(|(idx, dim)| match dim {
        hologram::ir::Dim::Static(n) => *n,
        hologram::ir::Dim::Dynamic => default_output_dim_value(idx, None),
        hologram::ir::Dim::Symbolic(name) => {
            default_output_dim_value(idx, Some(name.as_str()))  // LOSES symbolic!
        }
    })
    .collect();

// AFTER:
// Preserve symbolic dims in metadata - hologram's executor handles DimExpr resolution
let dims: Vec<OutputDim> = declared_shape
    .dims()
    .iter()
    .map(|dim| match dim {
        hologram::ir::Dim::Static(n) => OutputDim::Static(*n),
        hologram::ir::Dim::Dynamic => OutputDim::Dynamic,
        hologram::ir::Dim::Symbolic(name) => OutputDim::Symbolic(name.clone()),
    })
    .collect();
```

***

### Phase 4: Clean Up Resolution Logic

**File: `crates/hologram-ai-onnx/src/core/shapes.rs` (lines 55-72)**

Consider deprecating `resolve_symbolic_dimension()` or making it opt-in.

***

### Phase 5: Wire Up dim_exprs in Hologram (THE CRITICAL FIX)

**This phase addresses the execution blocker.** The infrastructure exists but isn't connected.

#### Background: Existing Infrastructure

**`KernelParams` in `/hologram/crates/backend/src/core/plan.rs:434-471`:**
```rust
pub struct KernelParams {
    pub dims: [usize; 4],               // Static dimension values (compile-time)
    pub dim_exprs: [Option<DimExpr>; 4], // Runtime resolution expressions ← EXISTS!
    // ...
}

impl KernelParams {
    pub fn resolve_dims(&self, input_shapes: &[[usize; 4]]) -> Self {
        // Resolves dim_exprs using actual input shapes ← EXISTS!
    }
}
```

**`DimExpr` variants:**
```rust
pub enum DimExpr {
    Static(usize),
    InputRef { input_id, dim_index },      // Reference to input dimension
    TotalElements { input_id },            // Product of all dims
    ProductOfDims { input_id_a, dim_a, input_id_b, dim_b },
    TotalElementsDiv { input_id, divisor },
    DimDiv { input_id, dim_index, divisor },
    PredecessorElementsDiv { predecessor_slot, divisor },
}
```

#### Change 7: Compiler - Populate dim_exprs for Gather

**File: `/hologram/crates/compiler/src/pipeline.rs` (lines 2640-2722)**

Currently, when building Gather params with symbolic shapes, the compiler falls back to `dims: [1, 1, ...]`. Instead, it should populate `dim_exprs`:

```rust
// BEFORE (lines 2643-2650):
let mut axis_size = 1;      // Hardcoded default
let mut inner_size = 1;     // Hardcoded default

// AFTER:
// When input shapes are symbolic, generate DimExpr instead of defaults
if input_shape_is_symbolic {
    params.dim_exprs[0] = Some(DimExpr::InputRef {
        input_id: weight_input_id,
        dim_index: 0  // vocab_size from weight shape
    });
    params.dim_exprs[1] = Some(DimExpr::InputRef {
        input_id: weight_input_id,
        dim_index: 1  // embedding_dim from weight shape
    });
    params.dim_exprs[2] = Some(DimExpr::TotalElements {
        input_id: indices_input_id  // num_indices from indices tensor
    });
}
```

#### Change 8: Executor - Call resolve_dims() Before Kernel Dispatch

**File: `/hologram/crates/backend/src/core/executor.rs`**

Before dispatching each kernel, resolve `dim_exprs` using actual input shapes:

```rust
// BEFORE:
kernel_fn(inputs, outputs, &params, tables)?;

// AFTER:
let resolved_params = if params.has_dim_exprs() {
    params.resolve_dims(&input_shapes)
} else {
    params.clone()
};
kernel_fn(inputs, outputs, &resolved_params, tables)?;
```

#### Why This Works

1. **Compile time**: When shapes are symbolic, compiler generates `DimExpr` formulas instead of hardcoded defaults
2. **Serialization**: `dim_exprs` are serialized into the .holo file
3. **Runtime**: Executor resolves `dim_exprs` using actual input tensor shapes
4. **Kernels**: Receive correctly resolved `dims` - no kernel changes needed

***

## Files to Modify

### hologram-ai-onnx (this repo)

| File | Changes |
|------|---------|
| `crates/hologram-ai-onnx/src/core/translator.rs` | Lines 234, 315, 702: Use `from_value_info_preserve_symbolic` |
| `crates/hologram-ai-onnx/src/translators/shape/split.rs` | Lines 69-76: Handle symbolic axis dims |
| `crates/hologram-ai-onnx/src/translators/shape/reshape.rs` | Lines 129-131: Preserve symbolic dims |
| `crates/hologram-ai-onnx/src/translators/shape/transpose.rs` | Line ~92: Remove `unwrap_or(1)` |
| `crates/hologram-ai-onnx/src/lib.rs` | Lines 809-823: Preserve symbolic in output metadata |
| `crates/hologram-ai-onnx/src/core/shapes.rs` | Lines 55-72: Deprecate or make resolution opt-in |

### hologram (external dependency - requires PR)

| File | Changes |
|------|---------|
| `/hologram/crates/compiler/src/pipeline.rs` | Lines 2640-2722: Populate `dim_exprs` for Gather when shapes are symbolic |
| `/hologram/crates/backend/src/core/executor.rs` | Call `params.resolve_dims()` before kernel dispatch |

***

## Why This Works

### Hologram's Backend Already Supports Symbolic Dimensions

From `crates/hologram-ai/src/runtime/executor.rs` (lines 240-296):

```rust
let resolved_dims: Vec<usize> = shape_expr
    .iter()
    .map(|(dim_idx, expr)| match expr {
        DimExpr::Static(n) => *n,
        DimExpr::InputRef { input_id, dim_index } => { /* resolve at runtime */ },
        DimExpr::TotalElements { input_id } => { /* sum all dims */ },
        // ... more DimExpr variants
    })
    .collect();
```

The executor already:

1. Receives actual input tensor shapes via `register_shape()`
2. Resolves `DimExpr` expressions at runtime
3. Allocates correct output buffer sizes based on actual dimensions

### Constant Folding Happens Naturally

Hologram's compiler performs constant folding in its optimization passes. If an operation has all constant inputs, it will be folded. We don't need to do this in the ONNX translator - hologram handles it.

***

## Verification

1. **Build check**: `cargo build -p hologram-ai-onnx`
2. **Existing tests**: `cargo test -p hologram-ai-onnx`
3. **E2E test with symbolic shapes**:
   * Compile a model with `dim_param="batch"` and `dim_param="seq_len"`
   * Verify symbolic dims appear in compiled IR (not resolved to 1/512)
4. **Runtime test**:
   * Execute compiled model with batch\_size=4, seq\_len=256
   * Verify output shapes are \[4, 256, ...] (not \[1, 512, ...])
   * Verify Gather kernel params show correct axis\_size/inner\_size (not 1, 1)

***

## Notes

* This is a **breaking change** for any code that relied on batch being resolved to 1
* Models that previously worked with hardcoded batch=1 should continue to work (batch=1 at runtime)
* Models that need variable batch sizes will now work correctly
* **Phase 5 requires changes to the hologram crate** - this will need a PR to the hologram repo
* The hologram infrastructure (`dim_exprs`, `resolve_dims()`) already exists - we're just wiring it up
