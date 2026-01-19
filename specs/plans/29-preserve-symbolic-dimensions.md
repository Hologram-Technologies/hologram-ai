# Plan: Preserve Symbolic Dimensions Through ONNX Translation

**Status: IMPLEMENTED - REDUCTION KERNEL DISPATCH ADDED**

## Implementation Progress

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ DONE | Use `from_value_info_preserve_symbolic` for graph inputs |
| Phase 2 | ✅ DONE | Fix translators that fail on symbolic dims (Split/Reshape/Transpose) |
| Phase 3 | ✅ DONE | Fix output metadata generation |
| Phase 4 | ⏳ OPTIONAL | Clean up resolution logic |
| Phase 5 | ✅ DONE | Wire up `dim_exprs` in hologram compiler/executor |
| Phase 6 | ✅ DONE | Dynamic workspace architecture |

## Commits

```
034e179 feat(hologram-ai-onnx): preserve symbolic dimensions through ONNX translation
```

## Key Architecture Change: Dynamic Workspaces

The fundamental issue with workspace allocation has been resolved by **not serializing pre-computed workspace sizes** for dynamic workspaces.

### Problem

With symbolic shapes, workspace sizes cannot be known at compile time. The previous approach:
1. Computed workspace sizes with static estimates at compile time
2. Serialized these estimates in `WorkspaceLayout.total_size`
3. Allocated workspace based on serialized estimates at executor construction
4. This caused overflows when estimates exceeded memory limits

### Solution

The new architecture:
1. **Dynamic regions don't update `total_size`** at compile time
2. **Executor defers allocation** for workspaces with dynamic regions
3. **Runtime resolution** computes actual sizes from input shapes
4. **`DynamicWorkspaceHandle`** allocates correct size at runtime

### Files Modified

**`/hologram/crates/backend/src/core/plan.rs`**:
- `add_region_dynamic()` now sets offset to placeholder (region index) and does NOT update `total_size`
- Added `WorkspaceSizeExpr` enum for symbolic workspace sizes
- Added `DynamicWorkspaceHandle` for runtime workspace management
- `WorkspaceLayout::resolve()` recomputes all offsets from scratch

**`/hologram/crates/backend/src/core/executor.rs`**:
- All constructors check `has_dynamic_regions()` before allocating
- If dynamic regions exist, workspace allocation is deferred
- `resolve_dynamic_workspace()` is called at runtime with actual input shapes
- Uses `DynamicWorkspaceHandle` for correctly-sized workspace

### API Usage

```rust
// Executor construction - workspace deferred for dynamic regions
let executor = PlanExecutor::new(plan, &backend)?;

// Register input shapes
executor.register_input_shape("input_ids", [1, 512, 1, 1]);
executor.register_input_shape("attention_mask", [1, 512, 1, 1]);

// Dynamic workspace resolved automatically on first execute
// Or explicitly preallocate:
executor.preallocate_dynamic_workspace(&[[1, 512, 1, 1], [1, 512, 1, 1]]);
```

---

## Previous Changes (Phase 1-5)

### ONNX Translation

1. **translator.rs**: Changed lines 232, 312, 511 to use `from_value_info_preserve_symbolic`
2. **shapes.rs**: Updated `resolve_symbolic_dimension()` to preserve non-batch symbolic dims as `Dim::Symbolic`
3. **lib.rs**: Updated output metadata generation to use `from_value_info_preserve_symbolic`
4. **split.rs**: Fixed to handle symbolic axis dims (allows identity pass-through for symbolic splits)
5. **reshape.rs**: Falls through to `reshape_dynamic` when symbolic dims detected (preserves names)
6. **transpose.rs**: Added safety check to skip constant folding for symbolic dims (builder already preserves them)

### Hologram Compiler/Executor

1. **DimExpr Resolution in Executor** - Executor calls `resolve_dims()` before kernel dispatch
2. **Workspace MIN_WORKSPACE_BYTES** - Compiler applies minimum to ALL workspace allocations
3. **MAX_GEMM_DIM increased** - From 128K to 512K for large transformer dimensions

---

## Verification

1. **Build check**: `cargo build -p hologram` ✅
2. **Backend tests**: `cargo test -p hologram-backend --lib` ✅ (384 tests pass)
3. **T5 Pipeline**: `cargo run -p hologram-ai --release -- run-pipeline` ✅ (runs without crashes)

---

## Notes

* This is a **breaking change** for any code that relied on batch being resolved to 1
* Models that previously worked with hardcoded batch=1 should continue to work
* Models with variable batch/sequence sizes now allocate correctly at runtime
* The architecture correctly handles both static and dynamic workspaces in the same plan
