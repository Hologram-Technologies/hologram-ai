# Critical Bug: Workspace Buffer Resolution in PlanExecutor

## Status

**CRITICAL**: Operations have correct buffer references, but workspace buffers aren't properly resolved during execution, causing kernel validation failures.

## Problem

After fixing buffer reference creation in the compiler, T5 encoder compilation produces correct BackendPlan:
- ✅ 183 dual-input operations (MatMul/GEMM) with `input_refs` containing 2 references
- ✅ Operations use `BufferRef::Workspace(slot)` for intermediate values
- ✅ Operations have varied buffer references (not all the same)

But execution fails with: **"GEMM kernel requires 2 inputs and 1 output"**

This means the GEMM kernel receives fewer than 2 input pointers, even though the operation has 2 `input_refs`.

## Root Cause

In `/hologram/crates/backend/src/executor.rs`, the execution flow is:

### Step 1: resolve_buffer_ref() (lines ~348-368)

```rust
fn resolve_buffer_ref(
    &self,
    buf_ref: &BufferRef,
    inputs: &[BufferHandle],
    outputs: &[BufferHandle],
) -> Option<BufferHandle> {
    match buf_ref {
        BufferRef::Input(idx) => inputs.get(*idx).copied(),
        BufferRef::Output(idx) => outputs.get(*idx).copied(),
        BufferRef::Workspace(slot) => {
            // ❌ PROBLEM: Creates BufferHandle from raw pointer
            self.workspace.as_ref().and_then(|ws| {
                let ptr = unsafe { ws.region_ptr(*slot) };
                if ptr.is_null() {
                    None  // ❌ Returns None if pointer is null!
                } else {
                    Some(BufferHandle::new(ptr as u64))  // ❌ Creates handle from pointer
                }
            })
        }
    }
}
```

### Step 2: execute() (lines ~227-238)

```rust
// Resolve BufferRef to BufferHandle for this op
let op_inputs: Vec<_> = self.plan.ops[i]
    .input_refs
    .iter()
    .filter_map(|buf_ref| self.resolve_buffer_ref(buf_ref, inputs, outputs))
    //         ^^^^^^^^^ filter_map removes None values!
    .collect();
```

**The Problem**: If `resolve_buffer_ref()` returns `None` for workspace buffers, `filter_map()` silently drops them! This means an operation with `input_refs=[Workspace(5), Workspace(10)]` might produce `op_inputs=[]` if workspace resolution fails.

### Step 3: execute_kernel() (lines ~301-340)

```rust
// Resolve buffer handles to actual data pointers via the backend
let input_ptrs: Vec<*const u8> = inputs
    .iter()
    .map(|&h| {
        backend
            .get_buffer_ptr(h)  // ❌ Backend doesn't know about workspace handles!
            .map(|p| p as *const u8)
            .ok_or(BackendError::InvalidHandle(h))
    })
    .collect::<Result<Vec<_>, _>>()?;
```

**The Problem**: The `BufferHandle` created from workspace pointer isn't registered in the backend's buffer map, so `get_buffer_ptr()` fails.

## Why This Fails

1. **Workspace handles aren't real backend handles**: `BufferHandle::new(ptr as u64)` creates a handle containing a raw pointer value, but this isn't registered with the backend
2. **Backend can't resolve workspace handles**: When `execute_kernel()` calls `backend.get_buffer_ptr(workspace_handle)`, it returns `None` because the backend doesn't know about this handle
3. **Silent failures with filter_map**: Workspace buffers that can't be resolved are silently dropped, so a 2-input operation becomes a 0-input operation

## Solution

Workspace buffers should bypass the BufferHandle system entirely. They already have raw pointers - just use them directly!

### Option 1: Direct Pointer Resolution (Recommended)

Modify `execute()` to resolve pointers directly for workspace buffers:

**File:** `/hologram/crates/backend/src/executor.rs`

```rust
pub fn execute(
    &mut self,
    inputs: &[BufferHandle],
    outputs: &mut [BufferHandle],
    backend: &mut dyn ProgramBackend,
) -> BackendResult<()> {
    // ... validation ...

    for i in 0..num_ops {
        let op = &self.plan.ops[i];

        // Resolve input buffer references directly to pointers
        let input_ptrs: Vec<*const u8> = op.input_refs
            .iter()
            .map(|buf_ref| self.resolve_buffer_ptr(buf_ref, inputs, outputs, backend))
            .collect::<Result<Vec<_>, _>>()?;

        // Resolve output buffer references directly to pointers
        let output_ptrs: Vec<*mut u8> = op.output_refs
            .iter()
            .map(|buf_ref| self.resolve_buffer_ptr_mut(buf_ref, inputs, outputs, backend))
            .collect::<Result<Vec<_>, _>>()?;

        // Execute kernel with pointers
        self.execute_kernel_direct(i, &input_ptrs, &output_ptrs)?;
    }

    Ok(())
}
```

Add new helper methods:

```rust
/// Resolve BufferRef directly to a read-only pointer.
fn resolve_buffer_ptr(
    &self,
    buf_ref: &BufferRef,
    inputs: &[BufferHandle],
    outputs: &[BufferHandle],
    backend: &dyn ProgramBackend,
) -> BackendResult<*const u8> {
    match buf_ref {
        BufferRef::Input(idx) => {
            let handle = inputs.get(*idx)
                .ok_or_else(|| BackendError::invalid_config(
                    format!("Input buffer {} not provided", idx)
                ))?;

            backend.get_buffer_ptr(*handle)
                .map(|p| p as *const u8)
                .ok_or(BackendError::InvalidHandle(*handle))
        }

        BufferRef::Output(idx) => {
            let handle = outputs.get(*idx)
                .ok_or_else(|| BackendError::invalid_config(
                    format!("Output buffer {} not provided", idx)
                ))?;

            backend.get_buffer_ptr(*handle)
                .map(|p| p as *const u8)
                .ok_or(BackendError::InvalidHandle(*handle))
        }

        BufferRef::Workspace(slot) => {
            // Direct pointer access for workspace - no BufferHandle needed!
            self.workspace.as_ref()
                .and_then(|ws| {
                    let ptr = unsafe { ws.region_ptr(*slot) };
                    if ptr.is_null() {
                        None
                    } else {
                        Some(ptr as *const u8)
                    }
                })
                .ok_or_else(|| BackendError::invalid_config(
                    format!("Workspace slot {} not available", slot)
                ))
        }
    }
}

/// Resolve BufferRef directly to a mutable pointer.
fn resolve_buffer_ptr_mut(
    &self,
    buf_ref: &BufferRef,
    inputs: &[BufferHandle],
    outputs: &[BufferHandle],
    backend: &dyn ProgramBackend,
) -> BackendResult<*mut u8> {
    // Similar to resolve_buffer_ptr but returns *mut u8
    match buf_ref {
        BufferRef::Input(idx) => {
            let handle = inputs.get(*idx)
                .ok_or_else(|| BackendError::invalid_config(
                    format!("Input buffer {} not provided", idx)
                ))?;

            backend.get_buffer_ptr(*handle)
                .ok_or(BackendError::InvalidHandle(*handle))
        }

        BufferRef::Output(idx) => {
            let handle = outputs.get(*idx)
                .ok_or_else(|| BackendError::invalid_config(
                    format!("Output buffer {} not provided", idx)
                ))?;

            backend.get_buffer_ptr(*handle)
                .ok_or(BackendError::InvalidHandle(*handle))
        }

        BufferRef::Workspace(slot) => {
            self.workspace.as_ref()
                .and_then(|ws| unsafe { ws.region_ptr(*slot) })
                .filter(|p| !p.is_null())
                .ok_or_else(|| BackendError::invalid_config(
                    format!("Workspace slot {} not available", slot)
                ))
        }
    }
}
```

Update `execute_kernel` to accept pointers directly:

```rust
fn execute_kernel_direct(
    &mut self,
    op_idx: usize,
    input_ptrs: &[*const u8],
    output_ptrs: &[*mut u8],
) -> BackendResult<()> {
    let op = &self.plan.ops[op_idx];

    // Get kernel function from table
    let kernel_fn = match op.kernel_id {
        crate::plan::KernelId::GEMM_STANDARD | crate::plan::KernelId::GEMM_STRASSEN => {
            self.plan.kernel_table.gemm.get(op.kernel_idx).copied()
        }
        // ... other kernel types ...
        _ => self.plan.kernel_table.gemm.first().copied(),
    };

    let kernel_fn = kernel_fn.ok_or_else(|| {
        BackendError::invalid_config(format!(
            "Kernel index {} out of bounds for kernel_id {:?}",
            op.kernel_idx, op.kernel_id
        ))
    })?;

    // Execute kernel directly with pointers (no further resolution needed)
    kernel_fn(input_ptrs, output_ptrs, &op.params)?;

    Ok(())
}
```

### Option 2: Register Workspace Buffers (More Complex)

Alternatively, register workspace buffer handles with the backend so they can be resolved. This requires:
1. Creating BufferHandles for each workspace slot
2. Registering them in the backend's buffer map
3. Managing their lifecycle

This is more complex and adds overhead, so **Option 1 is recommended**.

## Expected Behavior

After the fix:

1. **Operations with workspace inputs work**:
   ```
   Op with input_refs=[Workspace(5), Workspace(10)]
   → Resolves to actual pointers from workspace
   → Kernel receives 2 valid pointers
   → Executes successfully
   ```

2. **Mixed buffer types work**:
   ```
   Op with input_refs=[Input(0), Workspace(20)]
   → Input(0) resolves via backend.get_buffer_ptr()
   → Workspace(20) resolves directly from workspace
   → Kernel receives 2 valid pointers
   → Executes successfully
   ```

3. **No silent failures**:
   - Errors are explicit if buffers can't be resolved
   - No silent dropping via filter_map

## Testing

After implementing the fix:

```bash
cd /workspace

# Rebuild hologram-onnx with updated hologram
cargo build --release

# Test T5 encoder
RUST_LOG=info cargo run --release -- run --config configs/test-encoder.toml
```

Expected output:
```
Operation breakdown: 232 no-input (constants), 135 single-input, 183 dual-input, 29 multi-input
Model execution completed successfully
Output: [1, 128, 512] tensor
```

## Files to Modify

- `/hologram/crates/backend/src/executor.rs` - Add direct pointer resolution methods

## Success Criteria

- [ ] T5 encoder executes without "GEMM kernel requires 2 inputs" error
- [ ] All 183 dual-input operations receive 2 pointers
- [ ] Workspace buffers are properly resolved
- [ ] Model produces output tensor with correct shape [1, seq_len, 512]
- [ ] No silent failures or None values dropped

## Priority

**CRITICAL** - This is the final blocker for ONNX model execution through hologram-onnx runtime.
