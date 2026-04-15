# Plan 066: Single-Backend Execution Path

**Status:** Planned
**Created:** 2026-04-15
**Depends on:** Plan 065 (GPU buffer chaining)
**Goal:** Eliminate GPU↔CPU transitions by executing ALL ops on a single backend.

## Problem

The current hybrid execution (try GPU → fall back to CPU) creates ~15s of readback overhead per SD UNet forward pass. CPU kernel time is only 3.8s, but total execution is 19s because every GPU→CPU transition requires flush + readback.

## Architecture

### Single Dispatch Path

The tape executor selects a backend at the start and dispatches ALL ops through it:

```
if Metal available → all ops through MetalBackend
if WebGPU available → all ops through WebGpuBackend  
else → all ops through CpuBackend
```

No fallback to CPU for individual ops. Every buffer stays in the backend's native format for the entire execution.

### `ComputeBackend::dispatch_kernel`

New trait method that handles any `TapeKernel`:

```rust
fn dispatch_kernel(
    &self,
    kernel: &TapeKernel,
    inputs: &[GpuInput<'_>],
    out_buf: &mut OutputBuffer,
) -> ExecResult<KernelOutput>;
```

The default implementation falls back to CPU dispatch for each kernel type.
Metal backend overrides for all supported ops.

### Metal Kernels Needed (SD UNet)

| Op | Calls | Metal Kernel | Notes |
|----|-------|-------------|-------|
| Conv2d | 97 | im2col + SGEMM | ✓ Already have |
| Transpose | 192 | Permute indices | Buffer layout transform |
| InstanceNorm | 61 | Mean/var + normalize | Per-channel stats |
| Reshape | 369 | No-op | Same bytes, reinterpret shape |
| LayerNorm | 48 | Mean/var + normalize | Like RmsNorm + mean subtraction |
| Slice | 32 | Copy sub-range | Strided copy |
| Mul | 74 | Binary elementwise | ✓ Already have |
| Add | 50 | Binary elementwise | ✓ Already have |
| Gemm | 24 | SGEMM | ✓ Already have |
| Sigmoid | 17 | Unary elementwise | ✓ Already have |
| Concat | 14 | Memcpy regions | Concatenate buffers |
| Resize | 3 | Bilinear interp | Spatial upsampling |
| Softmax | 3 | Row-wise softmax | ✓ Already have |
| Erf | 16 | Unary elementwise | ✓ Already have |

### Buffer Management

- All intermediate buffers are `GpuBuffer` (Metal buffers on macOS)
- Reshape: return the same Metal buffer with different shape metadata
- Slice: new Metal buffer with copied sub-range
- Concat: new Metal buffer, copy from input buffers
- Transpose: new Metal buffer, permuted copy

### Expected Performance

With all ops on Metal and zero GPU↔CPU transitions:
- Kernel time: dominated by GPU compute (~3s for MatMul + Conv2d)
- Total time: **~3-5s** (target)
- Current: 19s (3.8s kernel + 15s readback overhead)

## Files

- `hologram/crates/hologram-exec/src/backend/mod.rs` — dispatch_kernel trait method
- `hologram/crates/hologram-exec/src/backend/metal.rs` — all Metal kernel implementations
- `hologram/crates/hologram-exec/src/tape.rs` — single-backend execution path
