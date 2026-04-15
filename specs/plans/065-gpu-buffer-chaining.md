# Plan 065: GPU Buffer Chaining for All Compute Backends

**Status:** In Progress
**Created:** 2026-04-15
**Goal:** Enable consecutive GPU ops to chain without CPU readback, closing the SD UNet performance gap from 20s to ~3s.

## Problem

The tape executor passes `&[u8]` between ops. When a MatMul produces a Metal GPU buffer, the next op (Add/Mul) must read the result back to CPU just to pass it as `&[u8]`. This CPU roundtrip per op dominates execution time:

- MatMul kernel time: **0.57s** (Metal GPU)
- Total UNet execution: **20s** (19.4s is readback + CPU elementwise)
- DiffusionBee target: **~3s** (everything stays on GPU)

## Design

### 1. `GpuBuffer` — Backend-agnostic GPU buffer handle

```rust
pub enum GpuBuffer {
    Metal(metal::Buffer),
    Wgpu(wgpu::Buffer),
}
```

Replaces `#[cfg(has_metal)] MetalBuffer(metal::Buffer)` in `KernelOutput`. Provides `byte_len()` and `readback_into()`.

### 2. `GpuInput<'a>` — Dual-path input for dispatch

```rust
pub enum GpuInput<'a> {
    Cpu(&'a [u8]),
    Gpu(&'a GpuBuffer),
}
```

Enables mixed GPU+CPU inputs (e.g., GPU activation + CPU weight).

### 3. `dispatch_*_chained` trait methods with backward-compatible defaults

```rust
fn dispatch_float_chained(&self, op, inputs: &[GpuInput], out_buf) -> KernelOutput {
    // Default: readback GPU inputs to CPU, delegate to dispatch_float
}
```

- CPU backend: default impl works unchanged
- Metal backend: overrides to pass `metal::Buffer` directly (zero-copy GPU→GPU)
- WebGPU backend: default impl until chained path is needed

### 4. Tape executor changes

- `gpu_bufs: Vec<Option<GpuBuffer>>` replaces `#[cfg(has_metal)] metal_bufs`
- Build `SmallVec<[GpuInput; 4]>` for each instruction's inputs
- Route float/matmul ops through `dispatch_*_chained`
- Lazy readback only when CPU-only ops consume GPU results

### 5. Metal backend: `dispatch_*_gpu` helpers

Internal helpers that accept `&metal::Buffer` directly instead of `&[u8]`:
- `dispatch_unary_gpu(pipeline, input: &metal::Buffer) -> metal::Buffer`
- `dispatch_binary_gpu(pipeline, a: &metal::Buffer, b: &metal::Buffer) -> metal::Buffer`

Eliminates `new_buffer_with_data` copies for GPU→GPU transitions.

## Expected Performance Impact

| Phase | UNet Forward |
|-------|-------------|
| Before (CPU-only) | >600s |
| Phase 1 (batch inflation fix) | ~20s |
| Phase 2 (Metal MatMul + lazy readback) | ~20s (readback bottleneck) |
| Phase 3 (GPU buffer chaining) | **~3-5s** (target) |

## Files

- `hologram/crates/hologram-exec/src/backend/mod.rs` — GpuBuffer, GpuInput, KernelOutput, trait methods
- `hologram/crates/hologram-exec/src/backend/metal.rs` — chained dispatch overrides
- `hologram/crates/hologram-exec/src/tape.rs` — executor gpu_bufs, GpuInput routing, elementwise GPU dispatch
- `hologram/crates/hologram-exec/src/backend/cpu.rs` — no changes (default impls)
