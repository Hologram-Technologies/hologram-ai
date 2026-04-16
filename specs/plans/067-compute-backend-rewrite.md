# Plan 067: ComputeBackend + ComputeMemory Rewrite

**Status:** Design
**Created:** 2026-04-16
**Replaces:** Plans 065, 066 (GPU buffer chaining, single-backend execution)

## Problem

hologram-exec's tape executor is CPU-first with GPU bolted on. The result:
- 587ms of actual compute, 24s of CPU↔GPU sync overhead
- ~112 ops fall back to CPU, each triggering a full GPU pipeline flush
- GpuBuffer/GpuInput/readback logic adds complexity without solving the problem
- Data constantly moves between CPU and GPU memory

## Architecture: Device-Native Execution

### Core Principle

**All data lives on one device. All computation happens on that device.**

No CPU↔GPU transfers during execution. When Metal is the backend, weights load
directly into Metal buffers. LUT tables (from uor-foundation) build directly
on GPU. Intermediate activations stay in Metal memory. The only CPU involvement
is launching the command buffer and reading back the final output.

### `ComputeMemory` — Device Memory Abstraction

```rust
/// Manages tensor allocation on a specific device.
pub trait ComputeMemory: Send + Sync {
    type Buffer: Send + Sync;

    /// Allocate a buffer of `byte_len` bytes on this device.
    fn alloc(&self, byte_len: usize) -> Self::Buffer;

    /// Load data from CPU bytes into a device buffer.
    /// Called once at initialization (weights, constants), never during execution.
    fn upload(&self, data: &[u8]) -> Self::Buffer;

    /// Read device buffer back to CPU bytes.
    /// Called once at the end (output tensor), never during execution.
    fn download(&self, buf: &Self::Buffer) -> Vec<u8>;

    /// Zero-copy reshape: return the same buffer with different metadata.
    fn reshape(&self, buf: &Self::Buffer) -> Self::Buffer;
}
```

Implementations:
- `CpuMemory` — `Buffer = Vec<u8>`, upload/download are no-ops
- `MetalMemory` — `Buffer = metal::Buffer`, upload creates SharedStorage buffer
- `WebGpuMemory` — `Buffer = wgpu::Buffer`, upload uses staging

### `ComputeBackend<M: ComputeMemory>` — Device Computation

```rust
/// Dispatches tensor operations on a specific device.
///
/// Every backend implements the full UOR computational model:
/// - Ring arithmetic (Z/256Z, LUT-based transforms)
/// - Float ops (matmul, conv2d, normalization, elementwise)
/// - Data movement (transpose, slice, concat, reshape)
///
/// UOR LUT tables are loaded onto the device via ComputeMemory::upload
/// at initialization and stay resident for the lifetime of execution.
pub trait ComputeBackend<M: ComputeMemory>: Send + Sync {
    /// Execute a single kernel with device-native buffers.
    fn dispatch(
        &self,
        kernel: &TapeKernel,
        inputs: &[&M::Buffer],
        output: &mut M::Buffer,
        params: &KernelParams,
    ) -> ExecResult<()>;

    /// Load UOR ring LUT tables onto the device.
    /// Called once at initialization. The tables stay on-device for all
    /// subsequent ring op dispatches.
    fn load_ring_tables(&mut self, tables: &[&[u8; 256]], memory: &M);

    /// Flush pending work (GPU command buffer commit + wait).
    fn flush(&self);
}
```

Implementations:
- `CpuBackend` — the existing CPU dispatch code, unchanged
- `MetalBackend` — ALL ops dispatch as Metal compute shaders
- `WebGpuBackend` — ALL ops dispatch as WebGPU compute shaders

### Tape Executor — Single Path

```rust
pub fn execute<M: ComputeMemory, B: ComputeBackend<M>>(
    tape: &EnumTape,
    memory: &M,
    backend: &B,
    weights: &[M::Buffer],  // pre-uploaded to device
    inputs: &[M::Buffer],   // pre-uploaded to device
) -> ExecResult<Vec<M::Buffer>> {
    let mut bufs: Vec<M::Buffer> = (0..tape.num_slots())
        .map(|_| memory.alloc(0))
        .collect();

    // Seed with weights and inputs (already on device).
    seed_buffers(&mut bufs, weights, inputs);

    // Execute: one dispatch per instruction, all on device.
    for instr in &tape.instructions {
        let input_refs: Vec<&M::Buffer> = instr.input_indices
            .iter()
            .map(|&idx| &bufs[idx as usize])
            .collect();

        backend.dispatch(
            &instr.kernel,
            &input_refs,
            &mut bufs[instr.output_idx as usize],
            &instr.params,
        )?;
    }

    // Single flush at the end (GPU: commit + wait).
    backend.flush();

    // Download outputs to CPU.
    Ok(tape.output_indices().iter()
        .map(|&idx| bufs[idx as usize].clone())
        .collect())
}
```

### Metal Kernel Coverage

Every TapeKernel variant needs a Metal shader. Current coverage:

| Category | Kernels | Metal Shader |
|----------|---------|-------------|
| MatMul | sgemm, batched_sgemm | ✓ |
| Conv2d | im2col + sgemm | ✓ |
| Elementwise unary | relu, sigmoid, silu, tanh, gelu, erf, exp, neg, abs, reciprocal | ✓ |
| Elementwise binary | add, mul, sub, div | ✓ |
| Normalization | rms_norm, layer_norm, instance_norm, softmax | ✓ |
| Data movement | transpose_4d, slice_copy, concat_copy, resize_nearest | ✓ |
| No-op | reshape (same buffer, different metadata) | ✓ (zero-copy) |
| **Missing** | **LUT-GEMM Q4/Q8** (for quantized weights) | **Need GPU dequant+GEMM** |
| **Missing** | **Ring ops** (byte-domain Z/256Z) | **Need GPU kernels or remove** |

### Weight Loading

Weights are loaded directly into device memory at archive load time:

```rust
fn load_weights_to_device<M: ComputeMemory>(
    plan: &LoadedPlan,
    memory: &M,
) -> Vec<M::Buffer> {
    plan.constants().iter()
        .map(|(_, data)| memory.upload(data))
        .collect()
}
```

For Metal: this creates `metal::Buffer` with `StorageModeShared` for each weight
tensor. The weights never exist as CPU `Vec<u8>` during execution.

### LUT Tables

LUT tables (from uor-foundation ring arithmetic) are built directly on the
target device:

```rust
fn build_lut_table<M: ComputeMemory>(
    table: &[u8; 256],
    memory: &M,
) -> M::Buffer {
    memory.upload(table)
}
```

For byte-domain ring ops, the LUT table lives on GPU and the ring kernel
reads from it. No CPU lookup during execution.

## Implementation Plan

### Phase 1: ComputeMemory + CpuMemory (non-breaking)
- Define `ComputeMemory` trait in `hologram-exec`
- Implement `CpuMemory` using `Vec<u8>` buffers
- Verify all existing tests pass (CpuMemory is a transparent wrapper)

### Phase 2: MetalMemory + weight upload
- Implement `MetalMemory` using `metal::Buffer`
- Add `load_weights_to_device` for Metal weight loading
- Verify Metal buffers are created at load time, not during execution

### Phase 3: Single-path executor
- New `execute()` function parameterized by `<M, B>`
- All ops dispatch through `backend.dispatch()` — no fallback
- One flush at the end
- Keep `execute_direct()` as legacy path for backwards compatibility

### Phase 4: Complete Metal kernel coverage
- Q4 dequant+GEMM kernel for Conv2dLut4/MatMulLut4
- Ring op kernels (or convert ring ops to float at compile time)
- Remove CPU fallback path

### Phase 5: Remove legacy code
- Remove `GpuBuffer`, `GpuInput`, `dispatch_*_chained`
- Remove `gpu_bufs`, lazy readback, threshold checks
- Remove dual-path execution in `execute_direct`
- `hologram-exec` becomes clean single-path

## Expected Performance

With all data on GPU and one flush at the end:
- No CPU↔GPU sync overhead during execution
- GPU command buffer encodes all ~1651 ops in one pass
- Single commit + wait_until_completed at the end
- Expected: **~2-5s** for SD v1.5 UNet (vs current 24s)

## Crate Structure

New crate: **`hologram-backend`** — contains all device abstractions.
`hologram-exec` becomes a thin executor that consumes `hologram-backend`.

```
hologram-backend/
  src/
    lib.rs           — ComputeMemory, ComputeBackend traits
    cpu.rs           — CpuMemory + CpuBackend (SIMD, Accelerate BLAS)
    metal.rs         — MetalMemory + MetalBackend (Apple GPU)
    webgpu.rs        — WebGpuMemory + WebGpuBackend (browser + wgpu)
    kernels/         — shared kernel definitions (shader source, params)
```

### Platform Support (priority order)
1. **macOS** — MetalBackend (Apple Silicon GPU)
2. **WASM** — WebGpuBackend (browser WebGPU API)
3. **x86_64** — CpuBackend (AVX2/FMA + optional MKL)
4. **iOS** — MetalBackend (shared with macOS)

### WebGPU / WASM Considerations
- WebGPU is async (command buffer submit → poll → readback). The
  `ComputeMemory::download` and `ComputeBackend::flush` methods need
  async variants or callback-based APIs for the WASM target.
- WGSL shader source replaces MSL for WebGPU kernels. Shared kernel
  logic can be templated or dual-compiled.
- `wasm32-unknown-unknown` target: no std::thread, no mmap. Memory
  management uses WebGPU's buffer mapping API.

## Files

- `hologram/crates/hologram-backend/` — new crate for device abstractions
- `hologram/crates/hologram-exec/` — simplified executor consuming hologram-backend
- `hologram/crates/hologram-exec/src/tape.rs` — keep as legacy, eventually thin wrapper
