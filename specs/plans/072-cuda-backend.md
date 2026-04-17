# Plan 072: CUDA Compute Backend

**Status:** Design
**Created:** 2026-04-17
**Depends on:** Plan 067 (ComputeBackend + ComputeMemory rewrite)
**Informed by:** [dmriding/kaio](https://github.com/dmriding/kaio) — Rust-native NVIDIA kernel framework

## Problem

hologram supports Metal (macOS), WebGPU (browser/wgpu), and CPU backends but has
no NVIDIA GPU support. CUDA is the dominant GPU compute platform for ML inference
on Linux servers and Windows workstations. The `hologram-backend` crate has a
placeholder `cuda.rs` with a single TODO comment.

## Research: kaio Pattern Assessment

We evaluated kaio, a Rust NVIDIA kernel authoring framework that generates PTX at
compile time via proc macros and achieves 92.5% of cuBLAS on tensor-core matmul.

| Pattern | Verdict | Rationale |
|---------|---------|-----------|
| **Proc-macro PTX codegen** | Skip | High effort, breaks consistency with Metal/WebGPU embedded shader pattern. hologram embeds `include_str!` shaders — CUDA should follow the same model. |
| **Tensor-core matmul, FlashAttention, INT8** | Defer (Phase 4) | High value but requires a working backend first. `FloatOp::Attention` and `FloatOp::Gemm(quant_b)` already accommodate these — no IR changes needed. |
| **Auto-tuning / hardware detection** | Adopt (Phase 3) | Direct fit. Extend existing `GpuFamily`/`OpThresholds` for NVIDIA SM generations. Small effort, high value. |
| **cuBLAS via cudarc** | Adopt (Phase 4) | `cudarc` has optional `cublas` feature. Simpler path to high-perf matmul than hand-writing tensor-core PTX. |
| **Layered architecture** | Skip | hologram already has equivalent layering. |

## Architecture

### Driver Bindings: cudarc 0.19

Use `cudarc` (0.19.x) for CUDA driver API:
- **Dynamic loading** — resolves `libcuda.so` / `nvcuda.dll` at runtime
- **No CUDA toolkit at compile time** — CI and dev machines don't need nvcc
- **No CUDA toolkit at runtime** — PTX is JIT-compiled by the NVIDIA driver itself
- **PTX module loading** — `CudaDevice::load_ptx()` loads embedded PTX strings
- **Optional cuBLAS** — `cublas` feature for optimized matmul in Phase 4

### Kernel Source Strategy

Ship **pre-compiled PTX** checked into the repo alongside CUDA C source:

```
hologram-backend/src/kernels/
  metal.msl          # Metal shaders (existing)
  cuda_kernels.cu    # CUDA C source (human-readable, for maintenance)
  cuda_kernels.ptx   # Pre-compiled PTX (loaded at runtime, no toolkit needed)
```

- `cuda_kernels.ptx` is loaded via `include_str!` + `CudaDevice::load_ptx()`
- `cuda_kernels.cu` is the source of truth — recompile to `.ptx` via `nvcc -ptx`
  when kernels change (developer with toolkit does this, checks in result)
- No NVRTC dependency at runtime — pure driver API path
- PTX is forward-compatible: NVIDIA drivers JIT-compile PTX to native SASS

### Memory Model

NVIDIA GPUs have **discrete memory** (PCIe/NVLink separated from CPU RAM).
This differs from Metal's unified memory:

- `upload` → host-to-device copy (`htod_sync_copy`)
- `download` → device-to-host copy (`dtoh_sync_copy`)
- `mmap` → returns `None` (no zero-copy file mapping to discrete GPU)
- `alias` → clone the `CudaSlice` Arc (reference-counted, same device allocation)
- `evict` → drop the `CudaSlice`, driver reclaims VRAM

### Stream Management

- Phase 1-3: default stream (synchronous, simple)
- Phase 4+: named streams for compute/transfer overlap
  - Stream 0: kernel dispatch
  - Stream 1: async weight prefetch for next layer
  - `cudarc` supports `CudaStream` creation and stream-ordered operations

### Relationship to Plan 067

Plan 067 defines the `ComputeMemory` + `ComputeBackend<M>` trait system in
`hologram-backend`. This plan adds `CudaMemory` + `CudaBackend` as the fourth
implementation.

The exec-level wiring (Phase 2) targets the *current* exec-level trait in
`hologram-exec/src/backend/mod.rs`. This is **temporary scaffolding** — when
Plan 067 Phase 3 (single-path executor) lands, the exec-level wrapper is removed
and the `hologram-backend` trait is used directly. Don't over-invest in the
exec-level code.

## Implementation

### Phase 1: Foundation — hologram-backend CudaMemory + CudaBackend

**`hologram-backend/Cargo.toml`**:
```toml
cuda = ["dep:cudarc"]

[dependencies.cudarc]
version = "0.19"
optional = true
features = ["driver"]
```

**`hologram-backend/src/cuda.rs`** — full implementation:
- `CudaMemory` wrapping `Arc<cudarc::driver::CudaDevice>`
- `ComputeMemory for CudaMemory` with `Buffer = CudaSlice<u8>`
- `CudaBackend` with device, loaded PTX module, ring table device buffers
- `CudaBackend::new() -> Option<Self>` — init device 0, load PTX, extract kernel functions
- `ComputeBackend<CudaMemory> for CudaBackend` — dispatch via `FloatOp` match

**`hologram-backend/src/kernels/cuda_kernels.cu`** + **`cuda_kernels.ptx`**:

Initial kernel set (matching Metal parity, ~30 kernels):

| Category | Kernels |
|----------|---------|
| Unary elementwise | relu, neg, abs_val, sigmoid, silu, tanh_act, exp_act, reciprocal, gelu, erf_act, sin_act, cos_act, log_act, sqrt_act |
| Binary elementwise | add_op, mul_op, sub_op, div_op |
| Linear algebra | sgemm (naive tiled 16×16), batched_sgemm, im2col |
| Normalization | softmax, rms_norm, layer_norm, instance_norm |
| Data movement | transpose_4d, slice_copy, concat_copy, resize_nearest |
| Ring ops | ring_lut, ring_binary_lut |

All kernels use `blockDim = 256` threads, standard grid-stride loop pattern.
sgemm uses 16×16 shared-memory tiling (naive, no tensor cores — matching
Metal's naive sgemm).

### Phase 2: Exec-level wiring

**`hologram-exec/build.rs`** — add:
```rust
println!("cargo::rustc-check-cfg=cfg(has_cuda)");
if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
    println!("cargo:rustc-cfg=has_cuda");
}
```

**`hologram-exec/Cargo.toml`** — forward feature:
```toml
cuda = ["hologram-backend/cuda"]
```

**`hologram-exec/src/backend/mod.rs`**:
- Add `#[cfg(has_cuda)] pub mod cuda;`
- Add `GpuBuffer::Cuda(cudarc::driver::CudaSlice<u8>)` variant
- Add `BackendSelector::Cuda`
- Update priority: Metal > CUDA > WebGPU > CPU
- Implement `byte_len()`, `try_clone()`, `readback_into()` for Cuda variant

**`hologram-exec/src/backend/cuda.rs`** — new file:
- Wraps `hologram_backend::cuda::CudaBackend`
- Implements exec-level `ComputeBackend` trait
- `dispatch_float` → upload, dispatch, return `KernelOutput::GpuBuffer`
- `dispatch_float_chained` → accept `GpuInput::Gpu` for GPU-resident buffers
- `flush` → `device.synchronize()`

**`.github/workflows/ci.yml`** — add compile-only CUDA job:
```yaml
cuda-compile:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - run: cargo check --features cuda -p hologram-backend
    - run: cargo check --features cuda -p hologram-exec
```

### Phase 3: Hardware detection

**`hologram-exec/src/backend/hardware.rs`**:
- Add `GpuFamily` variants: `NvidiaTuring` (SM 7.5), `NvidiaAmpere` (SM 8.x), `NvidiaAda` (SM 8.9), `NvidiaHopper` (SM 9.0)
- Add `HardwareCaps` fields: `cuda_sm_version: Option<u32>`, `tensor_cores: bool`
- Add `#[cfg(has_cuda)]` detection via `CudaDevice::new(0)` for compute capability and VRAM
- Add per-SM `OpThresholds`:

| SM Gen | Elementwise | MatMul | Softmax | Norm |
|--------|------------|--------|---------|------|
| Hopper (9.0) | 256KB | 32×32 | 256KB | 256KB |
| Ada (8.9) | 512KB | 48×48 | 512KB | 512KB |
| Ampere (8.0) | 1MB | 64×64 | 1MB | 1MB |
| Turing (7.5) | 2MB | 96×96 | 2MB | 2MB |

### Phase 4: CUDA-specific optimizations (future)

Post-stub optimizations informed by kaio's approach:

- **cuBLAS matmul** — `cudarc` cublas feature, drop-in for sgemm. Simplest path to high performance.
- **Tensor-core matmul** — `wmma` / `mma.sync` instructions for f16 compute. kaio achieves 92.5% cuBLAS as reference.
- **FlashAttention kernel** — tiled attention with shared memory, online softmax. kaio's attention kernels are a good reference.
- **INT8 quantized matmul** — `mma.sync` INT8 path for Q8_0 weights.
- **Multi-stream execution** — overlap compute and weight prefetch.
- **cuDNN integration** — optional, for conv2d and normalization ops.

## Testing

**Compile-only CI** (ubuntu-latest, no GPU): `cargo check --features cuda`

**Runtime tests** (skip without GPU):
```rust
#[test]
#[cfg(has_cuda)]
fn cuda_relu_dispatch() {
    let mem = match CudaMemory::new() {
        Some(m) => m,
        None => return, // No NVIDIA GPU
    };
    // ... verify output matches CPU reference
}
```

**Numerical validation**: all CUDA outputs must match CPU reference within
tolerance (same bounds as Metal tests). CPU backend is the reference
implementation.

## Files

| File | Action |
|------|--------|
| `hologram-backend/Cargo.toml` | Add cudarc 0.19 optional dep |
| `hologram-backend/src/cuda.rs` | Replace stub → CudaMemory + CudaBackend |
| `hologram-backend/src/kernels/cuda_kernels.cu` | New — CUDA C kernel source |
| `hologram-backend/src/kernels/cuda_kernels.ptx` | New — pre-compiled PTX |
| `hologram-exec/build.rs` | Add has_cuda cfg |
| `hologram-exec/Cargo.toml` | Forward cuda feature |
| `hologram-exec/src/backend/mod.rs` | Wire GpuBuffer::Cuda, BackendSelector::Cuda |
| `hologram-exec/src/backend/cuda.rs` | New — exec-level CUDA backend |
| `hologram-exec/src/backend/hardware.rs` | NVIDIA GpuFamily + OpThresholds |
| `.github/workflows/ci.yml` | Add cuda-compile job |
