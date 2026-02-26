# Claude Instructions for hologram-ai

## Project Overview

hologram-ai provides:
1. **ONNX Model Compilation** - Compile and execute ANY ONNX model via hologram backend
2. **Optimized AI Kernels** (Future) - Inject hardware-specific kernels (cuBLAS, oneDNN, Metal) via `KernelOverride`

**Repository:** https://github.com/uor-framework/hologram-ai
**License:** MIT OR Apache-2.0

## Current Status

| Model            | Status        | Notes                               |
| ---------------- | ------------- | ----------------------------------- |
| ResNet18         | WORKING       | End-to-end image classification     |
| T5               | IN PROGRESS   | Buffer fix done; encoder precision needs work |
| Stable Diffusion | FUTURE        | Complex multi-model pipeline        |
| Kernel Overrides | NOT STARTED   | See Goal 2 for implementation guide |

### T5 Status Details

- **Buffer indexing bug**: ✅ FIXED in hologram commit `e28fa52`
  - Before fix: correlation ~0.0001 (values at wrong positions)
  - After fix: correlation ~0.91 (values in correct positions)
- **Encoder**: ⚠️ Working but ~9% numerical difference from ONNX Runtime
  - RMSNorm: exact match
  - Full encoder: r=0.91 (close but not exact)
- **Decoder**: ⚠️ Runs but produces different tokens due to encoder differences
  - ONNX picks token 3, Hologram picks token 32079 at position 1
  - Encoder precision cascades to decoder output
- **End-to-end generation**: ❌ Produces gibberish ("started 2016 panhol...")
- **Vocab size bug**: ✅ Fixed - now infers 32128 from model output
- **no_repeat_ngram**: ✅ Fixed - added to config-based pipeline
- **Where broadcast**: ✅ Fixed in hologram
- **Gather (embedding)**: ✅ Verified correct - matches ONNX exactly
- **Remaining work**: Debug attention/softmax numerical precision to get encoder r>0.99

## Architecture

```
+-------------------------------------------------------------+
|                      hologram-ai                            |
|  ┌─────────────────────────────────────────────────────┐   |
|  │ hologram-ai-onnx: ONNX → .holo compilation          │   |
|  │ hologram-ai-common: Shared serialization/utilities  │   |
|  │ hologram-ai: CLI, runtime, config management        │   |
|  └─────────────────────────────────────────────────────┘   |
|                                                             |
|  Future: KernelOverride implementations                     |
|  - cuBLAS/cuDNN for NVIDIA GPUs                            |
|  - oneDNN for Intel/AMD CPUs (AVX-512)                     |
|  - MPS/Metal for Apple Silicon                             |
+----------------------------+--------------------------------+
                             | depends on
+----------------------------v--------------------------------+
|                       hologram                              |
|  - hologram_ir: Intermediate representation                |
|  - hologram-backend: Execution with KernelOverride trait   |
|  - SIMD-optimized fallback kernels                         |
+-------------------------------------------------------------+
```

### Key Principle

**hologram has ZERO knowledge of hologram-ai.** The dependency is strictly one-way. hologram-ai registers optimized kernels at runtime via `register_kernel_override()`.

## Project Goals

### Goal 1: Compile and Execute ANY ONNX Model

**This is the north star for all development.**

#### Model Progression

We're building ONNX support incrementally:

1. **ResNet18** (Image Classification) - WORKING
   - Basic CNN architecture
   - Conv2D, BatchNorm, ReLU, MaxPool, GlobalAvgPool, MatMul
   - Single input (image), single output (class logits)

2. **T5** (Text-to-Text) - CURRENT FOCUS
   - Encoder-decoder transformer architecture
   - Attention, LayerNorm, Embedding, Softmax
   - Multiple inputs (encoder input, decoder input)
   - Autoregressive generation
   - We need to pass a prompt to T5 and ask for a joke in English that responds in English

3. **Stable Diffusion** (Image Generation) - FUTURE
   - U-Net architecture with cross-attention
   - VAE encoder/decoder
   - CLIP text encoder
   - Complex multi-model pipeline

#### Success Criteria

A model is considered "working" when:
- ONNX → .holo compilation succeeds
- Inference produces numerically correct outputs
- Performance is reasonable (not orders of magnitude slower than reference)

### Goal 2: Hardware-Optimized Kernels (Future)

| Operation   | cuBLAS        | oneDNN       | Metal                   | Priority |
| ----------- | ------------- | ------------ | ----------------------- | -------- |
| MatMul      | sgemm/dgemm   | matmul       | MPSMatrixMultiplication | High     |
| BatchMatMul | gemmBatched   | batch_matmul | -                       | High     |
| Conv2D      | cuDNN         | convolution  | MPSCNNConvolution       | High     |
| LayerNorm   | custom kernel | layer_norm   | -                       | Medium   |
| Softmax     | custom kernel | softmax      | MPSSoftmax              | Medium   |
| GELU/SiLU   | custom kernel | eltwise      | -                       | Low      |

## Workspace Structure

```
hologram-ai/
├── Cargo.toml                    # Workspace manifest
├── CLAUDE.md                     # This file
├── crates/
│   ├── hologram-ai/              # Main crate: CLI, runtime, config
│   │   ├── src/
│   │   │   ├── cli/              # Command-line interface
│   │   │   ├── config/           # TOML pipeline configuration
│   │   │   ├── runtime/          # .holo execution
│   │   │   └── tokenizers/       # Pure Rust tokenizer implementations
│   │   └── tests/
│   ├── hologram-ai-onnx/         # ONNX compilation
│   │   ├── src/
│   │   │   ├── ops/              # ONNX operation translators
│   │   │   ├── parser.rs         # ONNX protobuf parsing
│   │   │   └── builder.rs        # hologram_ir graph building
│   │   └── examples/
│   └── hologram-ai-common/       # Shared utilities
│       └── src/
│           └── serialization.rs  # .holo file format
│   # Future kernel extensions (when Goal 2 begins):
│   # └── hologram-backend-extensions/  # KernelOverride implementations
│   #     ├── src/
│   #     │   ├── lib.rs              # Re-exports, init()
│   #     │   ├── detect.rs           # Hardware detection
│   #     │   └── provider.rs         # OptimizedKernels struct
│   #     └── backends/
│   #         ├── cuda/               # NVIDIA cuBLAS/cuDNN
│   #         │   ├── mod.rs
│   #         │   ├── context.rs
│   #         │   └── matmul.rs
│   #         ├── onednn/             # Intel oneDNN (CPU)
│   #         │   ├── mod.rs
│   #         │   ├── matmul.rs
│   #         │   └── layer_norm.rs
│   #         └── metal/              # Apple MPS
│   #             ├── mod.rs
│   #             └── matmul.rs
├── examples/                     # Model configs (T5, etc.)
├── specs/                        # Design documents and plans
│   └── plans/                    # Prompts for hologram team
└── tests/                        # Integration tests
```

## Build Commands

```bash
# Build everything
cargo build

# Build specific crate
cargo build -p hologram-ai-onnx

# Run tests
cargo test

# Check with clippy
cargo clippy --all-targets

# Generate docs
cargo doc --no-deps --open

# Run with feature flags (future kernel support)
cargo build -p hologram-backend-extensions --features cuda      # CUDA support
cargo build -p hologram-backend-extensions --features onednn    # oneDNN support
cargo build -p hologram-backend-extensions --features metal     # Metal support
cargo build -p hologram-backend-extensions --features all       # All backends
```

## Core Architecture Principles

**CRITICAL: This is the foundational principle of hologram-ai**

### Core Philosophy

**Everything runs through hologram.** The entire point of hologram is to be a unified computational compiler and runtime. This principle is non-negotiable.

### What This Means

1. **No External Runtime Dependencies for Core Functionality**
   - Do NOT add dependencies like `tokenizers`, `ndarray`, `candle`, etc. for runtime execution
   - All computational operations must compile to hologram IR
   - All execution must go through hologram backend
   - External crates are acceptable ONLY for:
     - Build-time tools (prost-build, etc.)
     - Development utilities (testing, benchmarking)
     - Data loading/parsing (serde, image loading, etc.)

2. **Compilation Target: .holo Files**
   - Tokenizers compile to .holo → execute on hologram backend
   - Models compile to .holo → execute on hologram backend
   - Post-processing compiles to .holo → execute on hologram backend
   - Everything is a computational graph executed by hologram

3. **Temporary Pure Rust Implementations**
   - When hologram_ir lacks necessary operations (Gather, String ops, etc.):
     - Implement algorithms in **pure Rust** (std library only)
     - Document as bridge until hologram_ir gains operations
     - Plan migration path to full hologram_ir implementation
   - Example: SentencePiece tokenizer implemented in pure Rust until hologram_ir supports string operations

4. **The Vision**
   ```
   Everything is a .holo file:
   ├── tokenizer.holo       (text → tokens)   Future: Full hologram_ir
   ├── encoder.holo         (tokens → hidden)  ✅ Working now
   ├── decoder.holo         (hidden → logits)  ✅ Working now
   └── post_process.holo    (logits → output)  Future: Full hologram_ir

   All execute on hologram backend.
   All benefit from hologram optimizations.
   All are config-driven and cacheable.
   ```

5. **Implementation Guidelines**
   - When implementing new functionality (tokenizers, custom ops, etc.):
     - First: Check if hologram_ir operations exist
     - If YES: Implement via hologram IR compilation
     - If NO: Implement in pure Rust (std only), document as bridge
     - Never: Add external runtime dependencies
   - Create issues/plans for hologram_ir enhancements needed
   - Maintain compilation to .holo even for bridge implementations

### Why This Matters

- **Unified optimization**: All operations benefit from hologram's SIMD kernels
- **Zero-copy execution**: Hologram workspace management
- **Consistent architecture**: One backend, one format, one execution model
- **Future-proof**: When hologram_ir gains operations, migrate seamlessly

### Examples

**✅ CORRECT - Pure Rust Implementation**:
```rust
// Implement SentencePiece unigram algorithm in pure Rust
// Uses only std::collections, no external tokenizer crates
impl SentencePieceTokenizer {
    fn tokenize_unigram(&self, text: &str) -> Vec<u32> {
        // Full Viterbi implementation in pure Rust
        // ...
    }
}
```

**❌ WRONG - External Runtime Dependency**:
```rust
use tokenizers::Tokenizer;  // NO! External runtime dep

impl SentencePieceTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.hf_tokenizer.encode(text, false)  // NO!
    }
}
```

## Kernel Override Implementation Guide (Future)

This section provides implementation guidance for Goal 2: Hardware-Optimized Kernels. **This work has not started yet** - the guide is provided as reference for when kernel work begins.

### KernelOverride Trait Reference

From `hologram-backend`, the trait you must implement:

```rust
pub trait KernelOverride: Send + Sync {
    /// Return Some(Ok(())) to use your implementation
    /// Return Some(Err(e)) to report an error
    /// Return None to fall back to hologram's default

    fn matmul(&self, a: &[f32], b: &[f32], c: &mut [f32],
              m: usize, k: usize, n: usize) -> Option<Result<()>>;

    fn batch_matmul(&self, a: &[f32], b: &[f32], c: &mut [f32],
                    batch: usize, m: usize, k: usize, n: usize) -> Option<Result<()>>;

    fn layer_norm(&self, input: &[f32], scale: &[f32], bias: &[f32], output: &mut [f32],
                  num_instances: usize, normalized_size: usize, epsilon: f32) -> Option<Result<()>>;

    fn softmax(&self, input: &[f32], output: &mut [f32],
               outer_size: usize, axis_size: usize, inner_size: usize) -> Option<Result<()>>;

    fn conv2d(&self, input: &[f32], weight: &[f32], bias: Option<&[f32]>, output: &mut [f32],
              batch: usize, in_channels: usize, out_channels: usize,
              in_height: usize, in_width: usize,
              kernel_h: usize, kernel_w: usize,
              stride_h: usize, stride_w: usize,
              pad_h: usize, pad_w: usize) -> Option<Result<()>>;
}
```

### Core Provider Pattern

```rust
// crates/hologram-backend-extensions/src/provider.rs
use hologram_backend::{register_kernel_override, KernelOverride, Result};

#[cfg(feature = "cuda")]
use crate::backends::cuda::CudaKernels;
#[cfg(feature = "onednn")]
use crate::backends::onednn::OneDnnKernels;
#[cfg(feature = "metal")]
use crate::backends::metal::MetalKernels;

pub struct OptimizedKernels {
    #[cfg(feature = "cuda")]
    cuda: Option<CudaKernels>,
    #[cfg(feature = "onednn")]
    onednn: Option<OneDnnKernels>,
    #[cfg(feature = "metal")]
    metal: Option<MetalKernels>,
}

impl OptimizedKernels {
    /// Auto-detect available hardware and initialize backends
    pub fn detect() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            cuda: CudaKernels::new().ok(),
            #[cfg(feature = "onednn")]
            onednn: OneDnnKernels::new().ok(),
            #[cfg(feature = "metal")]
            metal: MetalKernels::new().ok(),
        }
    }

    /// Check what backends are available
    pub fn available_backends(&self) -> Vec<&'static str> {
        let mut backends = vec![];
        #[cfg(feature = "cuda")]
        if self.cuda.is_some() { backends.push("cuda"); }
        #[cfg(feature = "onednn")]
        if self.onednn.is_some() { backends.push("onednn"); }
        #[cfg(feature = "metal")]
        if self.metal.is_some() { backends.push("metal"); }
        backends
    }
}

impl KernelOverride for OptimizedKernels {
    fn matmul(&self, a: &[f32], b: &[f32], c: &mut [f32],
              m: usize, k: usize, n: usize) -> Option<Result<()>> {
        // Priority: CUDA > Metal > oneDNN > None (hologram default)
        #[cfg(feature = "cuda")]
        if let Some(ref cuda) = self.cuda {
            return Some(cuda.matmul(a, b, c, m, k, n));
        }
        #[cfg(feature = "metal")]
        if let Some(ref metal) = self.metal {
            return Some(metal.matmul(a, b, c, m, k, n));
        }
        #[cfg(feature = "onednn")]
        if let Some(ref onednn) = self.onednn {
            return Some(onednn.matmul(a, b, c, m, k, n));
        }
        None  // Fall back to hologram's SIMD kernels
    }

    // ... implement other methods with same priority pattern
}
```

```rust
// crates/hologram-backend-extensions/src/lib.rs
mod provider;
pub mod backends;

pub use provider::OptimizedKernels;

use hologram_backend::register_kernel_override;
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize hologram-backend-extensions with auto-detected hardware.
///
/// Call this once at application startup before executing any hologram plans.
/// Safe to call multiple times - only the first call has effect.
///
/// # Example
///
/// ```rust
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     hologram_backend_extensions::init()?;
///
///     // Now run hologram plans - they'll use optimized kernels automatically
///     let backend = hologram_backend::cpu::CpuBackend::new();
///     // ...
///     Ok(())
/// }
/// ```
pub fn init() -> Result<(), &'static str> {
    let mut result = Ok(());
    INIT.call_once(|| {
        let kernels = OptimizedKernels::detect();
        tracing::info!("hologram-backend-extensions initialized with: {:?}", kernels.available_backends());
        result = register_kernel_override(Box::new(kernels));
    });
    result
}
```

### oneDNN Implementation Guide

oneDNN works on CPU (Intel, AMD, ARM) without requiring GPU drivers. Recommended starting point.

```rust
// crates/hologram-backend-extensions/backends/onednn/matmul.rs
use hologram_backend::Result;
use onednn::{engine::Engine, memory, primitive, stream::Stream};

pub struct OneDnnMatMul {
    engine: Engine,
    stream: Stream,
}

impl OneDnnMatMul {
    pub fn new() -> std::result::Result<Self, onednn::error::Error> {
        let engine = Engine::new(onednn::engine::Kind::Cpu, 0)?;
        let stream = Stream::new(&engine)?;
        Ok(Self { engine, stream })
    }

    pub fn execute(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        // Create memory descriptors
        let a_md = memory::Descriptor::new(&[m as i64, k as i64], memory::DataType::F32)?;
        let b_md = memory::Descriptor::new(&[k as i64, n as i64], memory::DataType::F32)?;
        let c_md = memory::Descriptor::new(&[m as i64, n as i64], memory::DataType::F32)?;

        // Create matmul primitive descriptor
        let matmul_pd = primitive::matmul::PrimitiveDescriptor::new(
            &self.engine,
            &a_md, &b_md, &c_md,
            None, // No attributes
        )?;

        // Create memory objects (zero-copy from slices)
        let a_mem = memory::Memory::new(&self.engine, &a_md, a.as_ptr() as *mut _)?;
        let b_mem = memory::Memory::new(&self.engine, &b_md, b.as_ptr() as *mut _)?;
        let c_mem = memory::Memory::new(&self.engine, &c_md, c.as_mut_ptr() as *mut _)?;

        // Execute
        let matmul = primitive::Primitive::new(&matmul_pd)?;
        matmul.execute(&self.stream, &[
            (primitive::matmul::Arg::Src, &a_mem),
            (primitive::matmul::Arg::Weights, &b_mem),
            (primitive::matmul::Arg::Dst, &c_mem),
        ])?;

        self.stream.wait()?;
        Ok(())
    }
}
```

### CUDA/cuBLAS Implementation Guide

For NVIDIA GPUs. Requires CUDA toolkit installed.

```rust
// crates/hologram-backend-extensions/backends/cuda/matmul.rs
use cublas_sys::*;

impl CudaContext {
    /// C = A @ B using cuBLAS sgemm
    /// Note: cuBLAS uses column-major, hologram uses row-major
    /// We compute C^T = B^T @ A^T to avoid transpose overhead
    pub fn matmul(
        &self,
        a: &[f32],        // Host memory (m x k, row-major)
        b: &[f32],        // Host memory (k x n, row-major)
        c: &mut [f32],    // Host memory (m x n, row-major)
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        // Allocate device memory
        let mut d_a: *mut f32 = std::ptr::null_mut();
        let mut d_b: *mut f32 = std::ptr::null_mut();
        let mut d_c: *mut f32 = std::ptr::null_mut();

        unsafe {
            cudaMalloc(&mut d_a as *mut _ as *mut _, (m * k * 4) as usize);
            cudaMalloc(&mut d_b as *mut _ as *mut _, (k * n * 4) as usize);
            cudaMalloc(&mut d_c as *mut _ as *mut _, (m * n * 4) as usize);

            // Copy to device
            cudaMemcpy(d_a as _, a.as_ptr() as _, m * k * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b as _, b.as_ptr() as _, k * n * 4, cudaMemcpyHostToDevice);

            // cuBLAS sgemm: C = alpha * A @ B + beta * C
            // For row-major: compute C^T = B^T @ A^T
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,  // B not transposed (but it's B^T in col-major view)
                CUBLAS_OP_N,  // A not transposed
                n as i32,     // rows of B^T = cols of B
                m as i32,     // cols of A^T = rows of A
                k as i32,     // shared dimension
                &alpha,
                d_b,          // B^T in column-major = B in row-major
                n as i32,     // leading dimension of B
                d_a,          // A^T in column-major = A in row-major
                k as i32,     // leading dimension of A
                &beta,
                d_c,
                n as i32,     // leading dimension of C
            );

            // Copy result back
            cudaMemcpy(c.as_mut_ptr() as _, d_c as _, m * n * 4, cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(d_a as _);
            cudaFree(d_b as _);
            cudaFree(d_c as _);
        }

        Ok(())
    }
}
```

### Hard Rules for Kernel Overrides

1. **Never modify hologram.** All optimizations are injected via KernelOverride.
2. **Graceful fallback.** Return `None` when hardware isn't available.
3. **Zero overhead when unused.** If hologram-ai isn't initialized, hologram runs at full speed.
4. **Benchmark everything.** Every kernel must prove it's faster than hologram's default.
5. **Match hologram's numerical precision.** Results must be bitwise identical or within 1e-6 relative error.
6. **Thread safety.** KernelOverride is Send + Sync; ensure all backend handles are thread-safe.
7. **No panics in kernels.** Return Result::Err instead of panicking.

### Benchmarking Requirements

Every PR adding kernel overrides must include benchmark comparisons using Criterion:

```rust
// crates/hologram-backend-extensions/benches/matmul.rs
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn matmul_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for size in [64, 128, 256, 512, 1024, 2048].iter() {
        let m = *size;
        let k = *size;
        let n = *size;
        let flops = 2 * m * k * n; // multiply-add = 2 FLOPs

        group.throughput(Throughput::Elements(flops as u64));

        let a: Vec<f32> = (0..m*k).map(|i| (i % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (i % 100) as f32 / 100.0).collect();
        let mut c = vec![0.0f32; m * n];

        // Benchmark hologram default
        group.bench_with_input(
            BenchmarkId::new("hologram_default", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    hologram_backend::simd::matmul_tiled(&a, &b, &mut c, m, k, n);
                });
            },
        );

        // Benchmark with hologram-backend-extensions
        #[cfg(feature = "onednn")]
        {
            use hologram_backend_extensions::backends::onednn::OneDnnKernels;
            let onednn = OneDnnKernels::new().unwrap();
            group.bench_with_input(
                BenchmarkId::new("hologram_backend_extensions_onednn", size),
                size,
                |bencher, _| {
                    bencher.iter(|| {
                        onednn.matmul(&a, &b, &mut c, m, k, n).unwrap();
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, matmul_benchmark);
criterion_main!(benches);
```

**Expected output format:**
```
matmul/hologram_default/1024
                        time:   [12.3 ms 12.5 ms 12.7 ms]
                        thrpt:  [171.2 GFLOP/s 172.0 GFLOP/s 174.5 GFLOP/s]

matmul/hologram_ai_onednn/1024
                        time:   [2.1 ms 2.2 ms 2.3 ms]
                        thrpt:  [933.2 GFLOP/s 976.0 GFLOP/s 1.02 TFLOP/s]
                        change: [-82.3% -82.4% -82.5%] (p < 0.05)
```

### Kernel Testing Strategy

**Per-Backend Unit Tests** (in `backends/onednn/mod.rs`):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_small() {
        let kernels = OneDnnKernels::new().expect("init failed");

        // 2x3 @ 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0f32; 4];

        kernels.matmul(&a, &b, &mut c, 2, 3, 2).unwrap();

        // Expected: [[22, 28], [49, 64]]
        assert!((c[0] - 22.0).abs() < 1e-5);
        assert!((c[1] - 28.0).abs() < 1e-5);
        assert!((c[2] - 49.0).abs() < 1e-5);
        assert!((c[3] - 64.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_identity() {
        let kernels = OneDnnKernels::new().expect("init failed");

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0f32; 4];

        kernels.matmul(&a, &identity, &mut c, 2, 2, 2).unwrap();

        for i in 0..4 {
            assert!((c[i] - a[i]).abs() < 1e-5);
        }
    }
}
```

**Integration Tests (against hologram reference):**
```rust
fn assert_numerical_match(
    hologram_output: &[f32],
    hologram_ai_output: &[f32],
    rel_tolerance: f32,
) {
    assert_eq!(hologram_output.len(), hologram_ai_output.len());
    for (i, (a, b)) in hologram_output.iter().zip(hologram_ai_output.iter()).enumerate() {
        let diff = (a - b).abs();
        let max_val = a.abs().max(b.abs()).max(1e-7);
        let rel_diff = diff / max_val;
        assert!(
            rel_diff < rel_tolerance,
            "Mismatch at index {}: hologram={}, hologram-ai={}, rel_diff={}",
            i, a, b, rel_diff
        );
    }
}
```

## Documentation Guidelines

### Working Documents Location
- **ALL working markdown files MUST go in `/workspace/docs/working/`** unless otherwise specified
- Keep `/workspace/docs/working/implementation.md` as the active TODO tracker
- Example config files go in `/workspace/configs/examples/`
- Planning documents should reference implementation docs from `docs/working/`

## Code Quality Standards

**CRITICAL: These standards are MANDATORY and NON-NEGOTIABLE**

### Production-Ready Code ONLY

**ABSOLUTE REQUIREMENT: Every piece of code in this project MUST be production-ready.**

- **NO stubs** - Period. Nothing is a stub.
- **NO TODOs** - Every function is complete.
- **NO placeholders** - All code is real, working code.
- **NO "simplistic" implementations** - Full, proper implementations only.
- **NO "in a real implementation" comments** - This IS the real implementation.
- **NO shortcuts** - Do it right or don't do it.

Any code that contains phrases like "in production you would...", "a real implementation would...", "simplified for demonstration", or similar disclaimers is **UNACCEPTABLE**. If you're writing it, write it properly. If a feature isn't ready, don't include it at all.

### Exception: Bridge Implementations

**ONLY exception to the NO TODOs rule**: When hologram compiler lacks necessary operations (e.g., Conv2d, BatchNorm, etc.):

1. **NEVER write inline TODO comments** like `// TODO: implement this`
2. **DO document with BRIDGE comment** explaining the situation:
   ```rust
   // BRIDGE IMPLEMENTATION: hologram compiler does not yet support Conv2d operations.
   // This translator correctly computes output shapes and extracts attributes, but maps
   // to OpKind::Copy as a temporary bridge. Once hologram gains Conv2d support, this
   // will be updated to use the proper OpKind::Conv2d variant.
   //
   // See hologram team prompt in /workspace/specs/plans/<operation>-support.md
   ```
3. **DO create a comprehensive prompt** for the hologram team in `/workspace/specs/plans/` directory
4. **DO implement everything except the final OpKind mapping** - parse attributes, compute shapes, handle all edge cases
5. **DO write tests** for the shape calculations and attribute parsing
6. **DO document the migration path** clearly

**Bridge implementations are ONLY acceptable when**:
- The limitation is in hologram compiler, not hologram-ai-onnx
- A prompt has been written for the hologram team
- All OTHER aspects are fully implemented (parsing, shapes, validation)
- The code clearly documents what's missing and where to find the solution

### Core Requirements

1. **NO TODOs, Placeholders, or Stubs**
   - Every function MUST be fully implemented
   - No `unimplemented!()` macros
   - No `todo!()` macros
   - No `panic!("not implemented")` or similar
   - All edge cases must be handled

2. **Complete Implementations**
   - Functions must do what they claim to do
   - No shortcuts or partial implementations
   - All error paths must be handled
   - No temporary workarounds

3. **Tests Required - Maximum Coverage**
   - **Write tests for ALL methods and functions** - aim for the highest test coverage possible
   - Every public function MUST have at least one test
   - Every private function with non-trivial logic MUST have tests
   - Unit tests in module files or `tests/` subdirectory
   - Integration tests in top-level `tests/` directory
   - Include edge cases and error conditions
   - Test symbolic shapes with variable dimensions
   - Test all code paths, including error paths
   - No code should be merged without corresponding tests

4. **Documentation**
   - All public APIs MUST have rustdoc comments
   - Include examples in rustdoc for non-trivial functions
   - Document panics, errors, and safety considerations
   - Explain symbolic shape handling where applicable

5. **Error Handling**
   - Use proper error types (thiserror, anyhow)
   - No `unwrap()` in production code (use `?` or proper error handling)
   - No `expect()` unless truly impossible conditions
   - Provide helpful error messages

## Testing Requirements

### Unit Tests
- For every module in `src/`
- Test all public functions
- Test error conditions
- Test edge cases (empty inputs, large inputs, etc.)
- Test with symbolic shapes (variable batch, seq_len)

### Integration Tests
- In `tests/` directory for each crate
- Test full compilation pipelines
- Test multi-operation graphs
- Test with real ONNX models (MNIST, ResNet, etc.)

### Symbolic Shape Tests
- CRITICAL: Validate variable batch sizes
- CRITICAL: Validate variable sequence lengths
- Test shape inference propagation
- Test dimension expressions (Conv output dims)

### Memory Tests
- Ensure no OOM with large models
- Profile memory usage during compilation
- Test graph partitioning with 3000+ node graphs

### E2E Tests
- Full workflow: ONNX → .holo → execution
- Compile with hologram-ai CLI
- Run with hologram CLI
- Verify output correctness

### Benchmark Tests (for Kernel Overrides)
Every PR adding kernel overrides must include benchmark comparisons:
```
matmul_1024x1024:
  hologram default: 12.3 ms
  hologram-ai cuBLAS: 1.2 ms (10.2x faster)
  hologram-ai oneDNN: 2.1 ms (5.9x faster)
```

## ONNX Operation Implementation Requirements

**CRITICAL: When implementing or updating ONNX operations, you MUST:**

1. **Write comprehensive tests** for every operation translator:
   - Test normal cases with various input shapes and data types
   - Test edge cases (empty inputs, scalars, large tensors)
   - Test error conditions (wrong input count, invalid attributes)
   - Test constant folding paths (if applicable)
   - Tests should go in the `#[cfg(test)] mod tests {}` section of the same file

2. **Implement constant folding** when inputs are constants:
   - Many ONNX operations (Shape, Gather, Cast, Concat, Unsqueeze, etc.) should perform constant folding
   - If all inputs are `NodeOp::Constant`, compute the result at compile time and return a new `Constant`
   - This enables shape inference chains to collapse: Shape → Gather → Cast → Range → all Constants
   - Constant folding is essential for handling dynamic shape computations in models like T5

3. **Example operation implementations with tests:**
   - `src/ops/advanced.rs` - Range, Cast (with constant folding + tests)
   - `src/ops/shape.rs` - Unsqueeze, Concat, Reshape (with constant folding + tests)
   - `src/ops/constant.rs` - Shape, ConstantOfShape (with constant folding + tests)
   - `src/ops/indexing.rs` - Gather (with constant folding)

4. **Verify constant folding works**:
   - Create a test with constant inputs
   - Assert the result is a `NodeOp::Constant`
   - Verify the constant data matches expected output

## Implementation Workflow

1. **Read existing code first** - Never modify without understanding
2. **Write tests first** - TDD approach preferred
3. **Implement fully** - No TODOs or stubs
4. **Verify with tests** - All tests must pass
5. **Document public APIs** - Rustdoc for all public items
6. **ZERO Clippy Errors and Warnings** - This is MANDATORY:
   - Run `cargo clippy --all-targets` and ensure **ZERO warnings**
   - Run `cargo check --all` to verify no compilation errors
   - A task is NOT complete until clippy reports 0 warnings
   - No unused imports, variables, or dead code without explicit `#[allow(...)]`
   - Use `#[allow(dead_code)]` sparingly and only with a comment explaining why
   - Fix warnings immediately - do not accumulate technical debt
   - Run `just test` to verify all tests pass (this also catches clippy issues)
7. **Update TODO tracker** - Mark items complete in `docs/working/implementation.md`

## Conventions

- **Rust edition:** 2024
- **MSRV:** Same as hologram (check hologram's Cargo.toml)
- **Error handling:** Use `thiserror` for library errors, `anyhow` for CLI
- **Logging:** Use `tracing` for debug/info output
- Use workspace dependencies where possible
- Proto files go in `proto/` subdirectories
- Build scripts handle proto compilation

## Commit Rules

- Write atomic, descriptive commit messages in imperative mood
- Ensure `cargo test` passes before every commit
- Include benchmark results for any kernel changes
- **ZERO clippy warnings required** - Run `cargo clippy --all-targets` before committing
- Never commit code with clippy warnings - fix them first

## Dependencies

Workspace-level dependencies (defined in root `Cargo.toml`):
- `hologram` - Core compute framework (local path)
- `prost = "0.13"` - Protocol Buffers runtime
- `thiserror = "2.0"` - Error types
- `anyhow = "1.0"` - Error handling
- `tracing = "0.1"` - Logging

Build dependency (per-crate):
- `prost-build = "0.13"` - Proto compilation

Future kernel dependencies (in hologram-backend-extensions, feature-gated):
- `onednn = "0.5"` - Intel oneDNN (CPU)
- `cuda-sys = "0.3"` - CUDA runtime
- `cublas-sys = "0.3"` - cuBLAS
- `metal = "0.29"` - Apple Metal

## Feature Flags

```toml
# crates/hologram-backend-extensions/Cargo.toml
[features]
default = []
cuda = ["dep:cuda-sys", "dep:cublas-sys"]
onednn = ["dep:onednn"]
metal = ["dep:metal"]
all = ["cuda", "onednn", "metal"]

[dependencies]
hologram-backend = { path = "../../hologram/crates/hologram-backend" }
thiserror = "2.0"
tracing = "0.1"

# Optional backend dependencies
cuda-sys = { version = "0.3", optional = true }
cublas-sys = { version = "0.3", optional = true }
onednn = { version = "0.5", optional = true }
metal = { version = "0.29", optional = true }
```
