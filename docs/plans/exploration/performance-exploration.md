# Performance Exploration Roadmap

A comprehensive plan for auditing hologram-backend performance, creating execution benchmarks, and comparing against industry baselines.

---

## Overview

**Problem:** hologram-onnx is a translator (ONNX → Hologram IR), but we don't know the actual inference performance on consumer hardware. Current benchmarks only measure compilation time, not execution.

**Goals:**
1. Audit hologram-backend to understand actual optimization implementations
2. Create execution benchmarks that measure real inference time
3. Compare against ONNX Runtime, TensorRT, and llama.cpp baselines

---

## Phase 1: Audit hologram-backend

**Goal:** Document what optimizations are actually implemented vs. claimed.

**Working Directory:** `/hologram/` (the hologram repository)
**Output Location:** `/workspace/docs/working/backend-audit-report.md` (hologram-onnx repo)

> **Note:** The audit explores code in the hologram repository, but findings are documented in hologram-onnx since that's where performance optimizations will be actioned.

### Tasks

1. [ ] **Audit backend structure**
   - Location: `/hologram/crates/backend/`
   - Document crate organization and main entry points
   - Identify execution engine architecture

2. [ ] **SIMD implementation audit**
   - Search for: `#[target_feature]`, `std::arch`, `_mm256`, `vld1q`, NEON intrinsics
   - Document which operations have SIMD paths
   - Check for runtime CPU feature detection (AVX2, AVX-512, NEON)
   - Verify vectorization width (128-bit, 256-bit, 512-bit)

3. [ ] **Conv2D decomposition audit**
   - Location: `/hologram/crates/compiler/` (decomposition passes)
   - Location: `/hologram/crates/backend/` (GEMM kernels)
   - Find actual Im2col + GEMM implementation
   - Document tiling strategy and cache optimization
   - Check if BLAS libraries are used (OpenBLAS, MKL, Accelerate)
   - Verify `decompose_function()` that hologram-onnx delegates to

4. [ ] **Memory management audit**
   - Check for memory pooling / arena allocation
   - Document tensor allocation strategy
   - Look for zero-copy optimizations

5. [ ] **Threading model audit**
   - Search for: `rayon`, `std::thread`, thread pool
   - Document parallelization strategy (data parallel, task parallel)
   - Check for NUMA awareness

6. [ ] **GPU support audit**
   - Search for: `wgpu`, `cuda`, `metal`, `vulkan`
   - Document GPU backend status (if any)

### Deliverable
Create `/workspace/docs/working/backend-audit-report.md` with findings table:

```markdown
| Optimization | Claimed | Actually Implemented | Location |
|--------------|---------|---------------------|----------|
| AVX2 SIMD    | Yes     | ?                   | ?        |
| Im2col+GEMM  | Yes     | ?                   | ?        |
| Thread pool  | ?       | ?                   | ?        |
| GPU support  | ?       | ?                   | ?        |
```

---

## Phase 2: Execution Benchmarks

**Goal:** Measure actual inference latency and throughput on real models.

**Working Directory:** `/workspace/` (hologram-onnx repository)

> **Note:** Benchmarks are added to hologram-onnx since we're measuring the full pipeline (ONNX → compile → execute).

### Benchmark Infrastructure

1. [ ] **Create benchmark harness**
   ```
   benches/execution/
   ├── mod.rs           # Common benchmark utilities
   ├── models.rs        # Model loading helpers
   ├── ops_bench.rs     # Individual operation benchmarks
   ├── e2e_bench.rs     # End-to-end model benchmarks
   └── memory_bench.rs  # Memory usage benchmarks
   ```

2. [ ] **Individual operation benchmarks** (`ops_bench.rs`)
   - MatMul: 128x128, 512x512, 1024x1024, 4096x4096
   - Conv2D: 3x3, 5x5, 7x7 kernels at various resolutions
   - Attention: Scaled dot-product attention at seq_len 128, 512, 2048
   - Element-wise: Add, Mul, ReLU, GELU on varying tensor sizes
   - Reduction: Sum, Mean, Max on varying dimensions

3. [ ] **End-to-end model benchmarks** (`e2e_bench.rs`)

   | Model | Type | Size | Input Shape |
   |-------|------|------|-------------|
   | MNIST CNN | Vision | ~100KB | [1, 1, 28, 28] |
   | MobileNetV2 | Vision | ~14MB | [1, 3, 224, 224] |
   | ResNet-50 | Vision | ~100MB | [1, 3, 224, 224] |
   | BERT-base | Text | ~440MB | [1, 128] |
   | T5-small | Text | ~240MB | [1, 128] |
   | Whisper-tiny | Audio | ~150MB | [1, 80, 3000] |

4. [ ] **Metrics to capture**
   ```rust
   struct BenchmarkResult {
       latency_ms: f64,           // Mean inference time
       latency_p99_ms: f64,       // 99th percentile
       throughput_samples_sec: f64,
       peak_memory_mb: f64,
       first_inference_ms: f64,   // Cold start
   }
   ```

5. [ ] **Memory benchmarks** (`memory_bench.rs`)
   - Peak memory during inference
   - Memory allocation count
   - Tensor reuse efficiency

### Benchmark Commands

```bash
# Run all execution benchmarks
cargo bench --bench execution

# Run specific model
cargo bench --bench execution -- resnet50

# Profile with perf
perf record cargo bench --bench execution -- matmul
perf report
```

### Deliverable
Create `/workspace/docs/working/benchmark-results.md` with:
- Hardware specs (CPU, RAM, OS)
- Raw benchmark numbers
- Flame graphs for hotspots
- Memory profiles

---

## Phase 3: Baseline Comparisons

**Goal:** Compare hologram performance against industry-standard runtimes.

**Working Directory:** `/workspace/` (hologram-onnx repository)

### Comparison Targets

1. [ ] **ONNX Runtime (CPU)**
   - Version: Latest stable
   - Execution providers: Default CPU, MLAS
   - Same models, same input shapes

2. [ ] **ONNX Runtime (GPU)** (if applicable)
   - CUDA execution provider
   - Document GPU specs

3. [ ] **TensorRT** (if NVIDIA GPU available)
   - FP32 and FP16 modes
   - Note: Requires ONNX→TensorRT conversion

4. [ ] **llama.cpp** (for applicable models)
   - BERT/GPT-2 if supported
   - Document quantization settings (F32, F16, Q8, Q4)

### Comparison Methodology

```markdown
## Fair Comparison Checklist
- [ ] Same model architecture and weights
- [ ] Same input shape and batch size
- [ ] Same precision (FP32 unless noted)
- [ ] Same hardware (document specs)
- [ ] Warm-up runs before measurement (10 iterations)
- [ ] Statistical significance (100+ runs, report std dev)
- [ ] Measure same metrics (latency, throughput, memory)
```

### Comparison Script

```bash
# scripts/benchmark_comparison.sh
#!/bin/bash

MODEL=$1  # e.g., resnet50.onnx

echo "=== Hologram ==="
cargo bench --bench execution -- $MODEL

echo "=== ONNX Runtime ==="
python scripts/ort_benchmark.py $MODEL

echo "=== TensorRT ==="
python scripts/trt_benchmark.py $MODEL  # Optional
```

### Python Baseline Scripts

1. [ ] **Create ONNX Runtime benchmark** (`scripts/ort_benchmark.py`)
   ```python
   import onnxruntime as ort
   import numpy as np
   import time

   # Warmup, then measure N iterations
   ```

2. [ ] **Create TensorRT benchmark** (`scripts/trt_benchmark.py`) - optional

3. [ ] **Create llama.cpp benchmark** (`scripts/llamacpp_benchmark.sh`) - for LLM models

### Deliverable
Create `/workspace/docs/working/comparison-report.md` with:

```markdown
## ResNet-50 (batch=1, 224x224)

| Runtime | Latency (ms) | Throughput (img/s) | Memory (MB) |
|---------|-------------|-------------------|-------------|
| Hologram | ? | ? | ? |
| ONNX Runtime CPU | ? | ? | ? |
| TensorRT FP32 | ? | ? | ? |
| TensorRT FP16 | ? | ? | ? |

## Analysis
- Hologram is X% faster/slower than ORT because...
- Bottleneck is...
```

---

## Phase 4: Optimization Recommendations

Based on findings, create actionable recommendations:

1. [ ] **Identify performance gaps**
   - Which operations are slowest?
   - Where does hologram lag behind baselines?

2. [ ] **Prioritize optimizations**
   - High impact, low effort first
   - Document expected gains

3. [ ] **Create optimization roadmap**
   - File: `/workspace/docs/plans/optimization-roadmap.md`

---

## Directory Structure

```
# Hologram repository (audit target)
/hologram/
└── crates/
    ├── backend/     # SIMD, kernels, execution (AUDIT)
    ├── compiler/    # Decomposition passes (AUDIT)
    ├── ir/          # Graph representation
    └── ...

# hologram-onnx repository (benchmarks & outputs)
/workspace/ (hologram-onnx)
├── benches/
│   └── execution/           # NEW: Execution benchmarks
│       ├── mod.rs
│       ├── ops_bench.rs
│       ├── e2e_bench.rs
│       └── memory_bench.rs
├── scripts/
│   ├── ort_benchmark.py     # NEW: ONNX Runtime baseline
│   ├── trt_benchmark.py     # NEW: TensorRT baseline (optional)
│   └── benchmark_comparison.sh
├── docs/
│   └── working/
│       ├── backend-audit-report.md   # Phase 1 output
│       ├── benchmark-results.md      # Phase 2 output
│       └── comparison-report.md      # Phase 3 output
└── test_models/             # NEW: Benchmark models
    ├── mnist.onnx
    ├── mobilenetv2.onnx
    ├── resnet50.onnx
    └── bert-base.onnx
```

---

## Success Criteria

- [ ] Backend audit completed with all optimizations documented
- [ ] Execution benchmarks running for 6+ models
- [ ] Baseline comparisons completed for ONNX Runtime (minimum)
- [ ] Performance gaps identified and quantified
- [ ] Optimization roadmap created with priorities

---

## Timeline Estimate

Not providing time estimates per project guidelines. Tasks should be completed in order:
1. Phase 1 (Backend Audit) - prerequisite for understanding what to benchmark
2. Phase 2 (Execution Benchmarks) - creates measurement infrastructure
3. Phase 3 (Baseline Comparisons) - requires Phase 2 complete
4. Phase 4 (Recommendations) - synthesizes all findings
