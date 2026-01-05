# TurboSparse Exploration Roadmap

A roadmap for testing and comparing TurboSparse sparsification techniques to determine if hologram should implement similar optimizations.

---

## Overview

TurboSparse is a sparsification technique that achieves **~90% neuron sparsity** in dense LLMs (and **~97% in MoE models**) through a novel **dReLU activation function**. This roadmap outlines how to evaluate whether hologram should implement similar techniques.

**Reference:** [TurboSparse Paper (arXiv:2406.05955)](https://arxiv.org/abs/2406.05955)

---

## TurboSparse Key Concepts

### The dReLU Innovation
```
Standard Gated MLP:    gate(x) * up(x)     → ~70% sparsity achievable
TurboSparse dReLU:     ReLU(gate(x)) * ReLU(up(x))  → ~90% sparsity
```

By applying ReLU to **both** gate and up-projection (not just gate), TurboSparse achieves dramatically higher sparsity while maintaining model quality.

### Performance Gains
- Dense models: **2-5x decoding speedup**
- MoE models: TurboSparse-Mixtral activates only **4.3B of 47B parameters**
- Mobile: **22.2x speedup** on smartphones without GPU

---

## Testing & Comparison Roadmap

### Phase 1: Baseline Measurements (Prerequisite)

**Goal:** Establish performance baselines for dense inference on representative models.

**Tasks:**
1. [ ] Select benchmark models:
   - Mistral-7B (dense LLM, target of TurboSparse)
   - Llama-2-7B (comparison point)
   - A smaller model for quick iteration (e.g., GPT-2 124M)

2. [ ] Create benchmark harness:
   ```rust
   // benches/sparse_comparison.rs
   fn benchmark_dense_inference(model: &HoloModel, inputs: &[Tensor]) -> BenchmarkResult {
       // Measure: latency, throughput, memory bandwidth
   }
   ```

3. [ ] Measure baseline metrics:
   - Tokens/second for autoregressive generation
   - Time-to-first-token latency
   - Peak memory usage
   - Activation memory bandwidth (GB/s)

4. [ ] Profile hotspots:
   - Which ops dominate inference time?
   - What % of time is in MatMul after FFN?

**Deliverables:**
- `docs/benchmarks/dense_baseline.md` with baseline numbers
- `benches/dense_inference.rs` benchmark suite

---

### Phase 2: Sparsity Analysis (No Code Changes)

**Goal:** Measure actual sparsity in real models to validate TurboSparse claims.

**Tasks:**
1. [ ] Instrument activation capture:
   ```rust
   // Add to hologram-backend or create analysis tool
   fn capture_activations(model: &HoloModel, input: &Tensor) -> ActivationMap {
       // Run inference, capture intermediate activations
   }

   fn analyze_sparsity(activations: &ActivationMap) -> SparsityReport {
       // For each layer: count zeros, compute density
   }
   ```

2. [ ] Analyze standard models (GELU/Swish):
   - What's the natural sparsity after GELU in Llama/Mistral?
   - Compare gate vs up-projection sparsity

3. [ ] Download TurboSparse models (if available):
   - TurboSparse-Mistral-7B from HuggingFace
   - Measure actual sparsity to verify ~90% claim

4. [ ] Create sparsity visualizations:
   - Histogram of activation values per layer
   - Sparsity heatmap across layers

**Deliverables:**
- `tools/sparsity_analyzer.rs` CLI tool
- `docs/analysis/sparsity_measurements.md` with findings

---

### Phase 3: Sparse Kernel Prototyping

**Goal:** Implement and benchmark sparse operations in isolation.

**Tasks:**
1. [ ] Implement sparse MatMul variants:
   ```rust
   // In hologram-backend or separate experiment crate

   // CSR format sparse × dense
   fn sparse_csr_matmul(sparse: &CsrTensor, dense: &Tensor) -> Tensor;

   // Threshold-based sparse (skip zeros dynamically)
   fn threshold_sparse_matmul(input: &Tensor, weight: &Tensor, threshold: f32) -> Tensor;

   // Bitmap sparse (precomputed zero mask)
   fn bitmap_sparse_matmul(input: &Tensor, weight: &Tensor, mask: &BitVec) -> Tensor;
   ```

2. [ ] Benchmark sparse vs dense at various sparsities:
   | Sparsity | Dense (ms) | CSR (ms) | Threshold (ms) | Bitmap (ms) |
   |----------|------------|----------|----------------|-------------|
   | 50%      | ?          | ?        | ?              | ?           |
   | 70%      | ?          | ?        | ?              | ?           |
   | 90%      | ?          | ?        | ?              | ?           |
   | 95%      | ?          | ?        | ?              | ?           |

3. [ ] Find crossover point:
   - At what sparsity does sparse beat dense?
   - Does this vary by matrix size?

4. [ ] Test with realistic shapes:
   - [batch=1, seq=1, hidden=4096] × [4096, 14336] (Mistral FFN)
   - [batch=8, seq=2048, hidden=4096] × [4096, 14336] (batched)

**Deliverables:**
- `experiments/sparse_kernels/` prototype implementations
- `docs/benchmarks/sparse_kernel_comparison.md`

---

### Phase 4: Pattern Detection Prototype

**Goal:** Detect dReLU patterns in ONNX graphs.

**Tasks:**
1. [ ] Implement dReLU pattern matcher in hologram-onnx:
   ```rust
   // src/ops/patterns.rs (new file)

   /// Detects: Mul(Relu(gate_proj), Relu(up_proj))
   pub fn detect_drelu_mlp(graph: &OnnxGraph) -> Vec<DReluPattern> {
       // Walk graph, find Mul nodes
       // Check if both inputs are Relu
       // Check if Relu inputs are projections from same source
   }

   pub struct DReluPattern {
       pub gate_relu: NodeId,
       pub up_relu: NodeId,
       pub mul_node: NodeId,
       pub down_proj: Option<NodeId>,  // MatMul after the Mul
   }
   ```

2. [ ] Test on TurboSparse models:
   - Does pattern detection find expected dReLU layers?
   - Any false positives/negatives?

3. [ ] Compare to standard models:
   - What patterns exist in GELU-based models?
   - Could we detect "sparse-likely" patterns more generally?

**Deliverables:**
- `src/ops/patterns.rs` with pattern detection
- Unit tests for pattern matching

---

### Phase 5: End-to-End Sparse Execution

**Goal:** Integrate sparse execution path and measure real-world gains.

**Tasks:**
1. [ ] Add sparsity annotation to IR:
   ```rust
   // In hologram-ir (or as custom metadata in hologram-onnx)
   pub struct SparsityHint {
       pub expected_density: f32,  // 0.1 = 90% sparse
       pub verified: bool,         // Set after runtime measurement
   }
   ```

2. [ ] Implement dispatch logic in backend:
   ```rust
   fn execute_matmul(ctx: &mut ExecContext, op: &MatMulOp) -> Result<Tensor> {
       let input = ctx.get_tensor(op.input)?;

       // Check sparsity hint
       if let Some(hint) = op.sparsity_hint {
           if hint.expected_density < 0.3 {
               // Try sparse path
               let actual_density = input.compute_density();
               if actual_density < 0.3 {
                   return sparse_matmul(input, op.weight);
               }
           }
       }

       // Dense fallback
       dense_matmul(input, op.weight)
   }
   ```

3. [ ] Full model benchmarks:
   - TurboSparse-Mistral-7B with sparse execution
   - Compare to dense baseline from Phase 1

4. [ ] Measure overhead:
   - Runtime sparsity checking cost
   - Format conversion overhead (dense → CSR)
   - Memory for sparse metadata

**Deliverables:**
- Working sparse execution path
- `docs/benchmarks/e2e_sparse_results.md`

---

### Phase 6: Decision Point

**Goal:** Decide whether to productionize based on evidence.

**Evaluation Criteria:**

| Criterion | Threshold | Measured |
|-----------|-----------|----------|
| Speedup on TurboSparse models | >1.5x | ? |
| Memory bandwidth reduction | >50% | ? |
| Implementation complexity | Reasonable | ? |
| Maintenance burden | Acceptable | ? |
| Pattern detection accuracy | >95% | ? |
| Runtime overhead | <5% | ? |

**Decision Matrix:**

| Outcome | Speedup | Overhead | Recommendation |
|---------|---------|----------|----------------|
| Win-Win | >2x | <5% | Full implementation in hologram stack |
| Marginal | 1.3-2x | 5-15% | hologram-onnx only, optional flag |
| Not Worth It | <1.3x | >15% | Document findings, no implementation |

**If implementing:**
- Phase 7: Production implementation (see Architecture section below)
- Add to `hologram-ir` and `hologram-backend`

**If not implementing:**
- Document why TurboSparse doesn't help hologram
- Identify what would change the decision (different hardware, different models)

---

### Phase Dependencies

| Phase | Focus | Dependencies |
|-------|-------|--------------|
| Phase 1 | Baseline | None |
| Phase 2 | Sparsity analysis | Phase 1 |
| Phase 3 | Sparse kernels | None (can parallel with 1-2) |
| Phase 4 | Pattern detection | None (can parallel) |
| Phase 5 | E2E integration | Phases 1-4 |
| Phase 6 | Decision | Phase 5 |

---

## Architecture: Where Should This Live?

If the evaluation shows positive results, here's where each component should be implemented:

```
┌─────────────────────────────────────────────────────────────────┐
│                      hologram-onnx                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Pattern Detection (dReLU recognition from ONNX ops)    │   │
│  │  - Detects ReLU(gate) * ReLU(up) pattern               │   │
│  │  - Annotates IR nodes with sparsity hints               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Sparsity annotations in IR
┌─────────────────────────────────────────────────────────────────┐
│                      hologram-ir                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Sparse Tensor Types                                    │   │
│  │  - TensorFormat enum (Dense, CSR, COO, BSR)            │   │
│  │  - SparsityHint metadata on tensors                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Sparse-annotated IR
┌─────────────────────────────────────────────────────────────────┐
│                    hologram-compiler                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Sparsity-Aware Optimization Passes                     │   │
│  │  - Propagate sparsity through graph                     │   │
│  │  - Convert dense ops → sparse ops when beneficial       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Optimized sparse IR
┌─────────────────────────────────────────────────────────────────┐
│                    hologram-backend                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Sparse Execution                                       │   │
│  │  - Sparse GEMM kernels (CSR × Dense, etc.)             │   │
│  │  - Runtime sparsity detection (optional fallback)       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Recommended Approach: Start Small

1. **Phase 1 (hologram-onnx only):**
   - Detect dReLU pattern
   - Emit `NodeOp::Custom("sparse_mlp", ...)`
   - Backend handles custom op with sparse GEMM
   - *Validates the approach with minimal scope*

2. **Phase 2 (promote to hologram-ir):**
   - If Phase 1 shows gains, add proper `SparseTensor` to IR
   - Move sparse ops from custom to first-class
   - Other frontends can now emit sparse hints

---

## Resources Required

**Models:**
- Mistral-7B (~14GB)
- TurboSparse-Mistral-7B (if available on HuggingFace)
- GPT-2 124M (quick iteration)

**Compute:**
- CPU benchmarking machine (consistent, isolated)
- Optional: GPU for comparison baseline

**External References:**
- [TurboSparse Paper](https://arxiv.org/abs/2406.05955)
- [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) (their execution framework)
- [sprs](https://crates.io/crates/sprs) (Rust sparse matrix library)

---

## Conclusion

This roadmap provides a structured approach to evaluate TurboSparse:

1. **Measure before optimizing** - Establish baselines (Phase 1-2)
2. **Prototype in isolation** - Test sparse kernels independently (Phase 3)
3. **Integrate carefully** - Add pattern detection before execution (Phase 4-5)
4. **Decide with data** - Use measured results to guide implementation (Phase 6)

The most impactful near-term addition would be **dReLU detection + sparse MatMul**, which could deliver 2-5x speedups with moderate implementation effort—but this must be validated empirically before committing to full implementation.
