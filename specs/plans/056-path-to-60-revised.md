# Plan 056: Path to 60 tok/s — Revised After Q2 Experiment

## Findings from Q2 Experiment

1. **Pure integer LUT-GEMM is SLOWER than AMX BLAS** on Apple Silicon (410ms vs 320ms/step)
2. **Q2 with 4 centroids produces gibberish** — 4 levels insufficient for weight matrices
3. **AMX ceiling is ~43 tok/s** regardless of quantization level (Q2/Q4/Q8/f32 all
   dequant to f32 then BLAS — the cached f32 is the same size)
4. **The 38 tok/s GGUF baseline** comes from ALL weights being Q4 via dequant→BLAS

## What Actually Works on Apple Silicon CPU

The only proven fast path: **Q4 dequant → cached f32 → BLAS sgemm (AMX)**

This is what GGUF achieves at 38-43 tok/s. The dequant happens once per weight
matrix (cached in WeightCache). Subsequent decode steps hit the f32 cache →
BLAS sgemm at full AMX throughput.

## Path to 40+ tok/s (ONNX matching GGUF)

**The single remaining blocker:** ONNX f32 weights need ALL large MatMuls
Q4-quantized with the dequant→BLAS path. Currently only ~45 of 155 get Q4'd.

**Fix:** Quantize weights at parameter registration time, not at op lowering
time. This decouples quantization from fusion passes:

1. During parameter registration (builder.rs lines 130-212):
   - For each f32 weight param with shape [K, N] where K≥256, N≥256:
   - Run `quantize_4bit()` on the f32 data
   - Register the Q4 constant via `builder.matmul_lut_4bit()`
   - Do NOT register the f32 constant
   - Track which TIDs became Q4 constants

2. During node lowering:
   - When a MatMul's weight input is a Q4 constant (tracked from step 1):
   - Emit `MatMulLut4` instead of `FloatOp::MatMul`
   - This works regardless of fusion passes

**Impact:** ONNX with `--quantize q4_0` matches GGUF at 38-43 tok/s.
Archive drops from 4.4 GB to ~0.6 GB.

## Path to 60+ tok/s (beyond AMX ceiling)

The AMX ceiling is real. To break it on CPU without Metal:

### Option A: Speculative Decoding with Batch Verification (2x multiplier)

Compile a third tape at seq=N (verification tape). Draft N tokens with seq=1
decode tape, verify all N in a single prefill-style forward pass.

- Draft: N × 25ms = N × 25ms
- Verify: 1 × (25 × N)ms ≈ 25ms for small N (BLAS amortizes batch)
- Total: (N+1) × 25ms for N+ accepted tokens
- At N=4 with 75% acceptance: 5 passes for 3 tokens ≈ 42ms/token → still 24 tok/s

Wait — prefill with seq=N costs N× more than decode with seq=1. There's no
free batching on CPU. Each position still needs K multiply-adds.

**Revised estimate:** Speculative won't help on CPU because verification is
O(N) — same total compute as N sequential decode steps.

### Option B: Reduce MatMul Count via Architecture Changes

- **Sliding window attention** (O(w) instead of O(seq) per layer) — reduces
  attention MatMul cost at long context. TinyLlama at seq=24 is already short.
- **Layer pruning** — skip layers whose outputs are near-identity. Requires
  analysis per model.
- **Sparse attention** — already implemented via `sparse_v`. Helps at long context.

### Option C: Multi-Core CPU Parallelism

BLAS already uses multi-core internally. But the tape executor runs ops
sequentially — independent ops (attention heads, FFN gate+up) could run in
parallel if the executor had level-parallel dispatch.

The Plan 039 analysis estimated this at 20-40% improvement. 43 × 1.3 = 56 tok/s.

This requires:
1. Identify ops in the same tape level that are independent
2. Dispatch them via rayon in parallel
3. Each thread needs its own output buffer (no sharing)
4. Synchronize at level boundaries

### Option D: Reduce Data Per Step (Still Valid Concept)

The Q2 pure-integer approach failed because the kernel was slower than AMX.
But what if we use Q2 dequant→BLAS (same as Q4 does)?

- Q2 dequant: 4 centroids × 2-bit indices → f32 expansion
- Cache the dequanted f32 (same as Q4)
- BLAS sgemm on cached f32 (identical speed)
- Archive size: 0.25 bytes/weight vs 0.5 for Q4

This doesn't help steady-state tok/s (BLAS cost is identical), but:
- **Faster initial load** (half the data to dequant)
- **Smaller archive** (0.3 GB vs 0.6 GB)
- **Lower RSS** (dequant cache is the same size, but archive mmap is smaller)

## Recommended Priority

1. **Quantize at param registration** → ONNX matches GGUF at 38-43 tok/s
2. **Level-parallel tape execution** → 43 × 1.3 = ~56 tok/s
3. **Batch verification for speculative** → 56 × 1.5 = ~84 effective tok/s
   (requires verification tape + analysis of batch benefit on CPU)

Steps 1+2 give 56 tok/s without speculative. Step 3 pushes to 84.
