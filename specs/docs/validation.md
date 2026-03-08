I don't have permission to write the file. Here is the filled-in validation documentation:

# Validation Harness — hologram-ai

## Validation Goals

The validation harness verifies numerical correctness by comparing `hologram-ai`
outputs against reference runtimes. It validates:

1. **Tensor-level numerical equivalence** — output tensors match reference
   implementations within dtype-appropriate tolerances (max absolute error,
   mean absolute error, cosine similarity)
2. **Greedy token agreement** — for GGUF models, top-1 greedy token matches
   llama.cpp reference on deterministic prompts
3. **Graph IR structural validity** — `AiGraph::validate()` passes (DAG check,
   tensor registration, no dangling references)
4. **Quantization correctness** — dequantized values match precomputed
   reference values from GGML source for each quant scheme
5. **Cross-backend consistency** — Metal backend produces identical outputs
   to CPU backend on the same model and inputs (Phase 3)

---

## Test Inputs

| Source | Description | Location |
|--------|-------------|----------|
| **Committed fixtures** | Minimal synthetic ONNX models (identity, tiny-mlp) for smoke tests | `tests/fixtures/onnx/` |
| **Generated inputs** | Shape/dtype test inputs created programmatically in test code | Integration test functions |
| **Downloaded models** | Real GGUF models (TinyLlama, etc.) cached locally, not committed | `~/.cache/hologram-ai/models/` |
| **Precomputed quant blocks** | Known quantized block values extracted from GGML source | Unit test constants |

Synthetic fixtures validate shape and dtype propagation only — they do not
contain meaningful weights. Numerical golden tests require real model files
acquired via `hologram-ai download` or manual placement.

---

## Reference Outputs

Reference outputs are produced by external runtimes invoked via subprocess:

| Format | Reference Runtime | Invocation |
|--------|-------------------|------------|
| ONNX | ONNX Runtime | `python -m onnxruntime.tools.ort_test_runner` via `.npz` serialization |
| GGUF | llama.cpp | `main` binary with `--temp 0` for deterministic greedy decoding |

Reference outputs are **not committed** to the repository. They are generated
on-demand during validation runs. This avoids repository bloat and ensures
tests always compare against the current reference implementation version.

For unit tests (quantization, graph validation), expected values are hardcoded
constants derived from reference implementations at development time.

---

## Tolerance Thresholds

| Dtype | max_abs_err | mean_abs_err | cosine_sim_min |
|-------|-------------|--------------|----------------|
| f32 | 1e-5 | 1e-6 | 0.9999 |
| f16 | 1e-3 | 1e-4 | 0.999 |
| Quantized | scheme-dependent | scheme-dependent | scheme-dependent |

For greedy token comparison (GGUF models), exact token match is required —
any mismatch is a failure.

Quantized tolerances are wider due to inherent precision loss. Q4_0 and Q8_0
use the f16 thresholds as a baseline; K-quants may require looser bounds
depending on block size and scale distribution.

---

## Running Validation

```bash
# Run all unit and integration tests (committed fixtures only)
cargo test

# Run ONNX validation against ORT (requires onnxruntime in PATH)
hologram-ai validate --onnx model.onnx --input input.json

# Run GGUF greedy token validation against llama.cpp
hologram-ai validate --gguf model.gguf --prompt "The capital of France is" --tokens 5

# Generate JSON report
hologram-ai validate --report report.json model.gguf

# Run reference tests (ignored by default; requires external tools)
cargo test --ignored

# CI nightly: reference tests with real models
LLAMACPP_BIN=/path/to/llama.cpp/main cargo test --ignored
```

Reference tests are tagged `#[ignore]` and run on a nightly CI schedule to
avoid blocking every push on expensive model downloads and subprocess execution.
