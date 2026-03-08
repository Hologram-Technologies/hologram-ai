I now have enough context to fill in the backend matrix template. Based on the architecture docs, ADRs, and roadmap:

# Backend Matrix — hologram-ai

## Supported Backends

| Backend | Status | Notes |
|---------|--------|-------|
| CPU (hologram-exec) | MVP | `KvExecutor` provides execution via O(1) LUT-based engine; all AI ops registered via `CustomOpRegistry` |
| Metal | Phase 3 | Apple Silicon GPU acceleration; quantized GEMM kernels (Q4_0, Q8_0); flash attention |
| CUDA | Phase 4 | NVIDIA GPU backend; multi-GPU tensor parallelism |
| WebGPU | Phase 4 | Browser and portable GPU target |

---

## Capability Flags

| Flag | Description | Backends |
|------|-------------|----------|
| `MatMulLut4` | Q4_0 quantized matrix multiply via LUT | All |
| `MatMulLut8` | Q8_0 quantized matrix multiply via LUT | All (Phase 2+) |
| `FlashAttention` | Fused flash attention kernel | Metal, CUDA |
| `QuantizedGemm` | Hardware-accelerated quantized GEMM | Metal, CUDA |
| `Bf16Compute` | BF16 arithmetic support | Metal, CUDA |

---

## Adding a New Backend

Backend support is added via `CustomOpRegistry` handler registration — not a separate trait or crate.

1. **Register custom op handlers** during lowering. Each AI-specific op (`MultiHeadAttention`, `RmsNorm`, `RotaryEmbedding`, `Dequantize`, etc.) requires a handler registered in `CustomOpRegistry`.

2. **Implement the handler functions.** Handlers receive input tensors and produce outputs. For GPU backends, handlers dispatch to Metal/CUDA/WebGPU compute shaders.

3. **No separate backend crate required.** All execution routes through `hologram::KvExecutor`. Backend-specific capability is declared by which ops are registered and how their handlers are implemented.

4. **Provide software fallback.** Every custom op must have a pure-Rust CPU fallback. GPU-optimized paths are selected at registration time based on device availability.

```rust
// Example: registering a backend-specific attention handler
let mut registry = CustomOpRegistry::new();

#[cfg(feature = "metal")]
registry.register(AttentionOpId, metal_flash_attention_handler);

#[cfg(not(feature = "metal"))]
registry.register(AttentionOpId, cpu_attention_handler);
```
