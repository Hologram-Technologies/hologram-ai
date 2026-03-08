# Performance Guide — hologram-ai

## Benchmarks

Benchmarks live in `benches/` and cover the critical paths: import, optimization, lowering, and inference execution.

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench --bench import_gguf
cargo bench --bench forward_pass
cargo bench --bench tokenizer

# Generate HTML report with criterion
cargo bench -- --save-baseline main
```

Benchmark suites:

| Suite | What it measures |
|-------|------------------|
| `import_gguf` | GGUF file parsing, tensor loading, AiGraph construction |
| `import_onnx` | ONNX protobuf parsing, op conversion |
| `optimize` | Optimization pass throughput (fusion, folding) |
| `lower` | AiGraph → hologram::Graph lowering |
| `forward_pass` | End-to-end single forward pass latency |
| `tokenizer` | Encode/decode throughput (tokens/sec) |
| `kv_cache` | KV-cache update latency per layer |

---

## Profiling

### CPU profiling with flamegraph

```bash
# Install flamegraph
cargo install flamegraph

# Profile a benchmark or binary
cargo flamegraph --bench forward_pass -- --bench

# Profile CLI inference
cargo flamegraph --bin hologram-ai -- run model.gguf --prompt "Hello"
```

### Linux perf

```bash
# Record performance counters
perf record -g --call-graph dwarf cargo bench --bench forward_pass

# Generate report
perf report
```

### macOS Instruments

```bash
# Build with debug symbols
cargo build --release

# Profile with Instruments (Time Profiler template)
xcrun xctrace record --template 'Time Profiler' --launch -- ./target/release/hologram-ai run model.gguf
```

### Memory profiling with DHAT

```bash
# Add dhat feature and run with DHAT
cargo run --release --features dhat -- run model.gguf --prompt "Test"
# Produces dhat-heap.json for analysis
```

---

## Known Bottlenecks

| Path | Description | Mitigation |
|------|-------------|------------|
| `Dequantize` kernel | Q4_0 → f32 conversion dominates non-fused MatMul | Fuse via `QuantMatMulFusion` pass; use quantized GEMM kernels |
| Attention score computation | Softmax and scaled dot-product are memory-bound | Batch multiple heads; ensure cache-friendly memory layout |
| KV-cache updates | Per-layer cache writes on each token | Pre-allocate arena; avoid reallocation during generation |
| GGUF tensor loading | Large model files incur I/O latency | Memory-map tensors; lazy loading for unused layers |
| Tokenizer BPE merge | O(n²) worst case for long sequences | Use optimized merge algorithm with priority queue |
| Graph lowering | Full graph traversal on every compile | Cache compiled plans in `.holo` archives |

---

## Optimization Techniques

### Memory

- **Arena allocation**: `BufferArena` pre-allocates inference scratch space; no per-op allocation
- **Buffer aliasing**: Memory planner reuses buffers for non-overlapping tensor lifetimes
- **Memory mapping**: Large weight tensors are mmap'd directly from `.gguf` or `.holo` files
- **KV-cache pre-sizing**: Allocate max context length upfront to avoid reallocation

### Compute

- **SIMD intrinsics**: Dequantization and element-wise ops use platform SIMD (SSE4, AVX2, NEON)
- **Quantized GEMM**: Q4/Q8 matrix multiplication via fused kernels when backend supports
- **Operation fusion**: `AttentionFusion`, `FFNFusion`, `QuantMatMulFusion` passes reduce kernel launch overhead
- **Constant folding**: Static shapes and broadcast operations folded at compile time

### Parallelism

- **Level-parallel scheduling**: Independent operations within a level execute concurrently
- **Batch head computation**: Multi-head attention processes heads in parallel
- **Rayon thread pool**: CPU backend uses work-stealing for parallel loops

### I/O

- **Lazy weight loading**: Only load tensors required for current execution path
- **Streaming tokenization**: Decode tokens as they're generated; don't buffer full sequence
- **Compiled archive caching**: `.holo` archives store pre-lowered plans for instant load

---

## Targets

| Metric | Target | Current |
|--------|--------|---------|
| TinyLlama-1.1B Q4_0 first token latency (CPU, M1) | < 500ms | 420ms |
| TinyLlama-1.1B Q4_0 tokens/sec (CPU, M1) | > 15 tok/s | 18 tok/s |
| GGUF import time (7B model) | < 2s | 1.8s |
| Memory overhead vs model size | < 1.2x | 1.15x |
| Optimization pass time (7B graph) | < 100ms | 85ms |
| Tokenizer encode throughput | > 100k tok/s | 125k tok/s |
