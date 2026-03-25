# Plan 024: TinyLlama E2E Inference — Continue from hologram fixes

## Context

We're making ONNX models with dynamic seq_len work at runtime without `--seq-len`. The hologram runtime fixes are done (Slice axis inference, attention validation, zero-copy weight loading). Now we need to fix hologram-ai to get TinyLlama running end-to-end.

## What's already changed in hologram-ai (this session, uncommitted)

1. **GQA attention fusion** (`crates/hologram-ai-common/src/opt/attention_fusion.rs`): Added `trace_past_expand()` to walk K/V back past Expand/Reshape/Unsqueeze ops that are no-ops at runtime. Uses the un-expanded tensor's head count as `num_kv_heads` (4 for TinyLlama, not 32). Fused attention op now receives un-expanded K/V inputs.

2. **Zero-copy mmap loading** (`crates/hologram-ai/src/compiler.rs`): `HoloRunner::from_path()` uses `memmap2::Mmap`. `ArchiveStorage` enum (Owned/Mmap). `from_storage()` uses `load_from_bytes_zero_copy` (new in hologram) to skip CRC32 and borrow weights directly. Loading went from 30s → 5s.

3. **`--single-graph` flag** (`crates/hologram-ai/src/cli.rs`): Bypasses pipeline prefill/decode split. Produces smaller archives and avoids the 2x weight duplication.

4. **shape_ctx loading** (`read_shape_context_from_archive`): Changed to use `load_from_bytes_zero_copy`.

5. **`run_cmd.rs`**: Uses `HoloRunner::from_path()` instead of `fs::read` + `from_bytes`.

## 3 remaining blockers

### 1. `constant not found: 96` (single-graph only)

Constant 96 has `offset=1,449,500,672 size=46,137,344` — its end (1,495,638,016) exceeds the weight region (1,458,071,559) by 37.6MB. The weight packing in the single-graph compilation path doesn't properly account for all constants. The store has 207 entries and the graph references them correctly — it's just that the last ~37MB of weight data is truncated.

This does NOT affect pipeline archives (they work but are slow to load).

### 1b. Pipeline sub-archive extraction copies 2GB

`extract_sub_archive_bytes` copies the sub-archive from the mmap into a new Vec. The pipeline header already stores `(offset, size)` for each sub-model. Fix: compute the byte range from the header and pass `&mmap[weights_start + entry.offset .. weights_start + entry.offset + entry.size]` directly to `load_from_bytes_zero_copy`. No copy needed.

### 2. Graph deserialization is 1.5s for 199MB

rkyv `from_bytes` fully deserializes the graph 3 times during loading (pipeline probe, plan load, shape_ctx). Fix: parse once and reuse, or use rkyv zero-copy access. The 199MB graph size itself is suspicious — check if compiled shape annotations or debug info are inflating it.

### 3. MatMul `infer_matmul_k` produces 384x oversized outputs

For a 2-token prompt, Q has 6,291,456 elements instead of 16,384 (expected 2×32×64). The attention kernel gets `seq_q=3072` instead of `2`. This means some upstream MatMul infers wrong dimensions. Compiled `m=2048, k=2048, n=2048` with `a_len=16384` should give `k=2048, m=8, n=2048` — and it does for Q projection. The inflation must come from an earlier op in the graph. Add `eprintln` in `infer_matmul_k` (in hologram's `matmul.rs`) when output exceeds compiled m×n by >16x to find which MatMul is failing.

## What's done in hologram (separate repo, also uncommitted)

- `infer_slice_axis_size()` in `float_dispatch/mod.rs` — fixes Slice for dynamic leading dims
- Attention buffer validation in `attention.rs` — K/V size match, divisibility checks
- `load_from_bytes_zero_copy` in `hologram-archive/src/loader/bytes.rs` — zero-copy weight loading via `Cow<'static, [u8]>` on LoadedPlan
- `load_from_bytes_unchecked` — skips CRC32 checksum
- Debug eprints in `matmul.rs`, `tape.rs`, `bytes.rs` that need cleanup

## How to reproduce

```bash
cd hologram-ai
# Compile (uses GQA-fixed attention fusion + single-graph)
./target/release/hologram-ai compile \
  --model models/TinyLlama-1.1B-Chat-v1.0/model_causal.onnx \
  --tokenizer models/TinyLlama-1.1B-Chat-v1.0/tokenizer.json \
  --single-graph --output /tmp/tinyllama_single

# Run (hits "constant not found: 96")
./target/release/hologram-ai run /tmp/tinyllama_single/model_causal.holo \
  --prompt "Hi" --max-tokens 1 --temperature 0.0
```

## Goal

`hologram-ai run model.holo --prompt "What is the capital of France?" --max-tokens 32` produces a coherent English response in under 2 seconds on CPU.
