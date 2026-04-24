# Plan 078: Path to 60+ tok/s — MLX Studio-Inspired Optimizations

## Context

Current decode throughput is 38-43 tok/s (GGUF Q4, M4 Max). ONNX path is broken at 2.7 tok/s due to archive bloat. The AMX ceiling is ~43 tok/s single-thread — BLAS sgemm is saturated. Inspired by MLX Studio's techniques (JANG auto-quantization, 5-layer caching, speculative decoding, warmup), adapted for hologram-ai's compiler architecture.

**Key constraint:** On Apple Silicon CPU, BLAS already uses multiple cores internally. Level-parallel execution may contend with BLAS rather than help. Must benchmark before committing.

---

## Phase 1: Fix ONNX Q4 Archive Bloat (5 GB → ~1.7 GB)

**Root cause (confirmed by investigation):** `hologram::compile()` serializes all
`ConstantData::Bytes` into the graph bytes section. Each sub-archive (prefill,
decode, verify) carries its own copy of ~1.7 GB (400 MB quantized + 1.2 GB
inlined f32 embeddings/norms/biases). 3 × 1.7 GB = ~5.1 GB.

**Fix:** Use the existing `PipelineWriter::build_with_shared_weights()` and
`WeightStore` (BLAKE3 content dedup) in hologram base to store identical
constants once. The `WeightDedupIndex` mechanism already supports this —
it maps component names to (offset, size) in a shared blob. At load time,
`LoadedPipeline` grafts shared weights into each sub-archive.

**Implementation (hologram base):**
1. After compiling all 3 sub-archives, extract large `ConstantData::Bytes`
   constants from each graph
2. Feed them through `WeightStore::insert()` which deduplicates via BLAKE3
3. Build `WeightDedupIndex` with per-component constant offsets
4. Call `build_with_shared_weights(shared_blob, dedup_index)`
5. Each sub-archive keeps only its graph structure + small constants

**Implementation (hologram-ai):**
1. In compiler.rs, switch from `build_to_file` to `build_with_shared_weights`
2. Extract constant blobs from each compiled sub-archive
3. Build WeightStore from all three components' constants
4. Pass shared weights + dedup index to PipelineWriter

**Files (hologram base):**
- `crates/hologram-archive/src/writer/pipeline_writer.rs` (build_with_shared_weights)
- `crates/hologram-archive/src/weight/dedup.rs` (WeightStore, WeightDedupIndex)
- `crates/hologram-archive/src/loader/pipeline.rs` (resolve on load)

**Files (hologram-ai):**
- `crates/hologram-ai/src/compiler.rs` (pipeline assembly)

**Impact:** 5 GB → ~1.7 GB (single copy of all constants). Theoretical minimum ~0.6 GB
(just Q4 constants) requires moving constants to weight section.
**Effort:** ~200 lines across both repos, 2 days

---

## Phase 2: Mixed-Precision Quantization (quality win, enables Phase 5)

**Inspired by:** MLX Studio's JANG system — importance-aware bit allocation per layer type.

**Approach:** Instead of uniform Q4, use Q8 for attention weights (QKV/O projections) and Q4 for MLP weights (gate/up/down). hologram-ai already supports both Q4_0 and Q8_0.

**Implementation:**
1. Add `QuantStrategy::Mixed` variant (attention: Q8, mlp: Q4)
2. In `resolve_encodings.rs`, classify weights by tensor name:
   - `q_proj/k_proj/v_proj/o_proj/attn` → Q8
   - `gate_proj/up_proj/down_proj/mlp/fc1/fc2` → Q4
   - Embedding/lm_head → Q8 (boundary protection)
3. Propagate tensor name map from `builder.rs` to `resolve_encodings`
4. CLI: `--quantize mixed` or `--quantize q8:q4`

**Files:**
- `crates/hologram-ai-common/src/lower/resolve_encodings.rs`
- `crates/hologram-ai-common/src/lower/builder.rs` (QuantStrategy enum + name map)
- `crates/hologram-ai/src/commands/run_cmd.rs` (CLI flag)

**Impact:** +2-5% quality at same speed. Higher attention precision → better speculative acceptance rates in Phase 5.
**Effort:** ~150 lines, 1 day

---

## Phase 3: KV Cache Persistence / Prefix Caching (TTFT win)

**Inspired by:** MLX Studio's persistent disk cache + warmup — skip prefill for repeated/shared context.

**Implementation:**
1. **Serialize KvCacheState** — add `save_to_file`/`load_from_file` to `KvCacheState` in hologram base. Simple format: header (write_pos, config, layer count) + raw buffer data.
2. **PrefixCacheManager** (new file `prefix_cache.rs`) — BLAKE3 hash of token IDs as cache key, LRU eviction by mtime, configurable max cache size.
3. **Integration in run_cmd.rs** — after tokenization, check prefix cache. If hit, load KvCacheState, skip prefill, jump to decode. After generation, store to cache.
4. **CLI:** `--kv-persist` flag, `--kv-cache-dir DIR`

**Files:**
- `hologram/crates/hologram-exec/src/kv_cache.rs` (serialize/deserialize)
- `hologram-ai/crates/hologram-ai/src/prefix_cache.rs` (new)
- `hologram-ai/crates/hologram-ai/src/commands/run_cmd.rs` (integration)

**Impact:** TTFT drops from ~300ms to ~5ms for cached prefixes. No decode tok/s gain, but dramatic UX improvement for chat/multi-turn.
**Effort:** ~350 lines, 2 days

---

## Phase 4: Level-Parallel Tape Execution (43 → ~56 tok/s)

**Problem:** `execute_direct()` in tape.rs processes instructions sequentially even
when multiple instructions in the same level are independent. Level boundaries
exist (`level_offsets`) but are only used for weight prefetch and eviction.

**Benchmark results (confirmed):**
- TinyLlama Q4: default BLAS 37.7 tok/s, single-thread 26.6 tok/s (ratio 1.42x)
- Qwen2 Q8: default 19.0 tok/s, single-thread 14.5 tok/s (ratio 1.31x)
- BLAS gets 30-42% from multi-core — there IS headroom for level parallelism.

**Existing infrastructure (hologram base):**
- `EnumTape.level_offsets` partitions instructions by level
- `ExecutionSchedule` from Kahn's algorithm has `ParallelLevel.node_ids`
- `consumer_counts` for liveness tracking (needs atomic ops for parallel)
- `level_weight_ranges` for prefetch (stays at level boundaries)

**Implementation (hologram base, tape.rs execute_direct):**
1. At each level boundary, check if level has ≥ 2 instructions
2. For parallel levels: spawn rayon tasks for `instructions[level_start..level_end]`
3. Each task reads from shared arena (immutable inputs), writes to pre-allocated
   output buffer (disjoint indices — safety invariant from tape builder)
4. Use `AtomicU32` for consumer_counts decrements in parallel eviction
5. Sync at level boundaries for weight prefetch and eviction

**Key concern:** Output buffer arena needs `&self` for reads and disjoint `&mut`
for writes. Options:
- `UnsafeCell<Vec<OutputBuffer>>` with debug assertions on disjoint access
- Split arena into immutable input view + write-only output slots

**Files (hologram base):**
- `crates/hologram-exec/src/tape.rs` (execute_direct, lines 1291-1729)
- `crates/hologram-exec/src/buffer/arena.rs` or output_buffer.rs (thread-safe access)

**Impact:** 43 × 1.3 = ~56 tok/s (confirmed headroom from benchmark)
**Effort:** ~200 lines, 3 days

---

## Phase 5: Separate-Model Speculative Decoding (56 → 67-84 tok/s)

**Key insight:** Plan 056 concluded speculative doesn't help because verification is O(N) — but that assumed self-speculative (same model). With a separate smaller draft model, draft cost is negligible.

**Implementation:**
1. **Draft model loading** — add `draft_plan`, `draft_tape`, `draft_weight_cache` to `HoloRunner`.
2. **Dual-model speculative loop** — modify `speculative.rs` for draft/target architecture.
3. **KV synchronization** — separate `KvCacheState` per model, sync on accept/reject.
4. **CLI:** `--draft-model PATH` flag.

**Files:**
- `hologram-ai/crates/hologram-ai/src/runner.rs` (draft model loading)
- `hologram-ai/crates/hologram-ai/src/speculative.rs` (dual-model loop)
- `hologram-ai/crates/hologram-ai/src/commands/run_cmd.rs` (CLI, draft KV init)

**Impact:** Conservative 56 × 1.2 = ~67 tok/s. Optimistic 56 × 1.5 = ~84 tok/s.
**Effort:** ~400 lines, 4 days

---

## Sequencing

```
Step 0: Benchmarks (BLAS saturation + batch amortization)
Step 1: Phase 1 (ONNX Q4 fix) — immediate 15x for ONNX
Step 2: Phase 2 (mixed precision) — quality win
Step 3: Phase 4 (level parallel) — only if benchmark shows headroom
Step 4: Phase 3 (KV persistence) — UX win
Step 5: Phase 5 (draft speculative) — final push to 60+
```

## Decisions

- Always download/compile from ONNX (not GGUF)
- hologram base changes should be independent PRs that also integrate with hologram-ai
- macOS first → WASM second → x86_64 third
