# Self-Describing Tensor Headers for hologram Runtime

## Context

Every runtime bug in this session — KV cache decode gibberish, batched MatMul wrong output size, Transpose no-op, shape mismatches — traces to the same root cause: **tensor buffers carry no metadata**. Kernels infer shapes from buffer sizes using hardcoded stride divisions, and every inference is an assumption that can be wrong.

The fix: attach a lightweight metadata header to every buffer in the arena, making tensors self-describing. Like TCP headers carry port/seq/flags with payload, tensor headers carry shape/dtype/layout with data.

## Design: `TensorMeta`

```rust
/// Lightweight metadata attached to every buffer in the arena.
/// 40 bytes fixed-size, no heap allocation. Stored in a parallel Vec
/// alongside buffers — NOT embedded in buffer bytes (zero-copy preserved
/// for mmap'd data). O(1) access by NodeId index.
///
/// Serializable via rkyv for archive persistence.
#[derive(Clone, Copy, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct TensorMeta {
    /// Number of dimensions (0 = scalar, max 8).
    pub ndim: u8,
    /// Element data type.
    pub dtype: FloatDType,
    /// Actual shape at runtime (not compiled shape).
    /// First `ndim` entries are valid.
    pub dims: [u32; 8],
}
```

Key design decisions:
- **Fixed-size `[u32; 8]`, `Copy`** — 40 bytes, fits in one cache line, no heap allocation
- **rkyv-serializable** — can persist in archives for shape validation at load time
- **Stored in parallel `Vec<Option<TensorMeta>>`** in BufferArena — NOT embedded in buffer bytes. Mmap'd buffers remain zero-copy. Buffer access `get(NodeId) → &[u8]` is unchanged.
- **O(1) metadata access** — `get_meta(NodeId) → Option<&TensorMeta>` is a single index lookup into the parallel vec, same pattern as existing `elem_sizes: Vec<u8>`
- **Zero `Vec<T>` per tensor** — `dims: [u32; 8]` is fixed-size, `Copy`, stack-allocated. No heap allocation per metadata entry. The only `Vec` is the arena's `metas: Vec<Option<TensorMeta>>` container (one allocation, parallel to existing `buffers` Vec). Avoids the double-storage and copy costs of `Vec<usize>` shapes.
- **Zero performance overhead on insert** — setting metadata is a 40-byte struct copy during `swap_insert` which already does `Vec::take` + `Option::replace` (the metadata copy is negligible vs the buffer swap)
- **Populated at every write** — swap_insert, move_slot, insert_borrowed all set metadata
- **Validated at kernel dispatch** — kernels check metadata matches their parameters (debug builds only for validation, zero cost in release)

## Implementation Plan

### Phase 1: Add TensorMeta to BufferArena (hologram base)

**File**: `crates/hologram-exec/src/buffer/arena.rs`

1. Add `metas: Vec<Option<TensorMeta>>` parallel to `buffers`
2. Update `swap_insert_with_elem_size` → `swap_insert_with_meta(id, buf, meta)`
3. Update `insert_borrowed` → `insert_borrowed_with_meta(id, data, meta)`
4. Add `get_meta(NodeId) → Option<&TensorMeta>`
5. Keep backward-compat: existing `swap_insert_with_elem_size` creates TensorMeta from elem_size + buffer length (1-D shape inferred)

### Phase 2: Populate TensorMeta at tape build time

**File**: `crates/hologram-exec/src/tape_builder.rs`

1. Add `output_meta: Option<TensorMeta>` to `TapeInstruction`
2. Populate from `node_shapes_map()` + `node_dtypes_map()` at build time
3. For ops with dynamic output shapes (variable-length), set `dims` to compiled shape (can be updated at runtime)

### Phase 3: Propagate through execution

**File**: `crates/hologram-exec/src/tape.rs`

1. After each kernel dispatch, store `output_meta` in the arena alongside the buffer
2. For passthrough/in-place ops, propagate input meta to output
3. For KvWrite: set output meta from cache state (actual seq, not compiled)
4. For InlineTranspose: compute output meta from input meta + perm

### Phase 4: Kernel validation (debug builds)

**File**: `crates/hologram-exec/src/float_dispatch/`

1. `dispatch_attention`: validate Q meta matches `[q_heads, seq_q, head_dim]` or `[seq_q, q_heads, head_dim]`
2. `dispatch_matmul`: validate A meta's last dim matches compiled k
3. `dispatch_kv_write`: validate input meta's stride matches `n_kv_heads * head_dim`
4. All validation behind `#[cfg(debug_assertions)]` — zero cost in release

### Phase 5: KV cache uses TensorMeta

**File**: `crates/hologram-exec/src/tape.rs` (KvWrite/KvRead dispatch)

1. KvWrite reads `heads_first` from TapeKernel AND validates against input meta's layout
2. KvWrite output meta has actual `total_seq` (not compiled) — eliminates stride inference
3. KvRead output meta matches the format the attention kernel expects

### Phase 6: Fix the actual attention layout bug

With TensorMeta in place, the fix becomes mechanical:
1. `find_pre_transpose_with_scale` returns a tensor — check its meta
2. If meta shows `[seq, kv_heads * head_dim]` (flat), the data isn't heads-first
3. KvWrite uses meta to determine whether transpose is needed
4. No more guessing from buffer length

---

## Files to Modify

| File | Change |
|------|--------|
| `hologram-exec/src/buffer/arena.rs` | Add `TensorMeta`, `metas` vec, meta-aware insert/get |
| `hologram-exec/src/tape.rs` | `TapeInstruction.output_meta`, propagate in execute loop |
| `hologram-exec/src/tape_builder.rs` | Populate `output_meta` from graph shapes |
| `hologram-exec/src/float_dispatch/attention.rs` | Debug validation of input metas |
| `hologram-exec/src/float_dispatch/matmul.rs` | Debug validation of input metas |
| `hologram-exec/src/kv_cache.rs` | Use meta for seq_len instead of stride division |
| `hologram-core/src/op/float_op.rs` | Add `TensorMeta` type (shared across crates) |

## Verification

1. All existing tests pass (backward compat via 1-D shape inference)
2. `tinyllama_logit_conformance` — prefill matches ORT
3. `tinyllama_decode_conformance` — decode matches ORT (the real test)
4. BERT + ResNet E2E tests pass
5. Debug build catches any shape mismatch via assertions

## Phase 7: Remove `force_single_graph` (hologram-ai)

Single-graph LLM compilation is a dead path:
- Decode runs at compiled seq=2048 for every single token (~2000x slower)
- KV cache interaction with single-graph is buggy and untestable
- Multi-component models (Whisper, SD, CALM) require pipeline by definition

Remove:
- `ModelCompiler.force_single_graph: bool` field
- `--single-graph` CLI flag
- Single-graph KV cache fallback in `HoloRunner::execute_with_kv`
- Single-graph KV cache setup in `run_cmd.rs`
- Related conformance tests that use `force_single_graph`

Keep pipeline as the ONLY LLM compilation path.

---

## What This Solves

| Bug | Current Root Cause | With TensorMeta |
|-----|-------------------|-----------------|
| KV cache decode gibberish | KvWrite infers seq from `len/stride`, wrong for flat data | Meta carries actual `[seq, heads, dim]` shape |
| Batched MatMul wrong size | `m = a_len / k` ignores batch dims | Meta has full N-D shape, batch detected from dims |
| Transpose no-op | No shape info at dispatch time | Meta carries input shape for physical transpose |
| `seq_k - seq_q` overflow | Wrong seq inferred from K buffer | Meta validates `seq_k >= seq_q` before attention |
| Single-graph confusion | Two code paths with different layout assumptions | Pipeline-only, one code path |
