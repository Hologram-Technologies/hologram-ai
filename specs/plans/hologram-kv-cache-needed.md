# Cross-Repo Change Request: KV-Cache Support in hologram

**Status:** Requested
**Date:** 2026-03-08
**Target repo:** hologram (base crate)
**Requesting repo:** hologram-ai
**Depends on:** `hologram-types-needed.md` (native float ops)

---

## Context

Autoregressive LLM inference re-computes the full attention sequence on every
token generation step. For a 1024-token context with TinyLlama 1.1B, this means
~215 seconds per step in debug builds and ~4-20 seconds in release builds. With
KV-cache, only the **new token's** K and V projections are computed; previous
K/V values are read from a persistent cache buffer. This reduces per-step cost
from O(seq_len^2) to O(seq_len).

### Current state

| Component | Status |
|-----------|--------|
| `FloatOp::Attention` kernel | Exists in `float_dispatch.rs` — computes full Q×K^T→softmax→V each call |
| `KvStore` | Exists — but "KV" means key-value dispatch table, not KV-cache |
| `KvCacheLayout` in hologram-ai | Computes total cache bytes, but returns `none()` (unused) |
| `LlmMetaSection` in hologram-ai | Has `n_kv_heads`, `head_dim`, `max_seq_len`, `kv_cache_bytes` fields |
| `LayerDescriptor` / `TensorPort` | Archive infrastructure exists, not populated for prefill/decode |
| Cache read/write ops | **Do not exist** |
| Prefill/decode dual-graph | **Not implemented** |

---

## 1. KV-Cache Graph Ops (Priority: Critical)

Add two new `FloatOp` variants for stateless cache slot access:

```rust
pub enum FloatOp {
    // ... existing ops ...

    /// Read cached K or V tensor for a layer up to `present_len`.
    /// Inputs: [cache_buffer]
    /// Params: layer index, present_len (from GraphInputs)
    /// Output: K or V slice [present_len, num_kv_heads, head_dim]
    KvCacheRead {
        layer: u32,
        is_value: bool,  // false = K cache, true = V cache
    },

    /// Write new K or V entries at position `present_len` in the cache.
    /// Inputs: [cache_buffer, new_kv_data]
    /// Params: layer index, present_len
    /// Output: updated cache_buffer (or void — mutation)
    KvCacheWrite {
        layer: u32,
        is_value: bool,
    },

    /// Attention with KV-cache: Q attends to cached K/V.
    /// Inputs: [Q, cached_K, cached_V]
    /// Unlike regular Attention which takes fresh K/V.
    CachedAttention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: Option<f32>,
        causal: bool,
    },
}
```

### Why stateless ops, not a session object

hologram is a **stateless kernel executor**. The consumer (hologram-ai or
hologram-cli) owns the cache buffer and passes it as a `GraphInput`. The ops
read/write at offsets computed from `present_len` (also a `GraphInput`). No
runtime session state lives inside hologram.

### Execution semantics

```
Cache buffer layout (per layer, per K/V):
  [max_seq_len × num_kv_heads × head_dim] f32 values

KvCacheWrite(layer=2, is_value=false):
  inputs[0] = cache_buffer (full)
  inputs[1] = new_k [1, num_kv_heads, head_dim]  (single position)
  params: present_len
  effect: writes new_k at cache_buffer[present_len, :, :]

KvCacheRead(layer=2, is_value=false):
  inputs[0] = cache_buffer (full)
  params: present_len
  output: cache_buffer[0..present_len+1, :, :]  (slice view)

CachedAttention:
  inputs: [Q, cached_K_slice, cached_V_slice]
  Q shape:       [1, num_heads, head_dim]         (single new token)
  cached_K shape: [present_len+1, num_kv_heads, head_dim]
  cached_V shape: [present_len+1, num_kv_heads, head_dim]
  output:        [1, num_heads, head_dim]
```

---

## 2. Special GraphInputs (Priority: Critical)

The executor needs a way to receive per-call scalar parameters:

```rust
// Reserved GraphInput names for KV-cache:
"kv_cache_k"     // &mut [u8] — full K cache buffer (all layers interleaved)
"kv_cache_v"     // &mut [u8] — full V cache buffer
"present_len"    // u64 — current sequence position
```

The consumer allocates these buffers once per session and passes them on
every `execute()` call, incrementing `present_len` each step.

### Mutable inputs

Current `GraphInputs` uses `Vec<u8>` (owned, cloned). For KV-cache to work
efficiently, the cache buffer must be **mutated in place** rather than
copied on every call:

```rust
pub enum GraphInput {
    Owned(Vec<u8>),
    MutableRef(*mut u8, usize),  // pointer + length
}
```

Or alternatively, `GraphInputs` could use `&mut [u8]` slices with lifetime
bounds. The exact API is up to hologram's design.

---

## 3. Dual-Graph Archive Layout (Priority: High)

For LLM inference, the archive should contain **two sub-graphs**:

| Graph | Purpose | When used |
|-------|---------|-----------|
| `prefill` | Process full prompt at once | First call only |
| `decode` | Process single new token with cached K/V | Every subsequent call |

The prefill graph:
- Takes `input_ids [seq_len]`, `kv_cache_k`, `kv_cache_v`
- Computes all K/V projections and writes them to cache
- Returns logits `[seq_len, vocab_size]`

The decode graph:
- Takes `input_ids [1]`, `kv_cache_k`, `kv_cache_v`, `present_len`
- Reads cached K/V, computes only the new token's K/V
- Returns logits `[1, vocab_size]`

### Archive representation

Use existing `LayerDescriptor` infrastructure:

```rust
// In the archive, two named layers:
LayerDescriptor {
    entrypoint: LayerEntrypoint::Named("lm.prefill"),
    inputs: vec![
        TensorPort { name: "input_ids", shape: vec![0], dtype: I64 },
        TensorPort { name: "kv_cache_k", shape: vec![...], dtype: F32 },
        TensorPort { name: "kv_cache_v", shape: vec![...], dtype: F32 },
    ],
    outputs: vec![
        TensorPort { name: "logits", shape: vec![0, vocab_size], dtype: F32 },
    ],
}
LayerDescriptor {
    entrypoint: LayerEntrypoint::Named("lm.decode"),
    inputs: vec![
        TensorPort { name: "input_ids", shape: vec![1], dtype: I64 },
        TensorPort { name: "kv_cache_k", shape: vec![...], dtype: F32 },
        TensorPort { name: "kv_cache_v", shape: vec![...], dtype: F32 },
        TensorPort { name: "present_len", shape: vec![1], dtype: U64 },
    ],
    outputs: vec![
        TensorPort { name: "logits", shape: vec![1, vocab_size], dtype: F32 },
    ],
}
```

### `execute_layer()` API

```rust
impl KvExecutor {
    /// Execute a named layer/sub-graph within a pipeline archive.
    pub fn execute_layer(
        &mut self,
        archive_bytes: &[u8],
        layer_name: &str,
        inputs: GraphInputs,
    ) -> Result<GraphOutputs>;
}
```

---

## 4. What hologram-ai Changes (After hologram adds support)

Once hologram ships the above, hologram-ai will:

1. **Add `AiOp::KvCacheRead { layer }` and `AiOp::KvCacheWrite { layer }`** to
   the AI IR
2. **Implement dual-graph lowering** — the compiler produces two `hologram::Graph`
   instances (prefill + decode) from a single `AiGraph`
3. **Write pipeline archives** using `PipelineWriter` with `LayerDescriptor`
   headers for both sub-graphs
4. **Populate `LlmMetaSection`** with real `KvCacheLayout` data (n_layers,
   n_kv_heads, head_dim, max_seq_len, total bytes)
5. **Delete `KvCacheLayout::none()` fallback** — always compute real layout
   for LLM models

### Generation loop changes (in hologram-cli `run_cmd.rs`)

```rust
// Allocate cache once
let cache_k = vec![0u8; meta.kv_cache_bytes / 2];
let cache_v = vec![0u8; meta.kv_cache_bytes / 2];
let mut present_len: u64 = 0;

// Prefill
let mut inputs = GraphInputs::new();
inputs.insert("input_ids", encode_tokens(&prompt_ids));
inputs.insert("kv_cache_k", &mut cache_k);
inputs.insert("kv_cache_v", &mut cache_v);
let logits = executor.execute_layer(archive, "lm.prefill", inputs)?;
present_len = prompt_ids.len() as u64;

// Decode loop
loop {
    let next_id = argmax(&logits);
    let mut inputs = GraphInputs::new();
    inputs.insert("input_ids", encode_token(next_id));
    inputs.insert("kv_cache_k", &mut cache_k);
    inputs.insert("kv_cache_v", &mut cache_v);
    inputs.insert("present_len", present_len.to_le_bytes());
    let logits = executor.execute_layer(archive, "lm.decode", inputs)?;
    present_len += 1;
    // ...
}
```

---

## 5. Performance Impact

| Metric | Without KV-cache | With KV-cache |
|--------|-----------------|---------------|
| Per-step compute | O(seq_len^2 × d) | O(seq_len × d) |
| Memory | O(1) (recomputed) | O(n_layers × seq_len × head_dim) |
| TinyLlama 1.1B, 128 tokens (release) | ~4-20s/step | ~0.1-0.5s/step |

For TinyLlama 1.1B with 22 layers, 4 KV heads, 64 head_dim, 2048 max_seq_len:
- K cache: 22 × 4 × 64 × 2048 × 4 bytes = ~45 MB
- V cache: same = ~45 MB
- Total: ~90 MB (easily fits in RAM)

---

## Priority Summary

| Item | Priority | Blocking? |
|------|----------|-----------|
| `FloatOp::KvCacheRead/Write/CachedAttention` | **Critical** | Yes |
| Mutable `GraphInputs` (or buffer refs) | **Critical** | Yes |
| `execute_layer()` API | **High** | Yes |
| `LayerDescriptor` population for prefill/decode | **High** | Yes |
| `LlmMetaSection` with real `KvCacheLayout` | Medium | No (workaround exists) |

---

## Relationship to `hologram-types-needed.md`

This spec **depends on** native float ops being added first. KV-cache ops
(`KvCacheRead`, `KvCacheWrite`, `CachedAttention`) are `FloatOp` variants
that operate on f32 buffers. Without `GraphOp::Float(FloatOp)`, there's no
place to put them.

Implementation order:
1. Native float ops (`hologram-types-needed.md`) — unblocks lowering
2. KV-cache ops (this spec) — unblocks efficient inference
3. Dual-graph archives — unblocks production LLM serving
