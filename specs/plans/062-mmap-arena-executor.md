# Plan 062: Mmap-Backed Arena Executor

**Status:** Ready for implementation
**Created:** 2026-04-09
**Scope:** hologram-exec (hologram base)
**Branch:** `feat/sd-correctness-and-output-types` (continues Plan 061)

## Context

The SD v1.5 pipeline OOMs during execution because `execute_direct` uses `Vec<Vec<u8>>` for activation buffers. Even with runtime eviction (dropping Vecs after their last consumer), the system allocator retains freed pages in its free-list — RSS never drops. `madvise(MADV_FREE)` on malloc'd memory is unreliable (the allocator manages its own page lifecycle). The pipeline peaks at 49 GiB RSS during UNet forward passes, triggering the OOM killer during VAE decode.

The `MmapLender` infrastructure (built in `buffer/lent.rs`) provides a contiguous mmap-backed region with `advise_free_region()` for reliable page reclamation. This plan wires it into the executor as the backing for all activation buffers when `checkpoint_enabled` is true.

## Design

### New type: `OutputBuffer`

Replace `&mut Vec<u8>` in all kernel signatures with `&mut OutputBuffer`. `OutputBuffer` is an enum:

```rust
pub enum OutputBuffer {
    /// Heap-allocated (default path, LLMs). Owns its allocation.
    Heap(Vec<u8>),
    /// Arena-backed (memory-pressure path, diffusion models).
    /// Points into a contiguous MmapLender region. Fixed capacity.
    Arena {
        /// Raw pointer to the start of this buffer's region within the arena.
        ptr: *mut u8,
        /// Current logical length (bytes written so far).
        len: usize,
        /// Maximum capacity (slot size in the arena, fixed at tape build time).
        capacity: usize,
    },
}
```

`OutputBuffer` implements the subset of `Vec<u8>` that kernels actually use:

| Method | Heap | Arena |
|--------|------|-------|
| `len()` | `vec.len()` | `self.len` |
| `clear()` | `vec.clear()` | `self.len = 0` |
| `resize(n, val)` | `vec.resize(n, val)` | assert `n <= capacity`, memset, set `len = n` |
| `extend_from_slice(s)` | `vec.extend_from_slice(s)` | assert `len + s.len() <= capacity`, copy, advance len |
| `as_ptr()` | `vec.as_ptr()` | `self.ptr` |
| `as_slice() -> &[u8]` | `&vec[..]` | `slice::from_raw_parts(ptr, len)` |
| `as_mut_slice() -> &mut [u8]` | `&mut vec[..]` | `slice::from_raw_parts_mut(ptr, len)` |
| `is_empty()` | `vec.is_empty()` | `self.len == 0` |
| `capacity()` | `vec.capacity()` | `self.capacity` |

**Alignment:** Arena regions are 16-byte aligned (enforced by the slot offset computation in `compute_slot_assignments`). This guarantees `bytemuck::cast_slice_mut::<u8, f32>` succeeds without the `alloc_f32_in` slow path.

**`alloc_f32_in` adaptation:** The helper becomes:
```rust
fn alloc_f32_in(out_buf: &mut OutputBuffer, n: usize) -> &mut [f32] {
    let start = out_buf.len();
    out_buf.resize(start + n * 4, 0);
    bytemuck::cast_slice_mut(&mut out_buf.as_mut_slice()[start..])
}
```

### Executor flow (checkpoint_enabled = true)

1. **Compute arena size:** For each slot (from `slot_assignments`), find the max `output_byte_hint` among all nodes assigned to that slot. Sum all slot sizes (16-byte aligned) to get total arena size.

2. **Allocate arena:** `MmapLender::new(total_arena_bytes)`. One contiguous mmap region.

3. **Create OutputBuffers:** For each node, create `OutputBuffer::Arena { ptr, len: 0, capacity }` pointing into the MmapLender at the node's slot offset.

4. **Execute instructions:** Kernels write via `&mut OutputBuffer` — identical API to before.

5. **Evict:** When a node's live count reaches zero, call `lender.advise_free_region(slot_offset, slot_capacity)` to return pages to the OS. Reset `OutputBuffer::Arena.len = 0`. The region remains mapped (no munmap) so the next node assigned to the same slot can reuse it by writing from offset 0.

6. **Writeback:** At the end, copy surviving output buffers into owned `Vec<u8>`s for insertion into the `BufferArena` (callers expect owned data).

### Executor flow (checkpoint_enabled = false, default)

No change. Use `OutputBuffer::Heap(Vec::with_capacity(hint))` as before. Pre-allocation fires. No eviction. This is the LLM path.

## Files to modify

| File | Change |
|------|--------|
| **New:** `hologram-exec/src/buffer/output_buffer.rs` | `OutputBuffer` enum + trait impls |
| `hologram-exec/src/buffer/mod.rs` | Add `pub mod output_buffer;` |
| `hologram-exec/src/float_dispatch/helpers.rs` | `alloc_f32_in` takes `&mut OutputBuffer` |
| `hologram-exec/src/float_dispatch/elementwise.rs` | `unary_map_into`, `binary_map_into` signatures |
| `hologram-exec/src/float_dispatch/matmul.rs` | `dispatch_matmul_into` and related |
| `hologram-exec/src/float_dispatch/mod.rs` | `dispatch_float_into` and wrappers |
| `hologram-exec/src/float_dispatch/norm.rs` | GroupNorm/LayerNorm/RmsNorm `_into` variants |
| `hologram-exec/src/float_dispatch/conv.rs` | Conv2d (no `_into` variant yet, but result → `extend_from_slice`) |
| `hologram-exec/src/tape.rs` | `dispatch_kernel`, `execute_direct`, inline kernels, LUT-GEMM |
| `hologram-exec/src/backend/mod.rs` | `BackendDispatch` trait methods |
| `hologram-exec/src/backend/cpu.rs` | CPU backend impl |
| `hologram-exec/src/backend/metal.rs` | Metal backend impl |
| `hologram-exec/src/backend/webgpu.rs` | WebGPU backend impl |
| `hologram-exec/src/kv/store.rs` | KV cache write paths |
| `hologram-exec/src/lib.rs` | Re-export `OutputBuffer` |

**Total: ~44 function signatures to update + 1 new file + executor wiring.**

## Migration strategy

1. Create `OutputBuffer` with `From<Vec<u8>>` and `Into<Vec<u8>>` conversions.
2. Add `Deref<Target=[u8]>` and `DerefMut` impls.
3. Update `alloc_f32_in` first (used by ~20 kernels indirectly).
4. Update `dispatch_kernel` to pass `&mut OutputBuffer` instead of `&mut Vec<u8>`.
5. Update each kernel file one at a time, verifying `cargo build` after each.
6. Wire the arena allocation into `execute_direct` for the `checkpoint_enabled` path.
7. Run the full hologram-exec test suite (1311 tests) to verify no regression.
8. Run the SD pipeline test to verify memory stays bounded.

## Verification

```
# Unit tests (must all pass)
cargo test -p hologram-exec --release

# TinyLlama baseline (must be ≥ 40 tok/s, checkpoint_enabled = false)
RUST_LOG=info cargo run --release -- run \
  models/TinyLlama-1.1B-Chat-v1.0/model.holo \
  --prompt "Question: What is the capital of France? Answer:" \
  --max-tokens 15 --temperature 0.0 --stop $'\n'

# SD pipeline (VAE must complete without OOM at spatial_scale=4)
cargo test --release -p hologram-ai --features e2e -- sd_pipeline_generates_image --nocapture

# Memory gate: peak RSS during UNet forward < 8 GiB
# (down from 49 GiB with Vec<Vec<u8>> + no eviction)
```

## Non-goals

- Changing the `BufferArena` API (that's the graph-level arena for I/O, not the tape-level working memory)
- Adding mmap support to WASM (the arena path is gated behind `checkpoint_enabled`; WASM models won't set it)
- Optimizing the non-checkpoint path (it works fine for LLMs already)
