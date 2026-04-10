# Plan 062: OutputBuffer + Mmap-Backed Eviction

**Status:** Phase 1 complete, Phase 2 planned
**Created:** 2026-04-09
**Updated:** 2026-04-10
**Scope:** hologram-exec (hologram base)
**Branch:** `feat/sd-correctness-and-output-types`

## Phase 1: OutputBuffer type + Mmap eviction (DONE)

Replaced `&mut Vec<u8>` in all 44 kernel dispatch signatures with `&mut OutputBuffer`, a three-variant enum:

- **Heap(Vec<u8>)**: default for LLMs. Zero overhead vs bare Vec.
- **Arena { ptr, len, capacity }**: borrows from MmapLender (for future block-level tiling).
- **Mmap(MmapBuffer)**: individually mmap'd. On drop, `munmap` immediately returns pages to the OS.

When `checkpoint_enabled = true`, buffers ≥256 KiB use `OutputBuffer::Mmap`. Eviction drops the buffer, calling `munmap` — RSS drops during execution (verified: 26 GiB → 22 GiB during SD UNet).

### Verified
- 1351 hologram-exec tests pass
- TinyLlama ONNX: 40.6 tok/s (Heap path, zero overhead)
- SD pipeline: denoising completes, munmap reclaims pages

### Limitation
The SD UNet's live working set at peak is ~20 GiB due to cross-block skip connections (residual adds spanning 100+ instructions). Mmap eviction reclaims pages as soon as buffers' last consumer finishes, but the simultaneous live set of one UNet forward pass is inherently large. RSS stays at ~20 GiB because the model itself requires that much concurrent memory.

## Phase 2: Block-level graph tiling (PLANNED)

To get under 1 GiB for SD UNet, the executor must process one transformer block at a time:

1. **Block identification**: analyze the tape's instruction graph to identify transformer block boundaries (22 blocks in SD UNet). Each block has self-attention + cross-attention + feedforward.

2. **Cross-block checkpointing**: skip connections that cross block boundaries (the residual add that connects block N's input to block N+1's output) are evicted after block N finishes and recomputed when block N+1 needs them.

3. **Per-block execution**: for each block, execute all its instructions (50-80 per block), then drop ALL non-checkpoint intermediates. Peak memory = one block's working set (~500 MiB) + checkpoint buffers (~200 MiB).

4. **Tape partitioning**: split `EnumTape::instructions` into block segments at compile time (via `level_offsets` or a new `block_offsets` annotation). The executor loops over blocks, calling `execute_direct` per block with isolated buffer pools.

### Files to modify
- `hologram-exec/src/tape.rs` — block-level execution loop
- `hologram-exec/src/tape_builder.rs` — block boundary annotation
- `hologram-ai-common/src/lower/builder.rs` — emit block markers during lowering
- `hologram-ai/src/compiler.rs` — propagate block structure through compilation

### Target
- SD UNet forward: < 1 GiB peak RSS
- No regression on TinyLlama (< 5 GiB, no block tiling needed)
