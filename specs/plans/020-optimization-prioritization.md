# Plan 020: Optimization Task Prioritization

## Context

The sprint has four remaining optimization areas (P2d, P3, P4, P5) with ~15
individual tasks. This plan assesses what has been done, what remains, and
provides a priority ordering based on impact, effort, and dependencies.

Design principle: hologram-ai is a compiler only (ADR-0016). All runtime
kernels live in hologram base. Tasks that require new FloatOps or kernel
changes are cross-repo and noted as such.

---

## Status Assessment

### P2d: Remaining decode optimizations — NOT STARTED (hologram base)

All three tasks exist only in Plan 018 and SPRINT.md. No implementation work
has begun in either repo:

| Task | Status | Repo | Notes |
|------|--------|------|-------|
| `dispatch_float_into` — buffer reuse | Not wired | hologram base | API exists, not connected to tape executor |
| `WeightCache` — cache deserialized quantized weights | Not wired | hologram base | Implementation exists at `kv/weight_cache.rs` |
| Level-aware tape execution for KV decode | Design only | hologram base | Split tape around KvWrite/KvRead per level |

**Key constraint:** f32 ONNX decode at 13.6 tok/s is near memory bandwidth
ceiling (4.1 GB weights x ~60 GB/s DDR = ~15 tok/s theoretical max). Further
speedup requires weight quantization (GGUF models).

### P3: Compiler fusion passes — PARTIALLY DONE

| Task | Status | Repo | Notes |
|------|--------|------|-------|
| SwiGLU fusion | **DONE** | hologram-ai | `swiglu_fusion.rs` wired into MVP pipeline |
| Add+RMSNorm residual fusion | Not started | Cross-repo | Needs `FloatOp::AddRmsNorm` + kernel in hologram base |
| QK-Norm + RoPE + KV-Store fusion | Design only | Cross-repo | Depends on stable tape executor |

### P4: Compilation speed — NOT STARTED

| Task | Status | Repo | Notes |
|------|--------|------|-------|
| Release profile with LTO | Not done | hologram-ai | Only `[profile.dev]` exists |
| Early convergence detection | Not done | hologram-ai | Fixpoint runs exactly 3 iterations; code duplicated 3x |
| Cache `topo_order` | Not done | hologram-ai | Called ~40x per compilation, builds 3 HashMaps each |
| Avoid double LLM compilation | Not done | hologram-ai | `compile_llm_pipeline` re-imports from disk |

### P5: Variable-length prefill — BLOCKED

| Task | Status | Repo | Notes |
|------|--------|------|-------|
| Wire ShapeContextGraph into execute() | Implemented but disabled | hologram-ai | SeqMode::Variable exists but disabled |
| SeqMode::Variable | Disabled | hologram-ai | Most recent commit disabled it |
| Hologram executor baked param resolution | **Blocker** | hologram base | FloatOp params (m/k/n, size) baked at compile time |

**Blocker:** hologram executor bakes FloatOp params at compile time. When
runtime buffer sizes differ from compiled values, results are wrong. Unblocking
requires hologram base to resolve baked params from runtime buffer sizes.

---

## Priority Order

### Tier 1: Quick wins — hologram-ai only, no dependencies

1. **Release profile with LTO** (P4)
   - Effort: Tiny (3 lines in Cargo.toml)
   - Impact: HIGH — 10-20% speedup for compilation and execution
   - File: `Cargo.toml`

2. **Extract shared `post_concretization_repair` function** (P4)
   - Effort: Small (refactor)
   - Impact: LOW directly, enables convergence detection
   - File: `crates/hologram-ai/src/compiler.rs`

3. **Early convergence detection in fixpoint loop** (P4)
   - Effort: Small
   - Impact: MEDIUM — saves up to 9 pass invocations
   - File: `crates/hologram-ai/src/compiler.rs`

### Tier 2: Compilation speed — hologram-ai only

4. **Cache `topo_order` on AiGraph** (P4)
   - Effort: Medium
   - Impact: MEDIUM — eliminates ~40 redundant HashMap constructions per compile
   - File: `crates/hologram-ai-common/src/ir/graph.rs`

5. **Avoid double LLM compilation** (P4)
   - Effort: Large
   - Impact: HIGH — saves ~50% of LLM compile time
   - File: `crates/hologram-ai/src/compiler.rs`

### Tier 3: Cross-repo — requires hologram base changes

6. **Wire `dispatch_float_into`** (P2d)
   - Effort: Medium (hologram base)
   - Impact: HIGH — eliminates ~1000 per-op allocations per decode token

7. **Wire `WeightCache` into tape executor** (P2d)
   - Effort: Medium (hologram base)
   - Impact: HIGH for GGUF — 5-10x overhead reduction for quantized weights

8. **Add+RMSNorm residual fusion** (P3)
   - Effort: Medium (cross-repo)
   - Impact: MEDIUM — 1 tensor + 1 dispatch eliminated per residual x N_layers

### Tier 4: Blocked / deferred

9. **Level-aware tape execution** (P2d) — tape executor maturity needed
10. **QK-Norm + RoPE + KV-Store fusion** (P3) — design first, cross-repo
11. **Variable-length prefill** (P5) — hologram executor baked param blocker

---

## Execution Plan

**Phase A (Tier 1):** Single commit. Release profile, fixpoint dedup,
convergence detection. All hologram-ai, no dependencies.

**Phase B (Tier 2):** Cached topo_order + avoid double LLM compilation.

**Phase C (Tier 3):** Coordinate with hologram base for dispatch_float_into,
WeightCache, and AddRmsNorm FloatOp.

**Phase D (Tier 4):** Park until prerequisites are met.

---

## Verification

- `cargo build --release` succeeds (Tier 1)
- `cargo test` passes after each change
- `cargo clippy -- -D warnings` clean
- LLM compilation time measured before/after Tier 2 changes
- Conformance tests pass after any fusion changes
