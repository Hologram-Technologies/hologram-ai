# Plan 071 — Rayon/Tokio Thread Hygiene

**Status**: draft
**Scope**: hologram (base repo), minor touch in hologram-ai
**Reference**: https://posthog.com/blog/untangling-rayon-and-tokio

## Context

Both hologram and hologram-ai use Rayon (data parallelism) and Tokio (async)
with zero configuration — both runtimes default to `num_cpus` threads,
creating 2x oversubscription when both are active. There is no backpressure
on async executor dispatch, and no container-aware thread budget.

PostHog documented the same class of bug causing 4.9x p99 latency regression
(460ms → 94ms after fix) in a Rust service mixing Rayon and Tokio. The core
lessons apply directly to hologram's `parallel` + `async` feature combination.

Today there is no multi-client serving layer, so the failure mode is latent.
These changes are preventive hygiene that become load-bearing the moment
hologram is used behind an async server.

## Current State

### hologram (base)

| Site | Primitive | Config | Threshold |
|------|-----------|--------|-----------|
| Level execution (`hologram-exec/src/parallel/mod.rs`) | `par_iter` | default pool | `PARALLEL_THRESHOLD=4`, `SMALL_BUFFER_BYTES=256` |
| LUT-GEMM columns (`hologram-exec/src/lut_gemm/parallel.rs`) | `par_iter_mut` | default pool | `PAR_COL_THRESHOLD=64` |
| Batch matmul (`hologram-exec/src/float_dispatch/matmul.rs`) | `par_chunks_mut` | default pool | `batch >= 2` |
| Schedule building (`hologram-graph/src/schedule/mod.rs`) | `rayon::join` | default pool | none |
| Async executor (`hologram-async/src/executor/mod.rs`) | `spawn_blocking` | default Tokio | none |
| Streaming executor (`hologram-async/src/stream/mod.rs`) | `spawn_blocking` + `mpsc(64)` | default Tokio | none |

**What's good**: `spawn_blocking` correctly avoids blocking Tokio I/O threads.
Adaptive thresholds in exec prevent dispatch overhead on small work.

**What's missing**: No thread pool configuration, no backpressure, no
container-aware defaults.

### hologram-ai

| Site | Primitive | Config | Threshold |
|------|-----------|--------|-----------|
| Subgraph optimization (`hologram-ai-common/src/opt/pipeline.rs`) | `par_iter` | default pool | none |
| Multi-component compilation (`hologram-ai/src/compiler.rs`) | `par_iter` | default pool | none |
| Model downloads (`hologram-ai/src/download/mod.rs`) | Tokio runtime | multi-thread default | N/A |

**What's missing**: No thresholds on compiler par_iter (small models pay
dispatch overhead for no benefit).

## Changes

### 1. Configurable Rayon thread pool (hologram base)

Create a `ThreadConfig` that builds a scoped Rayon `ThreadPool` instead of
using the global default:

```rust
pub struct ThreadConfig {
    /// Rayon worker threads. None = num_cpus.
    pub rayon_threads: Option<usize>,
}
```

Wire this through `ExecutionContext` or equivalent so callers can set it.
Default remains `num_cpus` for backwards compatibility. Container deployments
set it to cgroup CPU request count.

**Files**: `hologram-exec/src/lib.rs` (or new `hologram-exec/src/thread_config.rs`)

### 2. Backpressure semaphore on async dispatch (hologram base)

Gate `AsyncExecutor::execute()` and `execute_stream()` behind a
`tokio::sync::Semaphore` so concurrent in-flight executions are bounded:

```rust
pub struct AsyncExecutor {
    semaphore: Arc<Semaphore>,
}

impl AsyncExecutor {
    pub fn new(max_concurrent: usize) -> Self { ... }

    pub async fn execute(...) -> ExecResult<GraphOutputs> {
        let _permit = self.semaphore.acquire().await?;
        tokio::task::spawn_blocking(move || { ... }).await?
    }
}
```

Without this, a burst of `.execute()` calls grows Rayon's internal queue
without bound, causing latency to snowball.

**Files**: `hologram-async/src/executor/mod.rs`, `hologram-async/src/stream/mod.rs`

### 3. Tokio worker thread budget (hologram base)

When `hologram-async` builds a Tokio runtime (if it ever owns one), configure
worker threads to half of available cores, leaving the other half for Rayon
compute. Document the split rationale.

Today `hologram-async` doesn't build its own runtime (callers provide it),
so this is a documentation + API guideline item for now.

**Files**: `hologram-async/src/lib.rs` (doc comments / builder API)

### 4. Compiler par_iter thresholds (hologram-ai)

Add minimum-work checks before `par_iter` in the compiler:

- Subgraph optimization: skip `par_iter` when `subgraphs.len() < 3`
- Multi-component compilation: skip `par_iter` when `components.len() < 3`

Sequential fallback avoids Rayon dispatch overhead for the common case
(single-component models, models with 1-2 subgraphs).

**Files**: `hologram-ai-common/src/opt/pipeline.rs`, `hologram-ai/src/compiler.rs`

## Non-Goals

- Building a serving/inference server (no multi-client path exists yet)
- Paged attention or streaming token delivery (separate plans)
- GPU thread/warp configuration (Metal/WebGPU have their own dispatch)

## Verification

1. `cargo test` passes in both repos with `parallel` + `async` features
2. Compile a multi-component model (SD pipeline) and confirm par_iter
   threshold kicks in correctly (log output at debug level)
3. Confirm `AsyncExecutor::new(max_concurrent)` rejects beyond the semaphore
   limit (unit test: spawn N+1 executions, verify Nth+1 blocks until one
   completes)
4. In a container with 4 CPU requests / 8 CPU limits, verify Rayon spawns 4
   threads (not 8) when configured via `ThreadConfig`
