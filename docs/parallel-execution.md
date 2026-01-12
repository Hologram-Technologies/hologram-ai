# Parallel Execution Architecture

This document describes hologram's parallel execution architecture, which enables concurrent execution of independent computation groups across CPU cores.

## Overview

The parallel execution system automatically detects independent computation patterns (like Q, K, V projections in transformer attention) and executes them concurrently using rayon's thread pool. This provides significant speedup on multi-core systems.

## Architecture

### Execution Levels

Groups are organized into dependency levels:

```
Level 0: [input_group]              → Execute first
Level 1: [Q_group, K_group, V_group] → Execute in parallel (3 threads)
Level 2: [attention_group]           → Execute after Level 1 completes
```

Groups within the same level have no dependencies on each other and execute concurrently. The executor waits for all groups in a level to complete before moving to the next level.

### Thread-Safe Backend

The `ProgramBackend` and `ComputeBackend` traits use `&self` instead of `&mut self` to enable thread-safe parallel execution:

```rust
pub trait ProgramBackend: Send + Sync {
    fn allocate_buffer(&self, size: usize) -> Result<BufferHandle, BackendError>;
    fn execute_program(&self, program: &Program, config: &LaunchConfig) -> Result<(), BackendError>;
    // ... all methods take &self
}
```

Implementations use interior mutability for thread-safe state management:

| Component | Synchronization | Purpose |
|-----------|-----------------|---------|
| Buffer storage | `RwLock<HashMap>` | Concurrent reads during execution, exclusive writes for allocation |
| Handle counter | `AtomicU64` | Lock-free unique handle generation |
| Buffer pool | `Mutex<BufferPool>` | Infrequent allocation reuse |

### StreamingExecutor

The `StreamingExecutor` manages parallel execution of groups:

```rust
pub struct StreamingExecutor {
    executors: Vec<Mutex<PlanExecutor>>,  // Per-group executors
    levels: Vec<Vec<ExecutionGroupInfo>>, // Dependency levels
    prefetch_enabled: bool,               // OS-level prefetch hints
    release_enabled: bool,                // Memory release after completion
    stats: Mutex<StreamingStats>,         // Thread-safe statistics
}
```

#### Execution Flow

```rust
// Groups within a level execute in parallel
level.par_iter().try_for_each(|group| {
    self.execute_group(group, inputs, outputs, backend)
})
```

Each group has a dedicated `PlanExecutor` with its own workspace, eliminating conflicts during parallel execution.

### Memory Management

The executor provides OS-level memory hints for optimal performance:

- **Prefetching**: `madvise(MADV_WILLNEED)` hints for upcoming levels
- **Release**: `madvise(MADV_DONTNEED)` to release completed level memory

These are non-blocking hints with zero CPU overhead during actual computation.

## Attention Pattern Detection

For transformer models, the system automatically detects Q, K, V projection patterns:

### Supported Architectures

| Model Family | Pattern |
|--------------|---------|
| BERT/RoBERTa | `encoder.layer.N.attention.self.query/key/value` |
| GPT-2/OPT | `transformer.h.N.attn.c_attn` |
| LLaMA | `model.layers.N.self_attn.q_proj/k_proj/v_proj` |
| T5 | `encoder.block.N.layer.0.SelfAttention.q/k/v` |

### Usage

```rust
use hologram_ai_onnx::core::translate_graph_to_ir_with_groups;

// Translate with automatic attention pattern detection
let ir_func = translate_graph_to_ir_with_groups(graph)?;

// Get parallelizable group levels
let levels = ir_func.inner().parallel_groups();
for (level_idx, groups) in levels.iter().enumerate() {
    println!("Level {}: {} groups can run in parallel", level_idx, groups.len());
}
```

## Performance Characteristics

### Lock Contention

- **Allocation**: Brief mutex contention on buffer pool, then lock-free handle generation
- **Execution**: Lock-free after initial buffer pointer retrieval (read lock only)
- **Statistics**: Single mutex update after all levels complete

### Expected Speedup

For transformer models with Q, K, V parallel projections:
- Near-linear speedup for independent groups
- ~2.5x speedup on 4-core CPU for 3 parallel attention projections

### Memory Overhead

- Same as sequential execution - each group has dedicated workspace
- No additional memory for thread synchronization beyond executor mutexes

## API Reference

### StreamingExecutor

```rust
impl StreamingExecutor {
    /// Create executor with plans and dependency levels
    pub fn new(
        plans: Vec<BackendPlan>,
        levels: Vec<Vec<ExecutionGroupInfo>>,
        backend: &dyn ProgramBackend,
    ) -> BackendResult<Self>;

    /// Execute with parallel groups (thread-safe)
    pub fn execute(
        &self,
        inputs: &[BufferHandle],
        outputs: &[BufferHandle],
        backend: &dyn ProgramBackend,
    ) -> BackendResult<()>;

    /// Get execution statistics
    pub fn stats(&self) -> StreamingStats;

    /// Enable/disable prefetching
    pub fn set_prefetch(&mut self, enabled: bool);

    /// Enable/disable memory release
    pub fn set_release(&mut self, enabled: bool);
}
```

### StreamingStats

```rust
pub struct StreamingStats {
    pub total_ns: u64,              // Total execution time
    pub level_times_ns: Vec<u64>,   // Time per level
    pub max_concurrent_groups: usize, // Max groups in any level
    pub groups_executed: usize,     // Total groups executed
}
```

## Files

| File | Description |
|------|-------------|
| `/hologram/crates/backend/src/streaming_executor.rs` | StreamingExecutor implementation |
| `/hologram/crates/backend/src/traits.rs` | Thread-safe ProgramBackend/ComputeBackend traits |
| `/hologram/crates/backend/src/cpu/mod.rs` | CpuBackend with interior mutability |
| `/workspace/crates/hologram-ai-onnx/src/core/attention_detection.rs` | Attention pattern detection |
| `/workspace/crates/hologram-ai-onnx/src/core/translator.rs` | `translate_graph_to_ir_with_groups()` |

## Design Decisions

### Why Interior Mutability?

The backend uses `RwLock`/`AtomicU64`/`Mutex` instead of an actor model because:

1. **Read-heavy workload**: Compute operations only read buffer pointers
2. **Lock-free compute**: SIMD operations use raw pointers, no locking during actual work
3. **Low contention**: Allocation is infrequent compared to execution
4. **Simpler API**: Single backend instance shared across threads

### Why Per-Group Executors?

Each group has its own `PlanExecutor` with dedicated workspace:

1. **No buffer conflicts**: Groups write to separate output buffers
2. **No synchronization during compute**: Just lock the executor once at start
3. **Memory isolation**: Each workspace is independent

### Group Assignment Separation

Group assignment is handled at graph level by `attention_detection.rs`, not during individual node translation. This maintains separation of concerns:

- **Pattern detection**: Identifies parallelizable structures in the graph
- **Operation translation**: Converts ONNX ops to IR (group-agnostic)
