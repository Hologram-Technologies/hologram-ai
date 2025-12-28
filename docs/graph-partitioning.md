# Graph Partitioning Implementation Guide

## Overview

This document outlines the design and implementation requirements for true graph partitioning in the hologram ONNX compiler. Graph partitioning would enable compilation of very large models (3000+ nodes) that exceed available system RAM.

## Current Status

**Implemented** ✅:
- Automatic detection of large graphs (>500 nodes)
- Streaming weight extraction with deduplication
- Minimal compiler optimizations for large graphs
- Subprocess isolation per component

**Not Implemented** ❌:
- True graph partitioning (splitting graph into independent chunks)
- Schedule merging across partitions
- Cross-partition dependency handling

## Why It's Needed

### Current Memory Requirements

| Graph Size | Peak Memory (compilation) | Notes |
|------------|--------------------------|-------|
| 500 nodes  | ~2 GB                    | Fits in available RAM ✅ |
| 1223 nodes (text_encoder) | ~4 GB    | Fits with optimizations ✅ |
| 3052 nodes (UNet) | ~10-12 GB          | Exceeds available RAM ❌ |

### System Constraints

- Total RAM: 15 GB
- Available RAM: ~7-8 GB (after OS and services)
- UNet requirement: 10-12 GB
- **Gap**: Need ~3-5 GB more than available

## Design

### High-Level Approach

1. **Graph Analysis**:
   - Topologically sort the operation graph
   - Identify natural partition boundaries
   - Group nodes into chunks of ~500 nodes each

2. **Partition Creation**:
   - For each partition, create a subgraph
   - Handle boundary tensors (outputs of previous partition → inputs of current)
   - Preserve operation dependencies within partition

3. **Independent Compilation**:
   - Compile each partition separately (peak memory ~2 GB each)
   - Extract and stream weights for each partition
   - Generate partial schedules

4. **Schedule Merging**:
   - Merge partition schedules into final schedule
   - Resolve buffer ID conflicts
   - Handle cross-partition data flow

### Technical Challenges

#### 1. Subgraph Creation

**Challenge**: Extracting a valid subgraph from the full graph

**Requirements**:
- Identify all dependencies for nodes in the partition
- Create "virtual inputs" for tensors coming from previous partitions
- Mark "virtual outputs" for tensors needed by future partitions
- Preserve shapes and data types

**Example**:
```
Partition 1: nodes 0-499
  - Real inputs: model inputs
  - Real outputs: partition_output_250, partition_output_489
  - Compiles to: schedule_1.holo

Partition 2: nodes 500-999
  - Real inputs: model inputs + partition_output_250, partition_output_489
  - Real outputs: partition_output_750, partition_output_999
  - Compiles to: schedule_2.holo
```

#### 2. Buffer ID Management

**Challenge**: Buffer IDs must be unique across all partitions

**Current Structure**:
```rust
pub struct BufferAllocationPlan {
    pub buffers: HashMap<BufferId, BufferInfo>,
    pub first_use: HashMap<BufferId, usize>,
    pub last_use: HashMap<BufferId, usize>,
    pub peak_memory: usize,
    pub input_ids: HashMap<String, BufferId>,
    pub output_ids: HashMap<String, BufferId>,
}
```

**Solution Approach**:
- Offset buffer IDs per partition: `partition_idx * 1_000_000 + local_id`
- Track cross-partition buffers separately
- Rewrite buffer references during merge

#### 3. Cross-Partition Dependencies

**Challenge**: Handling data flow between partitions

**Approach**:
1. **Identify Boundary Tensors**:
   - Scan partition N for nodes used in partition N+1
   - Mark these as "partition outputs"

2. **Create Virtual Inputs**:
   - In partition N+1, create input placeholders for boundary tensors
   - Link these to the actual outputs from partition N during merge

3. **Buffer Continuity**:
   - Ensure boundary tensor buffers are accessible across partitions
   - May require buffer copying or shared memory regions

#### 4. Schedule Merging

**Challenge**: Combining partial schedules into a coherent final schedule

**Current Schedule Structure**:
```rust
pub struct ParallelSchedule {
    pub levels: Vec<ExecutionLevel>,
    pub total_ops: usize,
    pub max_parallelism: usize,
    pub buffer_plan: BufferAllocationPlan,
    pub stats: ScheduleStats,
}

pub struct ExecutionLevel {
    pub operations: Vec<LevelOperation>,
    pub buffer_allocations: Vec<BufferId>,
    pub buffer_deallocations: Vec<BufferId>,
}
```

**Merge Strategy**:
1. Concatenate execution levels from all partitions
2. Offset buffer IDs in each level
3. Merge buffer allocation plans
4. Update statistics (total_ops, max_parallelism)
5. Recalculate peak memory (sum across partitions is conservative)

## Implementation Plan

### Phase 1: Dependency Analysis (Est: 2-3 days)

**Goal**: Understand graph structure and find partition boundaries

**Tasks**:
- [x] Implement topological sort traversal
- [ ] Identify nodes with minimal cross-partition dependencies
- [ ] Create boundary tensor detection algorithm
- [ ] Write tests for dependency analysis

**Files**:
- `crates/compiler/src/partitioned.rs` - Add dependency analysis functions

### Phase 2: Subgraph Creation (Est: 3-4 days)

**Goal**: Extract valid subgraphs for each partition

**Tasks**:
- [ ] Implement subgraph extraction from node list
- [ ] Handle virtual inputs/outputs at boundaries
- [ ] Preserve shape and type information
- [ ] Write tests for subgraph creation

**Files**:
- `crates/compiler/src/partitioned.rs` - Add `create_partition_subgraph()`

**Key Code**:
```rust
fn create_partition_subgraph(
    &self,
    full_graph: &OperationGraph,
    partition_nodes: &[NodeId],
    boundary_inputs: &HashMap<NodeId, NodeId>,  // External node -> Virtual input
) -> Result<OperationGraph>
```

### Phase 3: Partition Compilation (Est: 2-3 days)

**Goal**: Compile each partition independently

**Tasks**:
- [ ] Compile partitions with minimal optimizations
- [ ] Extract and stream weights per partition
- [ ] Track cross-partition dependencies
- [ ] Write tests for partition compilation

**Files**:
- `crates/compiler/src/partitioned.rs` - Update `compile_streaming()`

### Phase 4: Schedule Merging (Est: 4-5 days)

**Goal**: Combine partition schedules into final schedule

**Tasks**:
- [ ] Implement buffer ID offsetting
- [ ] Merge buffer allocation plans
- [ ] Handle cross-partition buffer references
- [ ] Update schedule statistics
- [ ] Write comprehensive merge tests

**Files**:
- `crates/compiler/src/partitioned.rs` - Add `merge_schedules()`

**Key Code**:
```rust
fn merge_schedules(
    &self,
    partitions: Vec<(ParallelSchedule, HashMap<NodeId, BufferId>)>,
) -> Result<ParallelSchedule>
```

### Phase 5: Integration Testing (Est: 2-3 days)

**Goal**: Validate end-to-end partitioned compilation

**Tasks**:
- [ ] Test with sd-tiny (should work without partitioning)
- [ ] Test with UNet (3052 nodes, requires partitioning)
- [ ] Verify output correctness vs non-partitioned
- [ ] Performance benchmarking

**Success Criteria**:
- UNet compiles successfully on system with 8 GB available RAM
- Compiled model produces same outputs as non-partitioned version
- Peak memory during compilation stays under 8 GB

## Alternative Approaches

### 1. Increase System RAM

**Pros**: Simplest solution
**Cons**: Not always possible, doesn't scale to even larger models

### 2. Reduce Graph Size at ONNX Level

**Pros**: Could simplify model before translation
**Cons**: May lose accuracy, requires ONNX optimization tools

### 3. Lazy Compilation

**Pros**: Only compile nodes as needed during execution
**Cons**: Slower runtime, complex to implement

## Estimated Total Effort

- Development: 13-18 days
- Testing: 2-3 days
- Documentation: 1-2 days
- **Total**: 16-23 days

## References

- Current implementation: [crates/compiler/src/partitioned.rs](../crates/compiler/src/partitioned.rs)
- Graph structures: [hologram/crates/compiler/src/graph.rs](../../hologram/crates/compiler/src/graph.rs)
- Schedule structures: [hologram/crates/compiler/src/schedule.rs](../../hologram/crates/compiler/src/schedule.rs)
- Memory optimization docs: [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md)
