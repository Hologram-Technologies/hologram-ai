# Plan 27: Replace petgraph with Custom Graph for hologram

**Status**: Approved
**Created**: 2026-01-11
**Priority**: High - Foundational infrastructure change

## Status Tracking

- [ ] Phase 1: Create `hologram-graph` crate
- [ ] Phase 2: Implement subgraph composition
- [ ] Phase 3: Multi-threaded execution groups
- [ ] Phase 4: Update ONNX/GGUF translators
- [ ] Phase 5: Runtime executor updates
- [ ] Phase 6: Migration & cleanup

---

## hologram's Core Requirements

hologram is a **geometrical computation engine** with guarantees:
- **O(1) lookup** for all data
- **Zero-copy execution**
- **Fastest possible performance**
- **Multi-layer distributed execution** (layers in graphs)
- **Arbitrary compiled programs** (Python, TypeScript, ONNX, GGUF, etc.)

## Why petgraph Doesn't Fit

**petgraph problems for hologram**:
1. **NodeIndex instability** - Remove a node, indices shift. Breaks layer load/unload.
2. **No O(1) property lookup** - Finding nodes by property is O(n)
3. **No layer concept** - Must track externally
4. **Memory scattered** - Nodes allocated individually, not cache-friendly
5. **Edge ordering undefined** - Causing input ordering bugs
6. **Serialization mismatch** - Not designed for rkyv

We keep adding workarounds. Time to fix the root cause.

## Custom Graph Design for hologram

### Design Goals

1. **O(1) lookup** - Node by ID, edges by node, inputs by slot
2. **Stable indices** - No invalidation on remove (generational IDs)
3. **Subgraph composition** - Reference pre-compiled .holo files
4. **Execution groups** - Parallel execution of independent subgraphs
5. **Memory planes** - Zero-copy within isolated regions
6. **Compile-time dependency resolution** - Subgraph deps resolved before runtime
7. **Cache-friendly** - Arena-based, contiguous memory
8. **rkyv-native** - Zero-copy serialization

### Core Data Structures

```rust
// In /hologram/crates/graph/src/graph.rs

/// Stable node identifier (generational to handle removal)
#[derive(Archive, Serialize, Deserialize, Copy, Clone, Eq, PartialEq, Hash)]
pub struct NodeId {
    index: u32,       // Position in arena
    generation: u32,  // Incremented on reuse (stable across removals)
}

/// Input slot with source reference
#[derive(Archive, Serialize, Deserialize)]
pub struct InputSlot {
    pub source: InputSource,   // Where data comes from
    pub name: Option<String>,  // Original tensor name (for debugging)
}

/// Input can come from a node OR from a subgraph output
#[derive(Archive, Serialize, Deserialize)]
pub enum InputSource {
    /// From another node in this graph
    Node(NodeId),
    /// From a subgraph's output port
    Subgraph { subgraph: SubgraphId, output_port: usize },
    /// External graph input (runtime provided)
    GraphInput { index: usize },
}

/// A node in the computation graph
#[derive(Archive, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub op: NodeOp,
    pub shape: Shape,
    pub dtype: DType,
    pub group: GroupId,                    // Which execution group
    pub inputs: SmallVec<[InputSlot; 4]>,  // Explicit ordered inputs
}

/// Reference to a pre-compiled subgraph (like Docker layers)
#[derive(Archive, Serialize, Deserialize)]
pub struct SubgraphRef {
    pub id: SubgraphId,
    pub name: String,

    /// Source: embedded bytes or external .holo path
    pub source: SubgraphSource,

    /// Input ports (data this subgraph needs)
    pub inputs: Vec<TensorPort>,

    /// Output ports (data this subgraph produces)
    pub outputs: Vec<TensorPort>,

    /// Which execution group (for parallel execution)
    pub group: GroupId,

    /// Memory plane for zero-copy execution
    pub plane: PlaneId,
}

#[derive(Archive, Serialize, Deserialize)]
pub enum SubgraphSource {
    /// Embedded in this .holo file (like HOLM bundles)
    Embedded { offset: u64, size: u64 },
    /// Reference to external .holo file
    External { path: String, checksum: u32 },
    /// Inline nodes (not pre-compiled, compile-time expansion)
    Inline(Vec<NodeId>),
}

/// Tensor port at subgraph/graph boundary
#[derive(Archive, Serialize, Deserialize)]
pub struct TensorPort {
    pub name: String,
    pub shape: Shape,
    pub dtype: DType,
}

/// Execution group - nodes/subgraphs that can run in parallel
#[derive(Archive, Serialize, Deserialize)]
pub struct ExecutionGroup {
    pub id: GroupId,
    pub nodes: Vec<NodeId>,
    pub subgraphs: Vec<SubgraphId>,

    /// Dependencies on other groups (must complete before this runs)
    pub depends_on: Vec<GroupId>,

    /// Memory plane for this group
    pub plane: PlaneId,
}

/// Memory plane - isolated region for zero-copy execution
#[derive(Archive, Serialize, Deserialize)]
pub struct MemoryPlane {
    pub id: PlaneId,
    /// Pre-allocated size (bytes) or dynamic
    pub size: Option<usize>,
    /// Alignment requirement
    pub alignment: usize,
}

/// The computation graph with subgraph composition
#[derive(Archive, Serialize, Deserialize)]
pub struct ComputeGraph {
    // === Nodes (primitive operations) ===
    nodes: Vec<Option<Node>>,
    generations: Vec<u32>,
    free_slots: Vec<u32>,

    // === Subgraphs (composed pre-compiled units) ===
    pub subgraphs: Vec<SubgraphRef>,

    // === Execution plan (compile-time resolved) ===
    pub groups: Vec<ExecutionGroup>,
    pub group_order: Vec<GroupId>,  // Topological order for execution

    // === Memory planes ===
    pub planes: Vec<MemoryPlane>,

    // === Graph boundary ===
    pub graph_inputs: Vec<TensorPort>,
    pub graph_outputs: Vec<TensorPort>,

    // === Metadata ===
    pub metadata: GraphMetadata,
}
```

### Key Operations (all O(1))

```rust
impl ComputeGraph {
    /// Get node by ID - O(1)
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        let slot = &self.nodes.get(id.index as usize)?;
        if self.generations[id.index as usize] == id.generation {
            slot.as_ref()
        } else {
            None  // Stale ID (node was removed)
        }
    }

    /// Get inputs for node - O(1), already ordered by slot
    pub fn inputs(&self, id: NodeId) -> &[InputSlot] {
        self.node(id).map(|n| n.inputs.as_slice()).unwrap_or(&[])
    }

    /// Get subgraph by ID - O(1)
    pub fn subgraph(&self, id: SubgraphId) -> Option<&SubgraphRef> {
        self.subgraphs.get(id.0 as usize)
    }

    /// Get execution group - O(1)
    pub fn group(&self, id: GroupId) -> Option<&ExecutionGroup> {
        self.groups.get(id.0 as usize)
    }
}
```

### Subgraph Composition

```rust
impl ComputeGraph {
    /// Add a pre-compiled subgraph reference
    pub fn add_subgraph(&mut self, name: &str, source: SubgraphSource,
                         inputs: Vec<TensorPort>, outputs: Vec<TensorPort>) -> SubgraphId {
        let id = SubgraphId(self.subgraphs.len() as u64);
        let group = self.current_group;
        let plane = self.groups[group.0 as usize].plane;

        self.subgraphs.push(SubgraphRef {
            id, name: name.to_string(), source, inputs, outputs, group, plane
        });
        self.groups[group.0 as usize].subgraphs.push(id);
        id
    }

    /// Connect node to subgraph output
    pub fn connect_from_subgraph(&mut self, subgraph: SubgraphId, output_port: usize,
                                   to: NodeId, slot: usize) {
        if let Some(node) = self.nodes[to.index as usize].as_mut() {
            while node.inputs.len() <= slot {
                node.inputs.push(InputSlot::empty());
            }
            node.inputs[slot] = InputSlot {
                source: InputSource::Subgraph { subgraph, output_port },
                name: None,
            };
        }
    }
}
```

### Execution Groups (Multi-threaded Parallel Execution)

```rust
impl ComputeGraph {
    /// Create new execution group with its own memory plane
    pub fn create_group(&mut self) -> GroupId {
        let plane = self.create_plane();
        let id = GroupId(self.groups.len() as u64);
        self.groups.push(ExecutionGroup {
            id, nodes: Vec::new(), subgraphs: Vec::new(),
            depends_on: Vec::new(), plane,
        });
        id
    }

    /// Add dependency: `dependent` must wait for `dependency` to complete
    pub fn add_group_dependency(&mut self, dependent: GroupId, dependency: GroupId) {
        self.groups[dependent.0 as usize].depends_on.push(dependency);
    }

    /// Resolve execution order at compile time (topological sort of groups)
    pub fn resolve_execution_order(&mut self) -> Result<(), CycleError> {
        // Kahn's algorithm on groups
        let mut in_degree: Vec<usize> = self.groups.iter()
            .map(|g| g.depends_on.len())
            .collect();
        let mut queue: VecDeque<GroupId> = in_degree.iter().enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(i, _)| GroupId(i as u64))
            .collect();

        self.group_order.clear();
        while let Some(gid) = queue.pop_front() {
            self.group_order.push(gid);
            for (i, group) in self.groups.iter().enumerate() {
                if group.depends_on.contains(&gid) {
                    in_degree[i] -= 1;
                    if in_degree[i] == 0 {
                        queue.push_back(GroupId(i as u64));
                    }
                }
            }
        }

        if self.group_order.len() == self.groups.len() {
            Ok(())
        } else {
            Err(CycleError)
        }
    }

    /// Get groups that can execute in parallel (same level in dependency DAG)
    pub fn parallel_groups(&self) -> Vec<Vec<GroupId>> {
        // Groups with same "depth" in dependency DAG can run in parallel
        let mut levels: Vec<Vec<GroupId>> = Vec::new();
        let mut depth: Vec<usize> = vec![0; self.groups.len()];

        for &gid in &self.group_order {
            let d = self.groups[gid.0 as usize].depends_on.iter()
                .map(|dep| depth[dep.0 as usize] + 1)
                .max()
                .unwrap_or(0);
            depth[gid.0 as usize] = d;

            while levels.len() <= d {
                levels.push(Vec::new());
            }
            levels[d].push(gid);
        }
        levels
    }
}
```

### Memory Planes (Zero-Copy Regions)

```rust
impl ComputeGraph {
    /// Create isolated memory plane
    pub fn create_plane(&mut self) -> PlaneId {
        let id = PlaneId(self.planes.len() as u64);
        self.planes.push(MemoryPlane {
            id, size: None, alignment: 64, // Cache-line aligned
        });
        id
    }

    /// Set plane size (for pre-allocation)
    pub fn set_plane_size(&mut self, plane: PlaneId, size: usize) {
        self.planes[plane.0 as usize].size = Some(size);
    }
}
```

### Memory-Optimized Subgraph Loading

For low-memory execution, subgraphs can be loaded/unloaded dynamically:

```rust
/// Subgraph lifecycle state
#[derive(Archive, Serialize, Deserialize, Clone, Copy)]
pub enum SubgraphState {
    /// Not loaded - on disk/embedded, not in memory
    Unloaded,
    /// Prefetching - OS-level read-ahead in progress
    Prefetching,
    /// Loaded - ready for execution
    Loaded,
    /// Executing - currently running
    Executing,
}

/// Memory-optimized executor for subgraph loading/unloading
pub struct StreamingExecutor {
    /// Current state of each subgraph
    states: Vec<SubgraphState>,

    /// Memory-mapped file handles (for embedded subgraphs)
    mmap: Option<Mmap>,

    /// Active subgraph executors (loaded into memory)
    loaded: HashMap<SubgraphId, SubgraphExecutor>,

    /// Maximum concurrent loaded subgraphs (memory budget)
    max_loaded: usize,
}

impl StreamingExecutor {
    /// Prefetch next subgraph (async, overlaps with current execution)
    /// Uses OS-level madvise(MADV_WILLNEED) - no CPU overhead
    pub fn prefetch(&mut self, id: SubgraphId) {
        if self.states[id.0 as usize] == SubgraphState::Unloaded {
            let subgraph = &self.graph.subgraphs[id.0 as usize];
            match &subgraph.source {
                SubgraphSource::Embedded { offset, size } => {
                    // Tell OS to start reading this region
                    #[cfg(unix)]
                    unsafe {
                        libc::madvise(
                            self.mmap.as_ptr().add(*offset as usize) as *mut _,
                            *size as usize,
                            libc::MADV_WILLNEED,
                        );
                    }
                }
                SubgraphSource::External { path, .. } => {
                    // Start async file read
                    self.async_load(id, path);
                }
                SubgraphSource::Inline(_) => {}  // Always loaded
            }
            self.states[id.0 as usize] = SubgraphState::Prefetching;
        }
    }

    /// Release subgraph memory (after execution complete)
    /// Uses madvise(MADV_DONTNEED) - no CPU overhead
    pub fn release(&mut self, id: SubgraphId) {
        if let Some(_executor) = self.loaded.remove(&id) {
            let subgraph = &self.graph.subgraphs[id.0 as usize];
            match &subgraph.source {
                SubgraphSource::Embedded { offset, size } => {
                    // Tell OS we don't need these pages anymore
                    #[cfg(unix)]
                    unsafe {
                        libc::madvise(
                            self.mmap.as_ptr().add(*offset as usize) as *mut _,
                            *size as usize,
                            libc::MADV_DONTNEED,
                        );
                    }
                }
                _ => {}  // External files just drop the handle
            }
            self.states[id.0 as usize] = SubgraphState::Unloaded;
        }
    }

    /// Execute with automatic prefetch/release (layer-by-layer)
    pub fn execute_streaming(&mut self, inputs: &[Tensor]) -> Vec<Tensor> {
        let levels = self.graph.parallel_groups();

        for (level_idx, level) in levels.iter().enumerate() {
            // Prefetch NEXT level's subgraphs (overlaps with current execution)
            if level_idx + 1 < levels.len() {
                for &next_group in &levels[level_idx + 1] {
                    for &subgraph in &self.graph.groups[next_group.0 as usize].subgraphs {
                        self.prefetch(subgraph);
                    }
                }
            }

            // Execute current level (all groups in parallel)
            level.par_iter().for_each(|group_id| {
                let group = self.graph.group(*group_id).unwrap();
                self.execute_group(group);
            });

            // Release PREVIOUS level's subgraphs (free memory)
            if level_idx > 0 {
                for &prev_group in &levels[level_idx - 1] {
                    for &subgraph in &self.graph.groups[prev_group.0 as usize].subgraphs {
                        // Only release if not needed by future groups
                        if !self.is_needed_later(subgraph, level_idx) {
                            self.release(subgraph);
                        }
                    }
                }
            }
        }
    }
}
```

**Key performance guarantees**:
- **Prefetch** uses `madvise(MADV_WILLNEED)` - just a hint to OS, no CPU blocking
- **Release** uses `madvise(MADV_DONTNEED)` - marks pages as reclaimable, no copy
- **Execution** never waits - prefetch overlaps with previous subgraph execution
- **Zero overhead** during actual computation - all memory ops are between levels

## Subgraph Location Options

Subgraphs can be located in multiple ways:

```rust
pub enum SubgraphSource {
    /// Embedded in THIS .holo file (same file, different section)
    /// Used for: Transformer layers within a single model file
    Embedded { offset: u64, size: u64 },

    /// Reference to EXTERNAL .holo file
    /// Used for: Shared encoders, vocabulary embeddings, etc.
    External { path: String, checksum: u32 },

    /// Inline nodes (not pre-compiled, just grouped)
    /// Used for: Organizing nodes into logical groups within same graph
    Inline(Vec<NodeId>),
}
```

**Same file composition** example:
```
model.holo
├── [Header]
├── [Main Graph] - references embedded subgraphs by offset
├── [Subgraph: encoder.layer.0] @ offset 0x1000
├── [Subgraph: encoder.layer.1] @ offset 0x2000
├── [Subgraph: encoder.layer.2] @ offset 0x3000
└── [Shared Weights]
```

## Implementation Plan

### Phase 1: Create `hologram-graph` Crate

**New crate** at `/hologram/crates/graph/`:
```
/hologram/crates/graph/
├── Cargo.toml
└── src/
    ├── lib.rs           # Public API exports
    ├── ids.rs           # NodeId, SubgraphId, GroupId, PlaneId
    ├── node.rs          # Node, InputSlot, InputSource
    ├── subgraph.rs      # SubgraphRef, SubgraphSource, TensorPort
    ├── group.rs         # ExecutionGroup, MemoryPlane
    ├── graph.rs         # ComputeGraph
    └── builder.rs       # GraphBuilder
```

**Cargo.toml**:
```toml
[package]
name = "hologram-graph"
version.workspace = true
edition.workspace = true

[dependencies]
rkyv = { workspace = true, features = ["validation"] }
smallvec = { version = "1", features = ["serde"] }
rustc-hash = "2"

[dev-dependencies]
criterion = "0.5"
```

**Why a separate crate?**
- Clean separation of graph logic from IR operations
- Can be used independently (Python bindings, other frontends)
- No petgraph dependency
- Easier testing and benchmarking

### Phase 2: Implement Subgraph Composition

```rust
impl GraphBuilder {
    /// Reference an external pre-compiled .holo file
    pub fn reference_external(&mut self, name: &str, path: &str,
                               inputs: Vec<TensorPort>, outputs: Vec<TensorPort>) -> SubgraphId {
        self.graph.add_subgraph(name, SubgraphSource::External {
            path: path.to_string(),
            checksum: compute_checksum(path),
        }, inputs, outputs)
    }

    /// Embed a pre-compiled .holo inline (like HOLM bundles)
    pub fn embed_subgraph(&mut self, name: &str, holo_bytes: Vec<u8>,
                           inputs: Vec<TensorPort>, outputs: Vec<TensorPort>) -> SubgraphId {
        let offset = self.embedded_data.len() as u64;
        self.embedded_data.extend_from_slice(&holo_bytes);
        self.graph.add_subgraph(name, SubgraphSource::Embedded {
            offset, size: holo_bytes.len() as u64,
        }, inputs, outputs)
    }
}
```

### Phase 3: Multi-threaded Execution Groups

```rust
impl GraphBuilder {
    /// Start a new execution group (can run in parallel with others)
    pub fn begin_group(&mut self) -> GroupId {
        let id = self.graph.create_group();
        self.current_group = id;
        id
    }

    /// Mark dependency between groups
    pub fn group_depends_on(&mut self, group: GroupId, dependency: GroupId) {
        self.graph.add_group_dependency(group, dependency);
    }

    /// Finalize and resolve execution order at compile time
    pub fn finalize(&mut self) -> Result<(), CycleError> {
        self.graph.resolve_execution_order()
    }
}
```

### Phase 4: Update ONNX/GGUF Translators

**ONNX** (in `/workspace/crates/hologram-ai-onnx/src/`):
- Use explicit input slots
- Detect transformer layers → create groups
- Support referencing pre-compiled encoder/decoder subgraphs

**GGUF** (when implemented):
- Same pattern - explicit slots, groups, subgraph composition

### Phase 5: Runtime Executor Updates

**Multi-threaded group execution**:
```rust
impl Executor {
    pub fn execute(&self, graph: &ComputeGraph, inputs: &[Tensor]) -> Vec<Tensor> {
        // Get parallelizable groups
        let levels = graph.parallel_groups();

        for level in levels {
            // Execute all groups in this level in parallel
            level.par_iter().for_each(|group_id| {
                let group = graph.group(*group_id).unwrap();
                self.execute_group(group);
            });
        }
    }
}
```

### Phase 6: Migration & Cleanup

1. `impl From<OperationGraph> for ComputeGraph` (compatibility)
2. Update all graph consumers
3. Remove petgraph dependency
4. Update .holo format version

## Key Features

### Subgraph Composition Example
```rust
// Compose encoder + decoder from separate .holo files
let mut builder = GraphBuilder::new();

// Reference pre-compiled encoder
let encoder = builder.reference_external(
    "encoder",
    "encoder.holo",
    vec![TensorPort::new("input_ids", [1, 128], DType::I64)],
    vec![TensorPort::new("hidden_states", [1, 128, 512], DType::F32)],
);

// Reference pre-compiled decoder
let decoder = builder.reference_external(
    "decoder",
    "decoder.holo",
    vec![TensorPort::new("encoder_hidden", [1, 128, 512], DType::F32)],
    vec![TensorPort::new("logits", [1, 128, 32000], DType::F32)],
);

// Connect encoder output to decoder input
builder.connect_subgraphs(encoder, 0, decoder, 0);

let graph = builder.finalize()?;
```

### Parallel Execution Example
```rust
// Multi-head attention: Q, K, V can compute in parallel
let group_q = builder.begin_group();
// ... add Q projection nodes ...

let group_k = builder.begin_group();
// ... add K projection nodes ...

let group_v = builder.begin_group();
// ... add V projection nodes ...

// Attention group depends on Q, K, V
let group_attn = builder.begin_group();
builder.group_depends_on(group_attn, group_q);
builder.group_depends_on(group_attn, group_k);
builder.group_depends_on(group_attn, group_v);
// ... add attention nodes ...

// At runtime: Q, K, V execute in parallel on separate threads
// Attention waits for all three, then executes
```

## Summary

| Feature | Implementation | Benefit |
|---------|----------------|---------|
| **Subgraph composition** | `SubgraphRef`, `SubgraphSource` | Docker-like pre-compiled layers |
| **Parallel execution** | `ExecutionGroup`, `parallel_groups()` | Multi-threaded speedup |
| **Memory planes** | `MemoryPlane`, `PlaneId` | Zero-copy per-group isolation |
| **Compile-time deps** | `resolve_execution_order()` | No runtime dependency checks |
| **Stable indices** | Generational `NodeId` | Layer load/unload without invalidation |
| **Explicit inputs** | `InputSlot`, `InputSource` | No edge ordering ambiguity |
| **Streaming execution** | `StreamingExecutor` | Low-memory with zero runtime overhead |
| **Prefetch/release** | `madvise()` hints | OS-level, non-blocking memory management |

## Verification

1. **Unit tests**: Add/remove nodes, subgraph composition, group dependencies
2. **Parallel tests**: Verify groups execute in correct order with parallel speedup
3. **Subgraph tests**: External .holo reference, embedded subgraphs
4. **ONNX tests**: T5, BERT with layer-wise groups
5. **Performance**: O(1) lookup, memory efficiency, parallel throughput
