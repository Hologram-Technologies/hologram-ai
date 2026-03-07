use crate::ir::{AiGraph, AiParam, DType};

/// Layout descriptor for the KV-cache arena.
pub struct KvCacheLayout {
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub dtype: DType,
}

impl KvCacheLayout {
    /// No KV-cache (single forward pass, Phase 1 / MVP sentinel).
    pub fn none() -> Self {
        Self { n_layers: 0, n_kv_heads: 0, head_dim: 0, max_seq_len: 0, dtype: DType::F32 }
    }

    /// Total bytes consumed by the KV-cache.
    pub fn byte_size(&self) -> u64 {
        if self.n_layers == 0 { return 0; }
        let elem_bytes = match self.dtype {
            DType::F32 | DType::U8 => 4u64,
            DType::F16 | DType::BF16 => 2,
            DType::INT8 => 1,
            DType::INT4 => 1, // conservatively rounded up
        };
        // K + V for each layer
        2 * self.n_layers as u64
            * self.n_kv_heads as u64
            * self.max_seq_len as u64
            * self.head_dim as u64
            * elem_bytes
    }
}

/// High-level memory budget derived from graph analysis.
pub struct MemoryPlan {
    pub kv_cache_layout: KvCacheLayout,
    pub total_weight_bytes: u64,
    pub total_activation_bytes: u64,
}

/// Estimates memory requirements from `AiGraph` topology.
pub struct MemoryPlanner;

impl MemoryPlanner {
    /// Produce a conservative `MemoryPlan`.
    ///
    /// Sprint 001: counts weight bytes exactly; activations are estimated as
    /// a fixed multiple of the largest activation tensor found. KV-cache is
    /// caller-supplied via `KvCacheLayout`.
    pub fn plan(&self, graph: &AiGraph) -> anyhow::Result<MemoryPlan> {
        let total_weight_bytes = graph.params.values()
            .map(param_bytes)
            .sum();

        // Conservative activation estimate: max tensor footprint × node count.
        let param_ids: std::collections::HashSet<_> = graph.params.keys().copied().collect();
        let max_activation = graph.tensor_info.iter()
            .filter(|(tid, _)| !param_ids.contains(tid))
            .map(|(_, ti)| {
                let n_elems: u64 = ti.shape.iter()
                    .filter_map(|d| d.as_concrete())
                    .product();
                n_elems * 4
            })
            .max()
            .unwrap_or(0);

        let total_activation_bytes = max_activation * graph.nodes.len() as u64;

        Ok(MemoryPlan {
            kv_cache_layout: KvCacheLayout::none(),
            total_weight_bytes,
            total_activation_bytes,
        })
    }
}

fn param_bytes(param: &AiParam) -> u64 {
    match param {
        AiParam::Inline { data, .. } => data.len() as u64,
        AiParam::Mmap { len, .. } => *len,
    }
}
