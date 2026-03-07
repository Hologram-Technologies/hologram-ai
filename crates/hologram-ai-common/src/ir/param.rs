use std::path::PathBuf;
use super::graph::TensorInfo;

/// A model weight or constant tensor.
///
/// Weights remain in their source representation until the lowering pass
/// decides the dequantization strategy.
#[derive(Debug, Clone)]
pub enum AiParam {
    /// Small weights embedded directly in the graph.
    Inline { data: Vec<u8>, info: TensorInfo },
    /// Large weights memory-mapped from the source model file.
    Mmap { path: PathBuf, offset: u64, len: u64, info: TensorInfo },
}

impl AiParam {
    /// Construct an inline parameter from owned bytes.
    pub fn inline(data: Vec<u8>, info: TensorInfo) -> Self {
        Self::Inline { data, info }
    }

    /// Construct a memory-mapped parameter reference.
    pub fn mmap(path: PathBuf, offset: u64, len: u64, info: TensorInfo) -> Self {
        Self::Mmap { path, offset, len, info }
    }

    /// Metadata for this parameter.
    pub fn info(&self) -> &TensorInfo {
        match self {
            AiParam::Inline { info, .. } => info,
            AiParam::Mmap  { info, .. } => info,
        }
    }

    /// Whether this parameter has no backing data (invalid).
    pub fn is_empty(&self) -> bool {
        match self {
            AiParam::Inline { data, .. } => data.is_empty(),
            AiParam::Mmap  { len, .. }  => *len == 0,
        }
    }
}
