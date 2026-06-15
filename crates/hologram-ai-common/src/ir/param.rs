use super::graph::TensorInfo;
use std::path::PathBuf;
use std::sync::Arc;

/// A model weight or constant tensor.
///
/// Weights remain in their source representation until the lowering pass
/// decides the dequantization strategy.
///
/// `Inline` data uses `Arc<Vec<u8>>` so that `AiGraph::clone()` is cheap —
/// weight bytes are reference-counted, not deep-copied. This enables
/// parallel compilation of prefill/decode/verify graphs from cloned graphs.
#[derive(Debug, Clone)]
pub enum AiParam {
    /// Small weights embedded directly in the graph.
    Inline {
        data: Arc<Vec<u8>>,
        info: TensorInfo,
    },
    /// Large weights memory-mapped from the source model file.
    Mmap {
        path: PathBuf,
        offset: u64,
        len: u64,
        info: TensorInfo,
    },
}

impl AiParam {
    /// Construct an inline parameter from owned bytes.
    pub fn inline(data: Vec<u8>, info: TensorInfo) -> Self {
        Self::Inline {
            data: Arc::new(data),
            info,
        }
    }

    /// Construct a memory-mapped parameter reference.
    pub fn mmap(path: PathBuf, offset: u64, len: u64, info: TensorInfo) -> Self {
        Self::Mmap {
            path,
            offset,
            len,
            info,
        }
    }

    /// Metadata for this parameter.
    pub fn info(&self) -> &TensorInfo {
        match self {
            AiParam::Inline { info, .. } => info,
            AiParam::Mmap { info, .. } => info,
        }
    }

    /// Interpret inline data as f32 slice (for small constant params like scales).
    /// Returns `None` for mmap params or if the data isn't aligned/sized for f32.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            AiParam::Inline { data, .. } if data.len() >= 4 => bytemuck::try_cast_slice(data).ok(),
            _ => None,
        }
    }

    /// Whether this parameter is missing required backing data.
    ///
    /// Zero-sized tensors legitimately serialize as zero bytes.
    pub fn is_empty(&self) -> bool {
        match self {
            AiParam::Inline { data, info } => data.is_empty() && !tensor_is_zero_sized(info),
            AiParam::Mmap { len, info, .. } => *len == 0 && !tensor_is_zero_sized(info),
        }
    }
}

fn tensor_is_zero_sized(info: &TensorInfo) -> bool {
    info.shape.iter().any(|dim| dim.as_concrete() == Some(0))
}
