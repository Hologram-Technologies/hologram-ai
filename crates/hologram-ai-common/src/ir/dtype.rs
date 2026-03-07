/// Arithmetic or storage data type for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    INT8,
    INT4,
    U8,
}

impl DType {
    /// Size of a single element in bytes. Returns `None` for sub-byte types (INT4).
    pub fn byte_size(self) -> Option<usize> {
        match self {
            DType::F32  => Some(4),
            DType::F16  => Some(2),
            DType::BF16 => Some(2),
            DType::INT8 => Some(1),
            DType::U8   => Some(1),
            DType::INT4 => None,  // 4 bits — caller handles packing
        }
    }
}
