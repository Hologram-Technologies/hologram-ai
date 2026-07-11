/// Arithmetic or storage data type for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
    INT8,
    INT4,
    U8,
    INT16,
    INT32,
    INT64,
    BOOL,
}

impl DType {
    /// Size of a single element in bytes. Returns `None` for sub-byte types (INT4).
    pub fn byte_size(self) -> Option<usize> {
        match self {
            DType::F32 => Some(4),
            DType::F64 => Some(8),
            DType::F16 => Some(2),
            DType::BF16 => Some(2),
            DType::INT8 => Some(1),
            DType::U8 => Some(1),
            DType::INT16 => Some(2),
            DType::INT32 => Some(4),
            DType::INT64 => Some(8),
            DType::BOOL => Some(1),
            DType::INT4 => None, // 4 bits — caller handles packing
        }
    }

    /// Bytes a tensor of `elems` elements occupies, INCLUDING sub-byte packing:
    /// INT4 packs two codes per byte, so `⌈elems/2⌉`. This is the packing-aware
    /// companion to [`DType::byte_size`], which returns `None` for sub-byte types
    /// and so cannot size a packed tensor on its own. Use this everywhere a
    /// tensor's footprint is computed, so int4 is never mis-accounted as 0 bytes
    /// (a footprint-accounting seam — see the wasm residency history).
    pub fn packed_bytes(self, elems: u64) -> u64 {
        match self.byte_size() {
            Some(b) => elems * b as u64,
            None => match self {
                DType::INT4 => elems.div_ceil(2),
                _ => 0, // no other sub-byte type today
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_bytes_accounts_int4_as_half_never_zero() {
        // Whole-byte dtypes: elems × width.
        assert_eq!(DType::INT8.packed_bytes(100), 100);
        assert_eq!(DType::F32.packed_bytes(100), 400);
        // INT4 is ⌈elems/2⌉ — NOT 0 (the old `byte_size().unwrap_or(0)` seam) and
        // NOT floored (odd counts round up, matching the encoder's allocation).
        assert_eq!(DType::INT4.packed_bytes(100), 50);
        assert_eq!(DType::INT4.packed_bytes(101), 51);
        assert_ne!(DType::INT4.packed_bytes(100), 0, "int4 is never free");
    }
}
