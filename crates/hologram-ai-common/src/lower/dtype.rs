//! Canonical dtype tags for hologram 0.5.0 graph construction.
//!
//! In the UOR-native model a tensor's dtype is a `hologram_graph::DTypeId`
//! (a `u8` wire tag), not the deleted `FloatDType` enum. The canonical
//! tag values are fixed by hologram's backend wire encoding
//! (`hologram_backend::cpu::dtype`); we mirror them here so the runtime-core
//! crate stays free of a backend dependency (it must build `no_std` for
//! wasm/embedded — see CONFORMANCE.md class NS). The `dtype_tags_agree`
//! V&V check (class CF) holds these against the backend constants.

use hologram_graph::registry::DTypeId;

// Canonical numeric dtype tags (mirror of `hologram_backend::cpu::dtype`).
pub const DTYPE_BOOL: u8 = 0;
pub const DTYPE_U8: u8 = 1;
pub const DTYPE_I8: u8 = 2;
pub const DTYPE_U64: u8 = 3;
pub const DTYPE_I32: u8 = 4;
pub const DTYPE_I64: u8 = 5;
pub const DTYPE_F16: u8 = 6;
pub const DTYPE_BF16: u8 = 7;
pub const DTYPE_F32: u8 = 8;
pub const DTYPE_F64: u8 = 9;
pub const DTYPE_I4: u8 = 10;

/// Map a hologram-ai `ir::DType` to a canonical `DTypeId`.
///
/// Widening/lowering decisions (kept faithful to what hologram's kernels
/// accept):
/// - `F64` → `F32` (no f64 kernels in the backend),
/// - `INT4` → `I4` (packed 4-bit, two nibbles/byte — the canonical tag for
///   sub-byte quantized weights; hologram sizes it `div_ceil(n, 2)` and
///   unpacks each nibble in its dequant kernels),
/// - `INT16` → `I32` (no i16 tag in the canonical set).
pub fn ai_dtype_to_dtype_id(dtype: &crate::ir::DType) -> DTypeId {
    use crate::ir::DType;
    let tag = match dtype {
        DType::F32 => DTYPE_F32,
        DType::F64 => DTYPE_F32,
        DType::F16 => DTYPE_F16,
        DType::BF16 => DTYPE_BF16,
        DType::INT8 => DTYPE_I8,
        DType::INT4 => DTYPE_I4,
        DType::U8 => DTYPE_U8,
        DType::INT16 => DTYPE_I32,
        DType::INT32 => DTYPE_I32,
        DType::INT64 => DTYPE_I64,
        DType::BOOL => DTYPE_BOOL,
    };
    DTypeId(tag)
}

/// The default tensor dtype (`f32`) as a `DTypeId`.
pub fn default_dtype_id() -> DTypeId {
    DTypeId(DTYPE_F32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DType;

    #[test]
    fn f32_maps_to_canonical_tag_8() {
        assert_eq!(ai_dtype_to_dtype_id(&DType::F32), DTypeId(8));
    }

    #[test]
    fn integer_widening_is_faithful() {
        assert_eq!(ai_dtype_to_dtype_id(&DType::INT16), DTypeId(DTYPE_I32));
        assert_eq!(ai_dtype_to_dtype_id(&DType::INT4), DTypeId(DTYPE_I4));
        assert_eq!(ai_dtype_to_dtype_id(&DType::F64), DTypeId(DTYPE_F32));
    }
}
