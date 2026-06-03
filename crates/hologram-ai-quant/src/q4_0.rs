use alloc::vec::Vec;
/// Q4_0 block: 18 bytes — [f16 scale (2 bytes)][32 nibbles packed into 16 bytes].
///
/// Dequantization: weight = (nibble − 8) × scale
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Q4_0Block {
    /// Scale factor stored as raw f16 bits (little-endian).
    pub scale: u16,
    /// 32 4-bit weights packed two-per-byte (low nibble first).
    pub qs: [u8; 16],
}

/// Size of a single Q4_0 block in bytes.
pub const Q4_0_BLOCK_SIZE: usize = 18;

const _: [u8; Q4_0_BLOCK_SIZE] = [0u8; core::mem::size_of::<Q4_0Block>()];

/// Dequantize a single Q4_0 block to 32 f32 values.
pub fn dequant_q4_0_block(block: &Q4_0Block) -> [f32; 32] {
    let scale = half::f16::from_bits(block.scale).to_f32();
    let mut out = [0f32; 32];
    for (i, &byte) in block.qs.iter().enumerate() {
        let lo = (byte & 0x0F) as i32 - 8;
        let hi = ((byte >> 4) & 0x0F) as i32 - 8;
        out[2 * i] = lo as f32 * scale;
        out[2 * i + 1] = hi as f32 * scale;
    }
    out
}

/// Dequantize a raw Q4_0 byte slice to f32 values.
///
/// # Panics
/// Panics if `data.len()` is not a multiple of `Q4_0_BLOCK_SIZE` (18).
pub fn dequant_q4_0(data: &[u8]) -> Vec<f32> {
    assert_eq!(
        data.len() % Q4_0_BLOCK_SIZE,
        0,
        "Q4_0 data length must be a multiple of 18"
    );
    let blocks: &[Q4_0Block] = bytemuck::cast_slice(data);
    let mut out = Vec::with_capacity(blocks.len() * 32);
    for block in blocks {
        out.extend_from_slice(&dequant_q4_0_block(block));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn dequant_zero_block() {
        // scale = f16::from_bits(0) = 0.0, all qs = 0x88 (nibbles all 8, so offset 0)
        let block = Q4_0Block {
            scale: 0x3C00,
            qs: [0x88u8; 16],
        }; // scale = f16(1.0)
        let vals = dequant_q4_0_block(&block);
        // low nibble = 8 & 0xF = 8, 8 - 8 = 0; high nibble = 8, 8 - 8 = 0
        for v in vals.iter() {
            assert_eq!(*v, 0.0f32);
        }
    }

    #[test]
    fn dequant_known_block() {
        // scale = f16(1.0) = 0x3C00
        // qs[0] = 0x09 → lo nibble = 9 → 9-8=1, hi nibble = 0 → 0-8=-8
        let mut block = Q4_0Block {
            scale: 0x3C00,
            qs: [0x88u8; 16],
        };
        block.qs[0] = 0x09; // lo=9-8=1, hi=0-8=-8
        let vals = dequant_q4_0_block(&block);
        assert!((vals[0] - 1.0f32).abs() < 1e-5, "vals[0]={}", vals[0]);
        assert!((vals[1] - (-8.0f32)).abs() < 1e-5, "vals[1]={}", vals[1]);
        // remaining should be 0
        for v in &vals[2..] {
            assert_eq!(*v, 0.0f32);
        }
    }

    #[test]
    fn dequant_q4_0_slice() {
        // two identical blocks: scale=f16(2.0)=0x4000, all nibbles=8 (offset 0)
        let mut data = vec![0u8; 36];
        data[0] = 0x00;
        data[1] = 0x40; // f16(2.0) LE
        data[18] = 0x00;
        data[19] = 0x40;
        for b in &mut data[2..18] {
            *b = 0x88;
        }
        for b in &mut data[20..36] {
            *b = 0x88;
        }
        let out = dequant_q4_0(&data);
        assert_eq!(out.len(), 64);
        for v in out.iter() {
            assert_eq!(*v, 0.0f32);
        }
    }

    #[test]
    #[should_panic]
    fn dequant_q4_0_bad_length() {
        dequant_q4_0(&[0u8; 17]);
    }
}
