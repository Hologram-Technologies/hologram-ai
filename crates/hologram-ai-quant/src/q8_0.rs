/// Q8_0 block: 34 bytes — [f16 scale (2 bytes)][32 × i8].
///
/// Dequantization: weight = i8_value × scale
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Q8_0Block {
    /// Scale factor stored as raw f16 bits (little-endian).
    pub scale: u16,
    /// 32 quantized values as signed 8-bit integers.
    pub qs: [i8; 32],
}

/// Size of a single Q8_0 block in bytes.
pub const Q8_0_BLOCK_SIZE: usize = 34;

const _: [u8; Q8_0_BLOCK_SIZE] = [0u8; std::mem::size_of::<Q8_0Block>()];

/// Dequantize a single Q8_0 block to 32 f32 values.
pub fn dequant_q8_0_block(block: &Q8_0Block) -> [f32; 32] {
    let scale = half::f16::from_bits(block.scale).to_f32();
    let mut out = [0f32; 32];
    for (i, &q) in block.qs.iter().enumerate() {
        out[i] = q as f32 * scale;
    }
    out
}

/// Dequantize a raw Q8_0 byte slice to f32 values.
///
/// # Panics
/// Panics if `data.len()` is not a multiple of `Q8_0_BLOCK_SIZE` (34).
pub fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    assert_eq!(
        data.len() % Q8_0_BLOCK_SIZE,
        0,
        "Q8_0 data length must be a multiple of 34"
    );
    // SAFETY: Q8_0Block is repr(C, packed) with Pod; alignment is 1.
    let blocks: &[Q8_0Block] = bytemuck::cast_slice(data);
    let mut out = Vec::with_capacity(blocks.len() * 32);
    for block in blocks {
        out.extend_from_slice(&dequant_q8_0_block(block));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_zero_block() {
        let block = Q8_0Block {
            scale: 0x3C00,
            qs: [0i8; 32],
        }; // scale = f16(1.0)
        let vals = dequant_q8_0_block(&block);
        for v in vals.iter() {
            assert_eq!(*v, 0.0f32);
        }
    }

    #[test]
    fn dequant_known_values() {
        // scale = f16(0.5) = 0x3800; qs = [2, -4, 127, ...]
        let mut block = Q8_0Block {
            scale: 0x3800,
            qs: [0i8; 32],
        };
        block.qs[0] = 2;
        block.qs[1] = -4;
        block.qs[2] = 127;
        let vals = dequant_q8_0_block(&block);
        assert!((vals[0] - 1.0f32).abs() < 1e-4, "vals[0]={}", vals[0]);
        assert!((vals[1] - (-2.0f32)).abs() < 1e-4, "vals[1]={}", vals[1]);
        assert!((vals[2] - 63.5f32).abs() < 1e-3, "vals[2]={}", vals[2]);
    }

    #[test]
    fn dequant_q8_0_slice() {
        let mut data = vec![0u8; 34];
        // scale = f16(1.0) = 0x3C00 LE
        data[0] = 0x00;
        data[1] = 0x3C;
        // qs[0] = 5 → 5.0
        data[2] = 5u8;
        let out = dequant_q8_0(&data);
        assert_eq!(out.len(), 32);
        assert!((out[0] - 5.0f32).abs() < 1e-5);
    }

    #[test]
    #[should_panic]
    fn dequant_q8_0_bad_length() {
        dequant_q8_0(&[0u8; 33]);
    }
}
