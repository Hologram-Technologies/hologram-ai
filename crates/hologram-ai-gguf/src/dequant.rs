//! Dequantization routines for GGUF quantized weights.
//!
//! This module implements dequantization from various GGML quantization formats
//! to F32. The supported formats are:
//!
//! - F32, F16, BF16 (trivial conversion)
//! - Q8_0 (8-bit quantization)
//! - Q4_0 (legacy 4-bit)
//! - Q4_K (K-quant 4-bit, most common)

use crate::error::{GgufError, Result};
use crate::parser::GgmlType;

/// Dequantize raw bytes to F32 vector.
///
/// # Arguments
/// * `data` - Raw quantized data bytes
/// * `dtype` - GGML data type
/// * `n_elements` - Expected number of output elements
///
/// # Returns
/// Vector of F32 values
pub fn dequantize(data: &[u8], dtype: GgmlType, n_elements: usize) -> Result<Vec<f32>> {
    match dtype {
        GgmlType::F32 => dequant_f32(data, n_elements),
        GgmlType::F16 => dequant_f16(data, n_elements),
        GgmlType::BF16 => dequant_bf16(data, n_elements),
        GgmlType::Q8_0 => dequant_q8_0(data, n_elements),
        GgmlType::Q4_0 => dequant_q4_0(data, n_elements),
        GgmlType::Q4_K => dequant_q4_k(data, n_elements),
        other => Err(GgufError::UnsupportedQuantization(format!("{:?}", other))),
    }
}

/// Dequantize F32 (no-op, just reinterpret bytes).
fn dequant_f32(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    if data.len() < n_elements * 4 {
        return Err(GgufError::InvalidMetadata {
            key: "tensor_data".to_string(),
            message: format!("Expected {} bytes for F32, got {}", n_elements * 4, data.len()),
        });
    }

    let mut result = Vec::with_capacity(n_elements);
    for chunk in data[..n_elements * 4].chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().unwrap();
        result.push(f32::from_le_bytes(bytes));
    }
    Ok(result)
}

/// Dequantize F16 to F32.
fn dequant_f16(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    if data.len() < n_elements * 2 {
        return Err(GgufError::InvalidMetadata {
            key: "tensor_data".to_string(),
            message: format!("Expected {} bytes for F16, got {}", n_elements * 2, data.len()),
        });
    }

    let mut result = Vec::with_capacity(n_elements);
    for chunk in data[..n_elements * 2].chunks_exact(2) {
        let bytes: [u8; 2] = chunk.try_into().unwrap();
        let f16_val = half::f16::from_le_bytes(bytes);
        result.push(f16_val.to_f32());
    }
    Ok(result)
}

/// Dequantize BF16 to F32.
fn dequant_bf16(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    if data.len() < n_elements * 2 {
        return Err(GgufError::InvalidMetadata {
            key: "tensor_data".to_string(),
            message: format!("Expected {} bytes for BF16, got {}", n_elements * 2, data.len()),
        });
    }

    let mut result = Vec::with_capacity(n_elements);
    for chunk in data[..n_elements * 2].chunks_exact(2) {
        let bytes: [u8; 2] = chunk.try_into().unwrap();
        let bf16_val = half::bf16::from_le_bytes(bytes);
        result.push(bf16_val.to_f32());
    }
    Ok(result)
}

/// Dequantize Q8_0 format.
///
/// Q8_0 block structure (34 bytes per 32 elements):
/// - 2 bytes: f16 scale
/// - 32 bytes: int8 quantized values
///
/// Dequantization: x_i = scale * q_i
fn dequant_q8_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 34; // 2 (scale) + 32 (quants)

    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    if data.len() < n_blocks * BYTES_PER_BLOCK {
        return Err(GgufError::InvalidMetadata {
            key: "tensor_data".to_string(),
            message: format!("Insufficient data for Q8_0: expected {} bytes, got {}",
                           n_blocks * BYTES_PER_BLOCK, data.len()),
        });
    }

    let mut result = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        let block = &data[block_offset..block_offset + BYTES_PER_BLOCK];

        // Read scale (f16)
        let scale_bytes: [u8; 2] = block[0..2].try_into().unwrap();
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        // Dequantize values
        let quants = &block[2..34];
        for &quant in quants.iter().take(BLOCK_SIZE) {
            if result.len() >= n_elements {
                break;
            }
            let q = quant as i8;
            result.push(scale * q as f32);
        }
    }

    Ok(result)
}

/// Dequantize Q4_0 format.
///
/// Q4_0 block structure (18 bytes per 32 elements):
/// - 2 bytes: f16 scale
/// - 16 bytes: 32 x 4-bit quantized values (2 per byte)
///
/// Dequantization: x_i = scale * (q_i - 8)
fn dequant_q4_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 18; // 2 (scale) + 16 (quants)

    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    if data.len() < n_blocks * BYTES_PER_BLOCK {
        return Err(GgufError::InvalidMetadata {
            key: "tensor_data".to_string(),
            message: format!("Insufficient data for Q4_0: expected {} bytes, got {}",
                           n_blocks * BYTES_PER_BLOCK, data.len()),
        });
    }

    let mut result = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        let block = &data[block_offset..block_offset + BYTES_PER_BLOCK];

        // Read scale (f16)
        let scale_bytes: [u8; 2] = block[0..2].try_into().unwrap();
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        // Dequantize values (2 values per byte)
        let quants = &block[2..18];
        for &byte in quants.iter().take(16) {

            // Low nibble
            if result.len() < n_elements {
                let q_low = (byte & 0x0F) as i32 - 8;
                result.push(scale * q_low as f32);
            }

            // High nibble
            if result.len() < n_elements {
                let q_high = ((byte >> 4) & 0x0F) as i32 - 8;
                result.push(scale * q_high as f32);
            }
        }
    }

    Ok(result)
}

/// Dequantize Q4_K format (K-quant).
///
/// Q4_K block structure (144 bytes per 256 elements):
/// - 2 bytes: f16 scale (d)
/// - 2 bytes: f16 min (dmin)
/// - 12 bytes: 6-bit scales for 8 sub-blocks
/// - 4 bytes: 4-bit mins for 8 sub-blocks
/// - 128 bytes: 256 x 4-bit quantized values
///
/// This is a more complex format with per-sub-block scales.
fn dequant_q4_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 144;
    const N_SCALES: usize = 8; // 8 sub-blocks of 32 elements each

    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    if data.len() < n_blocks * BYTES_PER_BLOCK {
        return Err(GgufError::InvalidMetadata {
            key: "tensor_data".to_string(),
            message: format!("Insufficient data for Q4_K: expected {} bytes, got {}",
                           n_blocks * BYTES_PER_BLOCK, data.len()),
        });
    }

    let mut result = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        let block = &data[block_offset..block_offset + BYTES_PER_BLOCK];

        // Read super-block scales
        let d_bytes: [u8; 2] = block[0..2].try_into().unwrap();
        let d = half::f16::from_le_bytes(d_bytes).to_f32();

        let dmin_bytes: [u8; 2] = block[2..4].try_into().unwrap();
        let dmin = half::f16::from_le_bytes(dmin_bytes).to_f32();

        // Read sub-block scales (6 bits each, packed in 12 bytes)
        let scales_bytes = &block[4..16];
        let mut scales = [0u8; N_SCALES];
        let mut mins = [0u8; N_SCALES];

        // Unpack 6-bit scales
        scales[0] = scales_bytes[0] & 0x3F;
        scales[1] = (scales_bytes[0] >> 6) | ((scales_bytes[1] & 0x0F) << 2);
        scales[2] = (scales_bytes[1] >> 4) | ((scales_bytes[2] & 0x03) << 4);
        scales[3] = scales_bytes[2] >> 2;
        scales[4] = scales_bytes[3] & 0x3F;
        scales[5] = (scales_bytes[3] >> 6) | ((scales_bytes[4] & 0x0F) << 2);
        scales[6] = (scales_bytes[4] >> 4) | ((scales_bytes[5] & 0x03) << 4);
        scales[7] = scales_bytes[5] >> 2;

        // Read mins (4 bits each, from bytes 6-11)
        mins[0] = scales_bytes[6] & 0x0F;
        mins[1] = scales_bytes[6] >> 4;
        mins[2] = scales_bytes[7] & 0x0F;
        mins[3] = scales_bytes[7] >> 4;
        mins[4] = scales_bytes[8] & 0x0F;
        mins[5] = scales_bytes[8] >> 4;
        mins[6] = scales_bytes[9] & 0x0F;
        mins[7] = scales_bytes[9] >> 4;

        // Read quantized values
        let quants = &block[16..144];

        // Dequantize each sub-block
        for sb in 0..N_SCALES {
            let sc = d * scales[sb] as f32;
            let m = dmin * mins[sb] as f32;

            let quant_offset = sb * 16; // 32 values = 16 bytes
            for byte_idx in 0..16 {
                let byte = quants[quant_offset + byte_idx];

                // Low nibble
                if result.len() < n_elements {
                    let q = (byte & 0x0F) as f32;
                    result.push(sc * q - m);
                }

                // High nibble
                if result.len() < n_elements {
                    let q = ((byte >> 4) & 0x0F) as f32;
                    result.push(sc * q - m);
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_f32() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let result = dequant_f32(&bytes, 4).unwrap();
        assert_eq!(result, values);
    }

    #[test]
    fn test_dequant_f16() {
        let values: Vec<half::f16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let bytes: Vec<u8> = values.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let result = dequant_f16(&bytes, 3).unwrap();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
        assert!((result[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_dequant_bf16() {
        let values: Vec<half::bf16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(half::bf16::from_f32)
            .collect();
        let bytes: Vec<u8> = values.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let result = dequant_bf16(&bytes, 3).unwrap();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
        assert!((result[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_dequant_q8_0_basic() {
        // Create a simple Q8_0 block with known values
        let mut block = vec![0u8; 34];

        // Scale = 0.5 (as f16)
        let scale = half::f16::from_f32(0.5);
        block[0..2].copy_from_slice(&scale.to_le_bytes());

        // Quantized values: [2, 4, 6, ...]
        for i in 0..32 {
            block[2 + i] = (i * 2) as u8;
        }

        let result = dequant_q8_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);

        // First value: 0.5 * 0 = 0.0
        assert!((result[0] - 0.0).abs() < 0.01);
        // Second value: 0.5 * 2 = 1.0
        assert!((result[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dequant_q4_0_basic() {
        // Create a simple Q4_0 block with known values
        let mut block = vec![0u8; 18];

        // Scale = 1.0 (as f16)
        let scale = half::f16::from_f32(1.0);
        block[0..2].copy_from_slice(&scale.to_le_bytes());

        // Set first byte to pack values 8 and 8 (which become 0 after -8)
        block[2] = 0x88; // low nibble = 8, high nibble = 8

        let result = dequant_q4_0(&block, 2).unwrap();
        assert_eq!(result.len(), 2);

        // Both values: 1.0 * (8 - 8) = 0.0
        assert!((result[0] - 0.0).abs() < 0.01);
        assert!((result[1] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_dequant_insufficient_data() {
        let result = dequant_f32(&[0, 0, 0], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_quantization() {
        let result = dequantize(&[], GgmlType::Q2_K, 0);
        assert!(result.is_err());
    }
}
