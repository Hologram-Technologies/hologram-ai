//! Quantization cross-validation: hologram-ai-quant vs dispatch_float(Dequantize).
//!
//! Tier 1: Compare the compiler's dequantization (hologram-ai-quant) against
//! the runtime's (dispatch_float with FloatOp::Dequantize). Both must agree
//! bit-for-bit on the same Q4_0 input.

use hologram::hologram_exec::float_dispatch::dispatch_float;
use hologram::hologram_core::op::FloatOp;
use hologram_ai_quant::{dequant_q4_0, Q4_0_BLOCK_SIZE};

/// Build a Q4_0 block from scale (f16 bits) and quantized nibbles.
fn make_q4_0_block(scale_bits: u16, qs: &[u8; 16]) -> Vec<u8> {
    let mut block = Vec::with_capacity(Q4_0_BLOCK_SIZE);
    block.extend_from_slice(&scale_bits.to_le_bytes());
    block.extend_from_slice(qs);
    block
}

#[test]
fn quant_cross_validate_q4_0_zero_block() {
    // scale = f16(1.0) = 0x3C00, all nibbles = 8 → dequant to 0.0
    let data = make_q4_0_block(0x3C00, &[0x88; 16]);

    let quant_result = dequant_q4_0(&data);
    let dispatch_result =
        dispatch_float(&FloatOp::Dequantize, &[&data]).expect("dispatch_float(Dequantize) failed");
    let dispatch_f32: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&dispatch_result).to_vec();

    assert_eq!(quant_result.len(), dispatch_f32.len(), "length mismatch");
    for (i, (q, d)) in quant_result.iter().zip(dispatch_f32.iter()).enumerate() {
        assert!((q - d).abs() < 1e-10, "index {i}: quant={q} dispatch={d}");
    }
}

#[test]
fn quant_cross_validate_q4_0_known_values() {
    // scale = f16(2.0) = 0x4000, qs[0] = 0x09 → lo=1*2=2.0, hi=-8*2=-16.0
    let mut qs = [0x88u8; 16];
    qs[0] = 0x09;
    qs[1] = 0xFA; // lo = A(10) - 8 = 2 → 4.0, hi = F(15) - 8 = 7 → 14.0
    let data = make_q4_0_block(0x4000, &qs);

    let quant_result = dequant_q4_0(&data);
    let dispatch_result =
        dispatch_float(&FloatOp::Dequantize, &[&data]).expect("dispatch_float(Dequantize) failed");
    let dispatch_f32: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&dispatch_result).to_vec();

    assert_eq!(quant_result.len(), dispatch_f32.len());
    for (i, (q, d)) in quant_result.iter().zip(dispatch_f32.iter()).enumerate() {
        assert!((q - d).abs() < 1e-10, "index {i}: quant={q} dispatch={d}");
    }

    // Also verify known values from quant crate
    assert!(
        (quant_result[0] - 2.0).abs() < 1e-5,
        "qs[0] lo: expected 2.0, got {}",
        quant_result[0]
    );
    assert!(
        (quant_result[1] - (-16.0)).abs() < 1e-5,
        "qs[0] hi: expected -16.0, got {}",
        quant_result[1]
    );
    assert!(
        (quant_result[2] - 4.0).abs() < 1e-5,
        "qs[1] lo: expected 4.0, got {}",
        quant_result[2]
    );
    assert!(
        (quant_result[3] - 14.0).abs() < 1e-5,
        "qs[1] hi: expected 14.0, got {}",
        quant_result[3]
    );
}

#[test]
fn quant_cross_validate_q4_0_multi_block() {
    // Two blocks with different scales and values
    let block1 = make_q4_0_block(0x3C00, &[0x79; 16]); // scale=1.0, lo=1, hi=-1
    let block2 = make_q4_0_block(0x4000, &[0xA6; 16]); // scale=2.0, lo=-2, hi=2
    let data: Vec<u8> = block1.into_iter().chain(block2).collect();

    let quant_result = dequant_q4_0(&data);
    let dispatch_result =
        dispatch_float(&FloatOp::Dequantize, &[&data]).expect("dispatch_float(Dequantize) failed");
    let dispatch_f32: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&dispatch_result).to_vec();

    assert_eq!(quant_result.len(), 64, "2 blocks → 64 values");
    assert_eq!(dispatch_f32.len(), 64);
    for (i, (q, d)) in quant_result.iter().zip(dispatch_f32.iter()).enumerate() {
        assert!((q - d).abs() < 1e-10, "index {i}: quant={q} dispatch={d}");
    }
}

#[test]
fn quant_cross_validate_q4_0_negative_scale() {
    // Negative scale: f16(-0.5) = 0xB800
    let data = make_q4_0_block(0xB800, &[0x09; 16]);

    let quant_result = dequant_q4_0(&data);
    let dispatch_result =
        dispatch_float(&FloatOp::Dequantize, &[&data]).expect("dispatch_float(Dequantize) failed");
    let dispatch_f32: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&dispatch_result).to_vec();

    assert_eq!(quant_result.len(), dispatch_f32.len());
    for (i, (q, d)) in quant_result.iter().zip(dispatch_f32.iter()).enumerate() {
        assert!((q - d).abs() < 1e-10, "index {i}: quant={q} dispatch={d}");
    }
}

#[test]
fn quant_cross_validate_q4_0_extreme_nibbles() {
    // Test all nibble extremes: 0x00 (lo=0-8=-8, hi=0-8=-8) and 0xFF (lo=15-8=7, hi=15-8=7)
    let mut qs = [0x00u8; 16];
    qs[0] = 0x00; // min nibbles
    qs[1] = 0xFF; // max nibbles
    let data = make_q4_0_block(0x3C00, &qs); // scale = 1.0

    let quant_result = dequant_q4_0(&data);
    let dispatch_result =
        dispatch_float(&FloatOp::Dequantize, &[&data]).expect("dispatch_float(Dequantize) failed");
    let dispatch_f32: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&dispatch_result).to_vec();

    // Check extreme values
    assert!((quant_result[0] - (-8.0)).abs() < 1e-5, "min nibble lo");
    assert!((quant_result[1] - (-8.0)).abs() < 1e-5, "min nibble hi");
    assert!((quant_result[2] - 7.0).abs() < 1e-5, "max nibble lo");
    assert!((quant_result[3] - 7.0).abs() < 1e-5, "max nibble hi");

    // Cross-validate
    for (i, (q, d)) in quant_result.iter().zip(dispatch_f32.iter()).enumerate() {
        assert!((q - d).abs() < 1e-10, "index {i}: quant={q} dispatch={d}");
    }
}
