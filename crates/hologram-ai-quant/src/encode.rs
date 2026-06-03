//! f32 → per-channel symmetric int8 weight encoder (the inverse of the dequant
//! unpackers in this crate). `no_std`; pure arithmetic, no IR dependency.

use alloc::vec;
use alloc::vec::Vec;

/// `no_std`-safe f32 round-half-away-from-zero via libm (`f32::round` is not in core).
#[inline(always)]
fn round_f32(x: f32) -> f32 {
    libm::roundf(x)
}

/// Per-channel symmetric int8 encoding of a row-major `[k, n]` weight.
///
/// One scale per **column** (output channel `j`):
/// `scale_j = max_i |W[i,j]| / 127`. Returns `(q, scales)` where `q` is the
/// row-major `[k,n]` i8 weight and `scales` has length `n`. A column of all
/// zeros gets `scale = 1.0` so dequant reproduces zeros exactly.
pub fn encode_int8_per_channel(w: &[f32], k: usize, n: usize) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(w.len(), k * n, "weight length must equal k*n");
    let mut scales = vec![1.0f32; n];
    for (j, scale) in scales.iter_mut().enumerate() {
        let mut amax = 0.0f32;
        for i in 0..k {
            let a = w[i * n + j].abs();
            if a > amax {
                amax = a;
            }
        }
        if amax > 0.0 {
            *scale = amax / 127.0;
        }
    }
    let mut q = vec![0i8; k * n];
    for i in 0..k {
        for j in 0..n {
            let v = round_f32(w[i * n + j] / scales[j]);
            let clamped = v.clamp(-127.0, 127.0);
            q[i * n + j] = clamped as i8;
        }
    }
    (q, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_within_half_scale() {
        let (k, n) = (5usize, 3usize);
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.13 - 0.7).collect();
        let (q, scales) = encode_int8_per_channel(&w, k, n);
        assert_eq!(scales.len(), n);
        assert_eq!(q.len(), k * n);
        for i in 0..k {
            for j in 0..n {
                let deq = q[i * n + j] as f32 * scales[j];
                assert!(
                    (deq - w[i * n + j]).abs() <= scales[j] / 2.0 + 1e-6,
                    "elem ({i},{j}): deq {deq} vs {}",
                    w[i * n + j]
                );
            }
        }
    }

    #[test]
    fn zero_column_scale_one_and_exact_zero() {
        let (k, n) = (3usize, 2usize);
        // Column 1 is all zeros.
        let w = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let (q, scales) = encode_int8_per_channel(&w, k, n);
        assert_eq!(scales[1], 1.0);
        for i in 0..k {
            assert_eq!(q[i * n + 1], 0);
        }
    }

    #[test]
    fn max_abs_maps_to_127() {
        // A column whose max-abs element should hit ±127 after rounding.
        let (k, n) = (2usize, 1usize);
        let w = vec![0.5f32, -1.0];
        let (q, scales) = encode_int8_per_channel(&w, k, n);
        assert_eq!(scales[0], 1.0 / 127.0);
        assert_eq!(q[1], -127);
        assert_eq!(q[0], 64); // 63.5 rounds half-away-from-zero
    }

    #[test]
    fn negative_only_column_round_trips() {
        let (k, n) = (3usize, 1usize);
        let w = vec![-0.5f32, -1.0, -0.25];
        let (q, scales) = encode_int8_per_channel(&w, k, n);
        assert_eq!(scales[0], 1.0 / 127.0);
        for i in 0..k {
            let deq = q[i] as f32 * scales[0];
            assert!((deq - w[i]).abs() <= scales[0] / 2.0 + 1e-6);
        }
    }
}
