//! f32 → per-channel symmetric int8 weight encoder (the inverse of the dequant
//! unpackers in this crate). `no_std`; pure arithmetic, no IR dependency.

use alloc::vec;
use alloc::vec::Vec;

/// `no_std`-safe f32 round-half-away-from-zero via libm (`f32::round` is not in core).
#[inline(always)]
fn round_f32(x: f32) -> f32 {
    libm::roundf(x)
}

/// Per-channel symmetric int8 encoding of an **output-major** `[n, k]` weight —
/// each output channel `j`'s k-vector contiguous at `w[j*k .. j*k+k]`.
///
/// This is `hologram_types::weight_layout::OUTPUT_MAJOR`: the layout the fused
/// decode GEMV streams, and the layout a `[out, in]` checkpoint tensor is
/// *already* in. Deriving an artifact in this form therefore costs no transpose
/// — it removes one, along with the `k·n` f32 scratch buffer the transpose
/// needed.
///
/// The codec is identical to [`encode_int8_per_channel`]: same per-output-channel
/// symmetric scale `scale_j = max_i |W[i,j]| / 127`, same round-half-away-from-
/// zero, same clamp. Only the storage order of the codes differs, and by the
/// codec-invariance of the accumulation (`Σ aᵢ·d(cᵢ)` depends on the decoded
/// operands alone, never on how they were stored) the dequantized weight is
/// element-for-element the same. `scales_omajor_and_rowmajor_agree_and_the_codes_transpose`
/// witnesses exactly that.
///
/// Returns `(q, scales)` with `q` output-major `[n,k]` and `scales` of length
/// `n`. A channel of all zeros gets `scale = 1.0` so dequant reproduces zeros
/// exactly.
pub fn encode_int8_per_channel_omajor(w: &[f32], n: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(w.len(), n * k, "weight length must equal n*k");
    let mut scales = vec![1.0f32; n];
    let mut q = vec![0i8; n * k];
    for j in 0..n {
        let row = &w[j * k..j * k + k];
        // Contiguous scan — the row-major encoder walks this column with stride
        // `n`, touching a fresh cache line per element.
        let mut amax = 0.0f32;
        for &v in row {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        if amax > 0.0 {
            scales[j] = amax / 127.0;
        }
        let s = scales[j];
        for (i, &v) in row.iter().enumerate() {
            let clamped = round_f32(v / s).clamp(-127.0, 127.0);
            q[j * k + i] = clamped as i8;
        }
    }
    (q, scales)
}

/// Per-channel symmetric int8 encoding of a row-major `[k, n]` weight.
///
/// One scale per **column** (output channel `j`):
/// `scale_j = max_i |W[i,j]| / 127`. Returns `(q, scales)` where `q` is the
/// row-major `[k,n]` i8 weight and `scales` has length `n`. A column of all
/// zeros gets `scale = 1.0` so dequant reproduces zeros exactly.
///
/// This is `weight_layout::ROW_MAJOR`. A **graph constant's** bytes are always
/// this way — the compiler owns them and transposes them itself when it fuses —
/// so the compile-time quantization pass uses this encoder. A load-time-bound
/// weight, whose bytes we author, should use
/// [`encode_int8_per_channel_omajor`] instead.
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

    /// Codec-invariance, in our own encoder: the output-major and row-major
    /// encoders are the *same codec* under two storage orders. Same scales
    /// element-for-element, and the codes are exact transposes — so the weight
    /// each one decodes to is identical, and the accumulation `Σ aᵢ·d(cᵢ)`
    /// cannot tell them apart.
    ///
    /// This is what licenses `derive_quantized_artifact` to drop its transpose:
    /// the artifact changes bytes, never values.
    #[test]
    fn scales_omajor_and_rowmajor_agree_and_the_codes_transpose() {
        let (k, n) = (7usize, 5usize);
        // A `[k,n]` row-major weight, and its `[n,k]` output-major transpose.
        let w_kn: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.37 - 4.1).collect();
        let mut w_nk = vec![0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                w_nk[j * k + i] = w_kn[i * n + j];
            }
        }

        let (q_kn, s_kn) = encode_int8_per_channel(&w_kn, k, n);
        let (q_nk, s_nk) = encode_int8_per_channel_omajor(&w_nk, n, k);

        assert_eq!(
            s_kn, s_nk,
            "the per-output-channel scales are the same codec"
        );
        for i in 0..k {
            for j in 0..n {
                assert_eq!(
                    q_kn[i * n + j],
                    q_nk[j * k + i],
                    "code ({i},{j}) must be the same under both storage orders"
                );
            }
        }
    }

    /// A zero channel is exact under the output-major encoder too — the property
    /// the substrate's contract calls out ("an all-zero activation row is exact"),
    /// stated here for the weight side.
    #[test]
    fn omajor_zero_channel_gets_scale_one_and_exact_zeros() {
        let (n, k) = (2usize, 3usize);
        // Channel 1 (the second contiguous k-vector) is all zeros.
        let w = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let (q, scales) = encode_int8_per_channel_omajor(&w, n, k);
        assert_eq!(scales[1], 1.0);
        for i in 0..k {
            assert_eq!(q[k + i], 0);
        }
    }

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
