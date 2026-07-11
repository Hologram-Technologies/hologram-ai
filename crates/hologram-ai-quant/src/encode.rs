//! f32 → per-channel symmetric int8 / int4 weight encoder (the inverse of the
//! dequant unpackers in this crate, and of the substrate's `matmul_i{8,4}_pc_omajor`
//! decode kernels). `no_std`; pure arithmetic, no IR dependency.

use alloc::vec;
use alloc::vec::Vec;

/// `no_std`-safe f32 round-half-away-from-zero via libm (`f32::round` is not in core).
#[inline(always)]
fn round_f32(x: f32) -> f32 {
    libm::roundf(x)
}

/// The symmetric int4 code magnitude ceiling. Codes live in `-7..=7`: the same
/// symmetric choice as int8's `-127..=127` (the two's-complement `-8` is left
/// unused, so `+max` and `-max` map to `±7` — no asymmetry). The substrate's i4
/// value grid `I4_VALUES = [0..7, -8..-1]` decodes any nibble, so `-7..=7` is a
/// valid subset; dequant is `code · scale`, matching the `matmul_i4_pc_omajor`
/// kernel and the `Dequantize{I4}` reference test in the substrate.
const I4_MAX: f32 = 7.0;

/// Pack a signed 4-bit code (`-8..=7`) into its two's-complement nibble
/// (`code & 0x0F`) — e.g. `-2 → 0b1110`, `1 → 0b0001`, `-1 → 0b1111`. This is the
/// substrate's archive convention (`I4_VALUES`): nibble `0..=7 → 0..=7`,
/// `8..=15 → −8..=−1`.
#[inline(always)]
fn i4_nibble(code: i8) -> u8 {
    (code as u8) & 0x0F
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

/// Per-channel symmetric **int4** encoding of an **output-major** `[n, k]`
/// weight — each output channel `j`'s k-vector contiguous at `w[j*k .. j*k+k]`.
/// The int4 twin of [`encode_int8_per_channel_omajor`]: same per-output-channel
/// symmetric codec (`scale_j = max_i |W[j,i]| / 7`, round-half-away-from-zero,
/// symmetric clamp), the codes packed **two per byte, low nibble first** within
/// each channel's contiguous k-run — the `[n, k/2]` byte layout the substrate's
/// `matmul_i4_pc_omajor` decode GEMV streams (element `l` of column `j` = nibble
/// `l` of its `k/2`-byte span).
///
/// `k` must be even (a channel is a whole number of packed bytes) — the same
/// constraint `matmul_i4_pc_omajor` asserts. Returns `(packed, scales)` with
/// `packed` of length `n·k/2` and `scales` of length `n`. A channel of all zeros
/// gets `scale = 1.0` so dequant reproduces zeros exactly.
pub fn encode_int4_per_channel_omajor(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(w.len(), n * k, "weight length must equal n*k");
    assert!(
        k.is_multiple_of(2),
        "int4 output-major requires even k (packed nibbles)"
    );
    let mut scales = vec![1.0f32; n];
    let mut packed = vec![0u8; n * k / 2];
    for j in 0..n {
        let row = &w[j * k..j * k + k];
        let mut amax = 0.0f32;
        for &v in row {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        if amax > 0.0 {
            scales[j] = amax / I4_MAX;
        }
        let s = scales[j];
        let base = j * (k / 2); // byte offset of channel j's span (k even)
        for (i, &v) in row.iter().enumerate() {
            let code = round_f32(v / s).clamp(-I4_MAX, I4_MAX) as i8;
            let nib = i4_nibble(code);
            let byte = base + (i >> 1);
            packed[byte] |= if i & 1 == 0 { nib } else { nib << 4 };
        }
    }
    (packed, scales)
}

/// Per-channel symmetric **int4** encoding of a row-major `[k, n]` weight — the
/// int4 twin of [`encode_int8_per_channel`], for a graph CONSTANT whose bytes the
/// compiler owns and repacks itself (the substrate's `QuantTier::omajor_repack`
/// consumes exactly this `[k, n]` nibble order). One scale per output channel
/// `j` (`scale_j = max_i |W[i,j]| / 7`). Codes are packed two per byte over the
/// flat `[k, n]` order, low nibble first (element `s = i·n + j` → nibble `s`).
///
/// Returns `(packed, scales)` with `packed` of length `⌈k·n / 2⌉`. A column of
/// all zeros gets `scale = 1.0`.
pub fn encode_int4_per_channel(w: &[f32], k: usize, n: usize) -> (Vec<u8>, Vec<f32>) {
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
            *scale = amax / I4_MAX;
        }
    }
    let mut packed = vec![0u8; (k * n).div_ceil(2)];
    for i in 0..k {
        for j in 0..n {
            let s = i * n + j;
            let code = round_f32(w[i * n + j] / scales[j]).clamp(-I4_MAX, I4_MAX) as i8;
            let nib = i4_nibble(code);
            packed[s >> 1] |= if s & 1 == 0 { nib } else { nib << 4 };
        }
    }
    (packed, scales)
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

    /// Decode a packed nibble the way the substrate's `I4_VALUES` grid does:
    /// nibble `0..=7 → 0..=7`, `8..=15 → −8..=−1` (two's complement), low nibble
    /// first. This mirrors `i4_at` in the substrate kernel — the external decoder
    /// our encoder must feed.
    fn i4_decode(packed: &[u8], l: usize) -> i8 {
        let byte = packed[l >> 1];
        let nib = if l & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        if nib < 8 {
            nib as i8
        } else {
            nib as i8 - 16
        }
    }

    /// The packing convention is bit-exact with the substrate. A single column
    /// whose max is 7 pins `scale = 1.0`, so the codes ARE the values and the
    /// packed bytes are directly checkable against the archive nibble order
    /// (`Dequantize{I4}` reference: element `l` = nibble `l`, low nibble first,
    /// two's complement).
    #[test]
    fn int4_packing_matches_substrate_nibble_convention() {
        // w = [-2, 1, 0, 7] over one column (k=4, n=1): scale = 7/7 = 1.
        //   el0 = -2 → 0b1110 (low  nibble byte0)
        //   el1 =  1 → 0b0001 (high nibble byte0)  → byte0 = 0x1E
        //   el2 =  0 → 0b0000 (low  nibble byte1)
        //   el3 =  7 → 0b0111 (high nibble byte1)  → byte1 = 0x70
        let w = vec![-2.0f32, 1.0, 0.0, 7.0];
        let (packed, scales) = encode_int4_per_channel_omajor(&w, 1, 4);
        assert_eq!(scales[0], 1.0);
        assert_eq!(
            packed,
            vec![0x1E, 0x70],
            "packed nibbles must match archive order"
        );
        // And they decode back through the substrate grid, exactly.
        for (l, &want) in [(-2i8), 1, 0, 7].iter().enumerate() {
            assert_eq!(i4_decode(&packed, l), want);
        }
    }

    /// int4 round-trips within half a scale — the quantization-error bound the
    /// quality gate relies on (`|deq − w| ≤ scale/2`), per output channel.
    #[test]
    fn int4_omajor_round_trips_within_half_scale() {
        let (n, k) = (3usize, 8usize);
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.21 - 3.3).collect();
        let (packed, scales) = encode_int4_per_channel_omajor(&w, n, k);
        assert_eq!(packed.len(), n * k / 2);
        assert_eq!(scales.len(), n);
        for j in 0..n {
            for i in 0..k {
                let code = i4_decode(&packed, j * k + i);
                let deq = code as f32 * scales[j];
                assert!(
                    (deq - w[j * k + i]).abs() <= scales[j] / 2.0 + 1e-6,
                    "({j},{i}): deq {deq} vs {}",
                    w[j * k + i]
                );
            }
        }
    }

    /// int4, like int8, is one codec under two storage orders: same per-channel
    /// scales, and the decoded codes are exact transposes. This licenses the
    /// output-major artifact derivation to skip its transpose for int4 too.
    #[test]
    fn int4_omajor_and_rowmajor_agree_and_codes_transpose() {
        let (k, n) = (6usize, 4usize);
        let w_kn: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.29 - 5.1).collect();
        let mut w_nk = vec![0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                w_nk[j * k + i] = w_kn[i * n + j];
            }
        }
        let (p_kn, s_kn) = encode_int4_per_channel(&w_kn, k, n);
        let (p_nk, s_nk) = encode_int4_per_channel_omajor(&w_nk, n, k);
        assert_eq!(s_kn, s_nk, "per-output-channel scales are the same codec");
        for i in 0..k {
            for j in 0..n {
                assert_eq!(
                    i4_decode(&p_kn, i * n + j),
                    i4_decode(&p_nk, j * k + i),
                    "code ({i},{j}) must decode the same under both storage orders"
                );
            }
        }
    }

    /// A zero channel is exact (scale 1.0, all codes 0).
    #[test]
    fn int4_zero_channel_is_exact() {
        let (n, k) = (2usize, 4usize);
        let w = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let (packed, scales) = encode_int4_per_channel_omajor(&w, n, k);
        assert_eq!(scales[1], 1.0);
        for i in 0..k {
            assert_eq!(i4_decode(&packed, k + i), 0);
        }
    }
}
