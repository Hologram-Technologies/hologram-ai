//! Substrate contract for **int4**: our per-channel-symmetric i4 artifact bytes,
//! fed through a load-time-bound `OUTPUT_MAJOR` + W8A8 `Dequantize → MatMul`,
//! reach the fused output-major i4 decode GEMV (`matmul_i4_pc_omajor`) and
//! reproduce the exact i4 integer oracle — end-to-end, against the real compiler
//! and the real backend.
//!
//! This is the load-bearing int4 witness. The unit tests prove the encoder packs
//! nibbles to the substrate's `I4_VALUES` convention and the binder declares the
//! slot INT4 with a halved range; THIS proves the loop closes — that the bytes
//! our `encode_int4_per_channel_omajor` writes are the bytes the substrate's i4
//! kernel decodes, numerically. Fails-without: a wrong nibble order, code
//! convention, or scale would land off the integer oracle.
//!
//! Mirrors `omajor_w8a8_substrate_contract.rs` (int8), differing only where the
//! tier does: `quant_dtype = I4`, the weight input is `[k, n]` packed nibbles
//! (`k·n/2` bytes, output-major `[n, k/2]`), and the oracle decodes the i4 code
//! grid. The activation path is identical (W8A8: the row is quantized to i8), so
//! `act_quant` is unchanged — only the WEIGHT width differs.

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{InferenceSession, InputBuffer};
use hologram_graph::constant::ConstantEntry;
use hologram_graph::node::{Node, QuantAttrs};
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{Graph, GraphOp, InputSource, OpKind};
use hologram_types::{act_quant, weight_layout};
use smallvec::SmallVec;
use uor_foundation::WittLevel;

const DTYPE_F32: u8 = 8;
const DTYPE_I8: u8 = 2;
const DTYPE_I4: u8 = 10;

/// A deterministic f32 weight `[out, in] = [n, k]` (output-major: each output
/// channel's k-vector contiguous), the order our artifact is authored in. No
/// `rand`: identical on every host so a failure reproduces from the test name.
fn weight_omajor_f32(n: usize, k: usize) -> Vec<f32> {
    (0..n * k)
        .map(|i| ((i as f32) * 0.35 - 6.1).sin() * 2.7)
        .collect()
}

/// Decode nibble `l` of a packed span the way the substrate's `I4_VALUES` grid
/// does — the external decoder our encoder must feed.
fn i4_code(packed: &[u8], l: usize) -> i32 {
    let byte = packed[l >> 1];
    let nib = if l & 1 == 0 { byte & 0x0F } else { byte >> 4 };
    if nib < 8 {
        i32::from(nib)
    } else {
        i32::from(nib) - 16
    }
}

/// The i4 integer oracle, independent of the kernels:
/// ```text
/// sa   = max|a| / 127
/// qa[i]= clamp(round_half_away(a[i]/sa), -127, 127)
/// ŷ[j] = (Σ_i qa[i]·code[i][j]) · (sa · sw[j])      -- Σ is EXACT in i32
/// ```
/// `code[i][j]` is the i4 code of output channel `j`, input `i`, read from the
/// output-major packed nibbles (channel `j`'s k-run is `packed[j*k/2 ..]`).
/// Returns `(ŷ, Σ qa·code)`.
fn i4_integer_oracle(
    a: &[f32],
    packed_omajor: &[u8],
    scales: &[f32],
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<i64>) {
    let amax = a.iter().fold(0f32, |acc, v| acc.max(v.abs()));
    if amax == 0.0 {
        return (vec![0.0; n], vec![0; n]);
    }
    let sa = amax / 127.0;
    let qa: Vec<i32> = a
        .iter()
        .map(|&v| (v / sa).round().clamp(-127.0, 127.0) as i32)
        .collect();
    let mut acc = vec![0i64; n];
    for j in 0..n {
        let chan = &packed_omajor[j * (k / 2)..(j + 1) * (k / 2)];
        acc[j] = (0..k)
            .map(|i| i64::from(qa[i]) * i64::from(i4_code(chan, i)))
            .sum();
    }
    let y = (0..n).map(|j| acc[j] as f32 * (sa * scales[j])).collect();
    (y, acc)
}

/// Build `out = A · dequant(Wq_i4)` whose weight is a graph INPUT declaring
/// `quant_dtype = I4`, `OUTPUT_MAJOR`, W8A8, per-channel scales on `axis = 1`.
/// (A graph input is the simplest binding to execute in-process; the weightless
/// κ-constant form our decode path uses is witnessed for int8 in the companion
/// file and shares this same declaration.)
fn omajor_i4_graph(m: usize, k: usize, n: usize, scales: &[f32]) -> Vec<u8> {
    let mut g = Graph::new();
    let a_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, k as u64));
    let w_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(k as u64, n as u64));
    let o_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, n as u64));
    let v_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(n as u64));

    let sc = g.constants_mut().insert(ConstantEntry {
        bytes: scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
        dtype: DTypeId(DTYPE_F32),
        shape: v_sh,
    });
    let zc = g.constants_mut().insert(ConstantEntry {
        bytes: vec![0u8; n * 4],
        dtype: DTypeId(DTYPE_I8),
        shape: v_sh,
    });

    let a_in = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: a_sh,
    });
    g.add_input(a_in);
    let w_in = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_I4),
        output_shape: w_sh,
    });
    g.add_input(w_in);

    let dq = g.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([
            InputSource::Node(w_in),
            InputSource::Constant(sc),
            InputSource::Constant(zc),
        ]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: w_sh,
    });
    g.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: DTYPE_I4,
            scale_bits: 0,
            zero_point: 0,
            axis: 1,
            weight_layout: weight_layout::OUTPUT_MAJOR,
            act_quant: act_quant::W8A8_TOKEN_SYM,
        },
    );
    let mm = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(a_in), InputSource::Node(dq)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: o_sh,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: o_sh,
    });
    g.add_output(out);

    compile(g, BackendKind::Cpu, WittLevel::W32)
        .expect("a load-time-bound OUTPUT_MAJOR + W8A8 i4 per-channel weight must compile")
        .archive
}

/// Run the archive with activation `a` `[m,k]` and output-major packed i4 weight.
fn run(archive: &[u8], a: &[f32], packed_omajor: &[u8], m: usize, n: usize) -> Vec<f32> {
    let mut session = InferenceSession::load(archive, CpuBackend::new())
        .expect("the fused output-major i4 archive must load");
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bufs = [
        InputBuffer { bytes: &a_bytes },
        InputBuffer {
            bytes: packed_omajor,
        },
    ];
    let outs = session
        .execute(&bufs)
        .expect("the fused output-major i4 call must execute");
    let o = &outs[0].bytes;
    assert_eq!(o.len(), m * n * 4, "output is [m,n] f32");
    o.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// **The load-bearing int4 witness.** Our `encode_int4_per_channel_omajor` bytes,
/// run through the substrate's fused output-major i4 GEMV, reproduce the exact i4
/// integer oracle. This closes the loop: encoder ↔ kernel agree numerically.
#[test]
fn i4_artifact_reproduces_the_exact_integer_oracle() {
    // k even (packed nibbles) and well under K_MAX; n small keeps the oracle cheap.
    let (m, k, n) = (1usize, 64usize, 8usize);
    // Author the weight output-major and encode it exactly as the artifact does.
    let w_nk = weight_omajor_f32(n, k);
    let (packed, scales) = hologram_ai_quant::encode_int4_per_channel_omajor(&w_nk, n, k);
    assert_eq!(packed.len(), n * k / 2, "packed nibbles = n·k/2 bytes");

    // Off-grid activation so the activation rounding is real (as in the i8 file).
    let a: Vec<f32> = (0..k)
        .map(|i| ((i as f32) * 0.618_034 - 12.7).sin() * 3.3)
        .collect();

    let got = run(&omajor_i4_graph(m, k, n, &scales), &a, &packed, m, n);
    let (oracle, acc) = i4_integer_oracle(&a, &packed, &scales, k, n);

    for j in 0..n {
        let tol = oracle[j].abs() * 1e-5 + 1e-6;
        assert!(
            (got[j] - oracle[j]).abs() <= tol,
            "col {j}: the fused i4 call must reproduce the exact integer oracle \
             Σ qa·code = {} scaled by sa·sw; got {}, oracle {}. An off-by-rounding \
             miss means the nibble order / code grid / scale disagree with the kernel.",
            acc[j],
            got[j],
            oracle[j]
        );
    }
}

/// **The browser's actual i4 binding compiles.** Our staged decode path binds
/// every projection weight as a WEIGHTLESS κ constant (`ConstantEntry{bytes:[]}`
/// plus a `holospaces.kappa_map` naming the κ whose bytes arrive at
/// materialization) declaring `INT4` / `OUTPUT_MAJOR` / W8A8. The graph-input tests prove the
/// i4 kernel DECODES; this proves the substrate accepts the weightless-constant
/// FORM the browser κ-binder actually emits for int4 (the int8 twin is
/// `a_weightless_kappa_constant_can_declare_output_major` in the companion file).
/// Fails-without: the substrate refuses a weightless i4 OUTPUT_MAJOR constant, and
/// the whole browser int4 tier would be unreachable through our representation.
#[test]
fn a_weightless_kappa_i4_constant_can_declare_output_major() {
    let (m, k, n) = (1usize, 64usize, 8usize);
    let mut g = Graph::new();
    let a_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, k as u64));
    let w_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(k as u64, n as u64));
    let o_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, n as u64));
    let v_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(n as u64));

    // The weightless i4 weight, exactly as `AiParam::external_range` lowers for
    // int4: a constant with NO bytes, declared INT4. Its packed nibbles arrive
    // later, addressed by κ.
    let wc = g.constants_mut().insert(ConstantEntry {
        bytes: Vec::new(),
        dtype: DTypeId(DTYPE_I4),
        shape: w_sh,
    });
    g.add_extension(
        "holospaces.kappa_map",
        b"ConstantId(0):kappa-under-test\n".to_vec(),
    );
    let scales: Vec<f32> = (0..n).map(|j| 0.001 + (j as f32) * 1e-6).collect();
    let sc = g.constants_mut().insert(ConstantEntry {
        bytes: scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
        dtype: DTypeId(DTYPE_F32),
        shape: v_sh,
    });
    let zc = g.constants_mut().insert(ConstantEntry {
        bytes: vec![0u8; n * 4],
        dtype: DTypeId(DTYPE_I8),
        shape: v_sh,
    });

    let a_in = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: a_sh,
    });
    g.add_input(a_in);
    let dq = g.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([
            InputSource::Constant(wc),
            InputSource::Constant(sc),
            InputSource::Constant(zc),
        ]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: w_sh,
    });
    g.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: DTYPE_I4,
            scale_bits: 0,
            zero_point: 0,
            axis: 1,
            weight_layout: weight_layout::OUTPUT_MAJOR,
            act_quant: act_quant::W8A8_TOKEN_SYM,
        },
    );
    let mm = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(a_in), InputSource::Node(dq)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: o_sh,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: o_sh,
    });
    g.add_output(out);

    compile(g, BackendKind::Cpu, WittLevel::W32).unwrap_or_else(|e| {
        panic!(
            "a weightless (zero-byte) κ constant declaring INT4 + OUTPUT_MAJOR + W8A8 must \
             compile — this is the exact binding our browser int4 tier emits; got: {e}"
        )
    });
}

/// Schedule-independence at i4: one batched call at `m = 64` (our prefill chunk)
/// is byte-identical row-for-row to `m` single-row (`m = 1`, decode) calls — the
/// property our chunked-prefill seeder and step runner rely on, now for int4.
#[test]
fn i4_prefill_batch_equals_decode_step_bit_for_bit() {
    let (m, k, n) = (64usize, 64usize, 8usize);
    let w_nk = weight_omajor_f32(n, k);
    let (packed, scales) = hologram_ai_quant::encode_int4_per_channel_omajor(&w_nk, n, k);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.031_25)
        .collect();

    let batched = run(&omajor_i4_graph(m, k, n, &scales), &a, &packed, m, n);
    let single = omajor_i4_graph(1, k, n, &scales);
    for r in 0..m {
        let row = run(&single, &a[r * k..(r + 1) * k], &packed, 1, n);
        assert_eq!(
            &batched[r * n..(r + 1) * n],
            &row[..],
            "row {r}: a batched i4 call must equal the single-row call bit for bit \
             (the prefill/decode agreement the seeder needs, at int4)"
        );
    }
}
