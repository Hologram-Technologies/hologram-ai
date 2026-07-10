//! Substrate contract: a **load-time-bound** `OUTPUT_MAJOR` weight reaches the
//! fused output-major W8A8 decode GEMV, at every `m`.
//!
//! Our decode path compiles WEIGHTLESS — weights are `AiParam::External{kappa}`
//! and the bytes arrive at materialization — because that is what makes archives
//! dedupable across models and pageable under the wasm32 4 GiB ceiling. Through
//! substrate v0.7.2 that structurally excluded us from the fast kernel: the
//! compile-time fusion needs constant bytes, and the load-time fusion hardcoded
//! `bq_omajor: false`.
//!
//! v0.8.0's `QuantAttrs::weight_layout` is the way out. `docs/numerics/w8a8.md`
//! states that a load-time-bound `OUTPUT_MAJOR` weight takes the output-major
//! kernel *"at every `m`, decode and prefill alike"*. The substrate ships no
//! positive end-to-end witness for that sentence — every `OUTPUT_MAJOR`
//! declaration in its own test suite is a fail-loud rejection. Our decode path's
//! correctness would rest on an unwitnessed promise, so we witness it here,
//! against the real compiler and the real backend.
//!
//! Three properties, in the order they matter to us:
//!
//! 1. **Schedule-independence** (the proof's CL-MM02; substrate's
//!    `batched_integer_gemv_equals_row_by_row_bit_for_bit`): one batched call at
//!    `m = 64` is byte-identical, row for row, to `m` single-row calls. This is
//!    what our chunked-prefill seeder (`chunk = 64`) and step runner
//!    (`chunk = 1`) rely on to agree. Without it, adopting W8A8 would silently
//!    split prefill from decode — exactly what `OMAJOR_W8A8_MAX_M = 4` forces on
//!    *constant* weights, and what would break every equivalence in
//!    `features/suites/s3_execution/decode_plan.feature`.
//! 2. **Exactness** (CL-MM03 / CL-MM04): the accumulation is an exact i32 sum,
//!    so the kernel must reproduce the integer oracle, not merely come close.
//! 3. **No silent-wrong**: a declaration no kernel can serve is refused, never
//!    quietly served as `[k,n]`.
//!
//! On what actually witnesses what — worth stating, because these tests do not
//! all witness what their names suggest. I ran the null hypothesis: strip the
//! declaration to `ROW_MAJOR`/`W8A32`, feed `[k,n]` bytes, re-run. Property (1)
//! still passes. So did an earlier test asserting "the result differs from an
//! f32 reference" — a naive reference loop already differs from
//! `matmul_f32_blocked` by f32 reassociation noise. So did a third asserting
//! exactness on grid-aligned activations. Both were deleted rather than kept as
//! false assurance.
//!
//! Only (2) — agreement with the **exact integer** oracle — separates W8A8 from
//! W8A32, and under the control it fails as it must (`Σq·w = -19556`: W8A32
//! lands `-0.520301` where the oracle demands `-0.5081427`). It is therefore the
//! sole reason this file may claim the fused path is live; (1) is a property
//! test that rides on (2) having established it.
//!
//! Read `dot` in the companion proof as `Σ aᵢ · d(cᵢ)`: the codec `d` here is the
//! identity (i8), and `weight_layout` names the code stream's storage order,
//! which the proof's codec-invariance says cannot move the accumulation.

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

/// Deterministic pseudo-random i8 weight in `{-127..=127}`, and its per-output
/// scale. No `rand` dependency: the sequence must be identical on every host so
/// a failure is reproducible from the test name alone.
fn weight_i8(k: usize, n: usize) -> Vec<i8> {
    (0..k * n)
        .map(|i| {
            let v = ((i as u64).wrapping_mul(2_654_435_761) >> 7) % 255;
            (v as i64 - 127) as i8
        })
        .collect()
}

/// Row-major `[k,n]` → output-major `[n,k]`. This is the transpose the binder
/// owes the substrate when it declares `OUTPUT_MAJOR`. For us it is free: we
/// author these bytes at κ-materialization from a `[out,in]` source that is
/// *already* in this order.
fn to_output_major(w_kn: &[i8], k: usize, n: usize) -> Vec<i8> {
    let mut w_nk = vec![0i8; k * n];
    for i in 0..k {
        for j in 0..n {
            w_nk[j * k + i] = w_kn[i * n + j];
        }
    }
    w_nk
}

/// The W8A32 reference the contract is stated against: `y[j] = Σ_i a[i]·(w[i][j]·sw[j])`,
/// in f32, with the weight read `[k,n]`. No activation rounding.
fn reference_w8a32(
    a: &[f32],
    w_kn: &[i8],
    scales: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += a[r * k + i] * (f32::from(w_kn[i * n + j]) * scales[j]);
            }
            out[r * n + j] = acc;
        }
    }
    out
}

/// Build a graph `out = A · dequant(Wq)` whose weight is a graph **input**
/// declaring `OUTPUT_MAJOR` + W8A8.
///
/// **This is not the binding our decode path uses.** Our κ weights lower to a
/// zero-byte `ConstantEntry` plus a `holospaces.kappa_map` extension, and the
/// substrate's compiler refuses `OUTPUT_MAJOR` on *any* `InputSource::Constant`
/// — see `a_weightless_kappa_constant_cannot_yet_declare_output_major` below.
/// A graph input is the only binding that reaches the fused call today, so it is
/// what these kernel-behaviour witnesses must use. Read them as: *the kernel does
/// what the contract says, once you can reach it.*
///
/// The `Dequantize` node keeps its logical `[k,n]` shape — `weight_layout` is a
/// statement about the *bytes*, not the type. Scales and zero-points are
/// constants: they are not the weight, and the fused kernel needs per-channel
/// symmetric (`zp = 0`) scales along `axis = 1`.
fn omajor_graph(m: usize, k: usize, n: usize, scales: &[f32]) -> Vec<u8> {
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
    // The weight as a graph INPUT. The compiler has no bytes to transpose; the
    // declaration is what licenses the fused omajor call. NOT our binding — see
    // the doc comment above and the tripwire at the bottom of this file.
    let w_in = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_I8),
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
            quant_dtype: DTYPE_I8,
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
        .expect("a load-time-bound OUTPUT_MAJOR + W8A8 per-channel weight must compile")
        .archive
}

/// Run the archive with activation `a` `[m,k]` and output-major weight bytes.
fn run(archive: &[u8], a: &[f32], w_nk: &[i8], m: usize, n: usize) -> Vec<f32> {
    let mut session = InferenceSession::load(archive, CpuBackend::new())
        .expect("the fused output-major W8A8 archive must load");
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w_bytes: Vec<u8> = w_nk.iter().map(|v| *v as u8).collect();
    let bufs = [
        InputBuffer { bytes: &a_bytes },
        InputBuffer { bytes: &w_bytes },
    ];
    let outs = session
        .execute(&bufs)
        .expect("the fused output-major W8A8 call must execute");
    let o = &outs[0].bytes;
    assert_eq!(o.len(), m * n * 4, "output is [m,n] f32");
    o.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// **The load-bearing witness.** A weightless `OUTPUT_MAJOR` + W8A8 weight
/// compiles, runs, and its batched (`m = 64`, our prefill chunk) result is
/// byte-identical row-for-row to `m` single-row (`m = 1`, our decode step)
/// calls.
///
/// This is the property that lets our chunked-prefill seeder and our step runner
/// share one declaration without splitting numerics — the failure mode that
/// `OMAJOR_W8A8_MAX_M = 4` forces on *constant* weights, and that would break
/// every equivalence in `features/suites/s3_execution/decode_plan.feature`.
#[test]
fn prefill_batch_equals_decode_step_bit_for_bit() {
    // k = 64 is a whole number of E8 groups and far under `K_MAX = 133_144`;
    // n = 8 keeps the reference loop cheap. m = 64 is our real seeder chunk.
    let (m, k, n) = (64usize, 64usize, 8usize);
    let scales: Vec<f32> = (0..n).map(|j| 0.001 + (j as f32) * 1e-6).collect();
    let w_kn = weight_i8(k, n);
    let w_nk = to_output_major(&w_kn, k, n);

    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.031_25)
        .collect();

    let batched = run(&omajor_graph(m, k, n, &scales), &a, &w_nk, m, n);

    let single = omajor_graph(1, k, n, &scales);
    for r in 0..m {
        let row = run(&single, &a[r * k..(r + 1) * k], &w_nk, 1, n);
        assert_eq!(
            &batched[r * n..(r + 1) * n],
            &row[..],
            "row {r}: a batched W8A8 call must equal the single-row call bit for bit \
             (schedule-independence — the prefill/decode agreement our seeder needs)"
        );
    }
}

/// The **integer oracle** of `docs/numerics/w8a8.md` — "the oracle is the integer
/// restatement", implemented here independently of the kernels:
///
/// ```text
/// sa   = max|a| / 127
/// q[i] = clamp(round_half_away(a[i]/sa), -127, 127)
/// ŷ[j] = (Σ_i q[i]·w[i][j]) · (sa · sw[j])          -- Σ is EXACT in i32
/// ```
///
/// Returns `(ŷ, Σ q·w)` so a caller can assert the accumulation is the integer
/// the proof says it is (`dot`, CL-MM03), not merely close to it.
fn integer_oracle(
    a: &[f32],
    w_kn: &[i8],
    scales: &[f32],
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<i64>) {
    let amax = a.iter().fold(0f32, |acc, v| acc.max(v.abs()));
    if amax == 0.0 {
        return (vec![0.0; n], vec![0; n]); // the contract: an all-zero row is exact
    }
    let sa = amax / 127.0;
    let q: Vec<i32> = a
        .iter()
        .map(|&v| (v / sa).round().clamp(-127.0, 127.0) as i32)
        .collect();
    let mut acc = vec![0i64; n];
    for j in 0..n {
        acc[j] = (0..k)
            .map(|i| i64::from(q[i]) * i64::from(w_kn[i * n + j]))
            .sum();
    }
    let y = (0..n).map(|j| acc[j] as f32 * (sa * scales[j])).collect();
    (y, acc)
}

/// **The discriminator.** `prefill_batch_equals_decode_step_bit_for_bit` and
/// `grid_aligned_…` both still pass when the declaration is stripped to
/// ROW_MAJOR/W8A32 — I checked, by running that control. They witness nothing
/// about W8A8 on their own. Neither does "the result differs from an f32
/// reference": a naive reference loop already differs from `matmul_f32_blocked`
/// by f32 reassociation noise, so that test passes under W8A32 too.
///
/// The property that *cannot* hold under W8A32 is exactness. W8A8 accumulates
/// `Σ q·w` in i32 with no rounding (the proof's CL-MM03/CL-MM04: every partial
/// sum is bounded by `k·B² ≤ K_MAX·B² ≤ i32::MAX`, so every wrap is the
/// identity). So the kernel's output, divided by `sa·sw[j]`, must land on the
/// *integer* the oracle computes — while W8A32, which never rounds the
/// activation, cannot.
///
/// The test asserts three things, and the third is what gives it teeth:
///   1. the kernel agrees with the integer oracle to f32 writeback precision;
///   2. the deviation from the W8A32 reference obeys the stated bound
///      `|ŷ[j] − y[j]| ≤ (amax/254)·sw[j]·Σ_i|w[i][j]|`;
///   3. the oracle and the W8A32 reference are **separated** on this input by far
///      more than the tolerance in (1) — so (1) is a real constraint, and a
///      substrate that quietly ran W8A32 would fail it.
#[test]
fn w8a8_reproduces_the_exact_integer_oracle_which_w8a32_cannot() {
    let (m, k, n) = (1usize, 64usize, 8usize);
    let scales: Vec<f32> = (0..n).map(|j| 0.001 + (j as f32) * 1e-6).collect();
    let w_kn = weight_i8(k, n);
    let w_nk = to_output_major(&w_kn, k, n);

    // Deliberately off-grid: no `sa` represents these exactly, so the activation
    // rounding is real and the two hypotheses separate.
    let a: Vec<f32> = (0..k)
        .map(|i| ((i as f32) * 0.618_034 - 12.7).sin() * 3.3)
        .collect();

    let got = run(&omajor_graph(m, k, n, &scales), &a, &w_nk, m, n);
    let (oracle, acc) = integer_oracle(&a, &w_kn, &scales, k, n);
    let w8a32 = reference_w8a32(&a, &w_kn, &scales, m, k, n);

    let amax = a.iter().fold(0f32, |acc, v| acc.max(v.abs()));
    for j in 0..n {
        // (3) the discriminator has power: the hypotheses are far apart here.
        let tol = oracle[j].abs() * 1e-5 + 1e-7;
        let separation = (oracle[j] - w8a32[j]).abs();
        assert!(
            separation > tol * 50.0,
            "col {j}: the integer oracle and the W8A32 reference are not separated \
             on this input (sep {separation}, tol {tol}) — the test cannot tell them \
             apart and witnesses nothing. Choose a more adversarial activation."
        );

        // (1) the kernel IS the integer oracle. Fails under W8A32.
        assert!(
            (got[j] - oracle[j]).abs() <= tol,
            "col {j}: the fused output-major call must reproduce the exact integer \
             oracle Σq·w = {} scaled by sa·sw; got {}, oracle {}. If this is off by \
             roughly the activation-rounding error, the declaration was ignored and \
             W8A32 ran.",
            acc[j],
            got[j],
            oracle[j]
        );

        // (2) and the deviation from W8A32 obeys the substrate's stated bound.
        let abs_w: f32 = (0..k).map(|i| f32::from(w_kn[i * n + j]).abs()).sum();
        let bound = (amax / 254.0) * scales[j] * abs_w;
        let dev = (got[j] - w8a32[j]).abs();
        assert!(
            dev <= bound * 1.000_01 + 1e-6,
            "col {j}: deviation {dev} exceeds the substrate's stated W8A8 bound {bound}"
        );
    }
}

/// The declaration is honoured, not silently downgraded: an `OUTPUT_MAJOR`
/// weight the kernel cannot serve is refused, never read as `[k,n]`.
///
/// Here the scales are per-tensor (`axis = -1`), which the integer GEMV cannot
/// decode. A fallback would reinterpret `[n,k]` bytes as `[k,n]` and return a
/// plausible wrong answer. We depend on this refusal: it is what makes an
/// incorrect declaration in our binder a build failure rather than a subtly
/// wrong model.
#[test]
fn an_unservable_output_major_declaration_fails_loud_and_never_falls_back() {
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
        output_dtype: DTypeId(DTYPE_I8),
        output_shape: w_sh,
    });
    g.add_input(w_in);
    let dq = g.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([InputSource::Node(w_in)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: w_sh,
    });
    g.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: DTYPE_I8,
            scale_bits: 0.05f32.to_bits(),
            zero_point: 0,
            axis: -1, // per-tensor — the omajor GEMV needs per-channel
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

    let err = compile(g, BackendKind::Cpu, WittLevel::W32)
        .err()
        .expect("an unservable OUTPUT_MAJOR declaration must not compile");
    let msg = format!("{err}");
    assert!(
        msg.contains("OUTPUT_MAJOR"),
        "the refusal must name the declaration it could not honour, got: {msg}"
    );
}

/// **The blocker, pinned.** Our decode path binds every projection weight as a
/// *weightless constant*: `ConstantEntry { bytes: vec![] }` in the graph, plus a
/// `holospaces.kappa_map` extension naming the κ whose bytes arrive at
/// materialization. That is exactly the case `QuantAttrs::weight_layout`'s
/// docstring says it exists to serve — "a weightless compile … has no constant
/// bytes for the compiler to transpose".
///
/// But `validate_weight_layout_declarations` rejects on
/// `matches!(node.inputs.first(), Some(InputSource::Constant(_)))`, without ever
/// asking whether the constant has bytes. So the documented use-case is
/// unreachable through the representation the doc describes, and every model we
/// ship stays on W8A32.
///
/// This test compiles that graph and pins the refusal. It is a **tripwire**: when
/// upstream narrows the check to constants that actually carry bytes — the same
/// question `fuse_const_i8_decode` already asks one screen away, `Some(e) if
/// e.bytes.len() == want_len` — this test fails, and
/// `SUBSTRATE_ACCEPTS_OUTPUT_MAJOR_ON_WEIGHTLESS_CONSTANTS` flips to `true`,
/// turning the whole path on. Nothing here is asserted from prose.
#[test]
fn a_weightless_kappa_constant_cannot_yet_declare_output_major() {
    // Read through `black_box` so this is a runtime check rather than a `const`
    // assertion clippy folds away: if the flag flips, this must FAIL here, not be
    // optimized into nothing.
    let flag = std::hint::black_box(
        hologram_ai_common::lower::SUBSTRATE_ACCEPTS_OUTPUT_MAJOR_ON_WEIGHTLESS_CONSTANTS,
    );
    assert!(
        !flag,
        "the flag claims the substrate accepts OUTPUT_MAJOR on a weightless constant — \
         then this test must be inverted, not skipped"
    );

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

    // The weightless weight, exactly as `AiParam::External` lowers: a constant
    // with NO bytes. Its content arrives later, addressed by κ.
    let wc = g.constants_mut().insert(ConstantEntry {
        bytes: Vec::new(),
        dtype: DTypeId(DTYPE_I8),
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
            quant_dtype: DTYPE_I8,
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

    let err = compile(g, BackendKind::Cpu, WittLevel::W32).err().expect(
        "TRIPWIRE: a weightless (zero-byte) κ constant declaring OUTPUT_MAJOR now COMPILES. \
             Upstream has lifted the restriction. Flip \
             SUBSTRATE_ACCEPTS_OUTPUT_MAJOR_ON_WEIGHTLESS_CONSTANTS to true, invert this test, \
             and re-baseline the transcript oracles — W8A8 re-keys κ.",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("graph constant"),
        "the refusal must be the constant-binding one (not k, tier, axis, or act_quant), \
         got: {msg}"
    );
}
