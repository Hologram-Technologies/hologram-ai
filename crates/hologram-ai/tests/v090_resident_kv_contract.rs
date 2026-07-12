//! Substrate contract for the v0.9.0 resident-KV path, driven through OUR
//! runner. Before the decode driver is reworked to carry the K/V by κ-label
//! (ADR-0019), this proves the plumbing our driver will stand on:
//!
//! - a `KvCacheWrite(cache, new, pos)` graph compiles and loads through
//!   `HoloRunner`;
//! - the **addressed loop** (`intern_input` → `execute_addressed` → carry the
//!   output label forward) produces the exact ring-write bytes an independent
//!   host oracle produces — dtype-agnostic byte copy, so bit-identical;
//! - the write realizes as an in-place **κ-move** under sole ownership
//!   (`last_dispatched() == 0`, the copy kernel elided) and holds pool
//!   allocation **exactly constant** across steps (the confinement law under
//!   the 32-bit ledger);
//! - a **κ-lease** flips the move to an honest copy so the pre-image survives
//!   bit-intact (the speculative-rollback primitive), and releasing the lease
//!   restores the move.
//!
//! Mirrors the substrate's own `tests/lease.rs` / `tests/kv_cache_write.rs`,
//! but every call goes through `hologram_ai::HoloRunner` — the surface the
//! decode driver uses — so a gap in our wrapper fails here, in isolation, not
//! tangled in the decode rewrite. Rung 2 of the invariance ladder is not even
//! needed: a KvCacheWrite is an exact byte move, so equality is bitwise on any
//! lane.

use hologram_ai::HoloRunner;
use hologram_compiler::{compile, BackendKind};
use hologram_graph::{
    node::Node,
    registry::{DTypeId, ShapeDescriptor},
    Graph, GraphOp, InputSource, OpKind,
};
use smallvec::SmallVec;
use uor_foundation::WittLevel;

const DTYPE_F32: u8 = 8;
const DTYPE_I32: u8 = 4;

/// b, kv-heads, bucket rows, head dim. Small so the oracle is trivial to read.
const DIMS: (usize, usize, usize, usize) = (1, 2, 6, 4);

fn f32s(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 19 + seed * 23) % 47) as f32 - 23.0) * 0.031)
        .collect()
}
fn to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Graph: `KvCacheWrite(cache, new, pos) → output`, one fixed-bucket ring write.
fn write_graph(b: usize, hkv: usize, bucket: usize, d: usize) -> Vec<u8> {
    let mut g = Graph::new();
    let cache_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank4(
        b as u64,
        hkv as u64,
        bucket as u64,
        d as u64,
    ));
    let new_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b as u64, hkv as u64, 1, d as u64));
    let pos_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(1));
    let mut input = |sh, dt: u8| {
        let n = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(dt),
            output_shape: sh,
        });
        g.add_input(n);
        InputSource::Node(n)
    };
    let cache = input(cache_sh, DTYPE_F32);
    let new = input(new_sh, DTYPE_F32);
    let pos = input(pos_sh, DTYPE_I32);
    let w = g.add_node(Node {
        op: GraphOp::Op(OpKind::KvCacheWrite),
        inputs: SmallVec::from_iter([cache, new, pos]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: cache_sh,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(w)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: cache_sh,
    });
    g.add_output(out);
    compile(g, BackendKind::Cpu, WittLevel::W32)
        .expect("a KvCacheWrite graph must compile")
        .archive
}

/// Independent host oracle: write `new` `[planes,1,d]` into `cache`
/// `[planes,bucket,d]` at ring row `pos % bucket`. No kernel — plain Rust — so
/// it is a real check on the substrate write, not a tautology.
fn ring_write_oracle(
    cache: &[f32],
    new: &[f32],
    pos: u32,
    planes: usize,
    bucket: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = cache.to_vec();
    let row = (pos as usize) % bucket;
    for p in 0..planes {
        let dst = p * bucket * d + row * d;
        let src = p * d;
        out[dst..dst + d].copy_from_slice(&new[src..src + d]);
    }
    out
}

fn le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// **The addressed decode loop, through our runner.** Carry the cache label
/// forward across steps: each write hits the ring-write oracle bit-for-bit and
/// the write realizes as a κ-move (copy kernel elided). And the confinement
/// law: the pool allocation warms up in the first couple of steps (it grows its
/// recycled free-list), then holds **exactly constant** — it does NOT track the
/// step count, so a 10-step and a 1000-step decode occupy the same pool. The
/// witness runs well past the ring-wrap (`bucket`) so a per-step or
/// per-position leak would show as a rising tail.
#[test]
fn addressed_kv_write_moves_in_place_and_confines_the_pool() {
    let (b, hkv, bucket, d) = DIMS;
    let planes = b * hkv;
    // Warm-up steps before the pool reaches best-fit steady state, then a long
    // steady tail (across two full ring wraps) that must be perfectly flat.
    const WARMUP: usize = 3;
    let steps = WARMUP + 2 * bucket + 4;
    let mut runner = HoloRunner::from_bytes(write_graph(b, hkv, bucket, d))
        .expect("resident-KV archive must load through HoloRunner");

    let mut cache = f32s(planes * bucket * d, 1);
    let mut lc = runner.intern_input(&to_le(&cache));

    let mut allocs = Vec::with_capacity(steps);
    for step in 0..steps {
        let new = f32s(planes * d, 40 + step);
        let want = ring_write_oracle(&cache, &new, step as u32, planes, bucket, d);
        let ln = runner.intern_input(&to_le(&new));
        let lp = runner.intern_input(&(step as u32).to_le_bytes());
        let out = runner
            .execute_addressed(&[lc, ln, lp])
            .expect("addressed KvCacheWrite must execute");
        assert_eq!(out.len(), 1, "one output: the updated cache");
        assert_eq!(
            le_to_f32(runner.resolve(&out[0]).expect("updated cache resolves")),
            want,
            "step {step}: addressed KvCacheWrite must equal the ring-write oracle"
        );
        assert_eq!(
            runner.last_dispatched(),
            0,
            "step {step}: sole ownership must MOVE (copy kernel elided), not copy"
        );
        allocs.push(runner.pool_allocated_bytes());
        lc = out[0];
        cache = want;
    }

    // The steady-state tail is flat: O(1) memory, independent of step count.
    let steady = allocs[WARMUP];
    for (step, &a) in allocs.iter().enumerate().skip(WARMUP) {
        assert_eq!(
            a, steady,
            "step {step}: pool allocation must stay constant in steady state \
             (confinement); saw {a} vs {steady} — a per-step KV leak"
        );
    }
}

/// **The lease is a borrow.** Leasing the cache forces the honest copy so the
/// pre-image survives bit-intact — reject re-steps from it — and releasing the
/// lease restores the in-place move. Speculative accept/reject in κ terms,
/// through our runner.
#[test]
fn a_leased_cache_copies_preserving_the_preimage_then_moves_on_release() {
    let (b, hkv, bucket, d) = DIMS;
    let planes = b * hkv;
    let mut runner = HoloRunner::from_bytes(write_graph(b, hkv, bucket, d))
        .expect("resident-KV archive must load through HoloRunner");

    let cache = f32s(planes * bucket * d, 1);
    let pre = runner.intern_input(&to_le(&cache));
    assert!(runner.retain_label(&pre), "a resident cache must lease");
    assert_eq!(runner.leased_count(), 1);

    // Draft step on the borrowed cache: honest copy, pre-image intact.
    let draft = f32s(planes * d, 7);
    let want = ring_write_oracle(&cache, &draft, 4, planes, bucket, d);
    let ln = runner.intern_input(&to_le(&draft));
    let lp = runner.intern_input(&4u32.to_le_bytes());
    let out = runner.execute_addressed(&[pre, ln, lp]).unwrap();
    assert_eq!(le_to_f32(runner.resolve(&out[0]).unwrap()), want);
    assert_eq!(
        runner.last_dispatched(),
        1,
        "a borrowed cache must take the honest copy, not the move"
    );
    assert_eq!(
        le_to_f32(runner.resolve(&pre).expect("leased pre-image survives")),
        cache,
        "the leased pre-image must survive the step bit-intact (rollback point)"
    );

    // REJECT: step again from the same intact pre-image with real data.
    let real = f32s(planes * d, 8);
    let want2 = ring_write_oracle(&cache, &real, 4, planes, bucket, d);
    let ln2 = runner.intern_input(&to_le(&real));
    let lp2 = runner.intern_input(&4u32.to_le_bytes());
    let out2 = runner.execute_addressed(&[pre, ln2, lp2]).unwrap();
    assert_eq!(
        le_to_f32(runner.resolve(&out2[0]).unwrap()),
        want2,
        "rollback step must re-run from the pre-image"
    );

    // ACCEPT: release the lease; sole ownership restores the move.
    assert!(runner.release_label(&pre));
    assert_eq!(runner.leased_count(), 0);
    let ln3 = runner.intern_input(&to_le(&f32s(planes * d, 9)));
    let lp3 = runner.intern_input(&5u32.to_le_bytes());
    let out3 = runner.execute_addressed(&[pre, ln3, lp3]).unwrap();
    assert_eq!(
        runner.last_dispatched(),
        0,
        "sole ownership must restore the in-place move"
    );
    assert!(
        runner.resolve(&pre).is_none(),
        "the moved value is consumed — no stale pre-image lingers"
    );
    assert!(runner.resolve(&out3[0]).is_some());
}

/// Leasing a label that was never made resident must report `false`, not lie —
/// the honest-refusal half of the ownership contract.
#[test]
fn leasing_a_nonresident_label_is_refused() {
    let (b, hkv, bucket, d) = DIMS;
    let mut runner = HoloRunner::from_bytes(write_graph(b, hkv, bucket, d)).unwrap();
    // A label minted by a *different* session's interner is not resident here.
    let ghost = {
        let mut other = HoloRunner::from_bytes(write_graph(b, hkv, bucket, d)).unwrap();
        other.intern_input(&[1, 2, 3, 4])
    };
    assert!(
        !runner.retain_label(&ghost),
        "a non-resident label cannot lease"
    );
    let real = runner.intern_input(&to_le(&f32s(b * hkv * bucket * d, 1)));
    assert!(runner.retain_label(&real), "a resident label leases");
}
