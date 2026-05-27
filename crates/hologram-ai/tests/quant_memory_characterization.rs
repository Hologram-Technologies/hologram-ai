//! V&V memory characterization for the quantized forward (understand the limit).
//!
//! A peak-tracking global allocator measures the **peak concurrent heap bytes**
//! a quantized `Dequantize→MatMul` forward holds, as a function of layer count.
//! The per-layer slope reveals where the memory goes: ≈ the packed i8 weight
//! (`d²` bytes/layer) means the weight set is the only cost; a much larger
//! slope (≈ `4·d²`/layer) would mean a dense-f32 intermediate is being
//! materialized per layer despite the `MatMulDequant` fusion — the kind of
//! non-obvious limit the V&V framework exists to expose.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};

/// Peak-tracking allocator: `peak` is the high-water mark of live bytes.
struct PeakAlloc {
    live: AtomicUsize,
    peak: AtomicUsize,
}
unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let now = self.live.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            self.peak.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        self.live.fetch_sub(l.size(), Ordering::Relaxed);
        System.dealloc(p, l);
    }
}
#[global_allocator]
static A: PeakAlloc = PeakAlloc {
    live: AtomicUsize::new(0),
    peak: AtomicUsize::new(0),
};

fn reset_peak() {
    A.peak.store(A.live.load(Ordering::Relaxed), Ordering::Relaxed);
}
fn peak() -> usize {
    A.peak.load(Ordering::Relaxed)
}

/// `layers` chained `Dequantize(W_i8 [d,d]) → MatMul`, weights as packed i8
/// inputs, per-tensor scale=1, zp=0 (dequant = the integer value as f32).
fn quant_stack(d: u64, layers: u64) -> AiGraph {
    let mut nodes = Vec::new();
    let mut ti: HashMap<u32, TensorInfo> = HashMap::new();
    let mut params: HashMap<u32, AiParam> = HashMap::new();
    let row = shape_from_concrete(&[1, d]);
    let weight = shape_from_concrete(&[d, d]);
    ti.insert(0, TensorInfo::new(DType::F32, row.clone()));
    let zp = 1u32;
    let sc = 2u32;
    ti.insert(zp, TensorInfo::new(DType::INT8, shape_from_concrete(&[])));
    ti.insert(sc, TensorInfo::new(DType::F32, shape_from_concrete(&[])));
    params.insert(zp, AiParam::inline(vec![0u8], ti[&zp].clone()));
    params.insert(sc, AiParam::inline(1.0f32.to_le_bytes().to_vec(), ti[&sc].clone()));
    let mut inputs = vec![0u32];
    let mut next = 3u32;
    let mut prev = 0u32;
    for i in 0..layers {
        let wq = next;
        let dq = next + 1;
        let mm = next + 2;
        next += 3;
        ti.insert(wq, TensorInfo::new(DType::INT8, weight.clone()));
        ti.insert(dq, TensorInfo::new(DType::F32, weight.clone()));
        ti.insert(mm, TensorInfo::new(DType::F32, row.clone()));
        inputs.push(wq);
        nodes.push(AiNode::new(2 * i as u32, AiOp::Dequantize { axis: -1 }, vec![wq, sc, zp], vec![dq]));
        nodes.push(AiNode::new(2 * i as u32 + 1, AiOp::MatMul, vec![prev, dq], vec![mm]));
        prev = mm;
    }
    AiGraph {
        name: format!("q_d{d}_l{layers}"),
        nodes,
        inputs,
        outputs: vec![prev],
        input_names: Vec::new(),
        output_names: Vec::new(),
        params,
        tensor_info: ti,
        metadata: HashMap::new(),
        warnings: Vec::new(),
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
        topo_cache: Default::default(),
    }
}

/// Peak heap bytes held *during a forward* with weights already resident
/// (pre-interned), so the measurement isolates the per-forward workspace, not
/// the one-time weight interning.
fn forward_peak(d: u64, layers: u64) -> (usize, usize) {
    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(quant_stack(d, layers)))
        .expect("compile");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load");
    // Every Dequantize→MatMul must fuse, or the dense f32 weight is materialized.
    assert_eq!(
        runner.dequant_matmul_fused_count(),
        layers as usize,
        "all layers must fuse to MatMulDequant"
    );
    let sizes = runner.input_byte_sizes();
    let buffers: Vec<Vec<u8>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| if i == 0 { vec![0u8; n] } else { vec![(i % 7 + 1) as u8; n] })
        .collect();
    let refs: Vec<&[u8]> = buffers.iter().map(|v| v.as_slice()).collect();
    runner.execute(&refs).expect("warm forward"); // resident + steady state

    reset_peak();
    runner.execute(&refs).expect("measured forward");
    let fwd_peak = peak();
    (fwd_peak, runner.resident_bytes())
}

/// Same shape but plain f32 weights (no Dequantize) — the baseline to separate
/// the pool's generational retention from any quant-specific cost.
fn f32_stack(d: u64, layers: u64) -> AiGraph {
    let mut nodes = Vec::new();
    let mut ti: HashMap<u32, TensorInfo> = HashMap::new();
    let row = shape_from_concrete(&[1, d]);
    let weight = shape_from_concrete(&[d, d]);
    ti.insert(0, TensorInfo::new(DType::F32, row.clone()));
    let mut inputs = vec![0u32];
    let mut prev = 0u32;
    for i in 0..layers {
        let w = 1 + i as u32;
        let out = 1 + layers as u32 + i as u32;
        ti.insert(w, TensorInfo::new(DType::F32, weight.clone()));
        ti.insert(out, TensorInfo::new(DType::F32, row.clone()));
        inputs.push(w);
        nodes.push(AiNode::new(i as u32, AiOp::MatMul, vec![prev, w], vec![out]));
        prev = out;
    }
    AiGraph {
        name: format!("f_d{d}_l{layers}"), nodes, inputs, outputs: vec![prev],
        input_names: Vec::new(), output_names: Vec::new(), params: HashMap::new(),
        tensor_info: ti, metadata: HashMap::new(), warnings: Vec::new(),
        dim_vars: Default::default(), shape_constraints: Default::default(),
        subgraphs: HashMap::new(), tensor_names: HashMap::new(), topo_cache: Default::default(),
    }
}

fn f32_forward_peak(d: u64, layers: u64) -> usize {
    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(f32_stack(d, layers)))
        .expect("compile");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load");
    let sizes = runner.input_byte_sizes();
    let buffers: Vec<Vec<u8>> = sizes.iter().map(|&n| vec![1u8; n]).collect();
    let refs: Vec<&[u8]> = buffers.iter().map(|v| v.as_slice()).collect();
    runner.execute(&refs).expect("warm");
    reset_peak();
    runner.execute(&refs).expect("measured");
    peak()
}

#[test]
fn quantized_forward_workspace_is_characterized() {
    let d = 2048u64; // i8 weight = 4 MiB/layer; a dense-f32 [d,d] would be 16 MiB
    let mut prev: Option<(u64, usize)> = None;
    println!("d={d}: i8 weight = {} MiB/layer, dense-f32 [d,d] = {} MiB/layer",
        d * d / (1 << 20), d * d * 4 / (1 << 20));
    for layers in [2u64, 4, 8, 16] {
        let (fwd, resident) = forward_peak(d, layers);
        let slope = prev.map(|(pl, pp)| (fwd as i64 - pp as i64) / (layers as i64 - pl as i64));
        println!(
            "layers={layers:2}: forward peak {:6.1} MiB, resident {:6.1} MiB{}",
            fwd as f64 / (1 << 20) as f64,
            resident as f64 / (1 << 20) as f64,
            slope.map(|s| format!(", +{:.1} MiB/layer", s as f64 / (1 << 20) as f64)).unwrap_or_default(),
        );
        prev = Some((layers, fwd));
    }

    // Quant vs plain-f32 peak at the same shape: isolates quant-specific cost
    // (if quant ≈ f32, the per-layer growth is the pool's generational
    // retention of the weight set, not a dense-f32 dequant intermediate).
    let (q16, _) = forward_peak(d, 16);
    let f16 = f32_forward_peak(d, 16);
    println!(
        "layers=16 peak: quant(i8) {:.1} MiB vs f32 {:.1} MiB (f32 weights are 4× i8)",
        q16 as f64 / (1 << 20) as f64,
        f16 as f64 / (1 << 20) as f64
    );
}
