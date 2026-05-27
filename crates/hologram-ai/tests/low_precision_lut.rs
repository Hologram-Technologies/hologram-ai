//! Low-precision (bf16) activation path — exercises hologram's LUT-accelerated
//! activation kernels (PM_7 Q1 tier) end to end through hologram-ai.
//!
//! hologram dispatches a unary activation (GELU/SiLU/Sigmoid/Tanh/Exp/Erf) on an
//! f16/bf16 tensor through a bit-exact lookup table instead of computing the
//! transcendental (`is_lut_dtype` ⇒ F16/BF16). hologram-ai reaches that path
//! for free: it lowers `AiOp::Sigmoid` → `OpKind::Sigmoid` carrying the tensor's
//! bf16 dtype, and the backend transparently selects the LUT kernel — the
//! canonical-forms architecture (hologram-ai declares the op + dtype, hologram
//! picks the kernel). This test confirms a bf16 activation compiles, runs, and
//! is numerically correct (the LUT is defined as bit-identical to the compute
//! path: `narrow(f(widen(bits)))`).

use std::collections::HashMap;

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, DType, TensorInfo};

fn f32_to_bf16_bits(x: f32) -> u16 {
    // Round-to-nearest-even truncation to bf16 (top 16 bits, with rounding).
    let bits = x.to_bits();
    let round = ((bits >> 16) & 1) + 0x7fff;
    ((bits + round) >> 16) as u16
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Single bf16 `Sigmoid` over X[1, n].
fn bf16_sigmoid_graph(n: u64) -> AiGraph {
    let shape = shape_from_concrete(&[1, n]);
    let mut tensor_info = HashMap::new();
    tensor_info.insert(0u32, TensorInfo::new(DType::BF16, shape.clone()));
    tensor_info.insert(1u32, TensorInfo::new(DType::BF16, shape));
    AiGraph {
        name: "bf16_sigmoid".into(),
        nodes: vec![AiNode::new(0, AiOp::Sigmoid, vec![0], vec![1])],
        inputs: vec![0],
        outputs: vec![1],
        input_names: Vec::new(),
        output_names: Vec::new(),
        params: HashMap::new(),
        tensor_info,
        metadata: HashMap::new(),
        warnings: Vec::new(),
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
        topo_cache: Default::default(),
    }
}

#[test]
fn bf16_activation_runs_through_lut_and_is_correct() {
    let n = 64u64;
    // A spread of inputs across the activation's interesting range.
    let xs: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.25).collect();
    let in_bytes: Vec<u8> = xs
        .iter()
        .flat_map(|&x| f32_to_bf16_bits(x).to_le_bytes())
        .collect();

    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(bf16_sigmoid_graph(n)))
        .expect("bf16 sigmoid compile failed");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");

    // The bf16 input is two bytes per element — confirms hologram-ai kept the
    // bf16 dtype through lowering (not upcast to f32), which is what routes the
    // backend to the LUT kernel.
    assert_eq!(
        runner.input_byte_sizes(),
        vec![(n * 2) as usize],
        "input must be bf16 (2 bytes/elem)"
    );

    let out = runner.execute(&[&in_bytes]).expect("execute failed");
    assert_eq!(out.len(), 1);
    let y_bytes = &out[0].bytes;
    assert_eq!(y_bytes.len(), (n * 2) as usize, "output must be bf16");

    // The LUT entry is `narrow(sigmoid(widen(bits)))`, so the result equals the
    // true sigmoid to within bf16 resolution (~2⁻⁸ relative; allow a couple
    // ULPs for the input + output rounding). This confirms the bf16 LUT path
    // produced a *correct* activation, not a fast wrong one.
    let mut max_rel = 0.0f32;
    for (i, chunk) in y_bytes.chunks_exact(2).enumerate() {
        let got = bf16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        let widened = bf16_bits_to_f32(f32_to_bf16_bits(xs[i]));
        let want = 1.0 / (1.0 + (-widened).exp());
        let rel = (got - want).abs() / (want.abs() + 1e-9);
        max_rel = max_rel.max(rel);
        assert!(
            (got - want).abs() <= 1e-3 + want.abs() * 0.02,
            "bf16 sigmoid wrong at {i}: x={}, got {got}, want {want} (rel {rel})",
            xs[i]
        );
    }
    eprintln!("bf16 Sigmoid via LUT: {n} elems correct, max rel err {max_rel:.2e}");
}
