//! Dump SmolLM2 prefill logits for the EE-3 logit-parity check (vs ONNX Runtime).
//!
//! `cargo run -p hologram-ai --example dump_logits -- <model.onnx> <out.f32>`
//!
//! Compiles the model (with-past export → empty-past recompute prefill), runs a
//! forward on a fixed token sequence, and writes the raw `logits[1,S,vocab]` as
//! little-endian f32 to `<out.f32>`. `scripts/verify_logits_vs_ort.py` runs the
//! same tokens through ONNX Runtime with a genuinely-empty past and compares —
//! the authoritative numeric witness behind CONFORMANCE EE-3.
//!
//! The token sequence is fixed and printed so the Python side uses the same ids.

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};

// Must match scripts/verify_logits_vs_ort.py.
const TOKENS: &[i64] = &[1, 450, 7483, 310, 3444, 338, 263, 8444];

fn i64_le(vals: &[i64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: dump_logits <model.onnx> <out.f32>");
    let out_path = args
        .next()
        .expect("usage: dump_logits <model.onnx> <out.f32>");
    let s = TOKENS.len();

    let archive = ModelCompiler {
        seq_len_override: Some(s as u64),
        ..Default::default()
    }
    .compile(ModelSource::OnnxPath(model.into()))?;
    let mut runner = HoloRunner::from_bytes(archive.bytes)?;

    // input_ids = TOKENS, position_ids = 0..S, everything else (empty mask /
    // empty past) zero-filled at its declared length.
    let ports = runner.input_port_info();
    let mut bufs: Vec<Vec<u8>> = ports
        .iter()
        .map(|p| vec![0u8; p.element_count * if p.dtype == 5 { 8 } else { 4 }])
        .collect();
    bufs[runner.input_index_by_name("input_ids").expect("input_ids")] = i64_le(TOKENS);
    if let Some(i) = runner.input_index_by_name("position_ids") {
        bufs[i] = i64_le(&(0..s as i64).collect::<Vec<_>>());
    }
    let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
    let out = runner.execute(&refs)?;
    let logits = &out[runner.output_index_by_name("logits").expect("logits")].bytes;

    std::fs::write(&out_path, logits)?;
    println!("tokens={TOKENS:?} seq_len={s}");
    println!(
        "wrote {} logit bytes ({} f32) to {out_path}",
        logits.len(),
        logits.len() / 4
    );

    // Dump any extra named outputs requested via HOLO_DUMP_OUTPUTS (comma-sep),
    // e.g. RoPE cos/sin exposed as graph outputs, to localize a divergence.
    if let Ok(names) = std::env::var("HOLO_DUMP_OUTPUTS") {
        for name in names.split(',').filter(|n| !n.is_empty()) {
            if let Some(i) = runner.output_index_by_name(name) {
                let safe = name.replace('/', "_");
                let p = format!("{out_path}.{safe}");
                std::fs::write(&p, &out[i].bytes)?;
                println!("wrote {name} ({} f32) to {p}", out[i].bytes.len() / 4);
            } else {
                println!("output {name:?} not found");
            }
        }
    }
    Ok(())
}
