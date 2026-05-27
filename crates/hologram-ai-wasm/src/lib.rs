//! Browser (WebAssembly) entry point for hologram-ai — ADR-0017.
//!
//! GitHub Pages is static hosting with no server, so the platform runs
//! **client-side**: this crate is a `wasm-bindgen` wrapper over the **real**
//! runtime core (`hologram-exec` + `hologram-backend`), not a reimplementation.
//! It is built single-threaded (no `parallel`/rayon, which can't spawn threads
//! on wasm32) and exposes the platform's verbs over byte buffers, mirroring the
//! `hologram-ai run` CLI surface so the browser drives the same code paths.
//!
//! v1 exposes `describe` + `run` (the arbitrary-model forward path, class NS).
//! `compile` (ONNX→`.holo`) and `generate` are wired in follow-ups as the
//! shared run/compile core is factored out of the native facade (ADR-0017 §3).

use hologram_backend::CpuBackend;
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use serde::Serialize;
use wasm_bindgen::prelude::*;

type Session = InferenceSession<CpuBackend<BufferArena>>;

/// Install a panic hook that surfaces Rust panics in the browser console.
/// Idempotent; safe to call from JS on startup.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// One graph port: backend dtype tag + logical element count. Serialized to JS.
#[derive(Serialize, serde::Deserialize)]
pub struct Port {
    pub dtype: u8,
    pub dtype_name: String,
    pub element_count: usize,
    /// Byte size honoring sub-byte packing (i4 = 2 nibbles/byte).
    pub bytes: usize,
}

/// What `describe` returns: the model's input/output port table.
#[derive(Serialize, serde::Deserialize)]
pub struct ModelInfo {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

/// One output of `run`: dtype + decoded numeric values (f32/f64/i32/i64) or, for
/// dtypes without a simple host decode, an empty `values` with raw `bytes`.
#[derive(Serialize, serde::Deserialize)]
pub struct Output {
    pub dtype: u8,
    pub dtype_name: String,
    pub element_count: usize,
    pub values: Vec<f64>,
    pub decoded: bool,
}

fn dtype_name(tag: u8) -> &'static str {
    match tag {
        0 => "bool",
        1 => "u8",
        2 => "i8",
        3 => "u64",
        4 => "i32",
        5 => "i64",
        6 => "f16",
        7 => "bf16",
        8 => "f32",
        9 => "f64",
        10 => "i4",
        _ => "?",
    }
}

fn dtype_byte_width(tag: u8) -> usize {
    match tag {
        0..=2 => 1,
        6 | 7 => 2,
        4 | 8 => 4,
        3 | 5 | 9 => 8,
        _ => 4,
    }
}

fn port_byte_size(element_count: usize, tag: u8) -> usize {
    match tag {
        10 => element_count.div_ceil(2), // i4: 2 nibbles/byte
        _ => element_count * dtype_byte_width(tag),
    }
}

fn load(holo: &[u8]) -> Result<Session, JsValue> {
    let backend = CpuBackend::<BufferArena>::new();
    InferenceSession::load(holo, backend).map_err(|e| JsValue::from_str(&format!("load .holo: {e:?}")))
}

fn port(dtype: u8, element_count: usize) -> Port {
    Port {
        dtype,
        dtype_name: dtype_name(dtype).to_string(),
        element_count,
        bytes: port_byte_size(element_count, dtype),
    }
}

/// Inspect a compiled `.holo`: its input/output ports (dtype × element_count).
/// The compiled archive carries no tensor names, so ports are positional.
#[wasm_bindgen]
pub fn describe(holo: &[u8]) -> Result<JsValue, JsValue> {
    let session = load(holo)?;
    let info = ModelInfo {
        inputs: session
            .input_ports()
            .iter()
            .map(|p| port(p.dtype, p.element_count as usize))
            .collect(),
        outputs: session
            .output_ports()
            .iter()
            .map(|p| port(p.dtype, p.element_count as usize))
            .collect(),
    };
    serde_wasm_bindgen::to_value(&info).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Synthesize an input buffer for a port from a fill value (`None` ⇒ zeros).
/// Numeric fills are encoded per the port dtype; zeros are dtype-agnostic.
fn synth(byte_size: usize, element_count: usize, dtype: u8, fill: Option<f64>) -> Result<Vec<u8>, JsValue> {
    let Some(v) = fill else {
        return Ok(vec![0u8; byte_size]);
    };
    let mut out = Vec::with_capacity(byte_size);
    for _ in 0..element_count {
        match dtype {
            1 => out.push(v as u8),
            2 => out.push(v as i8 as u8),
            3 => out.extend_from_slice(&(v as u64).to_le_bytes()),
            4 => out.extend_from_slice(&(v as i32).to_le_bytes()),
            5 => out.extend_from_slice(&(v as i64).to_le_bytes()),
            8 => out.extend_from_slice(&(v as f32).to_le_bytes()),
            9 => out.extend_from_slice(&v.to_le_bytes()),
            _ => {
                return Err(JsValue::from_str(&format!(
                    "fill {v} unsupported for dtype {}; supply this input's bytes directly",
                    dtype_name(dtype)
                )))
            }
        }
    }
    Ok(out)
}

fn decode(bytes: &[u8], dtype: u8) -> (Vec<f64>, bool) {
    let conv = |w: usize, f: &dyn Fn(&[u8]) -> f64| -> Vec<f64> {
        bytes.chunks_exact(w).map(f).collect()
    };
    match dtype {
        8 => (conv(4, &|c| f32::from_le_bytes(c.try_into().unwrap()) as f64), true),
        9 => (conv(8, &|c| f64::from_le_bytes(c.try_into().unwrap())), true),
        4 => (conv(4, &|c| i32::from_le_bytes(c.try_into().unwrap()) as f64), true),
        5 => (conv(8, &|c| i64::from_le_bytes(c.try_into().unwrap()) as f64), true),
        _ => (Vec::new(), false),
    }
}

/// Run one forward pass over an arbitrary compiled model (mirrors `run --fill`).
///
/// `inputs` are the explicit input buffers by graph-input index; any entry that
/// is empty (`len == 0`) is synthesized from `fill` (a numeric constant, or
/// `NaN`/absent ⇒ zeros). Returns each output's dtype + decoded values.
#[wasm_bindgen]
pub fn run(holo: &[u8], inputs: JsValue, fill: Option<f64>) -> Result<JsValue, JsValue> {
    // `inputs` is a JS array of byte arrays (one per graph input); an empty
    // entry means "synthesize from `fill`". An empty/undefined array fills all.
    let inputs: Vec<Vec<u8>> = if inputs.is_undefined() || inputs.is_null() {
        Vec::new()
    } else {
        serde_wasm_bindgen::from_value(inputs)
            .map_err(|e| JsValue::from_str(&format!("inputs must be an array of byte arrays: {e}")))?
    };
    let mut session = load(holo)?;
    let in_ports: Vec<(u8, usize)> = session
        .input_ports()
        .iter()
        .map(|p| (p.dtype, p.element_count as usize))
        .collect();

    if !inputs.is_empty() && inputs.len() != in_ports.len() {
        return Err(JsValue::from_str(&format!(
            "expected {} input(s), got {}",
            in_ports.len(),
            inputs.len()
        )));
    }

    // Build owned buffers: explicit bytes where provided, else synthesized.
    let mut owned: Vec<Vec<u8>> = Vec::with_capacity(in_ports.len());
    for (i, &(dtype, elems)) in in_ports.iter().enumerate() {
        let want = port_byte_size(elems, dtype);
        let provided = inputs.get(i).map(|a| a.to_vec()).filter(|b| !b.is_empty());
        match provided {
            Some(b) if b.len() == want => owned.push(b),
            Some(b) => {
                return Err(JsValue::from_str(&format!(
                    "input[{i}] is {} bytes but the model expects {want}",
                    b.len()
                )))
            }
            None => owned.push(synth(want, elems, dtype, fill)?),
        }
    }

    let bufs: Vec<InputBuffer> = owned.iter().map(|b| InputBuffer { bytes: b }).collect();
    let outputs = session
        .execute(&bufs)
        .map_err(|e| JsValue::from_str(&format!("execute: {e:?}")))?;

    let out_ports = session.output_ports();
    let results: Vec<Output> = outputs
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let dtype = out_ports.get(i).map(|p| p.dtype).unwrap_or(8);
            let (values, decoded) = decode(&o.bytes, dtype);
            Output {
                dtype,
                dtype_name: dtype_name(dtype).to_string(),
                element_count: out_ports.get(i).map(|p| p.element_count as usize).unwrap_or(0),
                values,
                decoded,
            }
        })
        .collect();

    serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    // Fixture: MatMul(x[1,4], W=identity[4,4]) → y == x. One input.
    const PROBE: &[u8] = include_bytes!("probe.holo");

    #[wasm_bindgen_test]
    fn describe_reports_ports() {
        let info: ModelInfo = serde_wasm_bindgen::from_value(describe(PROBE).unwrap()).unwrap();
        assert_eq!(info.inputs.len(), 1);
        assert_eq!(info.outputs.len(), 1);
        assert_eq!(info.inputs[0].dtype_name, "f32");
        assert_eq!(info.inputs[0].element_count, 4);
        assert_eq!(info.inputs[0].bytes, 16);
    }

    #[wasm_bindgen_test]
    fn run_fill_executes_in_wasm() {
        // fill = 1.0 → x = [1,1,1,1]; identity matmul ⇒ output [1,1,1,1].
        let outs: Vec<Output> =
            serde_wasm_bindgen::from_value(run(PROBE, JsValue::NULL, Some(1.0)).unwrap()).unwrap();
        assert_eq!(outs.len(), 1);
        assert!(outs[0].decoded);
        assert_eq!(outs[0].values, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[wasm_bindgen_test]
    fn run_explicit_input_executes_in_wasm() {
        // x = [1,2,3,4] passed explicitly; identity ⇒ output == x.
        let x: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let inputs = serde_wasm_bindgen::to_value(&vec![x]).unwrap();
        let outs: Vec<Output> =
            serde_wasm_bindgen::from_value(run(PROBE, inputs, None).unwrap()).unwrap();
        assert_eq!(outs[0].values, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
