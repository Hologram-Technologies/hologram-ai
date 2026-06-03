//! Diagnostic harness: surface the hologram `BackendError` that
//! `InferenceSession` collapses to a detail-less `ExecError::Backend`.
//!
//! `session.execute` does `backend.dispatch(call).map_err(|_| ExecError::Backend)`,
//! discarding which kernel failed and why. This test wraps `CpuBackend` in a
//! pass-through backend that logs the offending `KernelCall` + `BackendError`
//! *before* the session erases it — so we can tell whether a forward failure
//! is an hologram bug (kernel shape mismatch / unsupported op) or an
//! hologram-ai lowering issue.
//!
//! Generalized over any ONNX model. Drive a specific architecture by setting
//! `HOLOGRAM_AI_DIAG_ONNX` to its `.onnx` path:
//!
//! ```ignore
//! HOLOGRAM_AI_LIVE=1 \
//! HOLOGRAM_AI_DIAG_ONNX=models/Qwen2-0.5B-Instruct/model.onnx \
//! cargo test --release -p hologram-ai --test diag_backend -- --nocapture --ignored
//! ```
//!
//! Skip-safe: needs `HOLOGRAM_AI_LIVE=1` + a valid `HOLOGRAM_AI_DIAG_ONNX`
//! pointing at an existing ONNX file.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use hologram_ai::{ModelCompiler, ModelSource};
use hologram_backend::{Backend, BackendError, CpuBackend, KernelCall, Workspace};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};

/// Backend that forwards to `CpuBackend` but records every failing call (not
/// just the first), so we can see whether a failure is isolated or systemic.
struct DiagBackend<W: Workspace> {
    inner: CpuBackend<W>,
    failures: Arc<Mutex<Vec<String>>>,
    n: Arc<Mutex<usize>>,
    max_failures: usize,
}

impl<W: Workspace> Clone for DiagBackend<W> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner,
            failures: self.failures.clone(),
            n: self.n.clone(),
            max_failures: self.max_failures,
        }
    }
}

impl<W: Workspace> Backend for DiagBackend<W> {
    type Bounds = <CpuBackend<W> as Backend>::Bounds;
    type WS = W;

    fn dispatch(&mut self, call: &KernelCall, ws: &mut W) -> Result<(), BackendError> {
        let n = {
            let mut g = self.n.lock().unwrap();
            *g += 1;
            *g
        };
        // Trace every dispatch under HOLOGRAM_AI_DIAG_TRACE=1 — useful
        // for seeing the call sequence leading up to a failure
        // (slot allocations, kernel order, op kinds).
        if std::env::var("HOLOGRAM_AI_DIAG_TRACE").is_ok() {
            let summary: String = format!("{call:?}").chars().take(220).collect();
            eprintln!("DIAG #{n} {summary}");
        }
        match self.inner.dispatch(call, ws) {
            Ok(()) => Ok(()),
            Err(e) => {
                let mut slot = self.failures.lock().unwrap();
                if slot.len() < self.max_failures {
                    let msg = format!("call #{n}: {call:?}\n  -> BackendError: {e}");
                    eprintln!("DIAG BACKEND FAIL {msg}");
                    slot.push(msg);
                }
                Err(e)
            }
        }
    }
}

fn diag_model_path() -> Option<PathBuf> {
    let env_path = std::env::var("HOLOGRAM_AI_DIAG_ONNX").ok()?;
    let p = PathBuf::from(env_path);
    p.exists().then_some(p)
}

#[ignore = "diagnostic harness: drive with HOLOGRAM_AI_LIVE=1 + HOLOGRAM_AI_DIAG_ONNX=<path>"]
#[test]
fn forward_surfaces_backend_error() {
    if std::env::var("HOLOGRAM_AI_LIVE").as_deref() != Ok("1") {
        eprintln!("skipping: set HOLOGRAM_AI_LIVE=1");
        return;
    }
    let Some(path) = diag_model_path() else {
        eprintln!("skipping: set HOLOGRAM_AI_DIAG_ONNX=<path/to/model.onnx>");
        return;
    };
    eprintln!("DIAG: driving forward for {}", path.display());

    // Use a short window — the diagnostic only needs enough to walk every kernel
    // class once.
    let archive = ModelCompiler {
        seq_len_override: Some(16),
        ..Default::default()
    }
    .compile(ModelSource::OnnxPath(path))
    .expect("compile");

    let failures = Arc::new(Mutex::new(Vec::new()));
    let backend = DiagBackend {
        inner: CpuBackend::<BufferArena>::new(),
        failures: failures.clone(),
        n: Arc::new(Mutex::new(0)),
        max_failures: 4,
    };
    let mut session = InferenceSession::load(&archive.bytes, backend).expect("load");

    let sizes: Vec<usize> = session
        .input_ports()
        .iter()
        .map(|p| {
            let w = match p.dtype {
                5 => 8, // i64
                _ => 4,
            };
            p.element_count as usize * w
        })
        .collect();
    let bufs_owned: Vec<Vec<u8>> = sizes.iter().map(|&s| vec![0u8; s]).collect();
    let bufs: Vec<InputBuffer> = bufs_owned
        .iter()
        .map(|b| InputBuffer { bytes: b })
        .collect();

    let result = session.execute(&bufs);
    match result {
        Ok(_) => eprintln!("DIAG: forward SUCCEEDED (no backend error)"),
        Err(e) => {
            let detail = failures.lock().unwrap().clone();
            let body = if detail.is_empty() {
                "<no kernel-level error captured — failure was outside dispatch>".into()
            } else {
                detail.join("\n---\n")
            };
            panic!("forward failed: {e:?}\nbackend failure detail:\n{body}");
        }
    }
}
