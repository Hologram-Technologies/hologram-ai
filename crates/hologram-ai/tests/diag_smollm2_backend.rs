//! Diagnostic: surface the hologram `BackendError` that `InferenceSession`
//! collapses to a detail-less `ExecError::Backend`.
//!
//! `session.execute` does `backend.dispatch(call).map_err(|_| ExecError::Backend)`,
//! discarding which kernel failed and why. This test wraps `CpuBackend` in a
//! pass-through backend that logs the offending `KernelCall` + `BackendError`
//! *before* the session erases it — so we can tell whether the SmolLM2 forward
//! failure is an hologram bug (e.g. a kernel shape mismatch / unsupported op) or
//! an hologram-ai lowering issue.
//!
//! Skip-safe: needs `HOLOGRAM_AI_LIVE=1` + the model (HOLOGRAM_AI_SMOLLM2_ONNX
//! or <workspace>/models/smollm2-135m/model.onnx).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use hologram_ai::{ModelCompiler, ModelSource};
use hologram_backend::{Backend, BackendError, CpuBackend, KernelCall, Workspace};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};

/// Backend that forwards to `CpuBackend` but records the first failing call.
struct DiagBackend<W: Workspace> {
    inner: CpuBackend<W>,
    fail: Arc<Mutex<Option<String>>>,
    n: Arc<Mutex<usize>>,
}

// Manual Clone: `CpuBackend<W>` is Clone for any `W` (it's a ZST), so we must
// not pick up the `W: Clone` bound a derive would add (BufferArena isn't Clone).
impl<W: Workspace> Clone for DiagBackend<W> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner,
            fail: self.fail.clone(),
            n: self.n.clone(),
        }
    }
}

impl<W: Workspace> Backend for DiagBackend<W> {
    type Bounds = <CpuBackend<W> as Backend>::Bounds;
    type WS = W;

    fn dispatch(&mut self, call: &KernelCall, ws: &mut W) -> Result<(), BackendError> {
        *self.n.lock().unwrap() += 1;
        match self.inner.dispatch(call, ws) {
            Ok(()) => Ok(()),
            Err(e) => {
                let mut slot = self.fail.lock().unwrap();
                if slot.is_none() {
                    let msg = format!("call #{}: {call:?}\n  -> BackendError: {e}", *self.n.lock().unwrap());
                    eprintln!("DIAG BACKEND FAIL {msg}");
                    *slot = Some(msg);
                }
                Err(e)
            }
        }
    }
}

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HOLOGRAM_AI_SMOLLM2_ONNX") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m/model.onnx");
    p.exists().then_some(p)
}

// Diagnostic harness, not a pass/fail contract: run on demand to surface the
// current forward-execution frontier. (As of now it stops at GQA Attention —
// hologram's MHA-only Attention kernel has no kv_heads field, so K/V must be
// expanded to full heads before the op.) Un-ignore once the forward runs.
#[ignore = "diagnostic harness: surfaces the live backend-execution frontier (currently GQA attention)"]
#[test]
fn smollm2_forward_surfaces_backend_error() {
    if std::env::var("HOLOGRAM_AI_LIVE").as_deref() != Ok("1") {
        eprintln!("skipping: set HOLOGRAM_AI_LIVE=1");
        return;
    }
    let Some(path) = model_path() else {
        eprintln!("skipping: SmolLM2 ONNX not found");
        return;
    };

    let archive = ModelCompiler {
        seq_len_override: Some(16),
        ..Default::default()
    }
    .compile(ModelSource::OnnxPath(path))
    .expect("compile");

    let fail = Arc::new(Mutex::new(None));
    let backend = DiagBackend {
        inner: CpuBackend::<BufferArena>::new(),
        fail: fail.clone(),
        n: Arc::new(Mutex::new(0)),
    };
    let mut session = InferenceSession::load(&archive.bytes, backend).expect("load");

    // Zero-fill every input port at its declared byte size.
    let sizes: Vec<usize> = session
        .input_ports()
        .iter()
        .map(|p| {
            let w = match p.dtype {
                5 => 8, // i64
                _ => 4, // f32 and friends
            };
            p.element_count as usize * w
        })
        .collect();
    let bufs_owned: Vec<Vec<u8>> = sizes.iter().map(|&s| vec![0u8; s]).collect();
    let bufs: Vec<InputBuffer> = bufs_owned.iter().map(|b| InputBuffer { bytes: b }).collect();

    let result = session.execute(&bufs);
    match result {
        Ok(_) => eprintln!("DIAG: forward SUCCEEDED (no backend error)"),
        Err(e) => {
            let detail = fail.lock().unwrap().clone();
            panic!(
                "forward failed: {e:?}\nfirst backend failure detail:\n{}",
                detail.unwrap_or_else(|| "<no kernel-level error captured — failure was outside dispatch>".into())
            );
        }
    }
}
