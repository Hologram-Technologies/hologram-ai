//! A selected model service.

use hologram_ai_bundle::Bundle;
use hologram_ai_core::{AiResult, InferenceModelManifest};

use crate::session::Session;

/// A model service selected from an [`crate::Application`]. Holds the
/// verified bundle bytes; the engine is initialized per [`Session`].
#[derive(Debug)]
pub struct Model {
    entry: String,
    engine: String,
    bundle: Vec<u8>,
    manifest: InferenceModelManifest,
}

impl Model {
    pub(crate) fn from_bundle(entry: &str, engine: &str, bundle: Vec<u8>) -> AiResult<Self> {
        let parsed = Bundle::parse(&bundle)?;
        Ok(Self {
            entry: entry.into(),
            engine: engine.into(),
            manifest: parsed.manifest().clone(),
            bundle,
        })
    }

    /// The callable service name.
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// The engine identifier.
    pub fn engine(&self) -> &str {
        &self.engine
    }

    /// The canonical model manifest.
    pub fn manifest(&self) -> &InferenceModelManifest {
        &self.manifest
    }

    /// BLAKE3 digest of the bundle — the model's content identity.
    pub fn bundle_digest(&self) -> [u8; 32] {
        blake3::hash(&self.bundle).into()
    }

    /// Create a session: verifies all bundle component digests, initializes
    /// the engine, and allocates all fixed-capacity runtime state. After
    /// this call the steady-state inference path is allocation-free.
    pub fn session(&self) -> AiResult<Session> {
        Session::new(self)
    }

    /// The deterministic bundle bytes backing this model (read-only).
    pub fn bundle_bytes(&self) -> &[u8] {
        &self.bundle
    }
}
