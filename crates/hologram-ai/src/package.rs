//! Packaging R4 inference bundles as `.holo` `InferenceModel` layers.
//!
//! All container bytes are produced and verified through the official
//! Hologram APIs (`HoloWriter` / `HoloLoader`) — this module never writes
//! the `.holo` wire format by hand.

use hologram_ai_core::{AiError, AiResult, ErrorCategory, ENGINE_UOR_R4};
use hologram_space::{address_bytes, AppManifest, Capabilities, CapabilitySet, Layer, Realization};

/// One model layer to package: a deterministic R4 inference bundle plus its
/// callable service identity.
#[derive(Debug, Clone)]
pub struct ModelLayer {
    /// Callable service name, unique within the application (e.g.
    /// `ai.default`). Non-empty.
    pub entry: String,
    /// Engine identifier (e.g. [`ENGINE_UOR_R4`]). Non-empty.
    pub engine: String,
    /// Deterministic bundle bytes (schema v1, see `hologram-ai-bundle`).
    pub bundle: Vec<u8>,
}

impl ModelLayer {
    /// A uor-r4 R4G1 model layer.
    pub fn uor_r4(entry: impl Into<String>, bundle: Vec<u8>) -> Self {
        Self {
            entry: entry.into(),
            engine: ENGINE_UOR_R4.into(),
            bundle,
        }
    }

    fn validate(&self) -> AiResult<()> {
        if self.entry.is_empty() {
            return Err(AiError::invalid_argument("model entry name is empty"));
        }
        if self.engine.is_empty() {
            return Err(AiError::invalid_argument("model engine id is empty"));
        }
        Ok(())
    }
}

/// The capability set declared by a model-only archive: none. Model
/// bundles are self-contained content; inference needs no storage, network,
/// or channel grants.
fn empty_requires() -> hologram_space::KappaLabel71 {
    CapabilitySet::new(Capabilities {
        storage_roots: Vec::new(),
        storage_quota_bytes: 0,
        network_fetch: false,
        network_announce: false,
        publish_channels: Vec::new(),
        subscribe_channels: Vec::new(),
        memory_max_bytes: 0,
        cpu_time_per_event_ms: 0,
        priority_weight: 0,
    })
    .kappa()
}

fn build_layers(models: &[ModelLayer]) -> AiResult<Vec<Layer>> {
    let mut layers = Vec::with_capacity(models.len());
    for m in models {
        m.validate()?;
        let content = address_bytes(&m.bundle);
        layers.push(Layer::inference_model(
            content,
            m.entry.clone(),
            m.engine.clone(),
        ));
    }
    Ok(layers)
}

fn assemble(manifest: AppManifest, blobs: Vec<(Vec<u8>, Vec<u8>)>) -> AiResult<Vec<u8>> {
    manifest
        .validate()
        .map_err(|e| AiError::new(ErrorCategory::ArchiveEncode, format!("manifest: {e:?}")))?;
    let mut writer = hologram_archive::HoloWriter::new();
    writer.set_app_manifest(manifest.canonicalize());
    for (kappa, content) in blobs {
        writer.add_content_blob(kappa, content);
    }
    let bytes = writer
        .finish()
        .map_err(|e| AiError::new(ErrorCategory::ArchiveEncode, format!("writer: {e}")))?;
    verify_archive(&bytes)?;
    Ok(bytes)
}

/// Verify a freshly written archive with the official reader before it is
/// returned to any caller.
fn verify_archive(bytes: &[u8]) -> AiResult<()> {
    let loader = hologram_archive::HoloLoader::from_bytes(bytes)
        .map_err(|e| AiError::new(ErrorCategory::ArchiveEncode, format!("verify: {e}")))?;
    let plan = loader
        .into_plan()
        .map_err(|e| AiError::new(ErrorCategory::ArchiveEncode, format!("verify: {e}")))?;
    let manifest_bytes = plan
        .app_manifest()
        .ok_or_else(|| AiError::new(ErrorCategory::ArchiveEncode, "verify: no app manifest"))?;
    let manifest = AppManifest::decode(manifest_bytes)
        .map_err(|e| AiError::new(ErrorCategory::ArchiveEncode, format!("verify: {e:?}")))?;
    manifest
        .validate()
        .map_err(|e| AiError::new(ErrorCategory::ArchiveEncode, format!("verify: {e:?}")))?;
    Ok(())
}

/// Build a self-contained (fat) model-only `.holo` archive: zero, one, or
/// many `InferenceModel` layers, `primary = None`.
pub fn build_model_archive(models: &[ModelLayer]) -> AiResult<Vec<u8>> {
    if models.is_empty() {
        return Err(AiError::invalid_argument(
            "a model-only archive needs at least one model layer",
        ));
    }
    let layers = build_layers(models)?;
    let manifest = AppManifest {
        primary: None,
        requires: empty_requires(),
        layers,
        children: Vec::new(),
    };
    let blobs = models
        .iter()
        .map(|m| {
            (
                address_bytes(&m.bundle).as_bytes().to_vec(),
                m.bundle.clone(),
            )
        })
        .collect();
    assemble(manifest, blobs)
}

/// Add model layers to an existing `.holo` application (which may have a
/// WASM or rootfs primary layer). The existing layers, content blobs, and
/// primary are preserved; manifest order remains initialization order.
pub fn add_model_layers(archive: &[u8], models: &[ModelLayer]) -> AiResult<Vec<u8>> {
    let loader = hologram_archive::HoloLoader::from_bytes(archive)
        .map_err(|e| AiError::new(ErrorCategory::ArchiveDecode, format!("loader: {e}")))?;
    let plan = loader
        .into_plan()
        .map_err(|e| AiError::new(ErrorCategory::ArchiveDecode, format!("loader: {e}")))?;
    let manifest_bytes = plan
        .app_manifest()
        .ok_or_else(|| AiError::new(ErrorCategory::ArchiveDecode, "no app manifest section"))?;
    let mut manifest = AppManifest::decode(manifest_bytes)
        .map_err(|e| AiError::new(ErrorCategory::ArchiveDecode, format!("manifest: {e:?}")))?;
    let mut blobs: Vec<(Vec<u8>, Vec<u8>)> = plan
        .content_blobs()
        .map_err(|e| AiError::new(ErrorCategory::ArchiveDecode, format!("blobs: {e}")))?
        .into_iter()
        .map(|(k, c)| (k.to_vec(), c.to_vec()))
        .collect();

    let new_layers = build_layers(models)?;
    manifest.layers.extend(new_layers);
    for m in models {
        blobs.push((
            address_bytes(&m.bundle).as_bytes().to_vec(),
            m.bundle.clone(),
        ));
    }
    assemble(manifest, blobs)
}
