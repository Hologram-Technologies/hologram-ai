//! Loading `.holo` applications and discovering model services.

use std::path::Path;

use hologram_ai_bundle::Bundle;
use hologram_ai_core::{
    AiError, AiResult, ErrorCategory, InferenceModelManifest, OperationDescriptor,
};
use hologram_space::{AppManifest, LayerKind};

use crate::model::Model;

/// A discovered model service. Metadata only — inspecting a descriptor
/// never initializes an inference engine.
#[derive(Debug, Clone)]
pub struct ModelDescriptor {
    entry: String,
    engine: String,
    content_kappa: String,
    manifest: InferenceModelManifest,
}

impl ModelDescriptor {
    /// The callable service name (e.g. `ai.default`).
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// The engine identifier (e.g. `uor-r4`).
    pub fn engine(&self) -> &str {
        &self.engine
    }

    /// The κ content label of the model bundle.
    pub fn content_kappa(&self) -> &str {
        &self.content_kappa
    }

    /// The canonical model manifest (operations, status policy, ABI
    /// versions, provenance).
    pub fn manifest(&self) -> &InferenceModelManifest {
        &self.manifest
    }

    /// The advertised operations (authoritative for invocation).
    pub fn operations(&self) -> &[OperationDescriptor] {
        &self.manifest.operations
    }
}

/// An opened `.holo` application.
pub struct Application {
    bytes: Vec<u8>,
    fingerprint: [u8; 32],
    manifest: AppManifest,
    descriptors: Vec<ModelDescriptor>,
}

impl Application {
    /// Open and verify an application from archive bytes.
    pub fn open_bytes(bytes: &[u8]) -> AiResult<Self> {
        let loader = hologram_archive::HoloLoader::from_bytes(bytes)
            .map_err(|e| AiError::new(ErrorCategory::ArchiveDecode, format!("loader: {e}")))?;
        let fingerprint = loader.fingerprint();
        let plan = loader
            .into_plan()
            .map_err(|e| AiError::new(ErrorCategory::ArchiveDecode, format!("loader: {e}")))?;
        let manifest_bytes = plan
            .app_manifest()
            .ok_or_else(|| AiError::new(ErrorCategory::ArchiveDecode, "no app manifest section"))?;
        let manifest = AppManifest::decode(manifest_bytes)
            .map_err(|e| AiError::new(ErrorCategory::ArchiveDecode, format!("manifest: {e:?}")))?;
        manifest
            .validate()
            .map_err(|e| AiError::new(ErrorCategory::ArchiveDecode, format!("manifest: {e:?}")))?;
        let blobs = plan
            .content_blobs()
            .map_err(|e| AiError::new(ErrorCategory::ArchiveDecode, format!("blobs: {e}")))?;
        let descriptors = discover_models(&manifest, &blobs)?;
        Ok(Self {
            bytes: bytes.to_vec(),
            fingerprint,
            manifest,
            descriptors,
        })
    }

    /// Open and verify an application from a `.holo` path.
    pub fn open_path(path: impl AsRef<Path>) -> AiResult<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| {
            AiError::new(
                ErrorCategory::ArchiveDecode,
                format!("read {}: {e}", path.as_ref().display()),
            )
        })?;
        Self::open_bytes(&bytes)
    }

    /// BLAKE3 fingerprint of the archive (its trailing footer).
    pub fn archive_fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }

    /// All declared model services (may be empty — an application is not
    /// required to contain models).
    pub fn models(&self) -> &[ModelDescriptor] {
        &self.descriptors
    }

    /// Select a model service explicitly by entry name.
    pub fn model(&self, entry: &str) -> AiResult<Model> {
        if !self.descriptors.iter().any(|d| d.entry == entry) {
            return Err(AiError::new(
                ErrorCategory::ModelSelection,
                format!("no model service named '{entry}'"),
            ));
        }
        self.load_model(entry)
    }

    /// The unambiguous default model: the archive's only service. With zero
    /// or several services this is an error — callers must select by entry.
    pub fn default_model(&self) -> AiResult<Model> {
        match self.descriptors.len() {
            1 => self.load_model(&self.descriptors[0].entry.clone()),
            n => Err(AiError::new(
                ErrorCategory::ModelSelection,
                format!("archive declares {n} model services; select one by entry name"),
            )),
        }
    }

    fn load_model(&self, entry: &str) -> AiResult<Model> {
        let layer = self
            .manifest
            .layers
            .iter()
            .find(|l| l.kind == LayerKind::InferenceModel && l.entry == entry)
            .ok_or_else(|| {
                AiError::new(ErrorCategory::ModelSelection, format!("no model '{entry}'"))
            })?;
        let bundle_bytes = find_blob(&self.bytes, layer.content.as_bytes())?.to_vec();
        Model::from_bundle(entry, &layer.aux, bundle_bytes)
    }
}

fn discover_models(
    manifest: &AppManifest,
    blobs: &[(&[u8], &[u8])],
) -> AiResult<Vec<ModelDescriptor>> {
    let mut out = Vec::new();
    for layer in &manifest.layers {
        if layer.kind != LayerKind::InferenceModel {
            continue;
        }
        let content = blobs
            .iter()
            .find(|(k, _)| *k == layer.content.as_bytes())
            .map(|(_, c)| *c)
            .ok_or_else(|| {
                AiError::new(
                    ErrorCategory::ArchiveDecode,
                    format!(
                        "model '{}' content blob missing (thin archive)",
                        layer.entry
                    ),
                )
            })?;
        let bundle = Bundle::parse(content)?;
        out.push(ModelDescriptor {
            entry: layer.entry.clone(),
            engine: layer.aux.clone(),
            content_kappa: String::from_utf8_lossy(layer.content.as_bytes()).into_owned(),
            manifest: bundle.manifest().clone(),
        });
    }
    Ok(out)
}

fn find_blob<'a>(archive: &'a [u8], kappa: &[u8]) -> AiResult<&'a [u8]> {
    let loader = hologram_archive::HoloLoader::from_bytes(archive)
        .map_err(|e| AiError::new(ErrorCategory::ArchiveDecode, format!("loader: {e}")))?;
    let plan = loader
        .into_plan()
        .map_err(|e| AiError::new(ErrorCategory::ArchiveDecode, format!("loader: {e}")))?;
    let blobs = plan
        .content_blobs()
        .map_err(|e| AiError::new(ErrorCategory::ArchiveDecode, format!("blobs: {e}")))?;
    blobs
        .into_iter()
        .find(|(k, _)| *k == kappa)
        .map(|(_, c)| c)
        .ok_or_else(|| {
            AiError::new(
                ErrorCategory::ArchiveDecode,
                "model content blob missing (thin archive)",
            )
        })
}
