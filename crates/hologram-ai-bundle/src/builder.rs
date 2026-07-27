//! Bundle writer. Allocation is allowed at build time; the produced bytes
//! are what the zero-copy parser later borrows from.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use hologram_ai_core::{
    AbiVersions, AiError, AiResult, ArtifactDescriptor, ArtifactRole, ErrorCategory,
    InferenceModelManifest, ModelProvenance, OperationDescriptor, StatusPolicy,
    MANIFEST_SCHEMA_VERSION,
};

use crate::validate::validate_manifest;
use crate::{BUNDLE_MAGIC, BUNDLE_SCHEMA_VERSION, HEADER_LEN};

/// A buffered component awaiting bundle assembly.
#[derive(Debug, Clone)]
struct Component {
    role: ArtifactRole,
    bytes: Vec<u8>,
    digest: [u8; 32],
}

/// Canonical position of a role in `ArtifactRole::ALL` (deterministic sort
/// key). Every closed-enum role is present in `ALL`; the fallback only
/// exists to keep this function total.
fn role_order(role: ArtifactRole) -> usize {
    ArtifactRole::ALL
        .iter()
        .position(|r| *r == role)
        .unwrap_or(usize::MAX)
}

/// Builder for schema v1 inference bundles.
///
/// Components may be added in any order; [`BundleBuilder::finish`] sorts
/// them into canonical [`ArtifactRole::ALL`] order so the output bytes are
/// independent of insertion order. `status_policy` defaults to
/// [`StatusPolicy::R4_DEFAULT`]; `abi` and `provenance` default to zeroed
/// placeholders — production callers should always set real values via
/// [`BundleBuilder::set_abi`] and [`BundleBuilder::set_provenance`].
#[derive(Debug, Clone)]
pub struct BundleBuilder {
    engine: String,
    artifact_format: String,
    model_name: String,
    operations: Vec<OperationDescriptor>,
    status_policy: StatusPolicy,
    abi: AbiVersions,
    provenance: ModelProvenance,
    components: Vec<Component>,
}

impl BundleBuilder {
    /// Start a bundle for `engine` (e.g. `ENGINE_UOR_R4`) and
    /// `artifact_format` (e.g. `ARTIFACT_FORMAT_R4G1`).
    pub fn new(engine: &str, artifact_format: &str, model_name: &str) -> Self {
        Self {
            engine: engine.into(),
            artifact_format: artifact_format.into(),
            model_name: model_name.into(),
            operations: Vec::new(),
            status_policy: StatusPolicy::R4_DEFAULT,
            abi: AbiVersions {
                compiler_abi: (0, 0, 0),
                r4g1_format: (0, 0),
                contract: (0, 0, 0),
                holo_format: 0,
            },
            provenance: ModelProvenance {
                source_repository: None,
                source_revision: None,
                source_digest: [0u8; 32],
                compiler_revision: String::new(),
                compile_options_digest: [0u8; 32],
                quality_report_digest: [0u8; 32],
            },
            components: Vec::new(),
        }
    }

    /// Set the advertised operations (authoritative for invocation).
    pub fn set_operations(mut self, operations: Vec<OperationDescriptor>) -> Self {
        self.operations = operations;
        self
    }

    /// Set the normalized status/abstention policy.
    pub fn set_status_policy(mut self, status_policy: StatusPolicy) -> Self {
        self.status_policy = status_policy;
        self
    }

    /// Set the compiler/runtime ABI versions.
    pub fn set_abi(mut self, abi: AbiVersions) -> Self {
        self.abi = abi;
        self
    }

    /// Set the deterministic compilation provenance.
    pub fn set_provenance(mut self, provenance: ModelProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// Add a component under a semantic role. Computes the BLAKE3 digest
    /// and buffers a copy of `bytes`. Rejects a duplicate role with
    /// [`ErrorCategory::BundleEncode`].
    pub fn add_component(&mut self, role: ArtifactRole, bytes: &[u8]) -> AiResult<()> {
        if self.components.iter().any(|c| c.role == role) {
            return Err(AiError::new(
                ErrorCategory::BundleEncode,
                format!("duplicate component role `{}`", role.name()),
            ));
        }
        self.components.push(Component {
            role,
            bytes: bytes.to_vec(),
            digest: *blake3::hash(bytes).as_bytes(),
        });
        Ok(())
    }

    /// The manifest as it will be embedded in the bundle: schema v1, with
    /// artifact descriptors in canonical role order. Useful for tests and
    /// for inspection before [`BundleBuilder::finish`].
    pub fn manifest(&self) -> InferenceModelManifest {
        let mut refs: Vec<&Component> = self.components.iter().collect();
        refs.sort_by_key(|c| role_order(c.role));
        InferenceModelManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            engine: self.engine.clone(),
            artifact_format: self.artifact_format.clone(),
            model_name: self.model_name.clone(),
            operations: self.operations.clone(),
            artifacts: refs
                .iter()
                .map(|c| ArtifactDescriptor {
                    role: c.role,
                    digest: c.digest,
                    length: c.bytes.len() as u64,
                })
                .collect(),
            status_policy: self.status_policy,
            abi: self.abi.clone(),
            provenance: self.provenance.clone(),
        }
    }

    /// Validate, assemble, and encode the bundle.
    ///
    /// Runs the schema v1 semantic rules (mandatory roles for uor-r4/R4G1,
    /// capability⇔processor consistency — the same rules as
    /// [`crate::Bundle::validate`]), sorts components into canonical role
    /// order, and encodes header + manifest + payloads with no padding.
    pub fn finish(mut self) -> AiResult<Vec<u8>> {
        self.components.sort_by_key(|c| role_order(c.role));
        let manifest = self.manifest();
        validate_manifest(&manifest)?;

        let manifest_bytes = manifest.encode();
        let manifest_len = u32::try_from(manifest_bytes.len()).map_err(|_| {
            AiError::new(
                ErrorCategory::BundleEncode,
                "encoded manifest exceeds u32 length prefix",
            )
        })?;

        let payload_len: usize = self.components.iter().map(|c| c.bytes.len()).sum();
        let mut out = Vec::with_capacity(HEADER_LEN + manifest_bytes.len() + payload_len);
        out.extend_from_slice(&BUNDLE_MAGIC);
        out.extend_from_slice(&BUNDLE_SCHEMA_VERSION.to_le_bytes());
        out.extend_from_slice(&manifest_len.to_le_bytes());
        out.extend_from_slice(&manifest_bytes);
        for c in &self.components {
            out.extend_from_slice(&c.bytes);
        }
        Ok(out)
    }
}
