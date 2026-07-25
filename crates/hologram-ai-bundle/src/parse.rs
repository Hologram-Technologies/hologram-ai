//! Bounded zero-copy bundle parser. All input is treated as hostile: every
//! offset uses checked arithmetic, lengths are bounded, and structural
//! failures never panic.

use alloc::format;
use alloc::string::String;

use hologram_ai_core::{
    AiError, AiResult, ArtifactRole, ErrorCategory, InferenceModelManifest, MANIFEST_SCHEMA_VERSION,
};

use crate::validate::validate_manifest;
use crate::{BUNDLE_MAGIC, BUNDLE_SCHEMA_VERSION, HEADER_LEN, MAX_BUNDLE_SIZE, MAX_COMPONENTS};

fn decode_err(message: impl Into<String>) -> AiError {
    AiError::new(ErrorCategory::BundleDecode, message)
}

/// A parsed schema v1 bundle, borrowing the input bytes.
///
/// `parse` validates structure only — magic, schema version, manifest
/// decodability, role uniqueness, component extents, and exact input
/// consumption. It does **not** hash component payloads: use
/// [`Bundle::parse_verified`] (recommended) or call
/// [`Bundle::verify_digests`] before any engine sees component bytes.
#[derive(Debug, Clone)]
pub struct Bundle<'a> {
    bytes: &'a [u8],
    manifest: InferenceModelManifest,
    /// Offset of the first component payload.
    payload_offset: usize,
}

impl<'a> Bundle<'a> {
    /// Parse a bundle, validating structure but not component digests.
    ///
    /// Errors: [`ErrorCategory::BundleDecode`] on malformed input (bad
    /// magic, truncation, oversize, duplicate roles, length inconsistencies,
    /// trailing bytes) and [`ErrorCategory::AbiMismatch`] on an unknown
    /// bundle or manifest schema version.
    pub fn parse(bytes: &'a [u8]) -> AiResult<Self> {
        if bytes.len() as u64 > MAX_BUNDLE_SIZE {
            return Err(decode_err(format!(
                "bundle size {} exceeds maximum {MAX_BUNDLE_SIZE}",
                bytes.len()
            )));
        }
        if bytes.len() < HEADER_LEN {
            return Err(decode_err("truncated bundle header"));
        }
        if bytes[0..4] != BUNDLE_MAGIC {
            return Err(decode_err("bad bundle magic"));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != BUNDLE_SCHEMA_VERSION {
            return Err(AiError::new(
                ErrorCategory::AbiMismatch,
                format!("unsupported bundle schema version {version}"),
            ));
        }
        let manifest_len = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        let manifest_end = HEADER_LEN
            .checked_add(manifest_len)
            .ok_or_else(|| decode_err("manifest length overflows address space"))?;
        let manifest_bytes = bytes
            .get(HEADER_LEN..manifest_end)
            .ok_or_else(|| decode_err("declared manifest length exceeds input"))?;
        let manifest = InferenceModelManifest::decode(manifest_bytes)
            .map_err(|e| decode_err(format!("manifest decode failed: {e}")))?;
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(AiError::new(
                ErrorCategory::AbiMismatch,
                format!(
                    "unsupported manifest schema version {}",
                    manifest.schema_version
                ),
            ));
        }
        if manifest.artifacts.len() > MAX_COMPONENTS {
            return Err(decode_err(format!(
                "component count {} exceeds maximum {MAX_COMPONENTS}",
                manifest.artifacts.len()
            )));
        }
        for (i, a) in manifest.artifacts.iter().enumerate() {
            if manifest.artifacts[..i].iter().any(|b| b.role == a.role) {
                return Err(decode_err(format!(
                    "duplicate component role `{}` in manifest",
                    a.role.name()
                )));
            }
        }

        // Component extents: payloads follow the manifest contiguously and
        // must consume the input exactly — no truncation, no trailing bytes.
        let mut end = manifest_end as u64;
        for a in &manifest.artifacts {
            end = end
                .checked_add(a.length)
                .ok_or_else(|| decode_err("component lengths overflow u64"))?;
        }
        if end > bytes.len() as u64 {
            return Err(decode_err("declared component lengths exceed input"));
        }
        if end < bytes.len() as u64 {
            return Err(decode_err("trailing bytes after final component"));
        }

        Ok(Self {
            bytes,
            manifest,
            payload_offset: manifest_end,
        })
    }

    /// Parse and immediately verify every component digest. This is the
    /// recommended entry point: the returned bundle is structurally sound
    /// and every component matches the BLAKE3 digest in its manifest
    /// descriptor.
    pub fn parse_verified(bytes: &'a [u8]) -> AiResult<Self> {
        let bundle = Self::parse(bytes)?;
        bundle.verify_digests()?;
        Ok(bundle)
    }

    /// The decoded manifest.
    pub fn manifest(&self) -> &InferenceModelManifest {
        &self.manifest
    }

    /// BLAKE3 digest over the whole bundle byte string. This is the model
    /// identity used for content addressing.
    pub fn bundle_digest(&self) -> [u8; 32] {
        *blake3::hash(self.bytes).as_bytes()
    }

    /// Borrow a component's payload by role (zero-copy).
    pub fn component(&self, role: ArtifactRole) -> Option<&'a [u8]> {
        let mut offset = self.payload_offset;
        for a in &self.manifest.artifacts {
            let len = usize::try_from(a.length).ok()?;
            let end = offset.checked_add(len)?;
            let slice = self.bytes.get(offset..end)?;
            if a.role == role {
                return Some(slice);
            }
            offset = end;
        }
        None
    }

    /// Iterate all components as `(role, payload)` pairs in manifest order
    /// (zero-copy).
    pub fn components(&self) -> impl Iterator<Item = (ArtifactRole, &'a [u8])> + '_ {
        let bytes = self.bytes;
        let mut offset = self.payload_offset;
        self.manifest.artifacts.iter().filter_map(move |a| {
            let len = usize::try_from(a.length).ok()?;
            let end = offset.checked_add(len)?;
            let slice = bytes.get(offset..end)?;
            offset = end;
            Some((a.role, slice))
        })
    }

    /// Verify every component payload against the BLAKE3 digest declared in
    /// its manifest descriptor. Must succeed before any engine consumes
    /// component bytes; failures are [`ErrorCategory::IntegrityMismatch`].
    pub fn verify_digests(&self) -> AiResult<()> {
        for a in &self.manifest.artifacts {
            let payload = self.component(a.role).ok_or_else(|| {
                decode_err(format!(
                    "component `{}` extent inconsistent with manifest",
                    a.role.name()
                ))
            })?;
            if blake3::hash(payload).as_bytes() != &a.digest {
                return Err(AiError::new(
                    ErrorCategory::IntegrityMismatch,
                    format!("digest mismatch for component `{}`", a.role.name()),
                ));
            }
        }
        Ok(())
    }

    /// Validate the schema v1 semantic rules: mandatory roles for
    /// uor-r4/R4G1 models, and capability⇔processor consistency. See the
    /// crate-level docs for the exact rules.
    pub fn validate(&self) -> AiResult<()> {
        validate_manifest(&self.manifest)
    }
}
