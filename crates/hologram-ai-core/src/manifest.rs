//! Model manifest, operations, capabilities, status policy, provenance.

use alloc::string::String;
use alloc::vec::Vec;

use crate::canon::{CanonError, CanonReader, CanonWriter};
use crate::value::ValueDescriptor;

/// Current bundle manifest schema version.
pub const MANIFEST_SCHEMA_VERSION: u16 = 1;

/// Engine identifier for uor-r4 R4G1 models.
pub const ENGINE_UOR_R4: &str = "uor-r4";

/// Artifact format identifier for scored R4G1 graphs.
pub const ARTIFACT_FORMAT_R4G1: &str = "R4G1";

/// A callable operation advertised by a model. The manifest is
/// authoritative: callers must discover operations, not assume them.
///
/// Standardized initial operation names: `generate`, `predict`, `embed`,
/// `classify`, `transcribe`. A model implements any subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationDescriptor {
    pub name: String,
    pub inputs: Vec<ValueDescriptor>,
    pub outputs: Vec<ValueDescriptor>,
    /// Whether the operation supports streaming events.
    pub streaming: bool,
}

impl OperationDescriptor {
    pub fn encode(&self, w: &mut CanonWriter) {
        w.str(&self.name);
        w.seq_len(self.inputs.len());
        for v in &self.inputs {
            v.encode(w);
        }
        w.seq_len(self.outputs.len());
        for v in &self.outputs {
            v.encode(w);
        }
        w.bool(self.streaming);
    }

    pub fn decode(r: &mut CanonReader<'_>) -> Result<Self, CanonError> {
        let name = r.str()?.into();
        let inputs = decode_seq(r, ValueDescriptor::decode)?;
        let outputs = decode_seq(r, ValueDescriptor::decode)?;
        let streaming = r.bool()?;
        Ok(Self {
            name,
            inputs,
            outputs,
            streaming,
        })
    }
}

/// Semantic role of a bundle component. Closed, versioned vocabulary —
/// semantic roles, never historical filenames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactRole {
    /// Validated scored R4G1 graph (deployable).
    Graph,
    /// Signature/input-projection artifact required by the R4G1 runtime.
    SignatureArtifact,
    /// Byte-level BPE tokenizer (required iff text input/output).
    Tokenizer,
    /// Complete R4 score report.
    ScoreReport,
    /// Image processor contract (required iff image input).
    ImageProcessor,
    /// Audio processor + sample-format contract (required iff audio input).
    AudioProcessor,
    /// Generation configuration.
    GenerationConfig,
    /// Vocabulary metadata.
    VocabularyMetadata,
    /// Label maps for classification.
    LabelMap,
    /// Modality normalization metadata.
    NormalizationMetadata,
    /// Certification / witness artifacts.
    Witnesses,
}

impl ArtifactRole {
    /// All roles, in canonical encoding order.
    pub const ALL: [Self; 11] = [
        Self::Graph,
        Self::SignatureArtifact,
        Self::Tokenizer,
        Self::ScoreReport,
        Self::ImageProcessor,
        Self::AudioProcessor,
        Self::GenerationConfig,
        Self::VocabularyMetadata,
        Self::LabelMap,
        Self::NormalizationMetadata,
        Self::Witnesses,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Graph => "graph",
            Self::SignatureArtifact => "signature-artifact",
            Self::Tokenizer => "tokenizer",
            Self::ScoreReport => "score-report",
            Self::ImageProcessor => "image-processor",
            Self::AudioProcessor => "audio-processor",
            Self::GenerationConfig => "generation-config",
            Self::VocabularyMetadata => "vocabulary-metadata",
            Self::LabelMap => "label-map",
            Self::NormalizationMetadata => "normalization-metadata",
            Self::Witnesses => "witnesses",
        }
    }

    fn discriminant(self) -> u16 {
        match self {
            Self::Graph => 0,
            Self::SignatureArtifact => 1,
            Self::Tokenizer => 2,
            Self::ScoreReport => 3,
            Self::ImageProcessor => 4,
            Self::AudioProcessor => 5,
            Self::GenerationConfig => 6,
            Self::VocabularyMetadata => 7,
            Self::LabelMap => 8,
            Self::NormalizationMetadata => 9,
            Self::Witnesses => 10,
        }
    }

    fn from_discriminant(d: u16) -> Result<Self, CanonError> {
        Self::ALL
            .into_iter()
            .find(|r| r.discriminant() == d)
            .ok_or(CanonError::UnknownDiscriminant)
    }
}

/// A bundle component descriptor: semantic role, length, content digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactDescriptor {
    pub role: ArtifactRole,
    /// BLAKE3 digest of the component bytes.
    pub digest: [u8; 32],
    /// Component length in bytes.
    pub length: u64,
}

impl ArtifactDescriptor {
    pub fn encode(&self, w: &mut CanonWriter) {
        w.u16(self.role.discriminant());
        w.bytes(&self.digest);
        w.u64(self.length);
    }

    pub fn decode(r: &mut CanonReader<'_>) -> Result<Self, CanonError> {
        let role = ArtifactRole::from_discriminant(r.u16()?)?;
        let digest_bytes = r.bytes()?;
        let digest: [u8; 32] = digest_bytes
            .try_into()
            .map_err(|_| CanonError::LengthOutOfRange)?;
        let length = r.u64()?;
        Ok(Self {
            role,
            digest,
            length,
        })
    }
}

/// Normalized status and abstention policy (engine-agnostic form).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatusPolicy {
    /// Serve answers backed by an exact observed context.
    pub serve_exact: bool,
    /// Serve answers backed by graph structure.
    pub serve_graph: bool,
    /// Serve answers over novel (unobserved) contexts.
    pub serve_novel: bool,
    /// Widen the context at most once before deciding.
    pub widen_once: bool,
    /// Abstain (rather than fabricate a token) when unresolved.
    pub abstain_when_unresolved: bool,
}

impl StatusPolicy {
    /// The default R4 policy: serve exact/graph, widen once, abstain when
    /// unresolved, never serve novel contexts without widening.
    pub const R4_DEFAULT: Self = Self {
        serve_exact: true,
        serve_graph: true,
        serve_novel: false,
        widen_once: true,
        abstain_when_unresolved: true,
    };

    pub fn encode(&self, w: &mut CanonWriter) {
        w.bool(self.serve_exact);
        w.bool(self.serve_graph);
        w.bool(self.serve_novel);
        w.bool(self.widen_once);
        w.bool(self.abstain_when_unresolved);
    }

    pub fn decode(r: &mut CanonReader<'_>) -> Result<Self, CanonError> {
        Ok(Self {
            serve_exact: r.bool()?,
            serve_graph: r.bool()?,
            serve_novel: r.bool()?,
            widen_once: r.bool()?,
            abstain_when_unresolved: r.bool()?,
        })
    }
}

/// Compiler/runtime ABI and artifact format versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiVersions {
    /// `uor-r4-api` facade version (major, minor, patch).
    pub compiler_abi: (u16, u16, u16),
    /// R4G1 artifact format version (major, minor).
    pub r4g1_format: (u16, u16),
    /// Inference operation contract version (major, minor, patch).
    pub contract: (u16, u16, u16),
    /// Hologram `.holo` format version used for packaging.
    pub holo_format: u16,
}

impl AbiVersions {
    pub fn encode(&self, w: &mut CanonWriter) {
        let (a, b, c) = self.compiler_abi;
        w.u16(a);
        w.u16(b);
        w.u16(c);
        let (a, b) = self.r4g1_format;
        w.u16(a);
        w.u16(b);
        let (a, b, c) = self.contract;
        w.u16(a);
        w.u16(b);
        w.u16(c);
        w.u16(self.holo_format);
    }

    pub fn decode(r: &mut CanonReader<'_>) -> Result<Self, CanonError> {
        Ok(Self {
            compiler_abi: (r.u16()?, r.u16()?, r.u16()?),
            r4g1_format: (r.u16()?, r.u16()?),
            contract: (r.u16()?, r.u16()?, r.u16()?),
            holo_format: r.u16()?,
        })
    }
}

/// Deterministic compilation provenance. Never contains machine-specific
/// paths, cache locations, or wall-clock timestamps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProvenance {
    /// Source repository (e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`) or
    /// `None` for an anonymous local source.
    pub source_repository: Option<String>,
    /// Immutable source revision (full commit SHA) when known.
    pub source_revision: Option<String>,
    /// BLAKE3 digest over the canonical source file list (paths + digests).
    pub source_digest: [u8; 32],
    /// uor-r4 compiler revision (git commit).
    pub compiler_revision: String,
    /// Canonical digest of the compile options.
    pub compile_options_digest: [u8; 32],
    /// BLAKE3 digest of the score/quality report.
    pub quality_report_digest: [u8; 32],
}

impl ModelProvenance {
    pub fn encode(&self, w: &mut CanonWriter) {
        w.opt_str(self.source_repository.as_deref());
        w.opt_str(self.source_revision.as_deref());
        w.bytes(&self.source_digest);
        w.str(&self.compiler_revision);
        w.bytes(&self.compile_options_digest);
        w.bytes(&self.quality_report_digest);
    }

    pub fn decode(r: &mut CanonReader<'_>) -> Result<Self, CanonError> {
        let source_repository = r.opt_str()?.map(Into::into);
        let source_revision = r.opt_str()?.map(Into::into);
        let source_digest = read_digest(r)?;
        let compiler_revision = r.str()?.into();
        let compile_options_digest = read_digest(r)?;
        let quality_report_digest = read_digest(r)?;
        Ok(Self {
            source_repository,
            source_revision,
            source_digest,
            compiler_revision,
            compile_options_digest,
            quality_report_digest,
        })
    }
}

fn read_digest(r: &mut CanonReader<'_>) -> Result<[u8; 32], CanonError> {
    r.bytes()?
        .try_into()
        .map_err(|_| CanonError::LengthOutOfRange)
}

/// The canonical inference-model manifest. This is the authoritative
/// description of a model's engine, operations, components, status policy,
/// ABI versions, and provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceModelManifest {
    /// Bundle manifest schema version ([`MANIFEST_SCHEMA_VERSION`]).
    pub schema_version: u16,
    /// Engine identifier (e.g. [`ENGINE_UOR_R4`]).
    pub engine: String,
    /// Artifact format identifier (e.g. [`ARTIFACT_FORMAT_R4G1`]).
    pub artifact_format: String,
    pub model_name: String,
    /// Advertised operations; authoritative for invocation.
    pub operations: Vec<OperationDescriptor>,
    /// Bundle components with digests, in canonical role order.
    pub artifacts: Vec<ArtifactDescriptor>,
    pub status_policy: StatusPolicy,
    pub abi: AbiVersions,
    pub provenance: ModelProvenance,
}

impl InferenceModelManifest {
    /// Canonical encoding (deterministic by construction).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = CanonWriter::new();
        w.u16(self.schema_version);
        w.str(&self.engine);
        w.str(&self.artifact_format);
        w.str(&self.model_name);
        w.seq_len(self.operations.len());
        for op in &self.operations {
            op.encode(&mut w);
        }
        w.seq_len(self.artifacts.len());
        for a in &self.artifacts {
            a.encode(&mut w);
        }
        self.status_policy.encode(&mut w);
        self.abi.encode(&mut w);
        self.provenance.encode(&mut w);
        w.finish()
    }

    /// Decode a canonical manifest, rejecting trailing bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, CanonError> {
        let mut r = CanonReader::new(bytes);
        let m = Self::decode_body(&mut r)?;
        r.finish()?;
        Ok(m)
    }

    fn decode_body(r: &mut CanonReader<'_>) -> Result<Self, CanonError> {
        let schema_version = r.u16()?;
        let engine = r.str()?.into();
        let artifact_format = r.str()?.into();
        let model_name = r.str()?.into();
        let operations = decode_seq(r, OperationDescriptor::decode)?;
        let artifacts = decode_seq(r, ArtifactDescriptor::decode)?;
        let status_policy = StatusPolicy::decode(r)?;
        let abi = AbiVersions::decode(r)?;
        let provenance = ModelProvenance::decode(r)?;
        Ok(Self {
            schema_version,
            engine,
            artifact_format,
            model_name,
            operations,
            artifacts,
            status_policy,
            abi,
            provenance,
        })
    }
}

pub(crate) fn decode_seq<T>(
    r: &mut CanonReader<'_>,
    f: impl Fn(&mut CanonReader<'_>) -> Result<T, CanonError>,
) -> Result<Vec<T>, CanonError> {
    let n = r.seq_len()?;
    let mut out = Vec::with_capacity(n.min(1 << 16) as usize);
    for _ in 0..n {
        out.push(f(r)?);
    }
    Ok(out)
}
