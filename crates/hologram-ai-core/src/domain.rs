//! Stable domain model for AI holospace applications.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Adapter boundary to `holospaces::realizations::Kappa`.
///
/// This crate does not mint, hash, verify, or otherwise pretend to implement
/// content addressing locally. `KappaRef` is a typed string wrapper for κ-labels
/// already produced by the hologram/holospaces substrate.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub struct KappaRef(String);

impl KappaRef {
    /// Create a new κ-label reference from an existing substrate-produced label.
    pub fn new(label: impl Into<String>) -> Self {
        Self(label.into())
    }

    /// Borrow the underlying κ-label string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for KappaRef {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for KappaRef {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl core::fmt::Display for KappaRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Declares how an AI app is packaged for holospaces provisioning.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppEntryKind {
    /// The app is a `.holo` archive executed through the `.holo` engine path.
    HoloFile,
    /// The app is a Wasm userland module bound to the `hg_*` container ABI.
    Userland,
    /// The app is provisioned from a devcontainer source.
    Devcontainer,
}

/// Content-addressed application manifest for an AI holospace app.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AiAppManifest {
    /// Canonical app manifest κ-label.
    pub app_kappa: KappaRef,
    /// Stable human-readable name.
    pub name: String,
    /// How the app is launched within holospaces.
    pub entry_kind: AppEntryKind,
    /// Model manifests the app expects to reference.
    pub model_kappas: Vec<KappaRef>,
    /// Default runner manifest κ-label, when one is pinned in the app manifest.
    pub default_runner_kappa: Option<KappaRef>,
}

/// Content-addressed model manifest for a compiled or importable model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Canonical model manifest κ-label.
    pub model_kappa: KappaRef,
    /// κ-label of the compiled `.holo` archive or other model artifact.
    pub archive_kappa: KappaRef,
    /// Stable model name.
    pub name: String,
    /// Optional human-readable description.
    pub description: Option<String>,
}

/// Describes the worker or engine path that will execute an inference request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunnerManifest {
    /// Canonical runner manifest κ-label.
    pub runner_kappa: KappaRef,
    /// Stable runner name.
    pub name: String,
    /// Execution kind.
    pub kind: RunnerKind,
}

/// High-level runner category.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunnerKind {
    /// Execute the model as a `.holo` archive through `HoloEngine::run`.
    HoloEngine,
    /// Execute the model in a capability-scoped Wasm worker/userland.
    UserlandWorker,
    /// Deterministic echo runner reserved for tests.
    TestEcho,
}

/// Canonicalized inference parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct InferenceParams {
    /// Canonical κ-label of the parameter object, when one already exists.
    pub params_kappa: Option<KappaRef>,
    /// Maximum number of output tokens requested.
    pub max_output_tokens: Option<u32>,
    /// Temperature expressed in thousandths to avoid floating-point ambiguity.
    pub temperature_milli: Option<u32>,
    /// Stop sequences evaluated by the runner.
    pub stop_sequences: Vec<String>,
}

/// Prompt/input payload submitted by a user or prior event stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prompt {
    /// Canonical prompt/input κ-label.
    pub prompt_kappa: KappaRef,
    /// User-visible text content.
    pub text: String,
}

/// Canonical request handed to a model runner.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Canonical request κ-label.
    pub request_kappa: KappaRef,
    /// Optional application manifest κ-label that originated the request.
    pub app_kappa: Option<KappaRef>,
    /// Model manifest κ-label selected for execution.
    pub model_kappa: KappaRef,
    /// Runner manifest κ-label selected for execution.
    pub runner_kappa: KappaRef,
    /// Prompt payload.
    pub prompt: Prompt,
    /// Canonical execution parameters.
    pub params: InferenceParams,
}

/// Provenance for a completed inference result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceProvenance {
    /// Request κ-label.
    pub request_kappa: KappaRef,
    /// PromptSubmitted event κ-label that introduced the request.
    pub input_event_kappa: KappaRef,
    /// Prompt/input κ-label.
    pub prompt_kappa: KappaRef,
    /// Model manifest κ-label.
    pub model_kappa: KappaRef,
    /// Runner manifest κ-label.
    pub runner_kappa: KappaRef,
    /// Worker identity κ-label.
    pub worker_kappa: KappaRef,
    /// Canonical parameter object κ-label when separately addressed.
    pub params_kappa: Option<KappaRef>,
    /// Output payload κ-label.
    pub output_kappa: KappaRef,
}

/// Completed inference payload returned by a runner.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceOutput {
    /// Request κ-label this output satisfies.
    pub request_kappa: KappaRef,
    /// Canonical output κ-label.
    pub output_kappa: KappaRef,
    /// User-visible output content.
    pub content: String,
    /// Output provenance.
    pub provenance: InferenceProvenance,
}

/// Append-only application event stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AiEvent {
    /// Registers a model manifest with the application.
    ModelRegistered {
        /// Event κ-label.
        event_kappa: KappaRef,
        /// Registered model manifest.
        manifest: ModelManifest,
    },
    /// Submits a prompt and creates a pending inference request.
    PromptSubmitted {
        /// Event κ-label.
        event_kappa: KappaRef,
        /// Request to enqueue.
        request: InferenceRequest,
    },
    /// Records that a worker has started executing a request.
    InferenceStarted {
        /// Event κ-label.
        event_kappa: KappaRef,
        /// Request κ-label.
        request_kappa: KappaRef,
        /// Model κ-label being executed.
        model_kappa: KappaRef,
        /// Runner manifest selected by the worker.
        runner: RunnerManifest,
        /// Worker identity κ-label.
        worker_kappa: KappaRef,
    },
    /// Records a completed inference result.
    InferenceCompleted {
        /// Event κ-label.
        event_kappa: KappaRef,
        /// Completed output payload.
        output: InferenceOutput,
    },
    /// Records a failed inference attempt.
    InferenceFailed {
        /// Event κ-label.
        event_kappa: KappaRef,
        /// Request κ-label.
        request_kappa: KappaRef,
        /// Model κ-label being executed.
        model_kappa: KappaRef,
        /// Runner manifest κ-label.
        runner_kappa: KappaRef,
        /// Worker identity κ-label.
        worker_kappa: KappaRef,
        /// Stable failure description.
        error: String,
    },
}

impl AiEvent {
    /// Borrow the event's κ-label irrespective of variant.
    pub fn event_kappa(&self) -> &KappaRef {
        match self {
            Self::ModelRegistered { event_kappa, .. }
            | Self::PromptSubmitted { event_kappa, .. }
            | Self::InferenceStarted { event_kappa, .. }
            | Self::InferenceCompleted { event_kappa, .. }
            | Self::InferenceFailed { event_kappa, .. } => event_kappa,
        }
    }
}

/// Reducer-visible pending job phase.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PendingPhase {
    /// Submitted but not yet claimed by a worker.
    Queued,
    /// Claimed by a worker and in progress.
    Running,
}

/// Pending inference job visible in the reducer projection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingInference {
    /// PromptSubmitted event κ-label.
    pub submission_event_kappa: KappaRef,
    /// Original request payload.
    pub request: InferenceRequest,
    /// Runner manifest once the job has started.
    pub runner: Option<RunnerManifest>,
    /// Worker identity once the job has started.
    pub worker_kappa: Option<KappaRef>,
    /// Current pending phase.
    pub phase: PendingPhase,
}

/// Completed inference projection entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletedInference {
    /// Completion event κ-label.
    pub completion_event_kappa: KappaRef,
    /// Original request payload.
    pub request: InferenceRequest,
    /// Completed output payload.
    pub output: InferenceOutput,
}

/// Failed inference projection entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailedInference {
    /// Failure event κ-label.
    pub failure_event_kappa: KappaRef,
    /// Original request payload, when the prompt has been seen.
    pub request: Option<InferenceRequest>,
    /// Model κ-label.
    pub model_kappa: KappaRef,
    /// Runner κ-label.
    pub runner_kappa: KappaRef,
    /// Worker κ-label.
    pub worker_kappa: KappaRef,
    /// Stable failure description.
    pub error: String,
}

/// Deterministic projection of the AI event stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AiView {
    /// Registered models by manifest κ-label.
    pub models: BTreeMap<KappaRef, ModelManifest>,
    /// Pending inference jobs by request κ-label.
    pub pending_jobs: BTreeMap<KappaRef, PendingInference>,
    /// Completed inference jobs by request κ-label.
    pub completed_jobs: BTreeMap<KappaRef, CompletedInference>,
    /// Failed inference jobs by request κ-label.
    pub failed_jobs: BTreeMap<KappaRef, FailedInference>,
}
