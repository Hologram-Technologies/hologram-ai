//! Deterministic reducer for AI app event streams.

use crate::domain::{
    AiEvent, AiView, CompletedInference, FailedInference, PendingInference, PendingPhase,
};

/// Fold an append-only AI event stream into stable application state.
///
/// This reducer is pure and deterministic. It does not execute models, spawn
/// workers, or perform side effects; it only projects prior events into `AiView`.
pub fn reduce(events: &[AiEvent]) -> AiView {
    let mut view = AiView::default();
    for event in events {
        apply_event(&mut view, event);
    }
    view
}

fn apply_event(view: &mut AiView, event: &AiEvent) {
    match event {
        AiEvent::ModelRegistered { manifest, .. } => {
            view.models
                .insert(manifest.model_kappa.clone(), manifest.clone());
        }
        AiEvent::PromptSubmitted {
            event_kappa,
            request,
        } => register_prompt(view, event_kappa, request),
        AiEvent::InferenceStarted {
            request_kappa,
            runner,
            worker_kappa,
            ..
        } => mark_started(view, request_kappa, runner, worker_kappa),
        AiEvent::InferenceCompleted {
            event_kappa,
            output,
        } => mark_completed(view, event_kappa, output),
        AiEvent::InferenceFailed {
            event_kappa,
            request_kappa,
            model_kappa,
            runner_kappa,
            worker_kappa,
            error,
        } => mark_failed(
            view,
            event_kappa,
            request_kappa,
            model_kappa,
            runner_kappa,
            worker_kappa,
            error,
        ),
    }
}

fn register_prompt(
    view: &mut AiView,
    event_kappa: &crate::KappaRef,
    request: &crate::InferenceRequest,
) {
    clear_terminal_state(view, &request.request_kappa);
    view.pending_jobs.insert(
        request.request_kappa.clone(),
        PendingInference {
            submission_event_kappa: event_kappa.clone(),
            request: request.clone(),
            runner: None,
            worker_kappa: None,
            phase: PendingPhase::Queued,
        },
    );
}

fn mark_started(
    view: &mut AiView,
    request_kappa: &crate::KappaRef,
    runner: &crate::RunnerManifest,
    worker_kappa: &crate::KappaRef,
) {
    let Some(existing) = view.pending_jobs.get_mut(request_kappa) else {
        return;
    };
    existing.runner = Some(runner.clone());
    existing.worker_kappa = Some(worker_kappa.clone());
    existing.phase = PendingPhase::Running;
}

fn mark_completed(
    view: &mut AiView,
    event_kappa: &crate::KappaRef,
    output: &crate::InferenceOutput,
) {
    let request = request_for_completion(view, output);
    view.pending_jobs.remove(&output.request_kappa);
    view.failed_jobs.remove(&output.request_kappa);
    let Some(request) = request else {
        return;
    };
    view.completed_jobs.insert(
        output.request_kappa.clone(),
        CompletedInference {
            completion_event_kappa: event_kappa.clone(),
            request,
            output: output.clone(),
        },
    );
}

fn request_for_completion(
    view: &AiView,
    output: &crate::InferenceOutput,
) -> Option<crate::InferenceRequest> {
    if let Some(pending) = view.pending_jobs.get(&output.request_kappa) {
        return Some(pending.request.clone());
    }
    if let Some(completed) = view.completed_jobs.get(&output.request_kappa) {
        return Some(completed.request.clone());
    }
    view.failed_jobs
        .get(&output.request_kappa)
        .and_then(|failed| failed.request.clone())
}

fn mark_failed(
    view: &mut AiView,
    event_kappa: &crate::KappaRef,
    request_kappa: &crate::KappaRef,
    model_kappa: &crate::KappaRef,
    runner_kappa: &crate::KappaRef,
    worker_kappa: &crate::KappaRef,
    error: &str,
) {
    let request = view
        .pending_jobs
        .get(request_kappa)
        .map(|pending| pending.request.clone())
        .or_else(|| {
            view.completed_jobs
                .get(request_kappa)
                .map(|completed| completed.request.clone())
        });
    view.pending_jobs.remove(request_kappa);
    view.completed_jobs.remove(request_kappa);
    view.failed_jobs.insert(
        request_kappa.clone(),
        FailedInference {
            failure_event_kappa: event_kappa.clone(),
            request,
            model_kappa: model_kappa.clone(),
            runner_kappa: runner_kappa.clone(),
            worker_kappa: worker_kappa.clone(),
            error: error.to_string(),
        },
    );
}

fn clear_terminal_state(view: &mut AiView, request_kappa: &crate::KappaRef) {
    view.completed_jobs.remove(request_kappa);
    view.failed_jobs.remove(request_kappa);
}
