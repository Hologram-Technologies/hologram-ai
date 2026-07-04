//! The honesty meta-gate as a CI-gating test (`just honesty`).
//!
//! Mechanically enforces the docs-as-code links: model ⇄ features bidirectional
//! coverage, tag discipline, and the status contract (`open` never gates; a
//! gating suite defers no work). See docs/conceptual-model/03-status-discipline.md.

use hologram_ai_model::{honesty::audit, workspace_root, Model};

#[test]
fn honesty_audit_passes() {
    let model = Model::load().expect("the conceptual model must load and validate");
    let report = audit(&model, &workspace_root()).expect("the honesty audit must pass");
    println!("honesty audit OK: {report:?}");
    assert!(report.suites >= 1, "at least one gating suite must exist");
    assert_eq!(
        report.features_on_disk,
        report.suites + report.targets,
        "every feature on disk must be a dictionary row (no orphans)"
    );
    assert!(
        report.browser_rows >= 1 && report.rust_rows >= 1,
        "both executors must carry rows"
    );
}
