//! The honesty meta-gate: mechanical enforcement of the docs-as-code links.
//!
//! Audits the model ⇄ features correspondence (bidirectional, no orphans), the
//! per-feature tag discipline, and the status contract (gating suites carry no
//! pending work; `open` rows only measure). Run as a CI-gating test by
//! `hologram-ai-conformance` (`just honesty`).

use crate::{Executor, Model, ModelError, Tier};
use std::collections::BTreeSet;
use std::path::Path;

/// Summary counts from a passing audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditReport {
    /// Gating rows (tier `suite`).
    pub suites: usize,
    /// Non-gating rows (tier `target`).
    pub targets: usize,
    /// `.feature` files found under `features/suites/`.
    pub features_on_disk: usize,
    /// Rows executed by the Rust cucumber runner.
    pub rust_rows: usize,
    /// Rows executed by the browser (cucumber-js) runner.
    pub browser_rows: usize,
}

/// Tags that mark deferred work; forbidden anywhere in a gating feature.
const PENDING_MARKERS: [&str; 4] = ["@pending", "@skip", "@wip", "@ignore"];

fn walk_features(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            walk_features(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "feature") {
            out.push(path);
        }
    }
    Ok(())
}

/// Run the audit against the repository rooted at `root`.
pub fn audit(model: &Model, root: &Path) -> Result<AuditReport, ModelError> {
    let fail = |msg: String| Err(ModelError::Invariant(msg));

    // 1. Every feature on disk is claimed by exactly one row (model.validate()
    //    already guarantees a feature is claimed at most once).
    let suites_dir = root.join("features/suites");
    let mut on_disk = Vec::new();
    walk_features(&suites_dir, &mut on_disk).map_err(|source| ModelError::Io {
        path: suites_dir.display().to_string(),
        source,
    })?;
    let on_disk_rel: BTreeSet<String> = on_disk
        .iter()
        .map(|p| {
            p.strip_prefix(root)
                .expect("walked paths live under root")
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    let claimed: BTreeSet<String> = model.rows.iter().map(|r| r.feature.clone()).collect();
    if let Some(orphan) = on_disk_rel.difference(&claimed).next() {
        return fail(format!(
            "feature `{orphan}` exists on disk but no dictionary row claims it"
        ));
    }

    // 2. Every row's feature exists, and its content upholds the tag + status
    //    discipline.
    let mut suites = 0;
    let mut targets = 0;
    let mut rust_rows = 0;
    let mut browser_rows = 0;
    for row in &model.rows {
        let path = root.join(&row.feature);
        let text = std::fs::read_to_string(&path).map_err(|source| ModelError::Io {
            path: format!("row `{}` feature {}", row.id, path.display()),
            source,
        })?;

        let row_tag = format!("@row:{}", row.id);
        if !text.contains(&row_tag) {
            return fail(format!(
                "feature `{}` is missing its `{row_tag}` tag",
                row.feature
            ));
        }
        let status_tag = format!("@status:{}", row.status);
        if !text.contains(&status_tag) {
            return fail(format!(
                "feature `{}` is missing its `{status_tag}` tag (dictionary says `{}`)",
                row.feature, row.status
            ));
        }
        let stage_tag = format!("@stage:{}", row.stage);
        if !text.contains(&stage_tag) {
            return fail(format!(
                "feature `{}` is missing its `{stage_tag}` tag",
                row.feature
            ));
        }

        match row.tier {
            Tier::Suite => {
                suites += 1;
                for marker in PENDING_MARKERS {
                    if text.contains(marker) {
                        return fail(format!(
                            "gating feature `{}` carries `{marker}` — a gating suite may not defer work",
                            row.feature
                        ));
                    }
                }
                if text.contains("@target") {
                    return fail(format!(
                        "gating feature `{}` is tagged @target",
                        row.feature
                    ));
                }
            }
            Tier::Target => {
                targets += 1;
                if !text.contains("@target") {
                    return fail(format!(
                        "non-gating feature `{}` must be tagged @target so no runner gates on it",
                        row.feature
                    ));
                }
            }
        }

        match row.executor {
            Executor::Rust => rust_rows += 1,
            Executor::Browser => browser_rows += 1,
        }
    }

    // 3. The status vocabulary itself upholds the contract.
    for level in &model.levels {
        let expect_gating = level.id != "open";
        if level.gating != expect_gating {
            return fail(format!(
                "status `{}` gating flag contradicts the discipline (expected {expect_gating})",
                level.id
            ));
        }
    }

    Ok(AuditReport {
        suites,
        targets,
        features_on_disk: on_disk_rel.len(),
        rust_rows,
        browser_rows,
    })
}
