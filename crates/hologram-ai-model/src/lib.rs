//! Typed registries for the hologram-ai conceptual model.
//!
//! Parses `model/{dictionary,status,oracles,usecases}.toml` into validated,
//! typed data — the single source of truth the BDD features, witnesses, and
//! the honesty meta-gate are checked against. This crate contains no pipeline
//! code; it changes only when the conceptual model changes
//! (`docs/architecture/ARCHITECTURE.md` §3).

pub mod honesty;

use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Errors loading or validating the conceptual model.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("reading {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("parsing {path}: {source}")]
    Parse {
        path: String,
        source: toml::de::Error,
    },
    #[error("model invariant violated: {0}")]
    Invariant(String),
}

/// One honesty level from `model/status.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct StatusLevel {
    pub id: String,
    pub summary: String,
    pub may_assert: String,
    pub gating: bool,
}

/// A committed oracle artifact with its integrity checksum.
#[derive(Debug, Clone, Deserialize)]
pub struct OracleArtifact {
    pub path: String,
    pub sha256: String,
}

/// One authoritative external source from `model/oracles.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Oracle {
    pub id: String,
    pub authority: String,
    pub source: String,
    #[serde(default)]
    pub pin: String,
    pub license: String,
    pub kind: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub artifacts: Vec<OracleArtifact>,
}

/// Config-derived quantities a witness must reproduce from the model's own
/// `config.json` (never from literals in pipeline code).
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct UseCaseExpects {
    pub hidden_size: u64,
    pub num_hidden_layers: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub vocab_size: u64,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
}

/// One use-case instance from `model/usecases.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct UseCase {
    pub id: String,
    pub canonical: bool,
    pub family: String,
    #[serde(default)]
    pub hf_repo: String,
    #[serde(default)]
    pub hf_revision: String,
    #[serde(default)]
    pub oracle: String,
    #[serde(default)]
    pub note: String,
    pub expects: UseCaseExpects,
}

/// The gating tier of a dictionary row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Implemented, gating, green.
    Suite,
    /// Defined behavior, measured/probed, non-gating.
    Target,
}

/// Which BDD executor runs a row's feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Executor {
    /// The `cucumber` runner in `hologram-ai-conformance`.
    Rust,
    /// cucumber-js + Playwright Chromium in `apps/web/bdd`.
    Browser,
}

/// One dictionary row from `model/dictionary.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Row {
    pub id: String,
    pub statement: String,
    pub stage: String,
    pub status: String,
    pub oracles: Vec<String>,
    pub tier: Tier,
    pub executor: Executor,
    pub feature: String,
}

#[derive(Debug, Deserialize)]
struct DictionaryFile {
    row: Vec<Row>,
}
#[derive(Debug, Deserialize)]
struct StatusFile {
    level: Vec<StatusLevel>,
}
#[derive(Debug, Deserialize)]
struct OraclesFile {
    oracle: Vec<Oracle>,
}
#[derive(Debug, Deserialize)]
struct UseCasesFile {
    usecase: Vec<UseCase>,
}

/// The loaded, validated conceptual model.
#[derive(Debug)]
pub struct Model {
    pub rows: Vec<Row>,
    pub levels: Vec<StatusLevel>,
    pub oracles: Vec<Oracle>,
    pub usecases: Vec<UseCase>,
}

/// The repository root (the directory holding `model/` and `features/`).
pub fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/hologram-ai-model -> crates/
    p.pop(); // crates/ -> repo root
    p
}

fn load_toml<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, ModelError> {
    let text = std::fs::read_to_string(path).map_err(|source| ModelError::Io {
        path: path.display().to_string(),
        source,
    })?;
    toml::from_str(&text).map_err(|source| ModelError::Parse {
        path: path.display().to_string(),
        source,
    })
}

impl Model {
    /// Load and validate the model from `<repo>/model/`.
    pub fn load() -> Result<Self, ModelError> {
        Self::load_from(&workspace_root())
    }

    /// Load and validate the model rooted at `root`.
    pub fn load_from(root: &Path) -> Result<Self, ModelError> {
        let dict: DictionaryFile = load_toml(&root.join("model/dictionary.toml"))?;
        let status: StatusFile = load_toml(&root.join("model/status.toml"))?;
        let oracles: OraclesFile = load_toml(&root.join("model/oracles.toml"))?;
        let usecases: UseCasesFile = load_toml(&root.join("model/usecases.toml"))?;
        let model = Model {
            rows: dict.row,
            levels: status.level,
            oracles: oracles.oracle,
            usecases: usecases.usecase,
        };
        model.validate()?;
        Ok(model)
    }

    /// Look up an oracle by id.
    pub fn oracle(&self, id: &str) -> Option<&Oracle> {
        self.oracles.iter().find(|o| o.id == id)
    }

    /// Look up a dictionary row by id.
    pub fn row(&self, id: &str) -> Option<&Row> {
        self.rows.iter().find(|r| r.id == id)
    }

    /// Look up a use-case by id.
    pub fn usecase(&self, id: &str) -> Option<&UseCase> {
        self.usecases.iter().find(|u| u.id == id)
    }

    /// The canonical use-case instance.
    pub fn canonical_usecase(&self) -> Result<&UseCase, ModelError> {
        let canon: Vec<&UseCase> = self.usecases.iter().filter(|u| u.canonical).collect();
        match canon.as_slice() {
            [one] => Ok(one),
            other => Err(ModelError::Invariant(format!(
                "exactly one canonical use-case required, found {}",
                other.len()
            ))),
        }
    }

    /// Whether a status level gates CI.
    pub fn level_gating(&self, status: &str) -> Result<bool, ModelError> {
        self.levels
            .iter()
            .find(|l| l.id == status)
            .map(|l| l.gating)
            .ok_or_else(|| ModelError::Invariant(format!("unknown status level `{status}`")))
    }

    fn validate(&self) -> Result<(), ModelError> {
        let inv = |msg: String| Err(ModelError::Invariant(msg));

        let expected_levels: BTreeSet<&str> = ["verified", "build", "open"].into();
        let got_levels: BTreeSet<&str> = self.levels.iter().map(|l| l.id.as_str()).collect();
        if got_levels != expected_levels {
            return inv(format!(
                "status vocabulary must be exactly {expected_levels:?}, got {got_levels:?}"
            ));
        }

        let mut oracle_ids = BTreeSet::new();
        for o in &self.oracles {
            if !oracle_ids.insert(o.id.as_str()) {
                return inv(format!("duplicate oracle id `{}`", o.id));
            }
            for a in &o.artifacts {
                if a.sha256.len() != 64 || !a.sha256.chars().all(|c| c.is_ascii_hexdigit()) {
                    return inv(format!(
                        "oracle `{}` artifact `{}` has a malformed sha256",
                        o.id, a.path
                    ));
                }
            }
        }

        let mut row_ids = BTreeSet::new();
        let mut features = BTreeSet::new();
        for r in &self.rows {
            if !row_ids.insert(r.id.as_str()) {
                return inv(format!("duplicate row id `{}`", r.id));
            }
            if !features.insert(r.feature.as_str()) {
                return inv(format!("feature `{}` claimed by two rows", r.feature));
            }
            if !matches!(r.stage.as_str(), "S0" | "S1" | "S2" | "S3" | "S4") {
                return inv(format!("row `{}` has unknown stage `{}`", r.id, r.stage));
            }
            if !got_levels.contains(r.status.as_str()) {
                return inv(format!("row `{}` has unknown status `{}`", r.id, r.status));
            }
            // Status discipline: `open` is exactly the non-gating tier.
            let gating = self.level_gating(&r.status)?;
            match (gating, r.tier) {
                (true, Tier::Suite) | (false, Tier::Target) => {}
                (true, Tier::Target) => {
                    return inv(format!(
                        "row `{}` has gating status `{}` but non-gating tier `target`",
                        r.id, r.status
                    ));
                }
                (false, Tier::Suite) => {
                    return inv(format!(
                        "row `{}` is `open` (measurement-only) but tier `suite` would gate on it",
                        r.id
                    ));
                }
            }
            if r.oracles.is_empty() {
                return inv(format!("row `{}` cites no oracle", r.id));
            }
            for oid in &r.oracles {
                if !oracle_ids.contains(oid.as_str()) {
                    return inv(format!("row `{}` cites unknown oracle `{oid}`", r.id));
                }
            }
            if !r.feature.starts_with("features/suites/") || !r.feature.ends_with(".feature") {
                return inv(format!(
                    "row `{}` feature path `{}` is outside features/suites/",
                    r.id, r.feature
                ));
            }
        }

        let mut usecase_ids = BTreeSet::new();
        for u in &self.usecases {
            if !usecase_ids.insert(u.id.as_str()) {
                return inv(format!("duplicate use-case id `{}`", u.id));
            }
            if !u.oracle.is_empty() && !oracle_ids.contains(u.oracle.as_str()) {
                return inv(format!(
                    "use-case `{}` cites unknown oracle `{}`",
                    u.id, u.oracle
                ));
            }
        }
        self.canonical_usecase()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_loads_and_validates() {
        let model = Model::load().expect("the committed conceptual model must validate");
        assert!(model.rows.len() >= 20, "the dictionary must stay populated");
        model.canonical_usecase().expect("one canonical instance");
    }

    #[test]
    fn open_rows_never_gate() {
        let model = Model::load().expect("model");
        for row in &model.rows {
            if row.status == "open" {
                assert_eq!(row.tier, Tier::Target, "open row `{}` must be target", row.id);
            }
        }
    }
}
