use std::collections::HashMap;
use super::dim_expr::{DimExpr, DimVarId};
use super::constraint::ShapeError;

/// Where a dimension variable was introduced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimVarSource {
    /// Imported from ONNX dim_param or GGUF metadata.
    Import,
    /// Inferred by shape propagation.
    Inferred,
    /// Specified by user configuration (e.g., --max-seq-len).
    UserConfig,
}

/// A named dimension variable with optional bounds.
#[derive(Debug, Clone)]
pub struct DimVarEntry {
    /// Human-readable name (e.g., "batch", "seq_len").
    pub name: String,
    /// Inclusive lower bound. None means unbounded below (treated as 0).
    pub lower: Option<u64>,
    /// Inclusive upper bound. None means unbounded above.
    pub upper: Option<u64>,
    /// If Some, this variable is fixed to a concrete value.
    pub fixed: Option<u64>,
    /// Where this variable was defined.
    pub source: DimVarSource,
}

/// Registry of all dimension variables in an `AiGraph`.
/// Variables are interned: each unique name maps to exactly one `DimVarId`.
#[derive(Debug, Clone, Default)]
pub struct DimVarTable {
    entries: Vec<DimVarEntry>,
    name_to_id: HashMap<String, DimVarId>,
}

impl DimVarTable {
    /// Intern a variable name, returning its ID.
    /// If the name already exists, returns the existing ID.
    pub fn intern(&mut self, name: &str) -> DimVarId {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = DimVarId(self.entries.len() as u32);
        self.entries.push(DimVarEntry {
            name: name.to_owned(),
            lower: None,
            upper: None,
            fixed: None,
            source: DimVarSource::Import,
        });
        self.name_to_id.insert(name.to_owned(), id);
        id
    }

    /// Intern with bounds. If the variable exists, tightens bounds
    /// (max of lowers, min of uppers — intersection semantics).
    pub fn intern_with_bounds(
        &mut self,
        name: &str,
        lower: Option<u64>,
        upper: Option<u64>,
        source: DimVarSource,
    ) -> DimVarId {
        let id = self.intern(name);
        let entry = &mut self.entries[id.0 as usize];
        // Tighten bounds via intersection.
        entry.lower = match (entry.lower, lower) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        entry.upper = match (entry.upper, upper) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        entry.source = source;
        id
    }

    /// Fix a variable to a concrete value. Validates against bounds.
    pub fn fix(&mut self, id: DimVarId, value: u64) -> Result<(), ShapeError> {
        let entry = &mut self.entries[id.0 as usize];
        if let Some(lo) = entry.lower {
            if value < lo {
                return Err(ShapeError::BoundsViolation {
                    var: entry.name.clone(),
                    value,
                    lower: Some(lo),
                    upper: entry.upper,
                });
            }
        }
        if let Some(hi) = entry.upper {
            if value > hi {
                return Err(ShapeError::BoundsViolation {
                    var: entry.name.clone(),
                    value,
                    lower: entry.lower,
                    upper: Some(hi),
                });
            }
        }
        entry.fixed = Some(value);
        Ok(())
    }

    /// Look up a variable by ID.
    pub fn get(&self, id: DimVarId) -> &DimVarEntry {
        &self.entries[id.0 as usize]
    }

    /// Look up a variable by name.
    pub fn lookup(&self, name: &str) -> Option<DimVarId> {
        self.name_to_id.get(name).copied()
    }

    /// Produce a substitution map for all fixed variables.
    pub fn fixed_substitutions(&self) -> HashMap<DimVarId, DimExpr> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                e.fixed.map(|v| (DimVarId(i as u32), DimExpr::Concrete(v)))
            })
            .collect()
    }

    /// Concretize all unfixed variables to their upper bound (MVP lowering).
    /// Fails if any variable has no upper bound.
    pub fn concretize_to_upper(&mut self) -> Result<(), ShapeError> {
        for entry in &mut self.entries {
            if entry.fixed.is_some() {
                continue;
            }
            match entry.upper {
                Some(hi) => entry.fixed = Some(hi),
                None => {
                    return Err(ShapeError::NoBound {
                        var: entry.name.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Iterate all variables.
    pub fn iter(&self) -> impl Iterator<Item = (DimVarId, &DimVarEntry)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, e)| (DimVarId(i as u32), e))
    }

    /// Returns the number of interned variables.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if no variables have been interned.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_returns_same_id() {
        let mut t = DimVarTable::default();
        let a = t.intern("batch");
        let b = t.intern("batch");
        assert_eq!(a, b);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn intern_with_bounds_tightens() {
        let mut t = DimVarTable::default();
        t.intern_with_bounds("seq_len", Some(1), Some(4096), DimVarSource::Import);
        t.intern_with_bounds("seq_len", Some(128), Some(2048), DimVarSource::UserConfig);
        let id = t.lookup("seq_len").unwrap();
        let e = t.get(id);
        assert_eq!(e.lower, Some(128));
        assert_eq!(e.upper, Some(2048));
    }

    #[test]
    fn fix_within_bounds() {
        let mut t = DimVarTable::default();
        let id = t.intern_with_bounds("batch", Some(1), Some(64), DimVarSource::Import);
        assert!(t.fix(id, 32).is_ok());
        assert_eq!(t.get(id).fixed, Some(32));
    }

    #[test]
    fn fix_out_of_bounds() {
        let mut t = DimVarTable::default();
        let id = t.intern_with_bounds("batch", Some(1), Some(64), DimVarSource::Import);
        assert!(t.fix(id, 100).is_err());
    }

    #[test]
    fn concretize_to_upper() {
        let mut t = DimVarTable::default();
        t.intern_with_bounds("batch", Some(1), Some(32), DimVarSource::Import);
        t.intern_with_bounds("seq_len", Some(1), Some(2048), DimVarSource::Import);
        t.concretize_to_upper().unwrap();
        let subs = t.fixed_substitutions();
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn concretize_fails_without_upper() {
        let mut t = DimVarTable::default();
        t.intern("dynamic_dim");
        assert!(t.concretize_to_upper().is_err());
    }
}
