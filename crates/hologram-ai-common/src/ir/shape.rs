use smallvec::SmallVec;

/// A dimension in a tensor shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Dim {
    /// Known at compile / lowering time.
    Concrete(u64),
    /// Named symbolic dim (e.g. `"batch"`, `"seq_len"`).
    Symbolic(String),
    /// Fully unknown; resolved at runtime.
    Dynamic,
}

impl Dim {
    /// Returns the concrete value if known.
    pub fn as_concrete(&self) -> Option<u64> {
        match self {
            Dim::Concrete(v) => Some(*v),
            _ => None,
        }
    }
}

impl From<u64> for Dim {
    fn from(v: u64) -> Self { Dim::Concrete(v) }
}

impl From<usize> for Dim {
    fn from(v: usize) -> Self { Dim::Concrete(v as u64) }
}

/// Compact tensor shape — up to 6 dims inline before heap allocation.
pub type Shape = SmallVec<[Dim; 6]>;

/// Construct a concrete `Shape` from a slice of `u64`.
pub fn shape_from_concrete(dims: &[u64]) -> Shape {
    dims.iter().copied().map(Dim::Concrete).collect()
}
