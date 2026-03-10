//! Execution context: typed metadata carried in `.holo` archives and at runtime.
//!
//! This module defines three key abstractions:
//!
//! 1. **`ExecContext`** — a trait for typed metadata sections embedded in compiled
//!    archives. Each implementation maps to a unique archive section. The compiler
//!    produces contexts at build time; the runtime reads them before execution.
//!
//! 2. **`ContextBundle`** — a composable container that collects multiple archive
//!    sections during compilation and embeds them in a single rebuild pass.
//!
//! 3. **`RuntimeContext`** — a trait for execution-time state that ops can read
//!    and write during graph evaluation (e.g., resolved dimensions, modality
//!    metadata from vision encoders).
//!
//! **Design intent:** `ExecContext` and `RuntimeContext` are reference definitions
//! that will migrate to the hologram base crate once the ecosystem adopts them.
//! For now they live here as the canonical contract.

use std::any::Any;
use std::collections::{BTreeMap, HashMap};

use hologram::hologram_archive::section::{EmbeddableSection, SECTION_CUSTOM_BASE};

/// Section kind for shape recipe metadata.
pub const SECTION_SHAPE_RECIPE: u32 = SECTION_CUSTOM_BASE + 0x20;

// ── ExecContext trait ───────────────────────────────────────────────────────

/// Typed execution context carried inside a `.holo` archive.
///
/// Extends [`EmbeddableSection`] with deserialization, enabling the runtime
/// to reconstruct typed metadata from raw archive bytes.
pub trait ExecContext: EmbeddableSection + Send + Sync + 'static {
    /// Unique section identifier (for static dispatch / deserialization).
    fn section_id() -> u32
    where
        Self: Sized;

    /// Deserialize from archive section bytes.
    fn from_context_bytes(bytes: &[u8]) -> anyhow::Result<Self>
    where
        Self: Sized;
}

// ── ContextBundle ───────────────────────────────────────────────────────────

/// Composable container for multiple archive sections.
///
/// Collects [`EmbeddableSection`] implementations during compilation,
/// then embeds them all into the archive in a single rebuild pass.
/// Sections are serialized eagerly on [`insert`](Self::insert) and stored as
/// plain bytes, keyed by section kind. [`BTreeMap`] guarantees deterministic
/// ordering in the archive.
pub struct ContextBundle {
    sections: BTreeMap<u32, Vec<u8>>,
}

impl ContextBundle {
    /// Create an empty bundle.
    pub fn new() -> Self {
        Self {
            sections: BTreeMap::new(),
        }
    }

    /// Insert or replace a section. Serializes immediately.
    pub fn insert(&mut self, section: &dyn EmbeddableSection) {
        self.sections
            .insert(section.section_kind(), section.to_bytes());
    }

    /// Insert a pre-serialized section by kind.
    pub fn insert_raw(&mut self, kind: u32, bytes: Vec<u8>) {
        self.sections.insert(kind, bytes);
    }

    /// Check whether a section kind is present.
    pub fn contains(&self, kind: u32) -> bool {
        self.sections.contains_key(&kind)
    }

    /// Get raw bytes for a section kind.
    pub fn get_raw(&self, kind: u32) -> Option<&[u8]> {
        self.sections.get(&kind).map(|v| v.as_slice())
    }

    /// Deserialize a typed [`ExecContext`] from the bundle.
    pub fn get<T: ExecContext>(&self) -> anyhow::Result<Option<T>> {
        match self.sections.get(&T::section_id()) {
            Some(bytes) => T::from_context_bytes(bytes).map(Some),
            None => Ok(None),
        }
    }

    /// Iterate over all `(kind, bytes)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &[u8])> {
        self.sections.iter().map(|(&k, v)| (k, v.as_slice()))
    }

    /// Number of sections in the bundle.
    pub fn len(&self) -> usize {
        self.sections.len()
    }

    /// Whether the bundle is empty.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }
}

impl Default for ContextBundle {
    fn default() -> Self {
        Self::new()
    }
}

// ── RuntimeContext trait ─────────────────────────────────────────────────────

/// Runtime execution context for graph executors.
///
/// **Reference definition** — intended to migrate to the `hologram` base crate.
/// Runtime implementations live there; hologram-ai only defines the contract.
///
/// The executor creates one `RuntimeContext` per inference invocation.
/// Ops read dimension bindings and typed metadata; modality-specific producers
/// write values that downstream consumers read (e.g., a vision encoder writes
/// spatial dims that cross-attention reads).
pub trait RuntimeContext: Send + Sync {
    /// Resolve a named dimension variable (e.g., `"batch"`, `"seq_len"`).
    /// Returns `None` if the variable is unbound.
    fn dim_value(&self, name: &str) -> Option<u64>;

    /// Read a typed metadata value by key.
    /// Returns `None` if the key is absent or the type doesn't match.
    fn get<T: Send + Sync + 'static>(&self, key: &str) -> Option<&T>;

    /// Write a typed metadata value by key.
    /// Used by ops that produce dynamic information (e.g., vision encoder
    /// writes spatial dims that cross-attention reads).
    fn set<T: Send + Sync + 'static>(&mut self, key: &str, value: T);
}

/// Simple reference implementation of [`RuntimeContext`].
///
/// Uses string-keyed maps for both dimensions and typed metadata.
/// Intended for tests and as a blueprint for the hologram base crate runtime.
pub struct SimpleRuntimeContext {
    dims: HashMap<String, u64>,
    store: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl SimpleRuntimeContext {
    /// Create an empty runtime context.
    pub fn new() -> Self {
        Self {
            dims: HashMap::new(),
            store: HashMap::new(),
        }
    }

    /// Bind a dimension variable.
    pub fn bind_dim(&mut self, name: impl Into<String>, value: u64) {
        self.dims.insert(name.into(), value);
    }

    /// Bind all dimensions from a [`ShapeRecipeSection`] and concrete values.
    ///
    /// `values[i]` is bound to `recipes.dim_vars[i]`.
    pub fn bind_from_recipes(&mut self, recipes: &ShapeRecipeSection, values: &[u64]) {
        for (i, name) in recipes.dim_vars.iter().enumerate() {
            if let Some(&v) = values.get(i) {
                self.dims.insert(name.clone(), v);
            }
        }
    }
}

impl Default for SimpleRuntimeContext {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeContext for SimpleRuntimeContext {
    fn dim_value(&self, name: &str) -> Option<u64> {
        self.dims.get(name).copied()
    }

    fn get<T: Send + Sync + 'static>(&self, key: &str) -> Option<&T> {
        self.store.get(key)?.downcast_ref::<T>()
    }

    fn set<T: Send + Sync + 'static>(&mut self, key: &str, value: T) {
        self.store.insert(key.to_string(), Box::new(value));
    }
}

// ── Shape recipe types ──────────────────────────────────────────────────────

/// A recipe describing how to resolve deferred (symbolic) op parameters at
/// execution time.
///
/// Embedded in the archive so the runtime can bind dynamic dimensions
/// (batch, seq_len) and patch graph ops accordingly.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ShapeRecipeSection {
    /// Named dynamic dimension variables (e.g., `["batch", "seq_len"]`).
    /// Index in this vec is used by `ParamRecipe::DimVar`.
    pub dim_vars: Vec<String>,
    /// Per-node recipes — only nodes with deferred (symbolic) params get entries.
    pub node_recipes: Vec<NodeShapeRecipe>,
}

/// Shape recipe for a single graph node.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct NodeShapeRecipe {
    /// Index into the compiled graph's node array.
    pub node_index: u32,
    /// Recipes for each deferred parameter in the op.
    /// The meaning of each slot depends on the op type (e.g., MatMul: [m, k, n]).
    pub params: Vec<ParamRecipe>,
}

/// A single op parameter that may require runtime resolution.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum ParamRecipe {
    /// Concrete value — already baked into the graph op at compile time.
    Concrete(u64),
    /// Dynamic dimension variable — index into `ShapeRecipeSection::dim_vars`.
    DimVar(u32),
    /// Product of a dim var and a constant (e.g., `seq_len * 2048`).
    Product(u32, u64),
    /// Truly dynamic — runtime infers from buffer sizes. No compile-time info.
    RuntimeInferred,
}

impl ParamRecipe {
    /// Resolve this recipe given a binding function for dim vars.
    pub fn resolve(&self, bindings: &[u64]) -> Option<u64> {
        match self {
            ParamRecipe::Concrete(v) => Some(*v),
            ParamRecipe::DimVar(idx) => bindings.get(*idx as usize).copied(),
            ParamRecipe::Product(idx, factor) => bindings.get(*idx as usize).map(|v| v * factor),
            ParamRecipe::RuntimeInferred => None, // Caller must infer from buffer
        }
    }
}

impl ShapeRecipeSection {
    /// Create an empty recipe section with the given dim var names.
    pub fn new(dim_vars: Vec<String>) -> Self {
        Self {
            dim_vars,
            node_recipes: Vec::new(),
        }
    }

    /// Zero-copy access from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<&ArchivedShapeRecipeSection, rkyv::rancor::Error> {
        rkyv::access::<ArchivedShapeRecipeSection, rkyv::rancor::Error>(bytes)
    }

    /// Deserialize from raw bytes into an owned `ShapeRecipeSection`.
    pub fn deserialize_from(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
    }
}

impl EmbeddableSection for ShapeRecipeSection {
    fn section_kind(&self) -> u32 {
        SECTION_SHAPE_RECIPE
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("ShapeRecipeSection serialization")
            .to_vec()
    }
}

impl ExecContext for ShapeRecipeSection {
    fn section_id() -> u32 {
        SECTION_SHAPE_RECIPE
    }

    fn from_context_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Self::deserialize_from(bytes).map_err(|e| anyhow::anyhow!("deserialize ShapeRecipe: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_shape_recipe() {
        let section = ShapeRecipeSection {
            dim_vars: vec!["batch".into(), "seq_len".into()],
            node_recipes: vec![NodeShapeRecipe {
                node_index: 42,
                params: vec![
                    ParamRecipe::DimVar(1),      // m = seq_len
                    ParamRecipe::Concrete(2048), // k = 2048
                    ParamRecipe::Concrete(2048), // n = 2048
                ],
            }],
        };

        let bytes = section.to_bytes();

        // Zero-copy access.
        let archived = ShapeRecipeSection::from_bytes(&bytes).unwrap();
        assert_eq!(archived.dim_vars.len(), 2);
        assert_eq!(archived.dim_vars[0].as_str(), "batch");
        assert_eq!(archived.node_recipes.len(), 1);
        assert_eq!(archived.node_recipes[0].node_index, 42);

        // Full deserialization.
        let deserialized = ShapeRecipeSection::deserialize_from(&bytes).unwrap();
        assert_eq!(deserialized.dim_vars, vec!["batch", "seq_len"]);
        assert_eq!(deserialized.node_recipes[0].params.len(), 3);
    }

    #[test]
    fn param_recipe_resolve() {
        let bindings = vec![1, 512]; // batch=1, seq_len=512

        assert_eq!(ParamRecipe::Concrete(2048).resolve(&bindings), Some(2048));
        assert_eq!(ParamRecipe::DimVar(0).resolve(&bindings), Some(1));
        assert_eq!(ParamRecipe::DimVar(1).resolve(&bindings), Some(512));
        assert_eq!(ParamRecipe::Product(1, 64).resolve(&bindings), Some(32768));
        assert_eq!(ParamRecipe::DimVar(99).resolve(&bindings), None);
    }

    #[test]
    fn embeddable_section_impl() {
        let section = ShapeRecipeSection::new(vec!["batch".into()]);
        assert_eq!(section.section_kind(), SECTION_SHAPE_RECIPE);
    }

    #[test]
    fn context_bundle_insert_and_get() {
        let recipes = ShapeRecipeSection {
            dim_vars: vec!["batch".into(), "seq_len".into()],
            node_recipes: vec![NodeShapeRecipe {
                node_index: 7,
                params: vec![ParamRecipe::DimVar(0)],
            }],
        };

        let mut bundle = ContextBundle::new();
        assert!(bundle.is_empty());

        bundle.insert(&recipes);
        assert_eq!(bundle.len(), 1);
        assert!(bundle.contains(SECTION_SHAPE_RECIPE));

        // Round-trip via typed get.
        let recovered: ShapeRecipeSection = bundle.get::<ShapeRecipeSection>().unwrap().unwrap();
        assert_eq!(recovered.dim_vars, vec!["batch", "seq_len"]);
        assert_eq!(recovered.node_recipes[0].node_index, 7);
    }

    #[test]
    fn context_bundle_insert_raw_and_iter() {
        let mut bundle = ContextBundle::new();
        bundle.insert_raw(0x100, vec![1, 2, 3]);
        bundle.insert_raw(0x200, vec![4, 5]);

        let entries: Vec<_> = bundle.iter().collect();
        assert_eq!(entries.len(), 2);
        // BTreeMap: sorted by kind.
        assert_eq!(entries[0].0, 0x100);
        assert_eq!(entries[1].0, 0x200);
        assert_eq!(entries[0].1, &[1, 2, 3]);
    }

    #[test]
    fn context_bundle_replace_section() {
        let mut bundle = ContextBundle::new();
        bundle.insert_raw(0x100, vec![1, 2, 3]);
        bundle.insert_raw(0x100, vec![7, 8, 9]);
        assert_eq!(bundle.len(), 1);
        assert_eq!(bundle.get_raw(0x100).unwrap(), &[7, 8, 9]);
    }

    #[test]
    fn simple_runtime_context_dims() {
        let mut ctx = SimpleRuntimeContext::new();
        ctx.bind_dim("batch", 4);
        ctx.bind_dim("seq_len", 512);

        assert_eq!(ctx.dim_value("batch"), Some(4));
        assert_eq!(ctx.dim_value("seq_len"), Some(512));
        assert_eq!(ctx.dim_value("unknown"), None);
    }

    #[test]
    fn simple_runtime_context_typed_store() {
        let mut ctx = SimpleRuntimeContext::new();

        ctx.set("vision.spatial_dims", (14u64, 14u64));
        ctx.set("vision.patch_count", 196u64);

        let spatial = ctx.get::<(u64, u64)>("vision.spatial_dims").unwrap();
        assert_eq!(*spatial, (14, 14));

        let count = ctx.get::<u64>("vision.patch_count").unwrap();
        assert_eq!(*count, 196);

        // Wrong type returns None.
        assert!(ctx.get::<String>("vision.patch_count").is_none());

        // Missing key returns None.
        assert!(ctx.get::<u64>("nonexistent").is_none());
    }

    #[test]
    fn simple_runtime_context_bind_from_recipes() {
        let recipes = ShapeRecipeSection::new(vec!["batch".into(), "seq_len".into()]);
        let mut ctx = SimpleRuntimeContext::new();
        ctx.bind_from_recipes(&recipes, &[1, 2048]);

        assert_eq!(ctx.dim_value("batch"), Some(1));
        assert_eq!(ctx.dim_value("seq_len"), Some(2048));
    }
}
