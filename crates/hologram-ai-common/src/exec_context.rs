//! Archive section abstraction for hologram-ai.
//!
//! hologram-ai attaches model-level metadata (component descriptors, vision
//! patch-prune config, …) to the compiled `.holo` archive as **custom
//! sections** — raw byte blobs keyed by a section kind. hologram's archive
//! exposes a flat `SectionKind`+bytes model (architecture §7/§8); the
//! hologram-ai compiler maps each [`Section`] here onto a custom archive
//! section at write time. This module owns only the byte abstraction, so the
//! runtime-core crate carries no archive dependency.
//!
//! Runtime shape resolution does not exist in the UOR-native model: hologram's
//! compiler derives every op parameter from the graph's concrete interned
//! shapes (`ShapeArgs::from_graph`). hologram-ai therefore supplies concrete
//! shapes at lowering time and carries no shape-projection / recipe / runtime
//! dim-binding machinery (architecture §5.1, §5.3).

use std::collections::BTreeMap;

/// Base for hologram-ai custom section kinds. Chosen well above hologram's own
/// reserved range so the two never collide when mapped into the archive.
pub const SECTION_CUSTOM_BASE: u32 = 0x6800;

/// Section kind for ViT patch-prune configuration.
pub const SECTION_PATCH_PRUNE: u32 = SECTION_CUSTOM_BASE + 0x22;

/// A typed model-level metadata blob embedded as a custom archive section.
pub trait Section {
    /// Unique custom-section kind.
    fn section_kind(&self) -> u32;
    /// Serialize to the section's wire bytes.
    fn to_bytes(&self) -> Vec<u8>;
}

/// Composable container collecting [`Section`] blobs during compilation, keyed
/// by section kind. [`BTreeMap`] gives deterministic archive ordering.
#[derive(Default)]
pub struct ContextBundle {
    sections: BTreeMap<u32, Vec<u8>>,
}

impl ContextBundle {
    pub fn new() -> Self {
        Self {
            sections: BTreeMap::new(),
        }
    }

    /// Insert or replace a section. Serializes immediately; empty blobs are skipped.
    pub fn insert(&mut self, section: &dyn Section) {
        let bytes = section.to_bytes();
        if bytes.is_empty() {
            return;
        }
        self.sections.insert(section.section_kind(), bytes);
    }

    /// Insert a pre-serialized section by kind.
    pub fn insert_raw(&mut self, kind: u32, bytes: Vec<u8>) {
        self.sections.insert(kind, bytes);
    }

    pub fn contains(&self, kind: u32) -> bool {
        self.sections.contains_key(&kind)
    }

    pub fn get_raw(&self, kind: u32) -> Option<&[u8]> {
        self.sections.get(&kind).map(|v| v.as_slice())
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, &[u8])> {
        self.sections.iter().map(|(&k, v)| (k, v.as_slice()))
    }

    pub fn len(&self) -> usize {
        self.sections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }
}

/// Configuration for runtime patch pruning (PixelPrune), emitted by the
/// `PatchPruneInjection` pass and read by the host runtime to preprocess the
/// image input before feeding the compiled ViT graph.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct PatchPruneContext {
    pub kept_indices_input: u32,
    pub pixel_input: u32,
    pub channels: u32,
    pub patch_h: u32,
    pub patch_w: u32,
    pub total_patches: u32,
    pub max_kept: u32,
}

impl PatchPruneContext {
    /// Deserialize from section bytes.
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let archived = rkyv::access::<rkyv::Archived<Self>, rkyv::rancor::Error>(bytes)
            .map_err(|e| anyhow::anyhow!("access PatchPruneContext: {e}"))?;
        rkyv::deserialize::<Self, rkyv::rancor::Error>(archived)
            .map_err(|e| anyhow::anyhow!("deserialize PatchPruneContext: {e}"))
    }
}

impl Section for PatchPruneContext {
    fn section_kind(&self) -> u32 {
        SECTION_PATCH_PRUNE
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map(|b| b.to_vec())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_patch_prune_context() {
        let ctx = PatchPruneContext {
            kept_indices_input: 1,
            pixel_input: 0,
            channels: 3,
            patch_h: 16,
            patch_w: 16,
            total_patches: 196,
            max_kept: 147,
        };
        let bytes = ctx.to_bytes();
        let recovered = PatchPruneContext::from_bytes(&bytes).expect("deserialize");
        assert_eq!(recovered, ctx);
    }

    #[test]
    fn context_bundle_collects_sections() {
        let mut bundle = ContextBundle::new();
        assert!(bundle.is_empty());
        bundle.insert(&PatchPruneContext {
            kept_indices_input: 1,
            pixel_input: 0,
            channels: 3,
            patch_h: 16,
            patch_w: 16,
            total_patches: 196,
            max_kept: 147,
        });
        assert_eq!(bundle.len(), 1);
        assert!(bundle.contains(SECTION_PATCH_PRUNE));
    }

    #[test]
    fn context_bundle_deterministic_order() {
        let mut bundle = ContextBundle::new();
        bundle.insert_raw(0x200, vec![4, 5]);
        bundle.insert_raw(0x100, vec![1, 2, 3]);
        let entries: Vec<_> = bundle.iter().collect();
        assert_eq!(entries[0].0, 0x100);
        assert_eq!(entries[1].0, 0x200);
    }
}
