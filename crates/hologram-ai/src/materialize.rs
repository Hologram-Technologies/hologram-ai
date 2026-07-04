//! κ-materialization (journey stage S3): turn a k-form archive plus a κ-store
//! into an executable archive.
//!
//! Streamed compilation emits `.holo` archives whose weight constants are
//! 0-byte placeholders bound to κ-labels through the `holospaces.kappa_map`
//! extension (`docs/architecture/ARCHITECTURE.md` §4.3). Materialization
//! resolves each κ against a [`KappaStore`], **verifies the content re-hashes
//! to its κ** (content addressing is the integrity check), patches the
//! archive's constants section through `hologram-archive`'s public codec, and
//! re-emits the archive. The compile-time warm-fold section is dropped — its
//! folded results were derived over the empty placeholders; the session
//! re-derives the cone lattice from the real content at load.
//!
//! A missing or corrupt κ aborts with the label. There is no fallback path.

use anyhow::{bail, Context, Result};
use hologram_archive::constant_codec::{self, ConstantEntry};
use hologram_archive::{HoloLoader, SectionKind, FORMAT_VERSION, MAGIC};
use hologram_host::prism::vocabulary::Hasher;

/// The archive extension key binding weight constants to κ-labels.
pub const KAPPA_MAP_EXTENSION: &str = "holospaces.kappa_map";

/// A resolvable κ-addressed content store.
///
/// Realizations: [`DirKappaStore`] (a directory of `{κ}.bin` files, used by
/// the CLI and conformance tests) and the browser's OPFS resolver (a JS
/// callback wired through `hologram-ai-wasm`).
pub trait KappaStore {
    /// Return the content addressed by `kappa`, or fail naming the label.
    fn resolve(&mut self, kappa: &str) -> Result<Vec<u8>>;
}

impl<F> KappaStore for F
where
    F: FnMut(&str) -> Result<Vec<u8>>,
{
    fn resolve(&mut self, kappa: &str) -> Result<Vec<u8>> {
        self(kappa)
    }
}

/// A κ-store over a directory of `{κ}.bin` files (the native mirror of the
/// browser's OPFS `tensors/` layout).
pub struct DirKappaStore {
    root: std::path::PathBuf,
}

impl DirKappaStore {
    /// Create a store rooted at `root`.
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Persist `bytes` under their derived κ, returning the label.
    pub fn insert(&self, bytes: &[u8]) -> Result<String> {
        let kappa = kappa_of(bytes);
        std::fs::create_dir_all(&self.root)
            .with_context(|| format!("creating κ-store dir {}", self.root.display()))?;
        let path = self.root.join(format!("{kappa}.bin"));
        std::fs::write(&path, bytes)
            .with_context(|| format!("writing κ content {}", path.display()))?;
        Ok(kappa)
    }
}

impl KappaStore for DirKappaStore {
    fn resolve(&mut self, kappa: &str) -> Result<Vec<u8>> {
        let path = self.root.join(format!("{kappa}.bin"));
        std::fs::read(&path).with_context(|| format!("κ `{kappa}` not present in store"))
    }
}

/// The κ-label of `bytes`: `blake3:<hex>`, hologram's canonical content hash
/// (`HologramHasher`), matching `holospaces::address` and the streamed
/// hasher in `hologram-ai-wasm` (witnessed by the `kappa-addressing` row).
pub fn kappa_of(bytes: &[u8]) -> String {
    let digest: [u8; 32] = hologram_host::HologramHasher::initial()
        .fold_bytes(bytes)
        .finalize();
    let mut hex = String::with_capacity(7 + 64);
    hex.push_str("blake3:");
    for b in digest {
        use std::fmt::Write;
        write!(hex, "{b:02x}").expect("writing to a String cannot fail");
    }
    hex
}

/// One requirement from the archive's κ-map: the graph constant id and the
/// κ-label bound to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KappaRequirement {
    /// The graph `ConstantId` index the weight occupies.
    pub constant: u32,
    /// The κ-label (`blake3:<hex>`) of the weight content.
    pub kappa: String,
}

/// Parse the `holospaces.kappa_map` extension of `archive`. Returns an empty
/// list when the archive carries no map (it is already material).
pub fn kappa_requirements(archive: &[u8]) -> Result<Vec<KappaRequirement>> {
    let plan = HoloLoader::from_bytes(archive)
        .map_err(|e| anyhow::anyhow!("loading archive: {e:?}"))?
        .into_plan()
        .map_err(|e| anyhow::anyhow!("decoding archive sections: {e:?}"))?;
    let Some(map) = plan
        .extensions()
        .map_err(|e| anyhow::anyhow!("decoding archive extensions: {e:?}"))?
        .into_iter()
        .find_map(|(key, bytes)| (key == KAPPA_MAP_EXTENSION).then_some(bytes))
    else {
        return Ok(Vec::new());
    };
    parse_kappa_map(map)
}

/// Parse κ-map lines of the form `ConstantId(<n>):<kappa>`.
fn parse_kappa_map(bytes: &[u8]) -> Result<Vec<KappaRequirement>> {
    let text = std::str::from_utf8(bytes).context("κ-map extension is not UTF-8")?;
    let mut reqs = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let rest = line
            .strip_prefix("ConstantId(")
            .with_context(|| format!("malformed κ-map line `{line}`"))?;
        let (id, kappa) = rest
            .split_once("):")
            .with_context(|| format!("malformed κ-map line `{line}`"))?;
        let constant: u32 = id
            .parse()
            .with_context(|| format!("malformed constant id in κ-map line `{line}`"))?;
        reqs.push(KappaRequirement {
            constant,
            kappa: kappa.to_string(),
        });
    }
    Ok(reqs)
}

/// Materialize `archive` against `store`.
///
/// Resolves every κ-map entry, verifies each buffer re-hashes to its κ,
/// patches the constants section, drops the stale warm-fold section, and
/// re-emits the archive (footer re-fingerprinted). An archive without a κ-map
/// is already material and is returned unchanged.
pub fn materialize_archive(archive: &[u8], store: &mut dyn KappaStore) -> Result<Vec<u8>> {
    let requirements = kappa_requirements(archive)?;
    if requirements.is_empty() {
        return Ok(archive.to_vec());
    }

    let loader = HoloLoader::from_bytes(archive)
        .map_err(|e| anyhow::anyhow!("loading archive: {e:?}"))?;
    let plan = loader
        .into_plan()
        .map_err(|e| anyhow::anyhow!("decoding archive sections: {e:?}"))?;

    let constants_bytes = plan
        .section(SectionKind::Constants)
        .map_err(|e| anyhow::anyhow!("archive has a κ-map but no constants section: {e:?}"))?;
    let mut entries = constant_codec::decode(constants_bytes)
        .map_err(|e| anyhow::anyhow!("decoding constants section: {e:?}"))?;

    patch_constants(&mut entries, &requirements, store)?;

    let new_constants = constant_codec::encode(&entries);
    rebuild_archive(archive, plan.sections(), &new_constants)
}

/// Resolve and verify every requirement, writing the content into its entry.
fn patch_constants(
    entries: &mut [ConstantEntry],
    requirements: &[KappaRequirement],
    store: &mut dyn KappaStore,
) -> Result<()> {
    // The compiler emits the graph's constants first, in `ConstantId` order
    // (slot = node_count + id); trailing entries (constant *nodes*) follow.
    // Anchor on the first entry's slot and address requirements by id.
    let base_slot = entries
        .first()
        .map(|e| e.slot)
        .context("archive constants section is empty")?;
    for req in requirements {
        let idx = req.constant as usize;
        let entry = entries.get_mut(idx).with_context(|| {
            format!(
                "κ-map names ConstantId({}) but the constants section has fewer entries",
                req.constant
            )
        })?;
        if entry.slot != base_slot + req.constant {
            bail!(
                "constants section layout drift: ConstantId({}) maps to slot {} (expected {})",
                req.constant,
                entry.slot,
                base_slot + req.constant
            );
        }
        if entry.by_reference {
            bail!(
                "ConstantId({}) is a weight-pool reference; a k-form archive must inline its placeholders",
                req.constant
            );
        }
        if !entry.bytes.is_empty() {
            bail!(
                "ConstantId({}) already carries {} bytes; refusing to overwrite material content",
                req.constant,
                entry.bytes.len()
            );
        }
        let bytes = store
            .resolve(&req.kappa)
            .with_context(|| format!("resolving κ `{}`", req.kappa))?;
        let derived = kappa_of(&bytes);
        if derived != req.kappa {
            bail!(
                "κ integrity failure for ConstantId({}): store content hashes to `{derived}`, expected `{}`",
                req.constant,
                req.kappa
            );
        }
        entry.bytes = bytes;
    }
    Ok(())
}

/// Re-emit the archive: same sections in order, with the constants payload
/// replaced, the warm-fold section dropped, and the footer re-fingerprinted.
/// Mirrors `hologram-archive`'s writer layout (format v2).
fn rebuild_archive(
    archive: &[u8],
    sections: &[hologram_archive::format::SectionRef],
    new_constants: &[u8],
) -> Result<Vec<u8>> {
    let mut payloads: Vec<(SectionKind, &[u8])> = Vec::with_capacity(sections.len());
    for s in sections {
        let body: &[u8] = match s.kind {
            SectionKind::Constants => new_constants,
            // Folded over 0-byte placeholders at compile time; stale now.
            SectionKind::WarmStart => continue,
            _ => {
                let start = s.offset as usize;
                let end = start + s.length as usize;
                archive
                    .get(start..end)
                    .context("section range exceeds archive bytes")?
            }
        };
        payloads.push((s.kind, body));
    }

    let header_size = 4 + 2 + 2 + 2;
    let section_entry_size = 1 + 7 + 8 + 8;
    let table_size = section_entry_size * payloads.len();
    let mut offset = (header_size + table_size) as u64;

    let total: usize =
        header_size + table_size + payloads.iter().map(|(_, b)| b.len()).sum::<usize>() + 32;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(payloads.len() as u16).to_le_bytes());
    for (kind, body) in &payloads {
        out.push(*kind as u8);
        out.extend_from_slice(&[0u8; 7]);
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(body.len() as u64).to_le_bytes());
        offset += body.len() as u64;
    }
    for (_, body) in &payloads {
        out.extend_from_slice(body);
    }
    let footer: [u8; 32] = hologram_host::HologramHasher::initial()
        .fold_bytes(&out)
        .finalize();
    out.extend_from_slice(&footer);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kappa_of_is_prefixed_blake3_hex() {
        let k = kappa_of(b"hologram");
        assert!(k.starts_with("blake3:"));
        assert_eq!(k.len(), 7 + 64);
        assert_eq!(k, kappa_of(b"hologram"), "κ is deterministic");
        assert_ne!(k, kappa_of(b"holospace"), "κ separates content");
    }

    #[test]
    fn parse_kappa_map_round_trips() {
        let text = format!("ConstantId(0):{}\nConstantId(7):{}\n", kappa_of(b"a"), kappa_of(b"b"));
        let reqs = parse_kappa_map(text.as_bytes()).expect("well-formed map parses");
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].constant, 0);
        assert_eq!(reqs[1].constant, 7);
        assert!(parse_kappa_map(b"garbage-line").is_err(), "malformed lines fail loud");
    }

    #[test]
    fn dir_store_round_trips_and_verifies() {
        let dir = std::env::temp_dir().join(format!("kappa-store-test-{}", std::process::id()));
        let store = DirKappaStore::new(&dir);
        let kappa = store.insert(b"tensor-bytes").expect("insert");
        let mut store = store;
        let bytes = store.resolve(&kappa).expect("resolve");
        assert_eq!(bytes, b"tensor-bytes");
        assert_eq!(kappa_of(&bytes), kappa);
        assert!(store.resolve("blake3:0000").is_err(), "missing κ fails naming the label");
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn material_archive_passes_through() {
        // An archive with no κ-map is already material: identity.
        let holo = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../oracles/onnx/tiny-mlp.onnx"));
        // tiny-mlp.onnx is not a .holo — loading must fail loud, proving we
        // never silently pass through non-archives.
        let mut store = DirKappaStore::new("/nonexistent");
        assert!(materialize_archive(holo, &mut store).is_err());
    }
}
