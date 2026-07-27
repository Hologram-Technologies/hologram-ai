# R4 inference bundle — schema v1 (normative)

The bundle is the single deterministic, opaque model blob stored as a
Hologram content blob per `InferenceModel` layer (ADR-0005). Implementation:
`hologram-ai-bundle`. All multi-byte integers are little-endian.

## Wire format

```text
offset  size  field
0       4     magic b"R4IB"
4       2     u16 bundle schema version (= 1)
6       4     u32 manifest length N
10      N     canonical InferenceModelManifest bytes (hologram-ai-core::canon)
10+N    …     component payloads, concatenated with no padding, in the exact
              order of manifest.artifacts (canonical ArtifactRole order)
```

- No padding, timestamps, filesystem paths, or random identifiers.
- `manifest.artifacts` carries, per component: semantic role (u16), BLAKE3
  digest (32 bytes), length (u64).
- Canonical manifest encoding: see `hologram-ai-core::canon` —
  length-prefixed, declaration-ordered, map-free. Identical inputs produce
  byte-identical bundles.

## Limits and parser rules

- `MAX_BUNDLE_SIZE = 1 << 32` (4 GiB), enforced as checked `u64`.
- `MAX_COMPONENTS = 11` (= `ArtifactRole::ALL.len()`); duplicate roles are
  rejected.
- Every offset/length uses checked arithmetic; truncated input, trailing
  bytes, hostile length prefixes, unknown discriminants, and bad magic are
  typed errors, never panics.
- A manifest whose embedded `schema_version ≠ MANIFEST_SCHEMA_VERSION` is an
  `AbiMismatch`; a bundle header version mismatch is an `AbiMismatch`.
- `verify_digests` MUST pass (every component BLAKE3-verified against the
  manifest) before any engine initialization (`IntegrityMismatch`).
- `parse_verified` = `parse` + `verify_digests` and is the recommended path.

## Component roles (closed, versioned vocabulary)

| Role | Mandatory | Contents |
|---|---|---|
| `graph` | yes (uor-r4/R4G1) | validated scored R4G1 graph |
| `signature-artifact` | yes (uor-r4/R4G1) | signature/input-projection artifact (the artifact historically named `tless_artifacts.bin`; that name is never public) |
| `score-report` | yes (uor-r4/R4G1) | complete R4 score report |
| `tokenizer` | iff text in/out | byte-level BPE tokenizer |
| `image-processor` | iff image input | image processor contract |
| `audio-processor` | iff audio input | audio processor + sample-format contract |
| `generation-config` | optional | generation configuration |
| `vocabulary-metadata` | optional | vocabulary metadata |
| `label-map` | optional | label maps |
| `normalization-metadata` | optional | modality normalization metadata |
| `witnesses` | optional | certification / witness artifacts |

Mandatory-role rules apply when `engine == "uor-r4" && artifact_format ==
"R4G1"`; unknown engines skip them (forward compatibility).

## Capability ⇔ processor consistency

Validated identically at build and load:

- any `Text` value in any operation ⇒ `tokenizer` required;
- a token-only model (no `Text` anywhere) may omit the tokenizer;
- any `Image` value ⇒ `image-processor` required;
- any `Audio` value ⇒ `audio-processor` required;
- a multimodal model carries every processor its operations require.

Violations are `ProcessorMismatch` errors at load.

## Excluded (never packaged)

`tless_store.bin`, corpus metadata/records, observation shards, cover
artifacts/reports, resumable checkpoints, temporary directories, source
model weights (unless the deployed runtime requires them), source/cache
absolute paths, wall-clock timestamps.

## Identity and provenance

`bundle_digest` (BLAKE3 over the whole bundle) is the model identity; the
`InferenceModel` layer's content κ addresses exactly these bytes. Manifest
provenance records: source repository, immutable source revision, source
content digest, compiler revision, compile-options digest, quality-report
digest, and ABI versions (compiler ABI, R4G1 format, inference contract,
`.holo` format).
