# Plan 069: Unified Streaming Archive Path

**Status:** Planned
**Created:** 2026-04-16
**Depends on:** Plan 067 (ComputeBackend rewrite)

## Problem

Two archive compilation paths exist:
1. **In-memory** (`compile_components` → `PipelineWriter::build()`) — used for LLM
2. **Streaming** (`PipelineWriter::build_to_file()`) — used for non-LLM single models

The TinyLlama "constant not found: 0" bug manifests because:
- Sub-archives for decode/verify have empty weight regions
- The loader expects deferred constants to resolve from the weight region
- The weight borrowing between sub-archives may not work correctly

## Solution

**Single path: always streaming, always pipeline.**

1. Remove `compile_components` (in-memory `PipelineWriter::build()`)
2. All compilations go through `PipelineWriter::build_to_file()`
3. Weights are always streamed from a `WeightSource` (file or mmap)
4. Extra sections (tokenizer, host meta) embedded via PipelineWriter
5. All archives are pipeline format with SECTION_PIPELINE header

### LLM Path Changes

Currently:
```
prefill graph → compile_one_component(weights=full) → Vec<u8>
decode graph  → compile_one_component(weights=None) → Vec<u8>
verify graph  → compile_one_component(weights=None) → Vec<u8>
PipelineWriter::build() → Vec<u8> → write to disk
```

New:
```
prefill graph → compile_one_component(weights=empty) → Vec<u8> (graph only)
decode graph  → compile_one_component(weights=empty) → Vec<u8> (graph only)
verify graph  → compile_one_component(weights=empty) → Vec<u8> (graph only)
PipelineWriter::new()
  .add_model_streaming("prefill", sub_archive, weight_source)
  .add_model("decode", sub_archive)
  .add_model("verify", sub_archive)
  .add_section(tokenizer)
  .build_to_file(output_path, scratch_path)
```

Only the first model gets `add_model_streaming` with the weight source.
Other models use `add_model` with their graph-only sub-archives.

## Acceptance Criteria

- TinyLlama compiles and generates coherent text at ≥40 tok/s
- SD UNet compiles and runs forward pass in ≤5s
- Only one archive compilation path exists
- All archives are pipeline format
