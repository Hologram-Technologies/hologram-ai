# Hugging Face acquisition

Owned by `hologram-ai-huggingface` (ADR-0006). `uor-r4` never networks; it
receives a verified local directory.

## CLI (via `hologram ai`)

```bash
hologram ai download HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision <full-commit-sha>

hologram ai compile HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision <full-commit-sha> --output smollm2.holo

hologram ai compile --source /path/to/model-source --output model.holo
```

## Immutability policy

- Default: a full 40-hex commit SHA is required.
- Branch/tag resolution happens only when explicitly requested; the resolved
  immutable commit is recorded in provenance and used for the cache key.
- A local source directory is validated (config + safetensors + tokenizer)
  and hashed as-is.

## Cache layout

```text
<cache>/hf/<blake3(repo + '\0' + revision)>/     # complete entry + manifest marker
<cache>/hf/.staging-<key>/                       # resumable in-progress download
<cache>/hf/.lock-<key>                           # per-entry lock
```

Downloads stage into `.staging-*` and complete by atomic rename after all
expected files validate. An interrupted staging directory is never a cache
hit; rerunning resumes it. Concurrent processes take the per-entry lock.

## Security

- The official `hf` executable is invoked with opaque argv (no shell, no
  interpolation).
- Tokens come from the environment (`HF_TOKEN` or a caller-supplied
  provider), are passed to the child environment only, and are never logged,
  persisted, or included in errors/progress.
- Repository identifiers and revisions are strictly validated; path
  traversal and symlink escapes are rejected.
- `--offline` serves cache hits and fails cache misses without network.

## Support honesty

Not every HF model is compilable: the model configuration is checked
against the uor-r4 compatibility surface, and unsupported models return a
typed `UnsupportedModel` / `UnsupportedCapability` error.
