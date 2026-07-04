#!/usr/bin/env bash
# The anti-hardcode gate (dictionary row `parametricity`): no canonical-instance
# constant may appear in generic pipeline code. Model identity lives in data
# (model/*.toml, the web catalogue JSON) and in tests — never in code paths.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0

# Strip inline `#[cfg(test)] mod tests` tails from a Rust file, then grep.
check_rust() {
    local file="$1" pattern="$2" label="$3"
    local body
    body=$(awk '/#\[cfg\(test\)\]/{exit} {print}' "$file")
    if grep -nE "$pattern" <<<"$body" >/dev/null; then
        echo "anti-hardcode: $label leaked into $file:"
        grep -nE "$pattern" <<<"$body" | head -5
        fail=1
    fi
}

# Canonical model dimensions/ids must not appear in pipeline code.
DIM_PATTERN='\b(4096|32000|49152|151936|151643|151645)\b'
# Model identities must not appear in Rust pipeline code.
ID_PATTERN='SmolLM2|TinyLlama|Qwen2\.5|phi-4|HuggingFaceTB'

for f in crates/hologram-ai-safetensors/src/*.rs crates/hologram-ai-wasm/src/*.rs; do
    check_rust "$f" "$DIM_PATTERN" "a canonical model dimension"
    check_rust "$f" "$ID_PATTERN" "a model identity"
done

# The web app: repo ids belong in the catalogue data file, not in TS code.
for f in $(find apps/web/src -name '*.ts' -o -name '*.tsx' | grep -v -E '\.test\.|/bdd/'); do
    if grep -nE 'huggingface\.co/(HuggingFaceTB|Qwen|TinyLlama|microsoft)/' "$f" >/dev/null 2>&1 ||
       grep -nE '"(HuggingFaceTB|Qwen)/[A-Za-z0-9.-]+"' "$f" >/dev/null 2>&1; then
        echo "anti-hardcode: a model identity leaked into $f:"
        grep -nE '(HuggingFaceTB|Qwen)/[A-Za-z0-9.-]+' "$f" | head -5
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "anti-hardcode: FAILED — move the constant into config/data or a test."
    exit 1
fi
echo "anti-hardcode: clean."
