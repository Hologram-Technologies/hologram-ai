#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_PATH="${ROOT_DIR}/models/t5-small/t5-pipeline.holo"
PROMPT="tell me a joke"

if [[ ! -f "${PIPELINE_PATH}" ]]; then
  echo "Missing pipeline: ${PIPELINE_PATH}" >&2
  exit 1
fi

echo "Benchmark: max_tokens=1"
echo "Pipeline: ${PIPELINE_PATH}"
echo "Prompt: ${PROMPT}"
echo ""

command -v time >/dev/null 2>&1 || {
  echo "time not found." >&2
  exit 1
}

time -p cargo run -p hologram-ai --release -- \
  run-pipeline "${PIPELINE_PATH}" --prompt "${PROMPT}" --max-tokens 1
