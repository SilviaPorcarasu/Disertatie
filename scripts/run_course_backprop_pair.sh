#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT_DIR:-/workspace/Disertatie}"
PY="${PYTHON_BIN:-/workspace/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${1:-${ROOT}/outputs/course_backprop_pair_${RUN_TAG}}"

mkdir -p "${OUT_ROOT}"
cd "${ROOT}"

TOPIC="backpropagation in a feedforward neural network with explicit gradient flow and weight updates"
OBJECTIVE="understand backward gradient propagation layer by layer"
STYLE="2D academic infographic, node-edge process diagram, high contrast arrows, no text"
RAG_QUERY="backpropagation chain rule gradient flow hidden layer local derivative weight update learning rate"

BASE_ARGS=(
  --engine diffusion
  --model-id "Wan-AI/Wan2.1-T2V-14B-Diffusers"
  --course-mode
  --course-template backprop
  --course-few-shot
  --topic "${TOPIC}"
  --audience "undergraduate students"
  --objective "${OBJECTIVE}"
  --style "${STYLE}"
  --scene-frames 12
  --fps 10
  --frames 17
  --steps 16
  --guidance 4.8
  --seed 52
  --no-fallback-local-on-fail
  --print-prompt
)

echo "[1/2] Backprop course baseline (no RAG)"
"${PY}" "${ROOT}/scripts/generate.py" \
  "${BASE_ARGS[@]}" \
  --output "${OUT_ROOT}/backprop_no_rag.mp4" | tee "${OUT_ROOT}/backprop_no_rag.log"

echo "[2/2] Backprop course + semantic RAG"
"${PY}" "${ROOT}/scripts/generate.py" \
  "${BASE_ARGS[@]}" \
  --use-rag --rag-mode semantic \
  --rag-query "${RAG_QUERY}" \
  --chunks-path "${ROOT}/data/book_chunks.jsonl" \
  --output "${OUT_ROOT}/backprop_rag_semantic.mp4" | tee "${OUT_ROOT}/backprop_rag_semantic.log"

echo "Done. Outputs in: ${OUT_ROOT}"
ls -lh "${OUT_ROOT}"/*.mp4
