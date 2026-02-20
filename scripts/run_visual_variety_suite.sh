#!/usr/bin/env bash
set -euo pipefail

PY="${PYTHON_BIN:-/workspace/.venv/bin/python}"
OUT_DIR="${1:-/workspace/Disertatie/outputs/visual_variety}"
MODEL_ID="${T2V_MODEL_ID:-Wan-AI/Wan2.1-T2V-14B-Diffusers}"
STEPS="${T2V_STEPS:-20}"
FRAMES="${T2V_FRAMES:-25}"
FPS="${T2V_FPS:-12}"
QUALITY="${T2V_QUALITY:-fast}"

mkdir -p "${OUT_DIR}"

run_case() {
  local slug="$1"
  local prompt="$2"
  local seed="$3"
  local out="${OUT_DIR}/${slug}.mp4"

  echo "==> ${slug}"
  "${PY}" /workspace/Disertatie/scripts/generate.py \
    --engine diffusion \
    --model-id "${MODEL_ID}" \
    --no-force-diagram-prompt \
    --prompt "${prompt}" \
    --negative "text, letters, numbers, watermark, logo, flicker, camera shake, abrupt cuts" \
    --quality "${QUALITY}" \
    --steps "${STEPS}" \
    --frames "${FRAMES}" \
    --fps "${FPS}" \
    --seed "${seed}" \
    --output "${out}"
}

run_case \
  "gradient_valley_ball" \
  "educational 2D animation of a smooth loss landscape with hills and valleys, a small red ball starts high and rolls downhill into a local minimum, velocity decreases near the basin, arrow trail shows descent direction, clean background, no text" \
  21

run_case \
  "overfitting_curves" \
  "split-screen educational animation: left panel shows smooth curve fitting trend, right panel shows highly wiggly curve chasing noise points, then unseen points appear and smooth curve generalizes better, clear color coding, no text" \
  22

run_case \
  "attention_links" \
  "stylized educational scene with token nodes in a horizontal line, one active token emits glowing weighted links to others, link thickness changes over time to show shifting attention, clean motion, no text" \
  23

run_case \
  "bias_variance_archery" \
  "metaphoric educational animation of three target boards: one with tightly clustered off-center hits, one with widely scattered hits, and one balanced near center, smooth transitions comparing bias and variance intuition, no text" \
  24

echo "Done. Outputs in: ${OUT_DIR}"
