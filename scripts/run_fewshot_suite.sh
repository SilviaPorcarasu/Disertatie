#!/usr/bin/env bash
set -euo pipefail

QUALITY="${1:-high}"
CLIP_SECONDS="${2:-5}"
DRY_RUN="${3:-0}"

OUT_ROOT="/workspace/Disertatie/outputs/fewshot_suite"
MODEL_ID="${T2V_MODEL_ID:-THUDM/CogVideoX-5b}"
SEED="${T2V_SEED:-42}"

topics=(
  "overfitting vs generalization"
  "gradient descent"
  "bias-variance tradeoff"
  "attention mechanism"
)

cd /workspace/Disertatie
mkdir -p "${OUT_ROOT}"

for topic in "${topics[@]}"; do
  slug="$(echo "${topic}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')"
  out_dir="${OUT_ROOT}/${slug}"
  out_file="${out_dir}/video.mp4"
  log_file="${out_dir}/run.log"
  mkdir -p "${out_dir}"

  cmd=(
    python3 scripts/generate.py
    --engine diffusion
    --model-id "${MODEL_ID}"
    --quality "${QUALITY}"
    --seed "${SEED}"
    --use-rag
    --rag-query "${topic}"
    --topic "${topic}"
    --audience "undergraduate students"
    --objective "understand the concept visually and intuitively"
    --style "flat academic infographic, high contrast, stable camera, no on-screen text"
    --seconds "${CLIP_SECONDS}"
    --output "${out_file}"
  )

  echo "=== ${topic} ==="
  echo "Output: ${out_file}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CMD: '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    continue
  fi

  "${cmd[@]}" | tee "${log_file}"
done

echo "Done. Outputs in: ${OUT_ROOT}"
