#!/usr/bin/env bash
set -euo pipefail

# Optional arg1: output path
OUT_PATH="${1:-/workspace/Disertatie/outputs/gradient_nolabels_demo.mp4}"

cd /workspace/Disertatie

T2V_EMBEDDING_MODEL_ID="${T2V_EMBEDDING_MODEL_ID:-sentence-transformers/all-MiniLM-L6-v2}" \
T2V_RAG_MIN_SCORE="${T2V_RAG_MIN_SCORE:-0.12}" \
T2V_RAG_MAX_CONTEXT_WORDS="${T2V_RAG_MAX_CONTEXT_WORDS:-140}" \
python3 scripts/generate.py \
  --engine diffusion \
  --quality high \
  --seed "${T2V_SEED:-42}" \
  --model-id "${T2V_MODEL_ID:-Wan-AI/Wan2.1-T2V-14B-Diffusers}" \
  --use-rag \
  --rag-query "backpropagation gradient flow chain rule weight updates neural network" \
  --topic "backpropagation gradient demo: forward flow then backward gradients then weight updates, clean 2D network animation, no on-screen text" \
  --audience "undergraduate students" \
  --objective "understand backward gradient flow and weight updates" \
  --style "flat technical infographic, high contrast, stable camera, smooth motion, no labels" \
  --seconds 5 \
  --fps 16 \
  --frames 33 \
  --steps 30 \
  --guidance 5.0 \
  --output "${OUT_PATH}"

echo "Saved: ${OUT_PATH}"
