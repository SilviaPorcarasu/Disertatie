#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT_DIR:-/workspace/Disertatie}"
PY="${PYTHON_BIN:-/workspace/.venv/bin/python}"
OUT_ROOT="${1:-${ROOT}/outputs/semantic_demo_pack}"
CHUNKS_PATH="${CHUNKS_PATH:-${ROOT}/data/book_chunks.jsonl}"
INPUT_BOOK="${INPUT_BOOK:-${ROOT}/data/2017_Book_AnIntroductionToMachineLearnin.pdf}"
MODEL_ID="${T2V_MODEL_ID:-Wan-AI/Wan2.1-T2V-14B-Diffusers}"
EMBED_MODEL="${T2V_EMBEDDING_MODEL_ID:-sentence-transformers/all-MiniLM-L6-v2}"
LORA_PATH="${T2V_LORA_PATH:-}"
LORA_SCALE="${T2V_LORA_SCALE:-0.7}"
LORA_TRIGGER="${T2V_LORA_TRIGGER:-acad_infov1}"

export T2V_RAG_MIN_SCORE="${T2V_RAG_MIN_SCORE:-0.16}"
export T2V_RAG_CANDIDATE_K="${T2V_RAG_CANDIDATE_K:-28}"
export T2V_RAG_MAX_CONTEXT_WORDS="${T2V_RAG_MAX_CONTEXT_WORDS:-110}"
export T2V_RAG_MIN_CHUNK_WORDS="${T2V_RAG_MIN_CHUNK_WORDS:-25}"

mkdir -p "${OUT_ROOT}"
cd "${ROOT}"

echo "[1/3] Environment check"
"${PY}" "${ROOT}/scripts/check_env.py"

echo "[2/3] Rebuild semantic RAG chunks + embedding index"
T2V_EMBEDDING_MODEL_ID="${EMBED_MODEL}" \
"${PY}" "${ROOT}/scripts/chunk_book.py" \
  --input "${INPUT_BOOK}" \
  --output "${CHUNKS_PATH}" \
  --build-embedding-index \
  --clean-chunks \
  --min-words 45 \
  --max-words 220

echo "[3/3] Generate 3 semantic demo videos"
while IFS='|' read -r slug topic query seed; do
  [[ -z "${slug}" ]] && continue
  out_file="${OUT_ROOT}/${slug}.mp4"
  log_file="${OUT_ROOT}/${slug}.log"

  cmd=(
    "${PY}" "${ROOT}/scripts/generate.py"
    --engine diffusion
    --model-id "${MODEL_ID}"
    --use-rag --rag-mode semantic
    --chunks-path "${CHUNKS_PATH}"
    --rag-query "${query}"
    --topic "${topic}"
    --audience "undergraduate students"
    --objective "understand the mechanism step by step"
    --style "2D academic infographic, flat vector diagram, high contrast nodes and thick directional arrows, no on-screen text"
    --seconds 4 --fps 12 --frames 25 --steps 30 --guidance 5.0
    --seed "${seed}"
    --print-prompt
    --output "${out_file}"
  )

  if [[ -n "${LORA_PATH}" ]]; then
    cmd+=(
      --lora-path "${LORA_PATH}"
      --lora-scale "${LORA_SCALE}"
      --lora-prompt-profile academic_infographic
      --lora-trigger "${LORA_TRIGGER}"
    )
  fi

  if [[ "${STRICT_DIFFUSION:-0}" == "1" ]]; then
    cmd+=(--no-fallback-local-on-fail)
  fi

  echo "==> ${slug}"
  printf 'CMD: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" | tee "${log_file}"
  echo "Saved: ${out_file}"
  echo "Log:   ${log_file}"
  echo

done <<'TOPICS'
gradient_backprop|gradient flow in backpropagation as a clean 2D process diagram|backpropagation gradient flow chain rule weight updates neural network|42
a_star_search|A* search on a weighted grid with open set closed set and optimal path reconstruction|A* search graph shortest path heuristic open set closed set|43
attention_qkv|self-attention mechanism showing query key value weighted links over tokens|self attention query key value token weighting context aggregation|44
TOPICS

echo "Done. Demo outputs: ${OUT_ROOT}"
ls -lh "${OUT_ROOT}"/*.mp4
