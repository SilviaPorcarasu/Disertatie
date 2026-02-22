#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT_DIR:-/workspace/Disertatie}"
PY="${PYTHON_BIN:-/workspace/.venv/bin/python}"
INPUT_BOOK="${INPUT_BOOK:-${ROOT}/data/2017_Book_AnIntroductionToMachineLearnin.pdf}"
CHUNKS_PATH="${CHUNKS_PATH:-${ROOT}/data/book_chunks.jsonl}"
MODEL_ID="${T2V_MODEL_ID:-Wan-AI/Wan2.1-T2V-14B-Diffusers}"
EMBED_MODEL="${T2V_EMBEDDING_MODEL_ID:-sentence-transformers/all-MiniLM-L6-v2}"

# Practical defaults for quick academic demos; override with env vars if needed.
FPS="${T2V_FPS:-8}"
FRAMES="${T2V_FRAMES:-17}"
STEPS="${T2V_STEPS:-16}"
GUIDANCE="${T2V_GUIDANCE:-4.8}"
SCENE_FRAMES="${T2V_SCENE_FRAMES:-12}"
HEIGHT="${T2V_HEIGHT:-480}"
WIDTH="${T2V_WIDTH:-832}"

STRICT_DIFFUSION="${STRICT_DIFFUSION:-1}"
SKIP_RAG_REBUILD="${SKIP_RAG_REBUILD:-0}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${1:-${ROOT}/outputs/semantic_comparison_${RUN_TAG}}"

mkdir -p "${OUT_ROOT}"
cd "${ROOT}"

export T2V_RAG_MIN_SCORE="${T2V_RAG_MIN_SCORE:-0.16}"
export T2V_RAG_CANDIDATE_K="${T2V_RAG_CANDIDATE_K:-28}"
export T2V_RAG_MAX_CONTEXT_WORDS="${T2V_RAG_MAX_CONTEXT_WORDS:-110}"
export T2V_RAG_MIN_CHUNK_WORDS="${T2V_RAG_MIN_CHUNK_WORDS:-25}"

SUMMARY_CSV="${OUT_ROOT}/summary.csv"
RESULTS_MD="${OUT_ROOT}/results.md"

cat > "${SUMMARY_CSV}" <<'CSV'
topic_slug,variant,status,model_used,video_path,log_path,retrieval_json,plan_json,seed,fps,frames,steps,guidance,start_utc,end_utc,elapsed_sec
CSV

cat > "${RESULTS_MD}" <<'MD'
# Semantic RAG Comparison Results

| Topic | Variant | Status | Model | Video |
|---|---|---|---|---|
MD

cat > "${OUT_ROOT}/run_config.txt" <<CFG
ROOT=${ROOT}
MODEL_ID=${MODEL_ID}
EMBED_MODEL=${EMBED_MODEL}
CHUNKS_PATH=${CHUNKS_PATH}
FPS=${FPS}
FRAMES=${FRAMES}
STEPS=${STEPS}
GUIDANCE=${GUIDANCE}
SCENE_FRAMES=${SCENE_FRAMES}
HEIGHT=${HEIGHT}
WIDTH=${WIDTH}
STRICT_DIFFUSION=${STRICT_DIFFUSION}
SKIP_RAG_REBUILD=${SKIP_RAG_REBUILD}
RUN_TAG=${RUN_TAG}
CFG

echo "[1/3] Environment check"
"${PY}" "${ROOT}/scripts/check_env.py"

if [[ "${SKIP_RAG_REBUILD}" != "1" ]]; then
  echo "[2/3] Rebuild semantic RAG chunks + embedding index"
  T2V_EMBEDDING_MODEL_ID="${EMBED_MODEL}" \
  "${PY}" "${ROOT}/scripts/chunk_book.py" \
    --input "${INPUT_BOOK}" \
    --output "${CHUNKS_PATH}" \
    --build-embedding-index \
    --clean-chunks \
    --min-words 45 \
    --max-words 220
else
  echo "[2/3] Skip RAG rebuild (SKIP_RAG_REBUILD=1)"
fi

echo "[3/3] Run per-topic comparison: no_rag vs rag_semantic"

run_variant() {
  local slug="$1"
  local topic="$2"
  local query="$3"
  local seed="$4"
  local variant="$5"

  local variant_dir="${OUT_ROOT}/${slug}/${variant}"
  local video_path="${variant_dir}/video.mp4"
  local log_path="${variant_dir}/run.log"
  local retrieval_path="${variant_dir}/retrieval.json"
  local plan_path="${variant_dir}/plan.json"
  local status="ok"
  local model_used="unknown"
  local start_utc end_utc start_epoch end_epoch elapsed

  mkdir -p "${variant_dir}"
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_epoch="$(date +%s)"

  local cmd=(
    "${PY}" "${ROOT}/scripts/generate.py"
    --engine diffusion
    --model-id "${MODEL_ID}"
    --topic "${topic}"
    --audience "undergraduate students"
    --objective "understand the mechanism step by step"
    --style "2D academic infographic, node-edge process diagram, high contrast arrows, no on-screen text"
    --fps "${FPS}"
    --frames "${FRAMES}"
    --steps "${STEPS}"
    --guidance "${GUIDANCE}"
    --scene-frames "${SCENE_FRAMES}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --seed "${seed}"
    --print-prompt
    --output "${video_path}"
  )

  if [[ "${variant}" == "rag_semantic" ]]; then
    cmd+=(
      --use-rag
      --rag-mode semantic
      --chunks-path "${CHUNKS_PATH}"
      --rag-query "${query}"
    )
  fi

  if [[ "${STRICT_DIFFUSION}" == "1" ]]; then
    cmd+=(--no-fallback-local-on-fail)
  fi

  printf '==> %s [%s]\n' "${slug}" "${variant}"
  printf 'CMD: '
  printf '%q ' "${cmd[@]}"
  printf '\n'

  set +e
  "${cmd[@]}" | tee "${log_path}"
  local cmd_rc="${PIPESTATUS[0]}"
  set -e

  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_epoch="$(date +%s)"
  elapsed="$((end_epoch - start_epoch))"

  if [[ "${cmd_rc}" -ne 0 ]] || [[ ! -s "${video_path}" ]]; then
    status="failed"
  fi

  if grep -q "Model used:" "${log_path}"; then
    model_used="$(grep 'Model used:' "${log_path}" | tail -n 1 | sed 's/^Model used: //')"
  fi

  [[ -f "${retrieval_path}" ]] || retrieval_path=""
  [[ -f "${plan_path}" ]] || plan_path=""

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${slug}" "${variant}" "${status}" "${model_used}" "${video_path}" "${log_path}" \
    "${retrieval_path}" "${plan_path}" "${seed}" "${FPS}" "${FRAMES}" "${STEPS}" "${GUIDANCE}" \
    "${start_utc}" "${end_utc}" "${elapsed}" >> "${SUMMARY_CSV}"

  printf '| %s | %s | %s | %s | `%s` |\n' \
    "${slug}" "${variant}" "${status}" "${model_used}" "${video_path}" >> "${RESULTS_MD}"
}

while IFS='|' read -r slug topic query seed; do
  [[ -z "${slug}" ]] && continue
  run_variant "${slug}" "${topic}" "${query}" "${seed}" "no_rag"
  run_variant "${slug}" "${topic}" "${query}" "${seed}" "rag_semantic"
done <<'TOPICS'
gradient_backprop|backpropagation in a feedforward neural network with explicit gradient flow and weight updates|backpropagation chain rule gradient flow hidden layer weight updates learning rate|52
a_star_search|A* search on a weighted grid with open set closed set and optimal path reconstruction|A* search graph shortest path heuristic open set closed set admissible heuristic|53
attention_qkv|self-attention mechanism showing query key value weighted links over tokens|self attention query key value token weighting context aggregation|54
TOPICS

echo "Done. Outputs preserved in: ${OUT_ROOT}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Results MD:  ${RESULTS_MD}"
find "${OUT_ROOT}" -name '*.mp4' -type f -print
