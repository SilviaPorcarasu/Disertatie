#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT_DIR:-/workspace/Disertatie}"
PY="${PYTHON_BIN:-/workspace/.venv/bin/python}"
INPUT_BOOK="${INPUT_BOOK:-${ROOT}/data/2017_Book_AnIntroductionToMachineLearnin.pdf}"
CHUNKS_PATH="${CHUNKS_PATH:-${ROOT}/data/book_chunks.jsonl}"
MODEL_ID="${T2V_MODEL_ID:-Wan-AI/Wan2.1-T2V-14B-Diffusers}"
EMBED_MODEL="${T2V_EMBEDDING_MODEL_ID:-sentence-transformers/all-MiniLM-L6-v2}"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${1:-${ROOT}/outputs/course_ml_pack_${RUN_TAG}}"
STRICT_DIFFUSION="${STRICT_DIFFUSION:-1}"
SKIP_RAG_REBUILD="${SKIP_RAG_REBUILD:-0}"
SEED_OFFSET="${SEED_OFFSET:-0}"

# Faster defaults for iterative thesis demos.
FPS="${T2V_FPS:-8}"
FRAMES="${T2V_FRAMES:-17}"
STEPS="${T2V_STEPS:-14}"
GUIDANCE="${T2V_GUIDANCE:-4.6}"
SCENE_FRAMES="${T2V_SCENE_FRAMES:-10}"
HEIGHT="${T2V_HEIGHT:-480}"
WIDTH="${T2V_WIDTH:-832}"

mkdir -p "${OUT_ROOT}"
cd "${ROOT}"

export T2V_RAG_MIN_SCORE="${T2V_RAG_MIN_SCORE:-0.16}"
export T2V_RAG_CANDIDATE_K="${T2V_RAG_CANDIDATE_K:-28}"
export T2V_RAG_MAX_CONTEXT_WORDS="${T2V_RAG_MAX_CONTEXT_WORDS:-110}"
export T2V_RAG_MIN_CHUNK_WORDS="${T2V_RAG_MIN_CHUNK_WORDS:-25}"

SUMMARY_CSV="${OUT_ROOT}/summary.csv"
RESULTS_MD="${OUT_ROOT}/results.md"

cat > "${SUMMARY_CSV}" <<'CSV'
topic_slug,variant,status,model_used,video_path,log_path,template,seed,fps,frames,steps,guidance,scene_frames,start_utc,end_utc,elapsed_sec
CSV

cat > "${RESULTS_MD}" <<'MD'
# Course ML Pack Results

| Topic | Variant | Status | Model | Video |
|---|---|---|---|---|
MD

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

echo "[3/3] Generate no_rag vs rag_semantic pairs for ML topics"

run_variant() {
  local slug="$1"
  local template="$2"
  local topic="$3"
  local objective="$4"
  local rag_query="$5"
  local seed="$6"
  local variant="$7"
  local final_seed
  final_seed="$((seed + SEED_OFFSET))"

  local variant_dir="${OUT_ROOT}/${slug}/${variant}"
  local video_path="${variant_dir}/video.mp4"
  local log_path="${variant_dir}/run.log"
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
    --course-mode
    --course-template "${template}"
    --course-few-shot
    --topic "${topic}"
    --audience "undergraduate students"
    --objective "${objective}"
    --style "2D academic infographic, node-edge process diagram, high contrast arrows, no text"
    --fps "${FPS}"
    --frames "${FRAMES}"
    --steps "${STEPS}"
    --guidance "${GUIDANCE}"
    --scene-frames "${SCENE_FRAMES}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --seed "${final_seed}"
    --print-prompt
    --output "${video_path}"
  )

  if [[ "${variant}" == "rag_semantic" ]]; then
    cmd+=(
      --use-rag
      --rag-mode semantic
      --chunks-path "${CHUNKS_PATH}"
      --rag-query "${rag_query}"
    )
  fi

  if [[ "${STRICT_DIFFUSION}" == "1" ]]; then
    cmd+=(--no-fallback-local-on-fail)
  fi

  echo "==> ${slug} [${variant}]"
  printf 'CMD: '
  printf '%q ' "${cmd[@]}"
  printf '\n'

  set +e
  "${cmd[@]}" | tee "${log_path}"
  local rc="${PIPESTATUS[0]}"
  set -e

  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_epoch="$(date +%s)"
  elapsed="$((end_epoch - start_epoch))"

  if [[ "${rc}" -ne 0 ]] || [[ ! -s "${video_path}" ]]; then
    status="failed"
  fi

  if grep -q "Model used:" "${log_path}"; then
    model_used="$(grep 'Model used:' "${log_path}" | tail -n 1 | sed 's/^Model used: //')"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${slug}" "${variant}" "${status}" "${model_used}" "${video_path}" "${log_path}" "${template}" \
    "${final_seed}" "${FPS}" "${FRAMES}" "${STEPS}" "${GUIDANCE}" "${SCENE_FRAMES}" \
    "${start_utc}" "${end_utc}" "${elapsed}" >> "${SUMMARY_CSV}"

  printf '| %s | %s | %s | %s | `%s` |\n' \
    "${slug}" "${variant}" "${status}" "${model_used}" "${video_path}" >> "${RESULTS_MD}"
}

while IFS='|' read -r slug template topic objective rag_query seed; do
  [[ -z "${slug}" ]] && continue
  run_variant "${slug}" "${template}" "${topic}" "${objective}" "${rag_query}" "${seed}" "no_rag"
  run_variant "${slug}" "${template}" "${topic}" "${objective}" "${rag_query}" "${seed}" "rag_semantic"
done <<'TOPICS'
gradient_descent|general|gradient descent optimization on a convex loss surface with update arrows|understand negative gradient direction and convergence behavior|gradient descent optimizer loss surface learning rate convergence|61
backpropagation|backprop|backpropagation in a feedforward neural network with explicit gradient flow and weight updates|understand backward gradient propagation layer by layer|backpropagation chain rule gradient flow hidden layer local derivative weight update learning rate|62
a_star_search|a_star|A* search on a weighted grid with open set closed set and optimal path reconstruction|understand heuristic-guided shortest path search|A* search shortest path heuristic open set closed set admissible heuristic g score h score|63
attention_qkv|attention|self-attention mechanism showing query key value weighted links over tokens|understand attention weight computation and contextual aggregation|self attention query key value attention weights softmax context vector multi-head|64
bias_variance|general|bias variance tradeoff with underfit vs overfit model behavior|understand underfitting and overfitting mechanisms|bias variance underfitting overfitting generalization error decomposition|65
classification_metrics|general|classification metrics in confusion-matrix style with precision recall F1 relationship|understand precision recall F1 trade-offs|classification metrics confusion matrix precision recall f1 threshold tradeoff|66
TOPICS

echo "Done. Output root: ${OUT_ROOT}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Results MD:  ${RESULTS_MD}"
find "${OUT_ROOT}" -name '*.mp4' -type f -print
