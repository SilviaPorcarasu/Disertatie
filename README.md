# Disertatie - t2v (Text-to-Video, Academic)

Clean project structure with two entry scripts:

- `scripts/chunk_book.py`: ingest document, paragraph chunking, embedding index build
- `scripts/generate.py`: generate video (optionally with semantic RAG)

## Structure

- `src/t2v/cli/`: command entry logic
- `src/t2v/config.py`: runtime/device/model config
- `src/t2v/models/`: pipeline loading and VRAM strategy
- `src/t2v/prompting/`: English-only academic prompt builder
- `src/t2v/rag/`: paragraph chunking + embedding retrieval
- `src/t2v/render/`: frame generation + export
- `data/`: source docs + generated RAG artifacts (`book_chunks.jsonl`, `book_chunks.embeddings.npz`)
- `outputs/`: generated video and frame snapshots

## Setup (once)

```bash
# One command bootstrap (recommended)
bash /workspace/Disertatie/scripts/bootstrap_env.sh

# OR manual setup
python3 -m venv /workspace/.venv
source /workspace/.venv/bin/activate
pip install --upgrade pip
pip install -r /workspace/Disertatie/requirements.txt

# Preflight checks (fails fast if something is missing)
python /workspace/Disertatie/scripts/check_env.py
```

## Docker (recommended)

```bash
cd /workspace/Disertatie

# Build image once
docker compose build t2v

# Sanity check dependencies inside container
docker compose run --rm t2v python scripts/check_env.py

# Chunk + embedding index
docker compose run --rm t2v python scripts/chunk_book.py \
  --input /workspace/Disertatie/data/2017_Book_AnIntroductionToMachineLearnin.pdf \
  --output /workspace/Disertatie/data/book_chunks.jsonl

# Deterministic local engine (no model download)
docker compose run --rm t2v python scripts/generate.py \
  --engine local \
  --topic "gradient descent explainer" \
  --use-rag \
  --seconds 4 \
  --output /workspace/Disertatie/outputs/local_demo.mp4
```

GPU diffusion run (requires NVIDIA Container Toolkit):

```bash
docker compose --profile gpu run --rm t2v-gpu python scripts/generate.py \
  --engine diffusion \
  --model-id "Wan-AI/Wan2.1-T2V-14B-Diffusers" \
  --topic "gradient flow in backpropagation" \
  --use-rag \
  --seconds 4 \
  --output /workspace/Disertatie/outputs/gpu_demo.mp4
```

If GPU is not visible in container, validate host runtime first:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Run

```bash
source /workspace/.venv/bin/activate

# 1) Build/refresh paragraph chunks + embedding index
python /workspace/Disertatie/scripts/chunk_book.py

# 2) Generate video with semantic RAG
python /workspace/Disertatie/scripts/generate.py \
  --topic "an explainer video showing how gradients propagate in backpropagation" \
  --audience "undergraduate students" \
  --objective "understand gradient flow and weight updates" \
  --seconds 4 \
  --use-rag
```

Default model is Wan 2.1 14B (`Wan-AI/Wan2.1-T2V-14B-Diffusers`) unless you override `--model-id` or `T2V_MODEL_ID`.
`--fallback-local-on-fail` is enabled by default, so if diffusion cannot load/run, export falls back to local deterministic rendering.

Quick run commands:

```bash
# Any topic (minimal input): topic only, robust defaults for objective/style
/workspace/.venv/bin/python /workspace/Disertatie/scripts/generate.py \
  --engine diffusion \
  --topic "explain A* search on a graph, step by step" \
  --seconds 4 \
  --output /workspace/Disertatie/outputs/any_topic.mp4

# Semantic RAG (recommended): retrieve cues -> scene plan -> per-scene generation
/workspace/.venv/bin/python /workspace/Disertatie/scripts/generate.py \
  --engine diffusion \
  --model-id "Wan-AI/Wan2.1-T2V-14B-Diffusers" \
  --use-rag \
  --rag-mode semantic \
  --scene-frames 16 \
  --topic "gradient flow in backpropagation" \
  --audience "undergraduate students" \
  --objective "understand gradient flow and weight updates" \
  --seconds 4 \
  --output /workspace/Disertatie/outputs/wan_semantic.mp4

# Overlay RAG (legacy mode): inject raw retrieved context in prompt
/workspace/.venv/bin/python /workspace/Disertatie/scripts/generate.py \
  --engine diffusion \
  --model-id "Wan-AI/Wan2.1-T2V-14B-Diffusers" \
  --use-rag \
  --rag-mode overlay \
  --topic "gradient flow in backpropagation" \
  --seconds 4 \
  --output /workspace/Disertatie/outputs/wan_overlay.mp4

# Local deterministic engine (no diffusion model download)
/workspace/.venv/bin/python /workspace/Disertatie/scripts/generate.py \
  --engine local \
  --use-rag \
  --rag-mode semantic \
  --topic "gradient descent explainer" \
  --seconds 4 \
  --output /workspace/Disertatie/outputs/local_semantic.mp4

# End-to-end demo scripts
bash /workspace/Disertatie/scripts/run_gradient_demo.sh
bash /workspace/Disertatie/scripts/run_fewshot_suite.sh high 5
bash /workspace/Disertatie/scripts/run_semantic_demo_pack.sh
bash /workspace/Disertatie/scripts/run_semantic_comparison_pack.sh
```

## Notes

- Project is English-only for prompt/query flow.
- RAG retrieval is semantic (embedding similarity), not lexical overlap.
- Chunking pipeline includes a robust cleaning pass by default:
  - removes TOC/index/bibliography/OCR-heavy chunks
  - deduplicates near-identical chunks
  - splits very long chunks
  - exports rejected chunks to `data/book_chunks.rejected.jsonl`

## Useful knobs

Chunk cleaning:

- `--min-words` (default: `45`)
- `--max-words` (default: `220`)
- `--save-rejected / --no-save-rejected`

RAG retrieval:

- `T2V_RAG_MIN_SCORE` (default: `0.18`)
- `T2V_RAG_CANDIDATE_K` (default: `24`)
- `T2V_RAG_MAX_CONTEXT_WORDS` (default: `80`)
- `T2V_RAG_MIN_CHUNK_WORDS` (default: `25`)

Generation overrides (CLI):

- `--model-id`
- `--engine` (`diffusion` or `local`)
- `--seconds` / `--frames`
- `--fps`
- `--steps`
- `--guidance`
- `--height` / `--width`
- `--lora-path` / `--lora-scale` / `--lora-weight-name`
- `--lora-prompt-profile` (`auto`, `none`, `academic_infographic`)
- `--lora-trigger` (prepend LoRA trigger token to prompts)
- `--lora-negative-boost / --no-lora-negative-boost`
- `--prune-hf-cache / --no-prune-hf-cache` (default: enabled)
- `--purge-hf-cache` (full delete, forces redownload)
- `--fallback-local-on-fail / --no-fallback-local-on-fail` (default: enabled)
- `T2V_PURGE_HF_CACHE=1` (enable full purge by default for every run)
- `T2V_CACHE_ROOT=/path/to/external/cache` (move HF/tmp cache to mounted external storage)
- `T2V_DTYPE=fp16|bf16` (Wan runs well with `fp16`; use `bf16` on supported GPUs)
- `T2V_RETRY_ON_BLACK=1` (default: enabled; one automatic retry if frames are near-black)
- `T2V_RETRY_ON_BLANK=1` (default: enabled; retry for near-black and washed-out outputs)
- `T2V_WHITE_RETRY_GUIDANCE` / `T2V_WHITE_RETRY_STEPS` (retry knobs for washed-out outputs)
- `--print-prompt` (debug final prompt content before inference)
- `--use-llm-planner` + `--planner-model-id` (scene planning with Llama before RAG/video)

Example model switch:

```bash
# Wan 2.1 14B (default GPU model)
/workspace/.venv/bin/python /workspace/Disertatie/scripts/generate.py \
  --model-id "Wan-AI/Wan2.1-T2V-14B-Diffusers" \
  --seconds 4 --fps 16 --steps 30 --guidance 5.0 --use-rag \
  --topic "a clean academic diagram showing gradient descent"

# Smaller CogVideoX (alternative)
/workspace/.venv/bin/python /workspace/Disertatie/scripts/generate.py \
  --model-id "THUDM/CogVideoX-2b" \
  --seconds 4 --fps 8 --steps 24 --guidance 5.0 --use-rag \
  --topic "a clean academic diagram showing gradient descent"

# LTX Video
/workspace/.venv/bin/python /workspace/Disertatie/scripts/generate.py \
  --model-id "Lightricks/LTX-Video-0.9.1" \
  --seconds 4 --fps 16 --steps 30 --guidance 3.0 --use-rag \
  --topic "a clean academic diagram showing gradient descent"

# Wan + LoRA style lock (academic infographic)
/workspace/.venv/bin/python /workspace/Disertatie/scripts/generate.py \
  --model-id "Wan-AI/Wan2.1-T2V-14B-Diffusers" \
  --lora-path "/workspace/models/lora/academic_infographic_v1" \
  --lora-scale 0.7 \
  --lora-prompt-profile academic_infographic \
  --lora-trigger "acad_infov1" \
  --topic "gradient flow in backpropagation as a clean 2D process diagram" \
  --use-rag --rag-mode semantic \
  --seconds 4 --fps 12 --frames 25 --steps 30 --guidance 5.0
```

Deterministic local academic diagrams (no video model download):

```bash
/workspace/.venv/bin/python /workspace/Disertatie/scripts/generate.py \
  --engine local \
  --topic "a clean academic diagram explaining gradient descent" \
  --use-rag --seconds 4 --fps 16 --height 720 --width 1280 \
  --output /workspace/Disertatie/outputs/local_demo.mp4
```

## Troubleshooting

If video export fails with ffmpeg/imageio backend errors:

```bash
source /workspace/.venv/bin/activate
pip install -r /workspace/Disertatie/requirements.txt
# optional system binary (if you control the OS/container)
apt-get update && apt-get install -y ffmpeg
```

Without ffmpeg, runtime falls back to exporting a `.gif` next to the requested output path.

If you run CogVideoX and tokenizer load fails with `protobuf` / `tiktoken` errors:

```bash
source /workspace/.venv/bin/activate
pip install protobuf tiktoken
```

If download fails with `No space left on device (os error 28)`:

```bash
df -h /
du -h -d 2 /root/.cache/huggingface/hub | sort -h | tail -n 20
```

For low storage quotas, keep automatic cache pruning enabled (default). It removes
unused model snapshots before each generation run and keeps only the active model
(plus embedding model when RAG is enabled).

If `/root` is full, remove stale model cache there:

```bash
rm -rf /root/.cache/huggingface/hub/models--Lightricks--LTX-Video-0.9.1
rm -rf /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers
```

Default cache root is `/workspace/.cache`; override with `T2V_CACHE_ROOT` for external storage mounts.
