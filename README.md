# t2v (Text-to-Video, Academic)

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

## Run

```bash
source /workspace/.venv/bin/activate

# 1) Build/refresh paragraph chunks + embedding index
python /workspace/t2v/scripts/chunk_book.py

# 2) Generate video with semantic RAG
python /workspace/t2v/scripts/generate.py \
  --topic "an explainer video showing how gradients propagate in backpropagation" \
  --audience "undergraduate students" \
  --objective "understand gradient flow and weight updates" \
  --seconds 4 \
  --use-rag
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
- `--purge-hf-cache` (only if you intentionally want full redownload)

Example model switch:

```bash
# Smaller CogVideoX
/workspace/.venv/bin/python /workspace/t2v/scripts/generate.py \
  --model-id "THUDM/CogVideoX-2b" \
  --seconds 4 --fps 8 --steps 24 --use-rag \
  --topic "a clean academic diagram showing gradient descent"

# LTX Video
/workspace/.venv/bin/python /workspace/t2v/scripts/generate.py \
  --model-id "Lightricks/LTX-Video-0.9.1" \
  --seconds 4 --fps 16 --steps 30 --guidance 3.0 --use-rag \
  --topic "a clean academic diagram showing gradient descent"
```

Deterministic local academic diagrams (no video model download):

```bash
/workspace/.venv/bin/python /workspace/t2v/scripts/generate.py \
  --engine local \
  --topic "a clean academic diagram explaining gradient descent" \
  --use-rag --seconds 4 --fps 16 --height 720 --width 1280 \
  --output /workspace/t2v/outputs/local_demo.mp4
```

## Troubleshooting

If download fails with `No space left on device (os error 28)`:

```bash
df -h /
du -h -d 2 /root/.cache/huggingface/hub | sort -h | tail -n 20
```

If `/root` is full, remove stale model cache there:

```bash
rm -rf /root/.cache/huggingface/hub/models--Lightricks--LTX-Video-0.9.1
rm -rf /root/.cache/huggingface/hub/models--THUDM--CogVideoX-5b
```

The runtime now forces HF cache + temp files to `/workspace/.cache`.
