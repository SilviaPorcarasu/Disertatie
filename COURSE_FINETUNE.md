# Course-Style Prompting and LoRA Fine-Tune

This guide focuses on making outputs look and behave like short course explainers (theory-first).

## 1) Generate Course-Style Baselines

Use either:

- single-topic backprop pair
- multi-topic ML pack (recommended for thesis tables)

```bash
cd /workspace/Disertatie
source /workspace/.venv/bin/activate
unset T2V_LORA_PATH T2V_LORA_SCALE T2V_LORA_TRIGGER

STRICT_DIFFUSION=1 bash /workspace/Disertatie/scripts/run_course_backprop_pair.sh
STRICT_DIFFUSION=1 bash /workspace/Disertatie/scripts/run_course_ml_pack.sh
SEED_OFFSET=100 STRICT_DIFFUSION=1 bash /workspace/Disertatie/scripts/run_course_ml_pack.sh
SEED_OFFSET=200 STRICT_DIFFUSION=1 bash /workspace/Disertatie/scripts/run_course_ml_pack.sh
```

This saves:
- `backprop_no_rag.mp4`
- `backprop_rag_semantic.mp4`
- their logs in the same folder

The multi-topic pack saves per-topic folders with:
- `no_rag/video.mp4`
- `rag_semantic/video.mp4`
- logs
- summary files (`summary.csv`, `results.md`)

`SEED_OFFSET` changes random seeds so you can generate diverse clips for LoRA.

## 2) Build LoRA Manifest from Saved Videos

```bash
python /workspace/Disertatie/scripts/build_lora_course_manifest.py \
  --input-root /workspace/Disertatie/outputs \
  --glob "**/*.mp4" \
  --output-jsonl /workspace/Disertatie/data/lora_course_manifest.jsonl \
  --trigger-token "acad_course_v1" \
  --split "0.8,0.1,0.1" \
  --seed 42
```

Outputs:
- `/workspace/Disertatie/data/lora_course_manifest.jsonl`
- `/workspace/Disertatie/data/lora_course_manifest.csv`

## 3) Recommended Training Setup (A100 80GB)

Few-shot note:
- in-prompt few-shot (`--course-few-shot`) improves structural consistency
- it does not create persistent learning by itself
- persistent learning comes from LoRA fine-tuning on curated video-caption pairs

Suggested starting hyperparameters for Wan LoRA:
- rank `16`
- alpha `32`
- dropout `0.05`
- learning rate `1e-4`
- batch size `1`
- gradient accumulation `8`
- bf16 enabled
- 3k to 6k steps

Target module set (typical transformer attention):
- `to_q`, `to_k`, `to_v`, `to_out.0`

## 4) Integrate LoRA Back in Inference

```bash
python /workspace/Disertatie/scripts/generate.py \
  --engine diffusion \
  --model-id "Wan-AI/Wan2.1-T2V-14B-Diffusers" \
  --course-mode \
  --course-template backprop \
  --lora-path "/workspace/models/lora/academic_course_v1" \
  --lora-scale 0.7 \
  --lora-prompt-profile academic_infographic \
  --lora-trigger "acad_course_v1" \
  --use-rag --rag-mode semantic \
  --rag-query "backpropagation chain rule gradient flow hidden layer local derivative weight update learning rate" \
  --topic "backpropagation in a feedforward neural network with explicit gradient flow and weight updates" \
  --objective "understand backward gradient propagation layer by layer" \
  --scene-frames 12 --fps 10 --frames 17 --steps 16 --guidance 4.8 \
  --no-fallback-local-on-fail \
  --output /workspace/Disertatie/outputs/backprop_course_lora_v1.mp4
```

## 5) Evaluation for Thesis Table

For each topic, keep both variants (`no_rag`, `rag_semantic`) and score:
- theoretical correctness
- step-order clarity
- temporal coherence
- visual clarity
- topic adherence

Use the same seed/settings for fair comparison.
