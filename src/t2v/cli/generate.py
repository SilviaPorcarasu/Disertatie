import argparse
import os
from pathlib import Path

import numpy as np

from t2v.config import build_runtime_config, clear_runtime_cache, setup_environment
from t2v.models.pipeline import load_pipeline
from t2v.prompting.education import (
    build_education_prompt,
    build_negative_prompt,
    normalize_request_to_english,
)
from t2v.rag.retrieve import retrieve_context
from t2v.render.export import export_outputs
from t2v.render.generate_video import generate_frames
from t2v.render.local_diagram import generate_local_diagram_frames


def _frames_look_blank(video_frames) -> bool:
    if not video_frames:
        return True

    indices = sorted({0, len(video_frames) // 2, len(video_frames) - 1})
    means = []
    stds = []
    for idx in indices:
        arr = np.asarray(video_frames[idx].convert("RGB"), dtype=np.uint8)
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))

    return max(stds) < 2.5 and (min(means) > 245.0 or max(means) < 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Educational text-to-video generator")
    parser.add_argument(
        "--topic",
        default="gradient flow in backpropagation",
        help="Educational topic to explain",
    )
    parser.add_argument(
        "--audience",
        default="high-school and early university students",
        help="Target audience",
    )
    parser.add_argument(
        "--objective",
        default="understand how gradients propagate backward and update weights",
        help="Learning objective",
    )
    parser.add_argument(
        "--style",
        default="flat infographic style",
        help="Visual style",
    )
    parser.add_argument(
        "--output",
        default="/workspace/t2v/outputs/demo.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--engine",
        choices=("diffusion", "local"),
        default="diffusion",
        help="Video engine: diffusion model or deterministic local diagram",
    )
    parser.add_argument(
        "--model-id",
        default="",
        help="Optional model id override (e.g. THUDM/CogVideoX-2b, Lightricks/LTX-Video-0.9.1)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Target duration in seconds (overrides default frame count if > 0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=0,
        help="Output FPS override (if > 0)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Frame count override (if > 0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Inference steps override (if > 0)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=0.0,
        help="Guidance scale override (if > 0)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Optional output height override (if > 0 and supported by model)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Optional output width override (if > 0 and supported by model)",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Use retrieved context from chunked book JSONL",
    )
    parser.add_argument(
        "--purge-hf-cache",
        action="store_true",
        help="Delete /workspace HF cache before run (forces model redownload)",
    )
    parser.add_argument(
        "--chunks-path",
        default="/workspace/t2v/data/book_chunks.jsonl",
        help="Path to chunked book JSONL",
    )
    parser.add_argument(
        "--rag-query",
        default="",
        help="Query for retrieval; defaults to topic if empty",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=3,
        help="Number of chunks to inject as context",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topic_en = normalize_request_to_english(args.topic)
    audience_en = normalize_request_to_english(args.audience)
    objective_en = normalize_request_to_english(args.objective)
    style_en = normalize_request_to_english(args.style)

    setup_environment()
    if args.model_id:
        os.environ["T2V_MODEL_ID"] = args.model_id
    if args.seconds and args.seconds > 0:
        os.environ["DURATION_SECONDS"] = str(args.seconds)
    if args.fps and args.fps > 0:
        os.environ["FPS"] = str(args.fps)
    if args.frames and args.frames > 0:
        os.environ["NUM_FRAMES"] = str(args.frames)
    if args.steps and args.steps > 0:
        os.environ["NUM_INFERENCE_STEPS"] = str(args.steps)
    if args.guidance and args.guidance > 0:
        os.environ["GUIDANCE_SCALE"] = str(args.guidance)
    if args.height and args.height > 0:
        os.environ["HEIGHT"] = str(args.height)
    if args.width and args.width > 0:
        os.environ["WIDTH"] = str(args.width)

    clear_runtime_cache(purge_hf_cache=args.purge_hf_cache)
    cfg = build_runtime_config(args.output)
    print(
        "Generation config:"
        f" engine={args.engine}"
        f" family={cfg.model_family}"
        f" model={cfg.model_id}"
        f" device={cfg.device}"
        f" frames={cfg.num_frames}"
        f" fps={cfg.fps}"
        f" seconds~={cfg.num_frames / max(cfg.fps, 1):.2f}"
        f" size={cfg.width or 'default'}x{cfg.height or 'default'}"
        f" steps={cfg.num_inference_steps}"
        f" guidance={cfg.guidance_scale}"
        f" dynamic_cfg={cfg.use_dynamic_cfg}"
    )
    print(
        "Cache paths:"
        f" HF_HOME={os.getenv('HF_HOME')}"
        f" HUB={os.getenv('HUGGINGFACE_HUB_CACHE')}"
        f" TMPDIR={os.getenv('TMPDIR')}"
    )
    pipe = None
    if args.engine == "diffusion":
        pipe = load_pipeline(cfg)

    reference_context = ""
    if args.use_rag:
        query = normalize_request_to_english(args.rag_query.strip() or args.topic)
        reference_context = retrieve_context(
            chunks_path=Path(args.chunks_path),
            query=query,
            top_k=args.rag_top_k,
        )
        print(f"RAG enabled (EN query: {query}). Retrieved context chars: {len(reference_context)}")

    prompt = build_education_prompt(
        topic=topic_en,
        audience=audience_en,
        objective=objective_en,
        style=style_en,
        reference_context=reference_context,
    )
    if args.engine == "local":
        video_frames = generate_local_diagram_frames(
            cfg,
            topic=topic_en,
            objective=objective_en,
            reference_context=reference_context,
        )
    else:
        negative_prompt = build_negative_prompt()
        video_frames = generate_frames(pipe, cfg, prompt, negative_prompt)
        if _frames_look_blank(video_frames):
            print("Detected blank/low-variance video. Retrying with stronger generation settings.")
            rescue_prompt = (
                prompt
                + " Fill the full frame with clearly visible colorful objects in every scene. "
                + "Use high contrast and avoid washed-out whites."
            )
            rescue_negative_prompt = (
                negative_prompt
                + ", blank frame, overexposed white frame, washed out frame, monochrome empty frame"
            )
            if cfg.model_family == "cogvideo":
                rescue_steps = max(cfg.num_inference_steps, 36 if "5b" in cfg.model_id.lower() else 20)
                rescue_guidance = 6.0 if "5b" in cfg.model_id.lower() else max(cfg.guidance_scale, 7.5)
            elif cfg.model_family == "ltx":
                rescue_steps = max(cfg.num_inference_steps, 34)
                rescue_guidance = max(cfg.guidance_scale, 3.5)
            elif cfg.model_family == "wan":
                rescue_steps = max(cfg.num_inference_steps, 34)
                rescue_guidance = max(cfg.guidance_scale, 5.5)
            else:
                rescue_steps = max(cfg.num_inference_steps, 28)
                rescue_guidance = max(cfg.guidance_scale, 5.5)
            video_frames = generate_frames(
                pipe,
                cfg,
                rescue_prompt,
                rescue_negative_prompt,
                num_inference_steps=rescue_steps,
                guidance_scale=rescue_guidance,
                use_dynamic_cfg=True,
            )

            if _frames_look_blank(video_frames):
                print(
                    "Warning: output is still low-variance. Try higher steps or shorter video "
                    "(e.g. NUM_INFERENCE_STEPS=45 NUM_FRAMES=9)."
                )

    try:
        export_outputs(video_frames, cfg)
    except Exception as exc:
        print(f"Video export failed: {exc}")
        return

    print(f"Saved video: {cfg.output_path}")
    if args.engine == "local":
        print("Model used: local-diagram")
    else:
        print(f"Model used: {cfg.model_id}")


if __name__ == "__main__":
    main()
