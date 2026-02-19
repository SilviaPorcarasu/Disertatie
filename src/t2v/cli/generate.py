import argparse
import os
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Educational text-to-video generator")
    default_chunks_path = str(
        Path(__file__).resolve().parents[3] / "data" / "book_chunks.jsonl"
    )
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
        "--quality",
        choices=("fast", "balanced", "high"),
        default="",
        help="Quality preset for default frames/steps/guidance when not explicitly overridden",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Optional deterministic seed (>=0)",
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
        default=default_chunks_path,
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
    if args.quality:
        os.environ["T2V_QUALITY_PROFILE"] = args.quality
    if args.seed is not None and args.seed >= 0:
        os.environ["T2V_SEED"] = str(args.seed)
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
        f" seed={cfg.seed if cfg.seed is not None else 'auto'}"
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
