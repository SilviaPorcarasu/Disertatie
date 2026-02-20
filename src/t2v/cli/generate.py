import argparse
import os
from pathlib import Path

import numpy as np

from t2v.config import build_runtime_config, clear_runtime_cache, setup_environment
from t2v.models.pipeline import load_pipeline
from t2v.planning.llama import (
    build_scene_plan_with_llama,
    plan_to_rag_query,
    plan_to_scene_directives,
)
from t2v.prompting.education import (
    build_education_prompt,
    build_negative_prompt,
    build_scene_plan,
    normalize_request_to_english,
    save_plan_json,
)
from t2v.rag.retrieve import (
    DEFAULT_EMBEDDING_MODEL_ID,
    retrieve_context,
    retrieve_cues,
    save_retrieval_json,
)
from t2v.render.export import export_outputs
from t2v.render.generate_video import generate_frames
from t2v.render.local_diagram import generate_local_diagram_frames


MODEL_PRESETS = {
    "cogvideo-5b": "THUDM/CogVideoX-5b",
    "cogvideo1.5-5b": "THUDM/CogVideoX1.5-5b",
    "wan2.1-14b": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "wan2.1-1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}


CANONICAL_TOPIC_PROMPTS = {
    "gradient_descent": (
        "2D educational infographic of gradient descent on a convex loss contour map. "
        "A red point starts far from the minimum and moves iteratively in the negative gradient direction. "
        "At each step, arrows show local slope direction and parameter update. "
        "Step size is larger at the beginning and smaller near convergence. "
        "Trajectory remains visible from start to end. Static camera, clean white background, no text."
    ),
    "backpropagation": (
        "2D educational diagram of neural network backpropagation. "
        "Forward pass goes left to right, then gradient signals propagate right to left through all layers. "
        "Each layer shows incoming gradient, local derivative effect, and outgoing gradient. "
        "Then weights update in the opposite direction of the gradient. "
        "Clean vector style, static camera, neutral background, no text."
    ),
    "neural_network_basics": (
        "2D educational diagram of a feedforward neural network. "
        "Input nodes connect to hidden layers and then output nodes with directional arrows. "
        "Animation highlights signal flow from input to output, then highlights learned weight updates. "
        "Consistent node identity, static camera, clean background, no text."
    ),
    "classification_metrics": (
        "2D educational infographic explaining classification metrics with a confusion-grid style layout. "
        "Animation shows prediction outcomes grouped into four quadrants, then highlights precision, recall, "
        "and F1 tradeoffs through animated arrows and color emphasis. "
        "Clear geometric elements, static camera, clean background, no text."
    ),
}


CANONICAL_NEGATIVE_PROMPT = (
    "blurry, noise, distortion, artifacts, flicker, text, letters, numbers, watermark, logo, "
    "people, landscape, random abstract animation, overexposed, underexposed, blank frame"
)


TEACHER_TOPIC_KEYWORDS = {
    "gradient_descent": ("gradient descent", "optimizer", "optimization", "loss surface"),
    "backpropagation": ("backprop", "back propagation", "gradient flow", "chain rule"),
    "classification_metrics": ("metric", "metrics", "precision", "recall", "f1", "accuracy", "confusion"),
    "neural_network_basics": ("neural network", "layers", "perceptron", "feedforward", "hidden layer"),
}


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _short_words(text: str, max_words: int) -> str:
    words = normalize_request_to_english(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _infer_teacher_canonical_topic(topic: str, objective: str) -> str:
    text = f"{topic} {objective}".lower()
    for canonical, keywords in TEACHER_TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return canonical
    return ""


def _build_teacher_prompt(topic: str, objective: str) -> str:
    topic_short = _short_words(topic, 12)
    objective_short = _short_words(objective, 16)
    return (
        f"2D teacher-style educational infographic explaining {topic_short}. "
        "Scene1 setup core entities and visual legend. "
        "Scene2 demonstrate mechanism step by step with clear directional arrows. "
        f"Scene3 show outcome linked to {objective_short}. "
        "Scene4 concise recap of full causal chain in one coherent diagram. "
        "Static camera, high contrast, large colored nodes and arrows, clean background, no text."
    )


def _build_academic_stable_prompt(
    topic: str,
    audience: str,
    objective: str,
    reference_context: str = "",
) -> str:
    topic_short = _short_words(topic, 10)
    objective_short = _short_words(objective, 14)
    audience_short = _short_words(audience, 8)
    context_short = _short_words(reference_context, 18) if reference_context else ""
    parts = [
        f"educational explainer scene about {topic_short}",
        f"for {audience_short}",
        f"goal: {objective_short}",
        "show one coherent visual sequence with explicit cause-effect motion",
        "use simple geometric objects, arrows, and clear transitions",
        "stable camera, balanced exposure, high contrast foreground",
        "no text, no numbers, no formulas, no logo, no watermark",
    ]
    if context_short:
        parts.append(f"reference facts: {context_short}")
    return normalize_request_to_english(". ".join(parts) + ".")


def _apply_diagram_lock_prompt(prompt: str) -> str:
    if not _env_flag("T2V_FORCE_DIAGRAM_PROMPT", True):
        return prompt
    lock_prefix = (
        "single 2D educational diagram, flat infographic vector look, "
        "large colored nodes connected by thick directional arrows, no photorealism"
    )
    lock_suffix = (
        "keep one continuous scene layout, static camera, clear foreground, "
        "high contrast edges, explicit causal arrow flow, no text"
    )
    return normalize_request_to_english(f"{lock_prefix}. {prompt}. {lock_suffix}.")


def _apply_diagram_lock_negative(negative_prompt: str) -> str:
    if not _env_flag("T2V_FORCE_DIAGRAM_PROMPT", True):
        return negative_prompt
    extra = (
        "photorealistic people, faces, landscapes, cinematic lighting, bokeh, lens flare, "
        "3d render, abstract particles, unrelated objects, scene changes"
    )
    return normalize_request_to_english(f"{negative_prompt}, {extra}")


def _frame_diagnostics(video_frames) -> dict[str, float]:
    if not video_frames:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "temporal_diff": 0.0,
        }
    indices = sorted({0, len(video_frames) // 2, len(video_frames) - 1})
    means = []
    stds = []
    min_vals = []
    max_vals = []
    sampled = []
    for idx in indices:
        arr = np.asarray(video_frames[idx].convert("RGB"), dtype=np.float32)
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))
        min_vals.append(float(arr.min()))
        max_vals.append(float(arr.max()))
        sampled.append(arr)
    temporal_diffs = []
    for i in range(len(sampled) - 1):
        temporal_diffs.append(float(np.mean(np.abs(sampled[i + 1] - sampled[i]))))
    return {
        "mean": float(sum(means) / len(means)),
        "std": float(sum(stds) / len(stds)),
        "min": float(min(min_vals)),
        "max": float(max(max_vals)),
        "temporal_diff": float(sum(temporal_diffs) / len(temporal_diffs)) if temporal_diffs else 0.0,
    }


def _detect_blank_mode(video_frames) -> str | None:
    if not video_frames:
        return "empty"

    diag = _frame_diagnostics(video_frames)
    mean = diag["mean"]
    std = diag["std"]
    min_val = diag["min"]
    max_val = diag["max"]
    temporal_diff = diag["temporal_diff"]

    if max_val <= 2.0 and mean <= 0.6:
        return "black"
    if min_val >= 250.0 and std <= 2.0:
        return "white"
    if mean <= 18.0 and std <= 6.0:
        return "underexposed"
    if mean >= 220.0 and std <= 6.0:
        return "washed_out"
    # "TV static" style failure: noisy high-variance frames with poor temporal coherence.
    if std >= 32.0 and 20.0 <= mean <= 235.0 and temporal_diff >= 24.0:
        return "noise_static"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Educational text-to-video generator")
    default_chunks_path = str(
        Path(__file__).resolve().parents[3] / "data" / "book_chunks.jsonl"
    )
    parser.add_argument(
        "--topic",
        default="an educational concept explained step by step",
        help="Educational topic to explain",
    )
    parser.add_argument(
        "--audience",
        default="high-school and early university students",
        help="Target audience",
    )
    parser.add_argument(
        "--objective",
        default="understand the core mechanism and practical intuition",
        help="Learning objective",
    )
    parser.add_argument(
        "--style",
        default="flat infographic style",
        help="Visual style",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Direct positive prompt override (bypasses educational prompt builder).",
    )
    parser.add_argument(
        "--negative",
        default="",
        help="Direct negative prompt override.",
    )
    parser.add_argument(
        "--canonical-topic",
        choices=tuple(CANONICAL_TOPIC_PROMPTS.keys()),
        default="",
        help="Use canonical short scientific prompt for selected topic.",
    )
    parser.add_argument(
        "--teacher-mode",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("T2V_TEACHER_MODE", False),
        help=(
            "Teacher autopilot: infer canonical educational prompt from user topic, "
            "apply stable defaults, and fallback to local diagram if diffusion fails."
        ),
    )
    parser.add_argument(
        "--academic-stable",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("T2V_ACADEMIC_STABLE", False),
        help=(
            "Stability-first preset for diffusion: compact structured prompt, "
            "safe Wan defaults, and overlay RAG by default."
        ),
    )
    parser.add_argument(
        "--output",
        default="/workspace/Disertatie/outputs/demo.mp4",
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
        help=(
            "Optional model id override (e.g. Wan-AI/Wan2.1-T2V-14B-Diffusers, "
            "Lightricks/LTX-Video-0.9.1, THUDM/CogVideoX-2b)"
        ),
    )
    parser.add_argument(
        "--model-preset",
        choices=tuple(MODEL_PRESETS.keys()),
        default="",
        help="Convenience model preset. Ignored if --model-id is provided.",
    )
    parser.add_argument(
        "--lora-path",
        default="",
        help=(
            "Optional LoRA path or HF repo id to load into diffusion pipeline "
            "(e.g. /workspace/loras/my_lora or org/repo)."
        ),
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=-1.0,
        help="Optional LoRA scale (e.g. 0.6 to 1.0).",
    )
    parser.add_argument(
        "--lora-weight-name",
        default="",
        help="Optional LoRA weight filename inside repo/folder (e.g. pytorch_lora_weights.safetensors).",
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
        "--stability-smoke",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("T2V_STABILITY_SMOKE", False),
        help=(
            "Apply known-stable CogVideo baseline (24f, 8fps, 32 steps, guidance 4, "
            "576x320, seed 42) unless explicit overrides are provided."
        ),
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
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use retrieved context from chunked book JSONL",
    )
    parser.add_argument(
        "--rag-mode",
        choices=("overlay", "semantic"),
        default="",
        help=(
            "RAG injection mode. `overlay` injects raw context text; "
            "`semantic` builds structured cues and scene plan."
        ),
    )
    parser.add_argument(
        "--purge-hf-cache",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("T2V_PURGE_HF_CACHE", False),
        help=(
            "Delete /workspace HF cache before run (forces model redownload). "
            "Can be enabled globally with T2V_PURGE_HF_CACHE=1."
        ),
    )
    parser.add_argument(
        "--prune-hf-cache",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("T2V_PRUNE_HF_CACHE", True),
        help=(
            "Prune HF model cache before run, keeping only current generation model "
            "(and embedding model if RAG is enabled). Default: enabled."
        ),
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
    parser.add_argument(
        "--scene-frames",
        type=int,
        default=int(os.getenv("T2V_SCENE_FRAMES", "16")),
        help="Frames per scene when using semantic RAG scene generation.",
    )
    parser.add_argument(
        "--use-llm-planner",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("T2V_USE_LLM_PLANNER", False),
        help="Use Llama planner to produce scene directives before generation.",
    )
    parser.add_argument(
        "--planner-model-id",
        default=os.getenv("T2V_PLANNER_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        help="Model id used for LLM planning.",
    )
    parser.add_argument(
        "--planner-max-new-tokens",
        type=int,
        default=int(os.getenv("T2V_PLANNER_MAX_NEW_TOKENS", "512")),
        help="Max new tokens for planner generation.",
    )
    parser.add_argument(
        "--planner-temperature",
        type=float,
        default=float(os.getenv("T2V_PLANNER_TEMPERATURE", "0.2")),
        help="Planner sampling temperature.",
    )
    parser.add_argument(
        "--planner-top-p",
        type=float,
        default=float(os.getenv("T2V_PLANNER_TOP_P", "0.9")),
        help="Planner nucleus sampling top-p.",
    )
    parser.add_argument(
        "--print-prompt",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("T2V_PRINT_PROMPT", False),
        help="Print final positive/negative prompts before generation.",
    )
    parser.add_argument(
        "--force-diagram-prompt",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("T2V_FORCE_DIAGRAM_PROMPT", True),
        help=(
            "Force strict 2D diagram-lock prompt constraints. "
            "Disable for broader illustrative visuals."
        ),
    )
    parser.add_argument(
        "--fallback-local-on-fail",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("T2V_FALLBACK_LOCAL_ON_FAIL", True),
        help=(
            "Fallback to deterministic local diagram generation if diffusion cannot load "
            "or output remains low-information."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_engine = args.engine
    academic_stable = bool(args.academic_stable and not args.prompt.strip())
    canonical_topic = args.canonical_topic.strip()
    topic_en = normalize_request_to_english(args.topic)
    audience_en = normalize_request_to_english(args.audience)
    objective_en = normalize_request_to_english(args.objective)
    style_en = normalize_request_to_english(args.style)
    prompt_override = normalize_request_to_english(args.prompt.strip())
    negative_override = normalize_request_to_english(args.negative.strip())
    rag_mode_requested = (args.rag_mode or "").strip().lower()
    if academic_stable and not rag_mode_requested:
        rag_mode_requested = "overlay"
    if academic_stable:
        args.use_rag = True
    rag_mode = rag_mode_requested if rag_mode_requested in {"overlay", "semantic"} else (
        "overlay" if args.use_rag else "overlay"
    )
    teacher_mode = bool(args.teacher_mode and not prompt_override and not args.use_rag)
    if teacher_mode and not canonical_topic:
        inferred = _infer_teacher_canonical_topic(topic_en, objective_en)
        if inferred:
            canonical_topic = inferred
            print(f"Teacher mode: inferred canonical topic '{canonical_topic}'.")
    use_rag_effective = bool(args.use_rag and not prompt_override)
    semantic_rag_mode = bool(use_rag_effective and rag_mode == "semantic")
    use_planner_effective = bool(
        args.use_llm_planner
        and not prompt_override
        and not canonical_topic
        and not teacher_mode
        and not semantic_rag_mode
    )
    planner_directives = ""
    planner_rag_query = ""

    setup_environment()
    if args.model_id:
        os.environ["T2V_MODEL_ID"] = args.model_id
    elif args.model_preset:
        os.environ["T2V_MODEL_ID"] = MODEL_PRESETS[args.model_preset]
    if args.lora_path:
        os.environ["T2V_LORA_PATH"] = args.lora_path
    if args.lora_scale is not None and args.lora_scale >= 0:
        os.environ["T2V_LORA_SCALE"] = str(args.lora_scale)
    if args.lora_weight_name:
        os.environ["T2V_LORA_WEIGHT_NAME"] = args.lora_weight_name
    if academic_stable and args.engine == "diffusion":
        if not os.getenv("T2V_DTYPE"):
            os.environ["T2V_DTYPE"] = "bf16"
        if not (args.fps and args.fps > 0):
            os.environ["FPS"] = "12"
        if not (args.frames and args.frames > 0):
            os.environ["NUM_FRAMES"] = "25"
        if not (args.steps and args.steps > 0):
            os.environ["NUM_INFERENCE_STEPS"] = "36"
        if not (args.guidance and args.guidance > 0):
            os.environ["GUIDANCE_SCALE"] = "4.5"
        if not (args.width and args.width > 0):
            os.environ["WIDTH"] = "832"
        if not (args.height and args.height > 0):
            os.environ["HEIGHT"] = "480"
        os.environ.setdefault("T2V_USE_DYNAMIC_CFG", "0")
        os.environ.setdefault("T2V_RETRY_ON_BLANK", "1")
        os.environ.setdefault("T2V_FORCE_DIAGRAM_PROMPT", "0")
        print("Academic stable preset enabled (bf16, 25f, 36steps, 832x480, overlay RAG).")
    if teacher_mode and args.engine == "diffusion":
        if not (args.fps and args.fps > 0):
            os.environ["FPS"] = "8"
        if not (args.frames and args.frames > 0):
            os.environ["NUM_FRAMES"] = "24"
        if not (args.steps and args.steps > 0):
            os.environ["NUM_INFERENCE_STEPS"] = "32"
        if not (args.guidance and args.guidance > 0):
            os.environ["GUIDANCE_SCALE"] = "4.2"
        if not (args.width and args.width > 0):
            os.environ["WIDTH"] = "720"
        if not (args.height and args.height > 0):
            os.environ["HEIGHT"] = "480"
        if args.seed is None or args.seed < 0:
            os.environ["T2V_SEED"] = "13"
        os.environ.setdefault("T2V_USE_DYNAMIC_CFG", "0")
        os.environ.setdefault("T2V_PROMPT_MAX_WORDS", "145")
        print("Teacher mode: stable defaults enabled (24f/8fps/32steps/guidance4.2/720x480).")
    if args.stability_smoke:
        # Conservative baseline known to reduce white/black collapse for CogVideo.
        if not (args.fps and args.fps > 0):
            os.environ["FPS"] = "8"
        if not (args.frames and args.frames > 0):
            os.environ["NUM_FRAMES"] = "24"
        if not (args.steps and args.steps > 0):
            os.environ["NUM_INFERENCE_STEPS"] = "32"
        if not (args.guidance and args.guidance > 0):
            os.environ["GUIDANCE_SCALE"] = "4.0"
        if not (args.width and args.width > 0):
            os.environ["WIDTH"] = "576"
        if not (args.height and args.height > 0):
            os.environ["HEIGHT"] = "320"
        if args.seed is None or args.seed < 0:
            os.environ["T2V_SEED"] = "42"
        os.environ.setdefault("T2V_USE_DYNAMIC_CFG", "0")
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
    force_diagram = bool(args.force_diagram_prompt)
    if academic_stable:
        force_diagram = False
    os.environ["T2V_FORCE_DIAGRAM_PROMPT"] = "1" if force_diagram else "0"

    if prompt_override:
        if args.use_llm_planner:
            print("Prompt override active: skipping LLM planner.")
        if args.use_rag:
            print("Prompt override active: skipping RAG retrieval.")
    elif semantic_rag_mode:
        if args.use_llm_planner:
            print("Semantic RAG mode: skipping LLM planner.")
        print("Semantic RAG mode enabled: retrieval cues -> scene plan -> per-scene generation.")
    elif canonical_topic:
        if args.use_llm_planner:
            print("Canonical topic prompt active: skipping LLM planner.")
        if args.use_rag:
            print("Canonical topic prompt active: skipping RAG retrieval.")
    elif teacher_mode and (args.use_llm_planner or args.use_rag):
        print("Teacher mode: planner/RAG disabled for stability.")
    if use_planner_effective:
        planner = build_scene_plan_with_llama(
            topic=topic_en,
            audience=audience_en,
            objective=objective_en,
            style=style_en,
            model_id=args.planner_model_id,
            max_new_tokens=max(128, args.planner_max_new_tokens),
            temperature=max(0.0, args.planner_temperature),
            top_p=min(max(0.0, args.planner_top_p), 1.0),
        )
        planner_directives = plan_to_scene_directives(planner)
        planner_rag_query = plan_to_rag_query(planner)
        topic_en = normalize_request_to_english(str(planner.get("topic", topic_en)))
        objective_en = normalize_request_to_english(str(planner.get("objective", objective_en)))
        style_en = normalize_request_to_english(str(planner.get("style", style_en)))
        if args.print_prompt:
            print("Planner scene directives:", planner_directives)
            print("Planner RAG query:", planner_rag_query)

    cfg = build_runtime_config(args.output)
    keep_model_ids = [cfg.model_id]
    if use_rag_effective:
        keep_model_ids.append(os.getenv("T2V_EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID))
    clear_runtime_cache(
        purge_hf_cache=args.purge_hf_cache,
        prune_hf_cache=bool(args.prune_hf_cache and not args.purge_hf_cache),
        keep_model_ids=keep_model_ids,
    )
    print(
        "Generation config:"
        f" engine={run_engine}"
        f" family={cfg.model_family}"
        f" model={cfg.model_id}"
        f" device={cfg.device}"
        f" dtype={cfg.dtype}"
        f" frames={cfg.num_frames}"
        f" fps={cfg.fps}"
        f" seconds~={cfg.num_frames / max(cfg.fps, 1):.2f}"
        f" size={cfg.width or 'default'}x{cfg.height or 'default'}"
        f" steps={cfg.num_inference_steps}"
        f" guidance={cfg.guidance_scale}"
        f" dynamic_cfg={cfg.use_dynamic_cfg}"
        f" seed={cfg.seed if cfg.seed is not None else 'auto'}"
    )
    lora_path = os.getenv("T2V_LORA_PATH", "").strip()
    if lora_path:
        print(
            "LoRA config:"
            f" path={lora_path}"
            f" scale={os.getenv('T2V_LORA_SCALE', '1.0')}"
            f" weight_name={os.getenv('T2V_LORA_WEIGHT_NAME', 'auto') or 'auto'}"
        )
    print(
        "Cache paths:"
        f" HF_HOME={os.getenv('HF_HOME')}"
        f" HUB={os.getenv('HUGGINGFACE_HUB_CACHE')}"
        f" TMPDIR={os.getenv('TMPDIR')}"
    )
    if args.purge_hf_cache:
        print("Cache cleanup: full HF purge enabled.")
    elif args.prune_hf_cache:
        kept = ", ".join(dict.fromkeys(keep_model_ids))
        print(f"Cache cleanup: pruned HF cache, kept model caches: {kept}")
    else:
        print("Cache cleanup: disabled.")
    if (
        run_engine == "diffusion"
        and cfg.model_family == "cogvideo"
        and cfg.guidance_scale > 5.5
    ):
        print(
            "Warning: guidance > 5.5 can destabilize CogVideo outputs. "
            "Recommended range: 3.5 to 5.0."
        )
    if (
        run_engine == "diffusion"
        and cfg.model_family == "cogvideo"
        and cfg.width
        and cfg.height
        and (cfg.width * cfg.height > 576 * 320)
    ):
        print(
            "Warning: high resolution increases white/black collapse risk on first pass. "
            "Consider starting with 576x320, then scale up."
        )
    if run_engine == "diffusion" and cfg.model_family == "cogvideo" and cfg.num_frames > 32:
        print("Warning: >32 frames can reduce stability; start with 24 or 32.")
    pipe = None
    if run_engine == "diffusion":
        try:
            pipe = load_pipeline(cfg)
        except Exception as exc:
            if args.fallback_local_on_fail:
                print(
                    "Diffusion pipeline unavailable; switching to local fallback."
                    f" reason={exc}"
                )
                run_engine = "local"
            else:
                raise

    reference_context = ""
    rag_cues: dict | None = None
    if use_rag_effective:
        query = normalize_request_to_english(
            args.rag_query.strip() or planner_rag_query or args.topic
        )
        if rag_mode == "semantic":
            rag_cues = retrieve_cues(
                chunks_path=Path(args.chunks_path),
                query=query,
                top_k=args.rag_top_k,
            )
            retrieval_path = save_retrieval_json(rag_cues, cfg.output_dir)
            print(
                "RAG semantic enabled"
                f" (EN query: {query})."
                f" concepts={len(rag_cues.get('concepts', []))}"
                f" hits={len(rag_cues.get('hits', []))}."
                f" Saved: {retrieval_path}"
            )
        else:
            reference_context = retrieve_context(
                chunks_path=Path(args.chunks_path),
                query=query,
                top_k=args.rag_top_k,
            )
            print(
                f"RAG overlay enabled (EN query: {query}). Retrieved context chars: {len(reference_context)}"
            )

    semantic_scene_mode = bool(run_engine == "diffusion" and semantic_rag_mode)
    scene_plan = None
    if semantic_rag_mode:
        cues_payload = rag_cues or {
            "query": "",
            "hits": [],
            "concepts": [],
            "visual_cues": [],
            "motion_cues": [],
            "constraints": [],
        }
        scene_frames = max(2, int(args.scene_frames))
        scene_plan = build_scene_plan(
            topic=topic_en,
            objective=objective_en,
            style=style_en,
            cues=cues_payload,
            scene_frames=scene_frames,
        )
        plan_path = save_plan_json(scene_plan, cfg.output_dir)
        total_scene_frames = scene_frames * len(scene_plan.get("scenes", []))
        print(
            "Semantic scene plan enabled:"
            f" scenes={len(scene_plan.get('scenes', []))}"
            f" scene_frames={scene_frames}"
            f" total_frames={total_scene_frames}."
            f" Saved: {plan_path}"
        )

    if run_engine == "local":
        if semantic_rag_mode and not reference_context and isinstance(rag_cues, dict):
            concepts = rag_cues.get("concepts", [])
            if isinstance(concepts, list):
                reference_context = ", ".join(str(x) for x in concepts[:16])
        if prompt_override:
            prompt = prompt_override
        elif canonical_topic:
            prompt = CANONICAL_TOPIC_PROMPTS[canonical_topic]
        elif teacher_mode:
            prompt = _build_teacher_prompt(topic_en, objective_en)
        elif academic_stable:
            prompt = _build_academic_stable_prompt(
                topic=topic_en,
                audience=audience_en,
                objective=objective_en,
                reference_context=reference_context,
            )
        else:
            prompt = build_education_prompt(
                topic=topic_en,
                audience=audience_en,
                objective=objective_en,
                style=style_en,
                reference_context=reference_context,
                planner_directives=planner_directives,
            )
        if args.print_prompt:
            print("Prompt:", prompt)
        video_frames = generate_local_diagram_frames(
            cfg,
            topic=topic_en,
            objective=objective_en,
            reference_context=reference_context,
        )
    elif semantic_scene_mode and scene_plan is not None:
        scenes = [
            scene
            for scene in scene_plan.get("scenes", [])
            if isinstance(scene, dict) and str(scene.get("prompt", "")).strip()
        ]
        if not scenes:
            print("Semantic scene plan has no valid scenes. Falling back to single prompt.")
            prompt = build_education_prompt(
                topic=topic_en,
                audience=audience_en,
                objective=objective_en,
                style=style_en,
                reference_context=reference_context,
                planner_directives=planner_directives,
            )
            if academic_stable:
                prompt = _build_academic_stable_prompt(
                    topic=topic_en,
                    audience=audience_en,
                    objective=objective_en,
                    reference_context=reference_context,
                )
            negative_prompt = negative_override or CANONICAL_NEGATIVE_PROMPT
            video_frames = generate_frames(pipe, cfg, prompt, negative_prompt)
        else:
            video_frames = []
            blank_scene_modes: list[str] = []
            global_negative = normalize_request_to_english(
                str(scene_plan.get("global_negative_prompt", ""))
            )
            for scene in scenes:
                scene_id = normalize_request_to_english(
                    str(scene.get("id") or scene.get("scene_id") or "?")
                )
                scene_prompt = _apply_diagram_lock_prompt(
                    normalize_request_to_english(str(scene.get("prompt", "")))
                )
                scene_negative = normalize_request_to_english(
                    str(scene.get("negative_prompt", ""))
                )
                if not scene_negative:
                    scene_negative = (
                        global_negative
                        or negative_override
                        or CANONICAL_NEGATIVE_PROMPT
                    )
                scene_negative = _apply_diagram_lock_negative(scene_negative)
                frames_for_scene = max(2, int(scene.get("frames", args.scene_frames)))
                print(f"Scene {scene_id} prompt: {scene_prompt}")
                print(f"Scene {scene_id} negative: {scene_negative}")
                scene_frames_out = generate_frames(
                    pipe,
                    cfg,
                    scene_prompt,
                    scene_negative,
                    num_frames=frames_for_scene,
                )
                scene_blank_mode = _detect_blank_mode(scene_frames_out)
                if scene_blank_mode is not None:
                    blank_scene_modes.append(scene_blank_mode)
                video_frames.extend(scene_frames_out)

            if blank_scene_modes:
                print(
                    "Semantic scene generation low-information scenes:"
                    f" {', '.join(blank_scene_modes)}"
                )
                if args.fallback_local_on_fail:
                    print("Switching to local deterministic fallback for final export.")
                    video_frames = generate_local_diagram_frames(
                        cfg,
                        topic=topic_en,
                        objective=objective_en,
                        reference_context=reference_context,
                    )
    else:
        if prompt_override:
            prompt = prompt_override
        elif canonical_topic:
            prompt = CANONICAL_TOPIC_PROMPTS[canonical_topic]
        elif teacher_mode:
            prompt = _build_teacher_prompt(topic_en, objective_en)
        elif academic_stable:
            prompt = _build_academic_stable_prompt(
                topic=topic_en,
                audience=audience_en,
                objective=objective_en,
                reference_context=reference_context,
            )
        else:
            prompt = build_education_prompt(
                topic=topic_en,
                audience=audience_en,
                objective=objective_en,
                style=style_en,
                reference_context=reference_context,
                planner_directives=planner_directives,
            )

        if negative_override:
            negative_prompt = negative_override
        elif canonical_topic or teacher_mode:
            negative_prompt = CANONICAL_NEGATIVE_PROMPT
        elif academic_stable:
            negative_prompt = (
                CANONICAL_NEGATIVE_PROMPT
                + ", chaotic noise pattern, static TV effect, random texture, extreme blur"
            )
        else:
            negative_prompt = build_negative_prompt()
        prompt = _apply_diagram_lock_prompt(prompt)
        negative_prompt = _apply_diagram_lock_negative(negative_prompt)
        if cfg.model_family == "cogvideo" and len(prompt.split()) > 60:
            print(
                "Warning: long prompt may hurt stability/topic adherence for CogVideo. "
                "Prefer <= 60 words for smoke tests."
            )
        if args.print_prompt:
            print("Prompt:", prompt)
            print("Negative prompt:", negative_prompt)
        video_frames = generate_frames(pipe, cfg, prompt, negative_prompt)
        retry_on_blank = _env_flag(
            "T2V_RETRY_ON_BLANK",
            _env_flag("T2V_RETRY_ON_BLACK", True),
        )
        blank_mode = _detect_blank_mode(video_frames)
        if retry_on_blank and blank_mode is not None:
            diag = _frame_diagnostics(video_frames)
            retry_prompt = prompt
            retry_negative_prompt = negative_prompt
            retry_guidance = cfg.guidance_scale
            retry_steps = cfg.num_inference_steps

            if blank_mode in {"black", "underexposed", "empty"}:
                retry_guidance = min(
                    cfg.guidance_scale,
                    float(os.getenv("T2V_BLACK_RETRY_GUIDANCE", "4.8")),
                )
                retry_steps = int(
                    os.getenv(
                        "T2V_BLACK_RETRY_STEPS",
                        str(max(22, int(round(cfg.num_inference_steps * 0.75)))),
                    )
                )
                retry_prompt = (
                    prompt
                    + " balanced midtone exposure, clearly lit foreground objects, neutral background, no dark empty scene."
                )
                retry_negative_prompt = (
                    negative_prompt
                    + ", dark empty background, silhouette-only scene, crushed shadows"
                )
            elif blank_mode == "white":
                retry_guidance = min(
                    cfg.guidance_scale,
                    float(os.getenv("T2V_WHITE_RETRY_GUIDANCE", "4.8")),
                )
                retry_steps = int(
                    os.getenv(
                        "T2V_WHITE_RETRY_STEPS",
                        str(max(28, int(round(cfg.num_inference_steps * 0.9)))),
                    )
                )
                retry_prompt = (
                    prompt
                    + " normal exposure, neutral gray background, visible edge contrast, no bright white screen."
                )
                retry_negative_prompt = (
                    negative_prompt
                    + ", white screen, overexposed frame, clipped highlights, blown-out exposure"
                )
            else:
                retry_guidance = min(
                    cfg.guidance_scale,
                    float(os.getenv("T2V_WASHED_RETRY_GUIDANCE", "4.5")),
                )
                retry_steps = int(
                    os.getenv(
                        "T2V_WASHED_RETRY_STEPS",
                        str(max(26, int(round(cfg.num_inference_steps * 0.85)))),
                    )
                )
                retry_prompt = (
                    prompt
                    + " stronger contrast, defined edges, saturated but natural colors, balanced midtones."
                )
                retry_negative_prompt = (
                    negative_prompt
                    + ", washed-out highlights, flat low-contrast frame, foggy haze, low-detail pastel blur"
                )
            if blank_mode == "noise_static":
                retry_guidance = min(
                    cfg.guidance_scale,
                    float(os.getenv("T2V_NOISE_RETRY_GUIDANCE", "4.2")),
                )
                retry_steps = int(
                    os.getenv(
                        "T2V_NOISE_RETRY_STEPS",
                        str(max(42, int(round(cfg.num_inference_steps * 1.25)))),
                    )
                )
                retry_prompt = (
                    prompt
                    + " coherent objects with smooth temporal continuity, stable edges, no random pixel noise."
                )
                retry_negative_prompt = (
                    negative_prompt
                    + ", TV static noise, random pixel pattern, granular snow, temporal incoherent texture"
                )

            print(
                "Detected low-information output. "
                f"mode={blank_mode} mean={diag['mean']:.2f} std={diag['std']:.2f} "
                f"min={diag['min']:.2f} max={diag['max']:.2f}. "
                "Retrying once with safer settings:"
                f" steps={retry_steps} guidance={retry_guidance} dynamic_cfg=False"
            )
            retried_frames = generate_frames(
                pipe,
                cfg,
                retry_prompt,
                retry_negative_prompt,
                num_inference_steps=retry_steps,
                guidance_scale=retry_guidance,
                use_dynamic_cfg=False,
            )
            retried_mode = _detect_blank_mode(retried_frames)
            if retried_mode is not None:
                retry_diag = _frame_diagnostics(retried_frames)
                print(
                    "Retry still low-information:"
                    f" mode={retried_mode} mean={retry_diag['mean']:.2f} std={retry_diag['std']:.2f}"
                )
                blank_mode = retried_mode
            else:
                video_frames = retried_frames
                blank_mode = None

        fallback_local_on_fail = bool(args.fallback_local_on_fail or teacher_mode)
        if fallback_local_on_fail and blank_mode is not None:
            print(
                "Diffusion output still low-information after retry. "
                f"mode={blank_mode}. Switching to local deterministic fallback."
            )
            video_frames = generate_local_diagram_frames(
                cfg,
                topic=topic_en,
                objective=objective_en,
                reference_context=reference_context,
            )

    try:
        export_outputs(video_frames, cfg)
    except Exception as exc:
        print(f"Video export failed: {exc}")
        return

    print(f"Saved video: {cfg.output_path}")
    if run_engine == "local":
        print("Model used: local-diagram")
    else:
        print(f"Model used: {cfg.model_id}")


if __name__ == "__main__":
    main()
