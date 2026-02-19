import os
import shutil
from dataclasses import dataclass

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RuntimeConfig:
    model_family: str
    model_id: str
    device: str
    dtype: torch.dtype
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    use_dynamic_cfg: bool
    output_path: str
    output_dir: str
    fps: int = 12
    height: int | None = None
    width: int | None = None


def setup_environment() -> None:
    # Keep HF caches on persistent volume and off container root disk.
    os.environ["XDG_CACHE_HOME"] = "/workspace/.cache"
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"
    os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    tmp_dir = "/workspace/.cache/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def clear_runtime_cache(*, purge_hf_cache: bool = False) -> None:
    try:
        import gc

        gc.collect()
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    tmp_dir = "/workspace/.cache/tmp"
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
    except Exception:
        pass

    if purge_hf_cache:
        hf_cache = "/workspace/.cache/huggingface"
        try:
            shutil.rmtree(hf_cache, ignore_errors=True)
            os.makedirs(hf_cache, exist_ok=True)
        except Exception:
            pass


def select_model_id(has_cuda: bool) -> str:
    # Allow quick model experiments without code edits.
    return os.getenv(
        "T2V_MODEL_ID",
        "THUDM/CogVideoX-5b" if has_cuda else "THUDM/CogVideoX-2b",
    )


def infer_model_family(model_id: str) -> str:
    model = model_id.lower()
    if "cogvideo" in model:
        return "cogvideo"
    if "ltx" in model:
        return "ltx"
    if "wan" in model:
        return "wan"
    if "hunyuanvideo" in model or "hunyuan-video" in model:
        return "hunyuanvideo"
    return "generic"


def build_runtime_config(output_path: str) -> RuntimeConfig:
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32
    model_id = select_model_id(has_cuda)
    model_family = infer_model_family(model_id)
    is_5b = model_family == "cogvideo" and "5b" in model_id.lower()

    fps_default = "8" if model_family == "cogvideo" else "16"
    fps = int(os.getenv("FPS", fps_default))
    if has_cuda and model_family == "cogvideo":
        default_frames = 25 if is_5b else 33
    elif has_cuda and model_family in {"ltx", "wan", "hunyuanvideo"}:
        default_frames = 49
    else:
        default_frames = 17 if has_cuda else 9
    num_frames_env = os.getenv("NUM_FRAMES")
    duration_seconds_env = os.getenv("DURATION_SECONDS")
    if num_frames_env is not None:
        num_frames = int(num_frames_env)
    elif duration_seconds_env is not None:
        num_frames = max(2, int(round(float(duration_seconds_env) * fps)))
    else:
        num_frames = default_frames

    if model_family == "wan" and num_frames % 4 != 1:
        # Wan requires `num_frames - 1` divisible by 4.
        lower = ((num_frames - 1) // 4) * 4 + 1
        upper = lower + 4
        if lower < 1:
            num_frames = upper
        else:
            num_frames = lower if abs(num_frames - lower) <= abs(upper - num_frames) else upper
    if has_cuda:
        if model_family == "cogvideo":
            default_steps = "28" if is_5b else "16"
            default_guidance = "6.0" if is_5b else "7.0"
        elif model_family == "ltx":
            default_steps = "30"
            default_guidance = "3.0"
        elif model_family == "wan":
            default_steps = "30"
            default_guidance = "5.0"
        elif model_family == "hunyuanvideo":
            default_steps = "30"
            default_guidance = "4.5"
        else:
            default_steps = "20"
            default_guidance = "5.0"
    else:
        default_steps = "5"
        default_guidance = "5.0"

    num_inference_steps = int(os.getenv("NUM_INFERENCE_STEPS", default_steps))
    guidance_scale = float(os.getenv("GUIDANCE_SCALE", default_guidance))
    use_dynamic_cfg = _env_flag("T2V_USE_DYNAMIC_CFG", default=(model_family == "cogvideo" and is_5b))
    output_dir = os.path.dirname(output_path) or "."
    height_env = os.getenv("HEIGHT")
    width_env = os.getenv("WIDTH")
    height = int(height_env) if height_env else None
    width = int(width_env) if width_env else None

    return RuntimeConfig(
        model_family=model_family,
        model_id=model_id,
        device=device,
        dtype=dtype,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        use_dynamic_cfg=use_dynamic_cfg,
        output_path=output_path,
        output_dir=output_dir,
        fps=fps,
        height=height,
        width=width,
    )
