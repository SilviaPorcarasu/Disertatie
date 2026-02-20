import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cache_root() -> Path:
    # Priority: explicit T2V cache root -> XDG cache root -> project default.
    root = (
        os.getenv("T2V_CACHE_ROOT", "").strip()
        or os.getenv("XDG_CACHE_HOME", "").strip()
        or "/workspace/.cache"
    )
    return Path(root)


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
    seed: int | None = None
    wan_flow_shift: float | None = None


def setup_environment() -> None:
    # Keep HF caches on persistent volume and off container root disk.
    cache_root = _cache_root()
    hf_root = Path(os.getenv("HF_HOME", "").strip()) if os.getenv("HF_HOME", "").strip() else cache_root / "huggingface"
    hub_root = (
        Path(os.getenv("HUGGINGFACE_HUB_CACHE", "").strip())
        if os.getenv("HUGGINGFACE_HUB_CACHE", "").strip()
        else (
            Path(os.getenv("HF_HUB_CACHE", "").strip())
            if os.getenv("HF_HUB_CACHE", "").strip()
            else hf_root / "hub"
        )
    )
    transformers_cache = (
        Path(os.getenv("TRANSFORMERS_CACHE", "").strip())
        if os.getenv("TRANSFORMERS_CACHE", "").strip()
        else hf_root
    )
    tmp_dir = Path(os.getenv("TMPDIR", "").strip()) if os.getenv("TMPDIR", "").strip() else cache_root / "tmp"

    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("HF_HOME", str(hf_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_root))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_root))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _hf_hub_cache_dir() -> Path:
    value = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE")
    if value:
        return Path(value)
    return _cache_root() / "huggingface" / "hub"


def _model_cache_dir_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def _prune_hf_model_cache(keep_model_ids: Iterable[str]) -> None:
    hub_dir = _hf_hub_cache_dir()
    if not hub_dir.exists():
        return

    keep_dir_names = {
        _model_cache_dir_name(model_id.strip())
        for model_id in keep_model_ids
        if model_id and model_id.strip()
    }

    for model_dir in hub_dir.glob("models--*"):
        if model_dir.name not in keep_dir_names:
            shutil.rmtree(model_dir, ignore_errors=True)

    locks_dir = hub_dir / ".locks"
    if locks_dir.exists():
        for lock_dir in locks_dir.glob("models--*"):
            if lock_dir.name not in keep_dir_names:
                shutil.rmtree(lock_dir, ignore_errors=True)


def clear_runtime_cache(
    *,
    purge_hf_cache: bool = False,
    prune_hf_cache: bool = False,
    keep_model_ids: list[str] | None = None,
) -> None:
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

    tmp_dir = Path(os.getenv("TMPDIR", str(_cache_root() / "tmp")))
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
    except Exception:
        pass

    if purge_hf_cache:
        hf_cache = Path(os.getenv("HF_HOME", str(_cache_root() / "huggingface")))
        try:
            shutil.rmtree(hf_cache, ignore_errors=True)
            os.makedirs(hf_cache, exist_ok=True)
        except Exception:
            pass
    elif prune_hf_cache:
        try:
            _prune_hf_model_cache(keep_model_ids or [])
        except Exception:
            pass


def select_model_id(has_cuda: bool) -> str:
    # Highest priority: explicit override for quick experiments.
    model_override = os.getenv("T2V_MODEL_ID", "").strip()
    if model_override:
        return model_override

    if has_cuda:
        return os.getenv("T2V_GPU_MODEL_ID", "Wan-AI/Wan2.1-T2V-14B-Diffusers")
    return os.getenv("T2V_CPU_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")


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


def _select_dtype(has_cuda: bool, model_family: str) -> torch.dtype:
    dtype_override = os.getenv("T2V_DTYPE", "").strip().lower()
    dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_override:
        selected = dtype_map.get(dtype_override)
        if selected is not None:
            return selected

    if not has_cuda:
        return torch.float32

    # Prefer bf16 on capable GPUs for Wan/CogVideo stability.
    if model_family in {"wan", "cogvideo"}:
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass

    # CogVideo models are prone to NaNs in fp16 on some GPUs.
    if model_family == "cogvideo":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def build_runtime_config(output_path: str) -> RuntimeConfig:
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    model_id = select_model_id(has_cuda)
    model_family = infer_model_family(model_id)
    dtype = _select_dtype(has_cuda, model_family)
    is_5b = model_family == "cogvideo" and "5b" in model_id.lower()
    default_quality_profile = "high" if has_cuda else "fast"
    if has_cuda and model_family == "cogvideo":
        default_quality_profile = os.getenv("T2V_COGVIDEO_DEFAULT_QUALITY", "balanced")
    quality_profile = os.getenv("T2V_QUALITY_PROFILE", default_quality_profile).strip().lower()
    if quality_profile not in {"fast", "balanced", "high"}:
        quality_profile = "balanced"

    fps_default = "8" if model_family == "cogvideo" else "16"
    fps = int(os.getenv("FPS", fps_default))
    if has_cuda and model_family == "cogvideo":
        if quality_profile == "fast":
            default_frames = 24 if is_5b else 21
        elif quality_profile == "balanced":
            default_frames = 32 if is_5b else 29
        else:
            default_frames = 49 if is_5b else 37
    elif has_cuda and model_family in {"ltx", "wan", "hunyuanvideo"}:
        if quality_profile == "fast":
            default_frames = 25
        elif quality_profile == "balanced":
            default_frames = 33
        else:
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
            if quality_profile == "fast":
                default_steps = "32" if is_5b else "20"
                default_guidance = "4.0" if is_5b else "5.0"
            elif quality_profile == "balanced":
                default_steps = "36" if is_5b else "24"
                default_guidance = "4.5" if is_5b else "5.5"
            else:
                default_steps = "40" if is_5b else "28"
                default_guidance = "5.0" if is_5b else "6.0"
        elif model_family == "ltx":
            default_steps = "24" if quality_profile == "fast" else ("30" if quality_profile == "balanced" else "38")
            default_guidance = "2.5" if quality_profile == "fast" else ("3.0" if quality_profile == "balanced" else "3.5")
        elif model_family == "wan":
            default_steps = "24" if quality_profile == "fast" else ("30" if quality_profile == "balanced" else "38")
            default_guidance = "4.5" if quality_profile == "fast" else "5.0"
        elif model_family == "hunyuanvideo":
            default_steps = "24" if quality_profile == "fast" else ("30" if quality_profile == "balanced" else "38")
            default_guidance = "4.0" if quality_profile == "fast" else ("4.5" if quality_profile == "balanced" else "5.0")
        else:
            default_steps = "16" if quality_profile == "fast" else ("20" if quality_profile == "balanced" else "28")
            default_guidance = "4.5" if quality_profile == "fast" else ("5.0" if quality_profile == "balanced" else "5.5")
    else:
        default_steps = "5"
        default_guidance = "5.0"

    num_inference_steps = int(os.getenv("NUM_INFERENCE_STEPS", default_steps))
    guidance_scale = float(os.getenv("GUIDANCE_SCALE", default_guidance))
    use_dynamic_cfg = _env_flag("T2V_USE_DYNAMIC_CFG", default=(model_family == "cogvideo" and is_5b))
    output_dir = os.path.dirname(output_path) or "."
    height_env = os.getenv("HEIGHT")
    width_env = os.getenv("WIDTH")
    seed_env = os.getenv("T2V_SEED", "").strip()
    height = int(height_env) if height_env else None
    width = int(width_env) if width_env else None
    seed = int(seed_env) if seed_env else None
    wan_flow_shift = None

    # Wan 2.1 official inference defaults use 480p or 720p with flow-shifted UniPC.
    if model_family == "wan":
        wan_official_resolution = _env_flag("T2V_WAN_OFFICIAL_RESOLUTION", True)
        if wan_official_resolution and (height is None or width is None):
            if quality_profile == "high":
                height, width = 720, 1280
            else:
                height, width = 480, 832
        flow_shift_env = os.getenv("T2V_WAN_FLOW_SHIFT", "").strip()
        if flow_shift_env:
            try:
                wan_flow_shift = float(flow_shift_env)
            except ValueError:
                wan_flow_shift = None
        if wan_flow_shift is None:
            if (height or 0) >= 700 or (width or 0) >= 1200:
                wan_flow_shift = 5.0
            else:
                wan_flow_shift = 3.0

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
        seed=seed,
        wan_flow_shift=wan_flow_shift,
    )
