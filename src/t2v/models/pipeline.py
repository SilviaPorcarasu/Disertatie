import os
import sys
import types

from t2v.config import RuntimeConfig


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _patch_wan_ftfy_bug() -> None:
    """
    Work around a diffusers Wan bug where `ftfy` can be missing and later used unguarded.
    """
    try:
        import ftfy  # type: ignore
    except Exception:
        ftfy = types.ModuleType("ftfy")
        ftfy.fix_text = lambda text: text  # type: ignore[attr-defined]
        sys.modules["ftfy"] = ftfy

    try:
        from diffusers.utils import import_utils as _iu

        _iu._ftfy_available = True
    except Exception:
        pass

    wan_mod = sys.modules.get("diffusers.pipelines.wan.pipeline_wan")
    if wan_mod is not None and not hasattr(wan_mod, "ftfy"):
        wan_mod.ftfy = sys.modules["ftfy"]


def load_pipeline(cfg: RuntimeConfig):
    try:
        from diffusers import DiffusionPipeline
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: diffusers. Install it before running diffusion engine."
        ) from exc

    if cfg.model_family == "wan":
        _patch_wan_ftfy_bug()

    pipe = DiffusionPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
    )

    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()

    # 5B often exceeds 20GB VRAM if fully moved to CUDA.
    low_vram_default = cfg.model_family in {"cogvideo", "ltx", "wan", "hunyuanvideo"}
    use_low_vram = _env_flag("T2V_LOW_VRAM", default=low_vram_default)
    use_sequential_offload = _env_flag("T2V_SEQUENTIAL_OFFLOAD", default=False)

    if cfg.device == "cuda":
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")
        if use_sequential_offload and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        elif use_low_vram and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(cfg.device)
    else:
        pipe = pipe.to(cfg.device)

    return pipe
