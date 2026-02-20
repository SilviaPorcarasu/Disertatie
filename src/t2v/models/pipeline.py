import os
import sys
import types

from t2v.config import RuntimeConfig


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _configure_wan_scheduler(pipe, flow_shift: float | None) -> None:
    if flow_shift is None:
        return
    try:
        from diffusers import UniPCMultistepScheduler
    except Exception:
        return
    try:
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=float(flow_shift),
        )
    except Exception:
        # Keep existing scheduler if replacement fails.
        return


def _apply_lora_if_requested(pipe, token: str | None) -> None:
    lora_path = os.getenv("T2V_LORA_PATH", "").strip()
    if not lora_path:
        return

    lora_scale = _env_float("T2V_LORA_SCALE", 1.0)
    weight_name = os.getenv("T2V_LORA_WEIGHT_NAME", "").strip() or None
    adapter_name = os.getenv("T2V_LORA_ADAPTER_NAME", "").strip() or "edu_lora"
    load_kwargs = {"adapter_name": adapter_name}
    if weight_name:
        load_kwargs["weight_name"] = weight_name
    if token:
        load_kwargs["token"] = token

    try:
        pipe.load_lora_weights(lora_path, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load LoRA from '{lora_path}'. "
            "Check path/repo id and optional weight file name."
        ) from exc

    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
        except Exception:
            pass
    if _env_flag("T2V_LORA_FUSE", True) and hasattr(pipe, "fuse_lora"):
        try:
            pipe.fuse_lora(lora_scale=lora_scale)
        except Exception:
            # LoRA is still active even if fuse is unavailable.
            pass


def _hf_offline_mode_enabled() -> bool:
    return (
        _env_flag("HF_HUB_OFFLINE", False)
        or _env_flag("TRANSFORMERS_OFFLINE", False)
        or _env_flag("HF_DATASETS_OFFLINE", False)
    )


def _is_offline_error(message: str) -> bool:
    text = message.lower()
    return "offline mode is enabled" in text or "offlinemodeisenabled" in text


def _load_with_offline_fallback(pipeline_cls, model_id: str, load_kwargs: dict):
    try:
        return pipeline_cls.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        if not (_hf_offline_mode_enabled() or _is_offline_error(str(exc))):
            raise

    local_kwargs = dict(load_kwargs)
    local_kwargs["local_files_only"] = True
    try:
        return pipeline_cls.from_pretrained(model_id, **local_kwargs)
    except Exception as cached_exc:
        raise RuntimeError(
            "HF offline mode is enabled but model cache is incomplete. "
            "Either unset HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE and rerun online, "
            "or predownload the model with snapshot_download and retry."
        ) from cached_exc


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

    pipeline_cls = DiffusionPipeline
    if cfg.model_family == "wan":
        _patch_wan_ftfy_bug()
        try:
            from diffusers import WanPipeline

            pipeline_cls = WanPipeline
        except Exception:
            pipeline_cls = DiffusionPipeline

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    load_kwargs = {
        "torch_dtype": cfg.dtype,
    }
    if token:
        # Explicit token forwarding avoids ambiguous auth state in unstable shells.
        load_kwargs["token"] = token

    try:
        pipe = _load_with_offline_fallback(
            pipeline_cls,
            cfg.model_id,
            load_kwargs,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "tiktoken" in message or "protobuf" in message:
            raise RuntimeError(
                "Tokenizer dependencies missing for diffusion model. "
                "Install with: pip install protobuf tiktoken"
            ) from exc
        if "readtimeout" in message or "timed out" in message:
            raise RuntimeError(
                "HF download timed out. Set HF_TOKEN and predownload model with "
                "snapshot_download (single worker), then rerun with HF_HUB_OFFLINE=1."
            ) from exc
        raise

    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
    if cfg.model_family == "wan" and _env_flag("T2V_WAN_OFFICIAL_SCHEDULER", False):
        _configure_wan_scheduler(pipe, cfg.wan_flow_shift)
    _apply_lora_if_requested(pipe, token)

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
