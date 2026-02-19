from __future__ import annotations

import inspect

import numpy as np
from PIL import Image

from t2v.config import RuntimeConfig


def _to_pil_frame(frame) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected frame shape: {arr.shape}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise RuntimeError(f"Unexpected channel count: {arr.shape[-1]}")

    return Image.fromarray(arr, mode="RGB")


def _extract_video_frames(output) -> list[Image.Image]:
    frames = getattr(output, "frames", None)
    if frames is None and isinstance(output, (tuple, list)) and output:
        frames = output[0]
    if frames is None:
        raise RuntimeError("Pipeline output has no `frames` field.")

    if isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]
        if frames.ndim != 4:
            raise RuntimeError(f"Unexpected ndarray video shape: {frames.shape}")
        if frames.shape[-1] in (1, 3, 4):
            pass
        elif frames.shape[1] in (1, 3, 4):
            pass
        elif frames.shape[0] in (1, 3, 4) and frames.shape[1] > 4:
            frames = np.transpose(frames, (1, 0, 2, 3))
        else:
            raise RuntimeError(f"Unsupported ndarray frame layout: {frames.shape}")
        return [_to_pil_frame(frames[i]) for i in range(frames.shape[0])]

    if not isinstance(frames, list):
        try:
            import torch

            if torch.is_tensor(frames):
                arr = frames.detach().cpu().numpy()
                if arr.ndim == 5:
                    arr = arr[0]
                if arr.ndim != 4:
                    raise RuntimeError(f"Unexpected tensor video shape: {arr.shape}")
                if arr.shape[-1] in (1, 3, 4):
                    pass
                elif arr.shape[1] in (1, 3, 4):
                    pass
                elif arr.shape[0] in (1, 3, 4) and arr.shape[1] > 4:
                    arr = np.transpose(arr, (1, 0, 2, 3))
                else:
                    raise RuntimeError(f"Unsupported tensor frame layout: {arr.shape}")
                return [_to_pil_frame(arr[i]) for i in range(arr.shape[0])]
        except Exception:
            pass
        raise RuntimeError(f"Unsupported `frames` type: {type(frames)}")

    if not frames:
        return []

    if isinstance(frames[0], list):
        frames = frames[0]

    return [_to_pil_frame(frame) for frame in frames]


def generate_frames(
    pipe,
    cfg: RuntimeConfig,
    prompt: str,
    negative_prompt: str,
    *,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    use_dynamic_cfg: bool | None = None,
):
    output_type = "np" if cfg.model_family == "wan" else "pil"
    call_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": cfg.num_frames,
        "num_inference_steps": cfg.num_inference_steps
        if num_inference_steps is None
        else num_inference_steps,
        "guidance_scale": cfg.guidance_scale if guidance_scale is None else guidance_scale,
        "use_dynamic_cfg": cfg.use_dynamic_cfg if use_dynamic_cfg is None else use_dynamic_cfg,
        "frame_rate": cfg.fps,
        "height": cfg.height,
        "width": cfg.width,
        "output_type": output_type,
    }

    accepted = set(inspect.signature(pipe.__call__).parameters.keys())
    filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted and v is not None}

    try:
        output = pipe(**filtered_kwargs)
    except Exception as exc:
        # Some pipelines expose output_type but reject certain values; retry once without it.
        if "output_type" in filtered_kwargs and "output_type" in str(exc).lower():
            retry_kwargs = dict(filtered_kwargs)
            retry_kwargs.pop("output_type", None)
            output = pipe(**retry_kwargs)
        else:
            raise
    return _extract_video_frames(output)
