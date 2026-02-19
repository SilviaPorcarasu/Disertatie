import os

import numpy as np

from t2v.config import RuntimeConfig


def _export_with_diffusers(video_frames, output_path: str, fps: int) -> bool:
    try:
        from diffusers.utils import export_to_video
    except Exception:
        return False
    export_to_video(video_frames, output_path, fps=fps)
    return True


def _export_with_imageio(video_frames, output_path: str, fps: int) -> bool:
    try:
        import imageio.v3 as iio
    except Exception:
        return False

    # imageio expects numpy arrays, not PIL Image objects.
    arrays = [np.asarray(frame.convert("RGB")) for frame in video_frames]
    iio.imwrite(output_path, arrays, fps=fps)
    return True


def export_outputs(video_frames, cfg: RuntimeConfig, **_ignored_kwargs) -> None:
    # Backward compatibility: older callers may still pass extra kwargs (e.g. topic).
    os.makedirs(cfg.output_dir, exist_ok=True)

    first = os.path.join(cfg.output_dir, "first_frame.png")
    middle = os.path.join(cfg.output_dir, "middle_frame.png")
    last = os.path.join(cfg.output_dir, "last_frame.png")

    video_frames[0].save(first)
    video_frames[len(video_frames) // 2].save(middle)
    video_frames[-1].save(last)

    if _export_with_diffusers(video_frames, cfg.output_path, fps=cfg.fps):
        return
    if _export_with_imageio(video_frames, cfg.output_path, fps=cfg.fps):
        return
    raise RuntimeError(
        "Cannot export video: install either `diffusers` or `imageio`."
    )
