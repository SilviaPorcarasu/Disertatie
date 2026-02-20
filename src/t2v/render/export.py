import os

import numpy as np

from t2v.config import RuntimeConfig


def _export_with_diffusers(video_frames, output_path: str, fps: int) -> str | None:
    try:
        from diffusers.utils import export_to_video
    except Exception as exc:
        return f"diffusers export unavailable: {exc}"

    try:
        export_to_video(video_frames, output_path, fps=fps)
        return None
    except Exception as exc:
        return f"diffusers export failed: {exc}"


def _export_with_imageio(video_frames, output_path: str, fps: int) -> str | None:
    try:
        import imageio.v3 as iio
    except Exception as exc:
        return f"imageio export unavailable: {exc}"

    try:
        # imageio expects numpy arrays, not PIL Image objects.
        arrays = [np.asarray(frame.convert("RGB")) for frame in video_frames]
        iio.imwrite(output_path, arrays, fps=fps)
        return None
    except Exception as exc:
        return f"imageio export failed: {exc}"


def _export_as_gif(video_frames, output_path: str, fps: int) -> tuple[str | None, str | None]:
    try:
        stem, _ext = os.path.splitext(output_path)
        gif_path = f"{stem}.gif"
        duration_ms = max(20, int(round(1000 / max(fps, 1))))
        frames = [frame.convert("RGB") for frame in video_frames]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        return gif_path, None
    except Exception as exc:
        return None, f"GIF export failed: {exc}"


def export_outputs(video_frames, cfg: RuntimeConfig, **_ignored_kwargs) -> None:
    # Backward compatibility: older callers may still pass extra kwargs (e.g. topic).
    os.makedirs(cfg.output_dir, exist_ok=True)

    first = os.path.join(cfg.output_dir, "first_frame.png")
    middle = os.path.join(cfg.output_dir, "middle_frame.png")
    last = os.path.join(cfg.output_dir, "last_frame.png")

    video_frames[0].save(first)
    video_frames[len(video_frames) // 2].save(middle)
    video_frames[-1].save(last)

    diffusers_error = _export_with_diffusers(video_frames, cfg.output_path, fps=cfg.fps)
    if diffusers_error is None:
        return
    imageio_error = _export_with_imageio(video_frames, cfg.output_path, fps=cfg.fps)
    if imageio_error is None:
        return

    gif_path, gif_error = _export_as_gif(video_frames, cfg.output_path, fps=cfg.fps)
    if gif_error is None and gif_path:
        print(
            "Video export backend unavailable for mp4. "
            f"Saved GIF fallback instead: {gif_path}"
        )
        return

    raise RuntimeError(
        "Cannot export media. Install system `ffmpeg` or `pip install imageio-ffmpeg`. "
        f"Details: {diffusers_error}; {imageio_error}; {gif_error}"
    )
