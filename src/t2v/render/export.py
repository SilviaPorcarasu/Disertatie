import os

from diffusers.utils import export_to_video

from t2v.config import RuntimeConfig


def export_outputs(video_frames, cfg: RuntimeConfig, **_ignored_kwargs) -> None:
    # Backward compatibility: older callers may still pass extra kwargs (e.g. topic).
    os.makedirs(cfg.output_dir, exist_ok=True)

    first = os.path.join(cfg.output_dir, "first_frame.png")
    middle = os.path.join(cfg.output_dir, "middle_frame.png")
    last = os.path.join(cfg.output_dir, "last_frame.png")

    video_frames[0].save(first)
    video_frames[len(video_frames) // 2].save(middle)
    video_frames[-1].save(last)

    export_to_video(video_frames, cfg.output_path, fps=cfg.fps)
