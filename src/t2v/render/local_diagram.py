from __future__ import annotations

import math
import textwrap
from typing import List

from PIL import Image, ImageDraw, ImageFont

from t2v.config import RuntimeConfig


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 4,
) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len = 12
    head_w = 7
    p1 = end
    p2 = (
        end[0] - head_len * math.cos(angle) + head_w * math.sin(angle),
        end[1] - head_len * math.sin(angle) - head_w * math.cos(angle),
    )
    p3 = (
        end[0] - head_len * math.cos(angle) - head_w * math.sin(angle),
        end[1] - head_len * math.sin(angle) + head_w * math.cos(angle),
    )
    draw.polygon([p1, p2, p3], fill=color)


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    max_width_chars: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> int:
    y = xy[1]
    for line in textwrap.wrap(text, width=max_width_chars):
        draw.text((xy[0], y), line, font=font, fill=fill)
        y += int(font.size * 1.35)
    return y


def _clean_context(reference_context: str) -> str:
    words = reference_context.replace("\n", " ").split()
    if not words:
        return ""
    return " ".join(words[:36])


def generate_local_diagram_frames(
    cfg: RuntimeConfig,
    *,
    topic: str,
    objective: str,
    reference_context: str = "",
) -> List[Image.Image]:
    width = cfg.width or 1280
    height = cfg.height or 720
    frames = max(2, cfg.num_frames)

    title_font = _font(42)
    subtitle_font = _font(22)
    small_font = _font(18)
    box_font = _font(20)

    context_line = _clean_context(reference_context)
    palette = {
        "bg": (246, 249, 252),
        "panel": (255, 255, 255),
        "ink": (22, 28, 36),
        "muted": (78, 92, 110),
        "accent": (0, 112, 243),
        "accent2": (14, 170, 100),
        "accent3": (243, 112, 33),
        "grid": (224, 232, 242),
    }

    out: List[Image.Image] = []
    for idx in range(frames):
        t = idx / max(frames - 1, 1)
        pulse = 0.5 + 0.5 * math.sin(t * math.pi * 2.0)

        im = Image.new("RGB", (width, height), palette["bg"])
        draw = ImageDraw.Draw(im)

        # Subtle grid background for technical feel.
        grid_step = 40
        for x in range(0, width, grid_step):
            draw.line([(x, 0), (x, height)], fill=palette["grid"], width=1)
        for y in range(0, height, grid_step):
            draw.line([(0, y), (width, y)], fill=palette["grid"], width=1)

        # Header panel.
        draw.rounded_rectangle(
            [(24, 20), (width - 24, 150)],
            radius=20,
            fill=palette["panel"],
            outline=(210, 220, 232),
            width=2,
        )
        draw.text((44, 34), "Academic Explainer Diagram", font=title_font, fill=palette["ink"])
        draw.text((46, 88), f"Topic: {topic}", font=subtitle_font, fill=palette["muted"])

        # Left panel: gradient descent curve.
        left_x0, left_y0 = 40, 180
        left_x1, left_y1 = width // 2 - 10, height - 40
        draw.rounded_rectangle(
            [(left_x0, left_y0), (left_x1, left_y1)],
            radius=18,
            fill=palette["panel"],
            outline=(210, 220, 232),
            width=2,
        )
        draw.text((left_x0 + 18, left_y0 + 12), "Loss Landscape", font=subtitle_font, fill=palette["ink"])

        # Axes.
        ox, oy = left_x0 + 70, left_y1 - 60
        draw.line([(ox, oy), (left_x1 - 35, oy)], fill=palette["muted"], width=3)
        draw.line([(ox, oy), (ox, left_y0 + 36)], fill=palette["muted"], width=3)
        draw.text((left_x1 - 115, oy + 12), "weights", font=small_font, fill=palette["muted"])
        draw.text((ox - 40, left_y0 + 46), "loss", font=small_font, fill=palette["muted"])

        # Draw convex curve.
        curve_pts = []
        curve_x0 = ox + 20
        curve_x1 = left_x1 - 55
        for i in range(140):
            u = i / 139.0
            x = curve_x0 + u * (curve_x1 - curve_x0)
            y = oy - (85 - 120 * (u - 0.66) ** 2)
            curve_pts.append((x, y))
        draw.line(curve_pts, fill=palette["accent"], width=5)

        # Animated point descending.
        p_u = 0.10 + 0.78 * t
        px = curve_x0 + p_u * (curve_x1 - curve_x0)
        py = oy - (85 - 120 * (p_u - 0.66) ** 2)
        r = int(8 + 2 * pulse)
        draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=palette["accent3"], outline=palette["ink"], width=2)
        _draw_arrow(draw, (px - 30, py - 30), (px - 5, py - 4), palette["accent3"], width=3)
        draw.text((px - 140, py - 52), "gradient step", font=small_font, fill=palette["accent3"])

        # Right panel: flow diagram.
        right_x0, right_y0 = width // 2 + 10, 180
        right_x1, right_y1 = width - 40, height - 40
        draw.rounded_rectangle(
            [(right_x0, right_y0), (right_x1, right_y1)],
            radius=18,
            fill=palette["panel"],
            outline=(210, 220, 232),
            width=2,
        )
        draw.text((right_x0 + 18, right_y0 + 12), "Training Loop", font=subtitle_font, fill=palette["ink"])

        boxes = [
            ("Forward pass", palette["accent"], (right_x0 + 34, right_y0 + 70)),
            ("Compute loss", palette["accent2"], (right_x0 + 34, right_y0 + 165)),
            ("Backpropagate", palette["accent3"], (right_x0 + 34, right_y0 + 260)),
            ("Update weights", (120, 90, 220), (right_x0 + 34, right_y0 + 355)),
        ]

        current = min(3, int(t * 4.0))
        for bi, (label, color, (bx, by)) in enumerate(boxes):
            bw, bh = 290, 68
            active = bi <= current
            fill = tuple(int(c * (1.0 if active else 0.35) + 210 * (0.0 if active else 0.65)) for c in color)
            draw.rounded_rectangle([(bx, by), (bx + bw, by + bh)], radius=14, fill=fill, outline=(205, 215, 225), width=2)
            draw.text((bx + 18, by + 20), label, font=box_font, fill=(15, 20, 30))
            if bi < len(boxes) - 1:
                _draw_arrow(draw, (bx + bw // 2, by + bh + 8), (bx + bw // 2, by + bh + 26), palette["muted"], width=3)

        # Bottom notes.
        if context_line:
            draw.rounded_rectangle(
                [(32, height - 92), (width - 32, height - 26)],
                radius=12,
                fill=(234, 243, 255),
                outline=(180, 205, 236),
                width=2,
            )
            _draw_wrapped_text(
                draw,
                f"RAG context: {context_line}",
                (46, height - 78),
                max_width_chars=max(40, width // 14),
                font=small_font,
                fill=(20, 35, 60),
            )
        else:
            draw.rounded_rectangle(
                [(32, height - 92), (width - 32, height - 26)],
                radius=12,
                fill=(241, 246, 250),
                outline=(200, 214, 228),
                width=2,
            )
            _draw_wrapped_text(
                draw,
                f"Objective: {objective}",
                (46, height - 78),
                max_width_chars=max(40, width // 14),
                font=small_font,
                fill=(26, 36, 50),
            )

        out.append(im)

    return out

