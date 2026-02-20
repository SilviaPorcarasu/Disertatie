from __future__ import annotations

import math
import re
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


_TERM_STOPWORDS = {
    "about",
    "across",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "into",
    "its",
    "the",
    "to",
    "understand",
    "what",
    "with",
}


def _infer_local_visual_mode(topic: str, objective: str) -> str:
    text = f"{topic} {objective}".lower()
    if any(k in text for k in {"attention", "token", "context"}):
        return "attention_links"
    if any(k in text for k in {"overfit", "generalization", "bias", "variance", "compare", "versus", "tradeoff"}):
        return "comparison_curves"
    if any(k in text for k in {"backprop", "gradient flow", "neural network", "layer"}):
        return "network_flow"
    return "valley_ball"


def _extract_terms(text: str, max_terms: int = 6) -> list[str]:
    ranked: list[str] = []
    for token in re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower()):
        if token in _TERM_STOPWORDS:
            continue
        if token not in ranked:
            ranked.append(token)
        if len(ranked) >= max_terms:
            break
    return ranked


def _build_stage_labels(topic: str, objective: str) -> list[str]:
    terms = _extract_terms(f"{topic} {objective}", max_terms=6)
    if len(terms) >= 4:
        return [
            f"Introduce {terms[0]}",
            f"Link {terms[1]} -> {terms[2]}",
            f"Update using {terms[3]}",
            f"Recap {terms[0]} flow",
        ]
    return [
        "Introduce core elements",
        "Show interactions",
        "Apply step-by-step update",
        "Recap final intuition",
    ]


def _draw_left_panel_visual(
    draw: ImageDraw.ImageDraw,
    mode: str,
    t: float,
    pulse: float,
    panel: tuple[int, int, int, int],
    palette: dict[str, tuple[int, int, int]],
    subtitle_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    left_x0, left_y0, left_x1, left_y1 = panel
    draw.rounded_rectangle(
        [(left_x0, left_y0), (left_x1, left_y1)],
        radius=18,
        fill=palette["panel"],
        outline=(210, 220, 232),
        width=2,
    )

    if mode == "comparison_curves":
        draw.text((left_x0 + 18, left_y0 + 12), "Model Comparison", font=subtitle_font, fill=palette["ink"])
        mid = (left_x0 + left_x1) // 2
        draw.line([(mid, left_y0 + 56), (mid, left_y1 - 18)], fill=(220, 228, 238), width=2)
        for col, is_smooth in ((left_x0 + 36, True), (mid + 24, False)):
            ox, oy = col, left_y1 - 64
            x1 = col + (mid - left_x0 - 58 if is_smooth else left_x1 - mid - 58)
            draw.line([(ox, oy), (x1, oy)], fill=palette["muted"], width=2)
            draw.line([(ox, oy), (ox, left_y0 + 76)], fill=palette["muted"], width=2)
            pts = []
            n = 110
            for i in range(n):
                u = i / (n - 1)
                x = ox + 14 + u * (x1 - ox - 18)
                if is_smooth:
                    y = oy - (80 - 105 * (u - 0.55) ** 2)
                else:
                    y = oy - (
                        70
                        - 95 * (u - 0.55) ** 2
                        + 16 * math.sin(18 * u + 9 * t)
                        + 8 * math.sin(33 * u + 12 * t)
                    )
                pts.append((x, y))
            draw.line(pts, fill=palette["accent"] if is_smooth else palette["accent3"], width=4)
            lx = ox + 20 + t * (x1 - ox - 40)
            ly = oy - (80 - 105 * (((lx - (ox + 14)) / max(x1 - ox - 18, 1)) - 0.55) ** 2)
            draw.ellipse([(lx - 6, ly - 6), (lx + 6, ly + 6)], fill=palette["accent2"])
        draw.text((left_x0 + 44, left_y0 + 34), "smooth", font=small_font, fill=palette["accent"])
        draw.text((mid + 36, left_y0 + 34), "overfit", font=small_font, fill=palette["accent3"])
        return

    if mode == "attention_links":
        draw.text((left_x0 + 18, left_y0 + 12), "Attention Dynamics", font=subtitle_font, fill=palette["ink"])
        y = left_y0 + (left_y1 - left_y0) // 2 + 20
        n = 7
        xs = [left_x0 + 70 + i * ((left_x1 - left_x0 - 140) / max(n - 1, 1)) for i in range(n)]
        q_idx = min(n - 1, int(t * n))
        for i, x in enumerate(xs):
            fill = palette["accent3"] if i == q_idx else (180, 198, 224)
            r = 13 if i == q_idx else 10
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=fill, outline=palette["ink"], width=2)
        for j, x in enumerate(xs):
            if j == q_idx:
                continue
            weight = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(0.9 * j + 6.0 * t))
            color = tuple(int((1.0 - weight) * c0 + weight * c1) for c0, c1 in zip((170, 184, 210), palette["accent"]))
            _draw_arrow(draw, (xs[q_idx], y - 5), (x, y - 40 - int(18 * weight)), color, width=max(2, int(1 + 4 * weight)))
        draw.text((left_x0 + 30, left_y0 + 48), "active token shifts over time", font=small_font, fill=palette["muted"])
        return

    if mode == "network_flow":
        draw.text((left_x0 + 18, left_y0 + 12), "Signal And Gradient Flow", font=subtitle_font, fill=palette["ink"])
        layer_x = [
            left_x0 + 88,
            left_x0 + 210,
            left_x0 + 332,
            left_x0 + 454,
        ]
        layer_y = [left_y0 + 130, left_y0 + 230, left_y0 + 330]
        for li, x in enumerate(layer_x):
            for yi, y in enumerate(layer_y):
                r = 14
                active = (li / max(len(layer_x) - 1, 1)) <= t
                fill = palette["accent2"] if active else (190, 202, 222)
                draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=fill, outline=palette["ink"], width=2)
                if li < len(layer_x) - 1:
                    for y2 in layer_y:
                        _draw_arrow(draw, (x + 16, y), (layer_x[li + 1] - 16, y2), (175, 190, 214), width=2)
        gx = layer_x[-1] - (layer_x[-1] - layer_x[0]) * t
        gy = layer_y[1] - 52
        _draw_arrow(draw, (gx + 40, gy), (gx - 24, gy), palette["accent3"], width=4)
        draw.text((left_x0 + 34, left_y0 + 48), "forward then backward update", font=small_font, fill=palette["muted"])
        return

    # Default: valley + descending ball.
    draw.text((left_x0 + 18, left_y0 + 12), "Concept Dynamics", font=subtitle_font, fill=palette["ink"])
    ox, oy = left_x0 + 70, left_y1 - 60
    draw.line([(ox, oy), (left_x1 - 35, oy)], fill=palette["muted"], width=3)
    draw.line([(ox, oy), (ox, left_y0 + 36)], fill=palette["muted"], width=3)
    draw.text((left_x1 - 95, oy + 12), "state x", font=small_font, fill=palette["muted"])
    draw.text((ox - 48, left_y0 + 46), "score", font=small_font, fill=palette["muted"])

    curve_pts = []
    curve_x0 = ox + 20
    curve_x1 = left_x1 - 55
    for i in range(140):
        u = i / 139.0
        x = curve_x0 + u * (curve_x1 - curve_x0)
        y = oy - (85 - 120 * (u - 0.66) ** 2)
        curve_pts.append((x, y))
    draw.line(curve_pts, fill=palette["accent"], width=5)

    p_u = 0.10 + 0.78 * t
    px = curve_x0 + p_u * (curve_x1 - curve_x0)
    py = oy - (85 - 120 * (p_u - 0.66) ** 2)
    r = int(8 + 2 * pulse)
    draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=palette["accent3"], outline=palette["ink"], width=2)
    _draw_arrow(draw, (px - 30, py - 30), (px - 5, py - 4), palette["accent3"], width=3)
    draw.text((px - 120, py - 52), "next state", font=small_font, fill=palette["accent3"])


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
    stage_labels = _build_stage_labels(topic, objective)
    visual_mode = _infer_local_visual_mode(topic, objective)

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

        # Left panel: visual mode inferred from topic/objective.
        left_x0, left_y0 = 40, 180
        left_x1, left_y1 = width // 2 - 10, height - 40
        _draw_left_panel_visual(
            draw,
            visual_mode,
            t,
            pulse,
            (left_x0, left_y0, left_x1, left_y1),
            palette,
            subtitle_font,
            small_font,
        )

        # Right panel: generic staged process.
        right_x0, right_y0 = width // 2 + 10, 180
        right_x1, right_y1 = width - 40, height - 40
        draw.rounded_rectangle(
            [(right_x0, right_y0), (right_x1, right_y1)],
            radius=18,
            fill=palette["panel"],
            outline=(210, 220, 232),
            width=2,
        )
        draw.text((right_x0 + 18, right_y0 + 12), "Process Stages", font=subtitle_font, fill=palette["ink"])

        boxes = [
            (stage_labels[0], palette["accent"], (right_x0 + 34, right_y0 + 70)),
            (stage_labels[1], palette["accent2"], (right_x0 + 34, right_y0 + 165)),
            (stage_labels[2], palette["accent3"], (right_x0 + 34, right_y0 + 260)),
            (stage_labels[3], (120, 90, 220), (right_x0 + 34, right_y0 + 355)),
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
