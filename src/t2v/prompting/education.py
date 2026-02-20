import json
import os
import re
from pathlib import Path


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_request_to_english(text: str) -> str:
    # English-only mode: do not translate, only normalize formatting.
    return _normalize_space(text)


VISUAL_MODE_HINTS = {
    "process_diagram": (
        "single process diagram with 3 to 5 stages, ordered transitions, clear cause-and-effect arrows"
    ),
    "comparison_diagram": (
        "single scene comparison with two or three variants, synchronized transitions, balanced layout"
    ),
    "single_diagram": "single coherent node-edge diagram, visible colored nodes and directed edges",
    "explainer": "clean teaching animation with simple geometric elements and explicit motion cues",
}


FEW_SHOTS = {
    "overfitting_generalization": (
        "shot1 setup: same dataset appears with train points highlighted. "
        "shot2 compare: two boundaries appear, one smooth and one overly wiggly. "
        "shot3 reveal: unseen points appear and smooth boundary classifies better. "
        "shot4 takeaway: overfit boundary fluctuates while smooth boundary stays stable."
    ),
    "gradient_descent": (
        "shot1 setup: contour map with an initial point far from minimum. "
        "shot2 motion: point moves along arrows following negative gradient. "
        "shot3 convergence: step size becomes smaller near basin minimum. "
        "shot4 takeaway: final point stabilizes and path remains visible."
    ),
    "bias_variance_tradeoff": (
        "shot1 setup: identical data shown with low, medium, and high complexity models. "
        "shot2 comparison: flexibility increases left to right. "
        "shot3 error behavior: underfit high bias on simple model, overfit high variance on complex model. "
        "shot4 takeaway: middle model balances both."
    ),
    "attention_mechanism": (
        "shot1 setup: token nodes aligned in sequence with faint links. "
        "shot2 focus: one query node emits weighted links with varying thickness. "
        "shot3 shift: next query changes strongest links and attention focus. "
        "shot4 takeaway: context paths update dynamically over time."
    ),
    "default": (
        "shot1 setup, shot2 transformation, shot3 result, shot4 recap with ordered arrows and clear state transitions."
    ),
}


DIRECTOR_RULES = (
    "balanced midtone exposure with no clipped highlights and no crushed shadows",
    "large foreground elements centered and clearly visible in every frame",
    "consistent object identity across frames, no abrupt scene resets",
    "smooth monotonic motion; avoid jitter, flicker, and sudden camera cuts",
    "arrows and motion paths must show causal direction step by step",
)


_TOPIC_STOPWORDS = {
    "about",
    "across",
    "after",
    "and",
    "animation",
    "are",
    "between",
    "by",
    "can",
    "clear",
    "clip",
    "demonstrate",
    "explain",
    "explainer",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "mechanism",
    "no",
    "of",
    "on",
    "or",
    "scene",
    "show",
    "simple",
    "step",
    "the",
    "to",
    "understand",
    "visual",
    "with",
}


def _infer_visual_mode(user_request: str) -> str:
    text = user_request.lower()
    if any(keyword in text for keyword in {"gradient", "backprop", "training", "optimization"}):
        return "process_diagram"
    if any(keyword in text for keyword in {"compare", "versus", "vs", "tradeoff"}):
        return "comparison_diagram"
    if any(keyword in text for keyword in {"diagram", "schematic", "graph"}):
        return "single_diagram"
    return "explainer"


def _select_few_shot(topic: str) -> str:
    t = topic.lower()
    if "overfitting" in t or "generalization" in t:
        return FEW_SHOTS["overfitting_generalization"]
    if "gradient" in t or "descent" in t or "backprop" in t:
        return FEW_SHOTS["gradient_descent"]
    if "bias" in t and "variance" in t:
        return FEW_SHOTS["bias_variance_tradeoff"]
    if "attention" in t:
        return FEW_SHOTS["attention_mechanism"]
    return FEW_SHOTS["default"]


def _build_scene_timeline(topic: str, objective: str, visual_mode: str) -> str:
    mode = VISUAL_MODE_HINTS.get(visual_mode, VISUAL_MODE_HINTS["explainer"])
    return (
        f"timeline with four beats: setup for {topic}; mechanism demonstration; "
        f"consequence/result linked to objective {objective}; short recap. "
        f"keep structure: {mode}."
    )


def _extract_topic_terms(topic: str, objective: str, max_terms: int = 8) -> list[str]:
    text = f"{topic} {objective}".lower()
    terms = re.findall(r"[a-z][a-z0-9\-]{2,}", text)
    ranked = []
    for term in terms:
        if term in _TOPIC_STOPWORDS:
            continue
        if term not in ranked:
            ranked.append(term)
        if len(ranked) >= max_terms:
            break
    return ranked


def _build_topic_constraints(topic: str, objective: str) -> str:
    terms = _extract_topic_terms(topic, objective)
    if not terms:
        return topic
    return ", ".join(terms)


def _build_strict_scene_actions(topic: str, objective: str) -> str:
    topic_short = _truncate_words(_normalize_space(topic), 9)
    objective_short = _truncate_words(_normalize_space(objective), 11)
    return (
        f"scene1 setup core entities for {topic_short}; "
        f"scene2 show directional process with arrows; "
        f"scene3 show update effect linked to {objective_short}; "
        "scene4 recap full causal chain with one continuous diagram"
    )


def _truncate_words(text: str, max_words: int, *, add_ellipsis: bool = False) -> str:
    words = _normalize_space(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    clipped = " ".join(words[:max_words])
    if add_ellipsis:
        return f"{clipped} ..."
    return clipped


def _truncate_reference_context(reference_context: str, max_words: int = 28) -> str:
    text = _normalize_space(reference_context.replace("\n", " "))
    if not text:
        return "None"
    return _truncate_words(text, max_words=max_words)


def _word_count(text: str) -> int:
    return len(_normalize_space(text).split())


def _to_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_education_prompt(
    topic: str,
    audience: str,
    objective: str,
    style: str,
    reference_context: str = "",
    planner_directives: str = "",
) -> str:
    topic_en = normalize_request_to_english(topic)
    audience_en = normalize_request_to_english(audience)
    objective_en = normalize_request_to_english(objective)
    style_en = normalize_request_to_english(style)
    visual_mode = _infer_visual_mode(topic_en)
    context = _truncate_reference_context(reference_context, max_words=40)
    planner_short = _truncate_words(
        normalize_request_to_english(planner_directives),
        max_words=40,
    )
    few_shot = _truncate_words(_select_few_shot(topic_en), max_words=30)
    style_short = _truncate_words(style_en, max_words=16)
    objective_short = _truncate_words(objective_en, max_words=20)
    audience_short = _truncate_words(audience_en, max_words=10)
    mode_hint = VISUAL_MODE_HINTS.get(visual_mode, VISUAL_MODE_HINTS["explainer"])
    timeline = _truncate_words(
        _build_scene_timeline(topic_en, objective_short, visual_mode),
        max_words=26,
    )
    direction_rules = _truncate_words("; ".join(DIRECTOR_RULES), max_words=30)
    # Keep outputs tightly topic-locked unless explicitly relaxed.
    strict_topic = _env_flag("T2V_STRICT_TOPIC", True)
    topic_constraints = _truncate_words(
        _build_topic_constraints(topic_en, objective_en),
        max_words=12,
    )
    strict_scene_actions = _truncate_words(
        _build_strict_scene_actions(topic_en, objective_en),
        max_words=26,
    )

    required_parts = [
        "clean academic explainer animation with explicit instructional staging",
        f"topic: {topic_en}",
        f"for {audience_short}",
        f"learning goal: {objective_short}",
        f"visual structure: {mode_hint}",
        f"timeline: {timeline}",
        f"strict scene actions: {strict_scene_actions}",
        f"topic constraints: depict only {topic_constraints}",
        f"direction rules: {direction_rules}",
        "single coherent scene with one evolving diagram, no scene jumps",
        "balanced lighting, neutral background, visible colored nodes, arrows, and smooth transitions",
        "every frame must contain clear visual elements, avoid empty or black frames",
        "camera mostly static with very slow pan only when needed to preserve context",
        "no readable letters, words, numbers, formulas, captions, logos, or watermarks",
    ]
    if strict_topic:
        required_parts.append(
            "exclude unrelated objects, humans, animals, landscapes, cinematic effects, and abstract decorative particles"
        )

    optional_parts = []
    if style_short:
        optional_parts.append(f"style: {style_short}")
    if few_shot:
        optional_parts.append(f"few-shot reference sequence: {few_shot}")
    if context != "None":
        optional_parts.append(f"reference facts to depict: {context}")
    if planner_short:
        optional_parts.append(f"scene directives: {planner_short}")

    prompt_max_words = max(90, _to_int_env("T2V_PROMPT_MAX_WORDS", 170))
    force_style = _env_flag("T2V_FORCE_STYLE", True)
    if force_style and style_short and f"style: {style_short}" in optional_parts:
        optional_parts.remove(f"style: {style_short}")
        required_parts.append(f"style lock: {style_short}")

    selected_optional = list(optional_parts)
    prompt = ". ".join(required_parts + selected_optional) + "."
    while _word_count(prompt) > prompt_max_words and selected_optional:
        selected_optional.pop()
        prompt = ". ".join(required_parts + selected_optional) + "."

    return _truncate_words(prompt, max_words=prompt_max_words)


def build_negative_prompt() -> str:
    return (
        "gibberish text, letters, words, numbers, equations, subtitles, watermark, logo, camera shake, flicker, "
        "photorealistic people, cluttered background, unreadable symbols, black screen, blank frame, underexposed scene, "
        "overexposed scene, washed-out highlights, flat low-contrast frame, single-color frame, invisible foreground, "
        "hard cuts between unrelated scenes, fantasy landscapes, fireworks, decorative particles, unrelated objects, "
        "cinematic bokeh, depth-of-field blur, lens flare, handheld camera, abrupt zoom"
    )


def _get_cue_list(cues: dict | None, key: str) -> list[str]:
    if not isinstance(cues, dict):
        return []
    value = cues.get(key, [])
    if not isinstance(value, list):
        return []
    return [normalize_request_to_english(str(v)) for v in value if str(v).strip()]


def _compose_scene_prompt(
    topic: str,
    style: str,
    goal: str,
    visual_cue: str,
    motion_cue: str,
    *,
    max_words: int = 40,
) -> str:
    base = (
        f"teacher-style {style} diagram about {topic}. "
        f"{goal}. "
        f"{visual_cue}. "
        f"{motion_cue}. "
        "same layout, stable camera, minimal motion, no text."
    )
    return _truncate_words(base, max_words=max_words)


def build_scene_plan(
    topic: str,
    objective: str,
    style: str,
    cues: dict,
    scene_frames: int = 16,
) -> dict:
    topic_en = normalize_request_to_english(topic)
    objective_en = normalize_request_to_english(objective)
    style_en = normalize_request_to_english(style)

    visual_cues = _get_cue_list(cues, "visual_cues")
    motion_cues = _get_cue_list(cues, "motion_cues")
    constraints_from_cues = _get_cue_list(cues, "constraints")

    if not visual_cues:
        visual_cues = [
            "clean node-arrow diagram with high contrast",
            "structured process diagram with consistent objects",
        ]
    if not motion_cues:
        motion_cues = [
            "step-by-step highlights in causal order",
            "smooth monotonic transitions without hard cuts",
        ]

    scene_frames = max(2, int(scene_frames))
    global_constraints = [
        "same layout, stable camera, minimal motion",
        "consistent object identity across frames",
        "no readable text, numbers, or formulas",
        "high contrast and visible foreground elements",
    ]
    for c in constraints_from_cues[:4]:
        if c not in global_constraints:
            global_constraints.append(c)

    scene_defs = [
        (
            "intro",
            f"intro: core entities and layout for {topic_en}",
            visual_cues[0],
            motion_cues[0],
        ),
        (
            "step1",
            f"step 1: show first mechanism stage for {topic_en}",
            visual_cues[min(1, len(visual_cues) - 1)],
            motion_cues[min(1, len(motion_cues) - 1)],
        ),
        (
            "step2",
            f"step 2: show causal update linked to {objective_en}",
            visual_cues[0],
            motion_cues[0],
        ),
        (
            "recap",
            f"recap: summarize the process and outcome {objective_en}",
            visual_cues[min(1, len(visual_cues) - 1)],
            motion_cues[min(1, len(motion_cues) - 1)],
        ),
    ]

    global_negative_prompt = (
        "blurry, noise, distortion, artifacts, flicker, text, letters, numbers, "
        "watermark, logo, overexposed, underexposed, blank frame, camera shake, "
        "scene jump, hard cut, unrelated object"
    )
    scenes = []
    for scene_id, goal, visual_cue, motion_cue in scene_defs:
        prompt = _compose_scene_prompt(
            topic=topic_en,
            style=style_en or "flat infographic",
            goal=goal,
            visual_cue=visual_cue,
            motion_cue=motion_cue,
            max_words=40,
        )
        scenes.append(
            {
                "id": scene_id,
                "goal": goal,
                "prompt": prompt,
                "negative_prompt": global_negative_prompt,
                "frames": scene_frames,
            }
        )

    return {
        "topic": topic_en,
        "style": style_en or "flat infographic",
        "global_negative_prompt": global_negative_prompt,
        "global_constraints": global_constraints,
        "scenes": scenes,
    }


def save_plan_json(plan: dict, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "plan.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=True, indent=2)
    return out_path
