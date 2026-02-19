import os
import re


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_request_to_english(text: str) -> str:
    # English-only mode: do not translate, only normalize formatting.
    return _normalize_space(text)


VISUAL_MODE_HINTS = {
    "process_diagram": (
        "single process diagram, ordered state transitions, clear cause and effect arrows"
    ),
    "comparison_diagram": (
        "single frame comparison with two or three variants and synchronized transitions"
    ),
    "single_diagram": "single coherent node-edge diagram with focused motion",
    "explainer": "minimal abstract teaching diagram with simple geometric elements",
}


FEW_SHOTS = {
    "overfitting_generalization": (
        "show same dataset with two boundaries: smooth simple curve vs complex wiggly curve; "
        "then unseen points reveal better generalization of the smooth model."
    ),
    "gradient_descent": (
        "show loss landscape/contours and one point descending in steps toward minimum; "
        "step size decreases near convergence."
    ),
    "bias_variance_tradeoff": (
        "show low/medium/high complexity models side-by-side; prediction flexibility increases; "
        "underfit and overfit errors rise at opposite ends."
    ),
    "attention_mechanism": (
        "show token nodes and weighted links; for each query, strongest links shift; "
        "highlight focused context paths over time."
    ),
    "default": (
        "show the core concept with minimal objects, directional arrows, and ordered process states."
    ),
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


def _truncate_words(text: str, max_words: int) -> str:
    words = _normalize_space(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _truncate_reference_context(reference_context: str, max_words: int = 28) -> str:
    text = _normalize_space(reference_context.replace("\n", " "))
    if not text:
        return "None"
    return _truncate_words(text, max_words=max_words)


def build_education_prompt(
    topic: str,
    audience: str,
    objective: str,
    style: str,
    reference_context: str = "",
) -> str:
    topic_en = normalize_request_to_english(topic)
    audience_en = normalize_request_to_english(audience)
    objective_en = normalize_request_to_english(objective)
    style_en = normalize_request_to_english(style)
    visual_mode = _infer_visual_mode(topic_en)
    context = _truncate_reference_context(reference_context, max_words=28)
    few_shot = _truncate_words(_select_few_shot(topic_en), max_words=24)
    style_short = _truncate_words(style_en, max_words=14)
    objective_short = _truncate_words(objective_en, max_words=18)
    audience_short = _truncate_words(audience_en, max_words=10)
    mode_hint = VISUAL_MODE_HINTS.get(visual_mode, VISUAL_MODE_HINTS["explainer"])

    prompt_parts = [
        "clean 2d academic infographic animation, high contrast, stable camera, no text or symbols",
        f"topic: {topic_en}",
        f"for {audience_short}",
        f"learning goal: {objective_short}",
        f"visual structure: {mode_hint}",
        f"scene behavior: {few_shot}",
        f"style: {style_short}",
        "use geometric shapes, arrows, and smooth transitions only",
        "no letters, words, numbers, formulas, captions, logos, or watermarks",
    ]
    if context != "None":
        prompt_parts.append(f"reference facts: {context}")

    prompt = ". ".join(prompt_parts) + "."
    prompt_max_words = max(40, int(os.getenv("T2V_PROMPT_MAX_WORDS", "80")))
    return _truncate_words(prompt, max_words=prompt_max_words)


def build_negative_prompt() -> str:
    return (
        "gibberish text, letters, words, numbers, equations, subtitles, watermark, logo, camera shake, flicker, "
        "photorealistic people, cluttered background, unreadable symbols"
    )
