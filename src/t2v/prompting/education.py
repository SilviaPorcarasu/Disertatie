import re


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_request_to_english(text: str) -> str:
    # English-only mode: do not translate, only normalize formatting.
    return _normalize_space(text)


def _infer_visual_mode(user_request: str) -> str:
    text = user_request.lower()
    if any(keyword in text for keyword in {"diagram", "schematic", "graph"}):
        return "diagram"
    return "explainer"


def _truncate_reference_context(reference_context: str, max_words: int = 45) -> str:
    text = _normalize_space(reference_context.replace("\n", " "))
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


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

    mode = _infer_visual_mode(topic_en)
    if mode == "diagram":
        mode_block = (
            "Create one coherent academic diagram (not a storyboard). "
            "Keep layout stable and emphasize directional flow with arrows and node groups. "
        )
    else:
        mode_block = (
            "Create an academic explainer video with a stable camera and minimal transitions. "
            "Animate one consistent diagrammatic system to show causal flow step-by-step. "
        )

    base = (
        f"English-only academic text-to-video, {style_en}, clean 2D technical graphics, high contrast. "
        f"Audience: {audience_en}. Learning objective: {objective_en}. User request: {topic_en}. "
        + mode_block
        + "Use English terminology and concise English labels only. "
        "Keep visible educational objects: nodes, arrows, blocks, equations, symbols, highlights. "
        "Prioritize semantic correctness and visual clarity. "
        "No gibberish text or long paragraph text inside the frame."
    )

    if not reference_context:
        return base

    context = _truncate_reference_context(reference_context, max_words=45)
    if not context:
        return base

    return base + " Factual context (English, concise): " + context


def build_negative_prompt() -> str:
    return (
        "empty white background, blank scene, plain paper only, no objects, "
        "gibberish text, random unreadable letters, non-English text, "
        "long paragraph text in frame, photorealistic humans, camera shake, "
        "flicker, watermark, logo, glitch artifacts"
    )
