from __future__ import annotations

import json
import re
from typing import Any


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    loaded = json.loads(candidate)
                    if isinstance(loaded, dict):
                        return loaded
                except Exception:
                    return None
    return None


def _fallback_plan(topic: str, objective: str, style: str) -> dict[str, Any]:
    topic_clean = _normalize_space(topic)
    objective_clean = _normalize_space(objective)
    style_clean = _normalize_space(style)
    return {
        "topic": topic_clean,
        "objective": objective_clean,
        "style": style_clean,
        "rag_query": topic_clean,
        "scenes": [
            {
                "title": "setup",
                "visual_action": f"introduce the main elements for {topic_clean}",
                "key_concepts": [topic_clean],
                "rag_query": topic_clean,
            },
            {
                "title": "mechanism",
                "visual_action": f"show the core mechanism step by step: {objective_clean}",
                "key_concepts": [objective_clean],
                "rag_query": f"{topic_clean} {objective_clean}",
            },
            {
                "title": "result",
                "visual_action": "show the result and causal effects clearly",
                "key_concepts": ["result", "causal flow"],
                "rag_query": f"{topic_clean} result explanation",
            },
            {
                "title": "recap",
                "visual_action": "recap the process with concise visual summary",
                "key_concepts": ["summary", "recap"],
                "rag_query": f"{topic_clean} summary",
            },
        ],
    }


def _sanitize_plan(data: dict[str, Any], *, topic: str, objective: str, style: str) -> dict[str, Any]:
    plan = _fallback_plan(topic, objective, style)

    for key in ("topic", "objective", "style", "rag_query"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            plan[key] = _normalize_space(value)

    scenes = data.get("scenes")
    if isinstance(scenes, list) and scenes:
        clean_scenes = []
        for idx, raw in enumerate(scenes[:6], start=1):
            if not isinstance(raw, dict):
                continue
            title = _normalize_space(str(raw.get("title", f"scene{idx}")))
            visual_action = _normalize_space(str(raw.get("visual_action", "")))
            rag_query = _normalize_space(str(raw.get("rag_query", plan["rag_query"])))
            key_concepts_raw = raw.get("key_concepts", [])
            key_concepts = []
            if isinstance(key_concepts_raw, list):
                key_concepts = [
                    _normalize_space(str(item))
                    for item in key_concepts_raw
                    if str(item).strip()
                ][:8]

            if not visual_action:
                continue
            clean_scenes.append(
                {
                    "title": title or f"scene{idx}",
                    "visual_action": visual_action,
                    "key_concepts": key_concepts,
                    "rag_query": rag_query or plan["rag_query"],
                }
            )
        if clean_scenes:
            plan["scenes"] = clean_scenes
    return plan


def _planner_instruction(topic: str, audience: str, objective: str, style: str) -> str:
    return (
        "You are an educational video planner. Return JSON only.\n"
        "Schema:\n"
        "{\n"
        '  "topic": "...",\n'
        '  "objective": "...",\n'
        '  "style": "...",\n'
        '  "rag_query": "...",\n'
        '  "scenes": [\n'
        '    {"title":"...", "visual_action":"...", "key_concepts":["..."], "rag_query":"..."}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- 4 concise scenes for a 4-6 second educational clip.\n"
        "- visual_action must be concrete and drawable, no text overlays.\n"
        "- keep focus strictly on the requested topic.\n"
        "- rag_query should be semantic, concept-focused.\n"
        "\n"
        f"Topic: {topic}\n"
        f"Audience: {audience}\n"
        f"Objective: {objective}\n"
        f"Style: {style}\n"
    )


def build_scene_plan_with_llama(
    *,
    topic: str,
    audience: str,
    objective: str,
    style: str,
    model_id: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception:
        return _fallback_plan(topic, objective, style)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        generator = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        prompt = _planner_instruction(topic, audience, objective, style)
        outputs = generator(
            prompt,
            max_new_tokens=max(128, int(max_new_tokens)),
            do_sample=temperature > 0.0,
            temperature=max(0.0, float(temperature)),
            top_p=min(max(0.0, float(top_p)), 1.0),
            return_full_text=False,
        )
        if not outputs:
            return _fallback_plan(topic, objective, style)
        generated = str(outputs[0].get("generated_text", ""))
        parsed = _extract_first_json_object(generated)
        if parsed is None:
            return _fallback_plan(topic, objective, style)
        return _sanitize_plan(parsed, topic=topic, objective=objective, style=style)
    except Exception:
        return _fallback_plan(topic, objective, style)


def plan_to_scene_directives(plan: dict[str, Any], *, max_scenes: int = 4) -> str:
    scenes = plan.get("scenes", [])
    if not isinstance(scenes, list):
        return ""
    rows = []
    for idx, scene in enumerate(scenes[:max(1, max_scenes)], start=1):
        if not isinstance(scene, dict):
            continue
        action = _normalize_space(str(scene.get("visual_action", "")))
        if not action:
            continue
        title = _normalize_space(str(scene.get("title", f"scene{idx}")))
        rows.append(f"scene{idx} {title}: {action}")
    return " ; ".join(rows)


def plan_to_rag_query(plan: dict[str, Any]) -> str:
    queries = []
    main_query = plan.get("rag_query")
    if isinstance(main_query, str) and main_query.strip():
        queries.append(_normalize_space(main_query))
    scenes = plan.get("scenes", [])
    if isinstance(scenes, list):
        for scene in scenes[:6]:
            if not isinstance(scene, dict):
                continue
            q = scene.get("rag_query")
            if isinstance(q, str) and q.strip():
                queries.append(_normalize_space(q))
            concepts = scene.get("key_concepts", [])
            if isinstance(concepts, list):
                for c in concepts[:4]:
                    s = _normalize_space(str(c))
                    if s:
                        queries.append(s)
    if not queries:
        return ""
    # Deduplicate while preserving order.
    merged = list(dict.fromkeys(queries))
    return " ".join(merged[:20])

