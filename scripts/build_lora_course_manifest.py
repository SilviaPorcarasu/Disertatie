from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def _infer_topic_tag(path: Path) -> str:
    text = path.as_posix().lower()
    if any(k in text for k in ("backprop", "gradient_flow", "gradient_backprop")):
        return "backprop"
    if any(k in text for k in ("a_star", "astar", "shortest_path")):
        return "a_star"
    if any(k in text for k in ("attention", "qkv")):
        return "attention"
    return "general"


def _build_caption(topic_tag: str, trigger: str) -> str:
    prefix = f"{trigger}, " if trigger else ""
    templates = {
        "backprop": (
            "2D course infographic of neural network backpropagation: forward pass through layers, "
            "loss at output, backward chain-rule gradient flow, and weight update direction opposite gradient, "
            "stable camera, high contrast arrows, no text"
        ),
        "a_star": (
            "2D course infographic of A* search: open set expansion with heuristic guidance, "
            "closed set updates, predecessor links, and shortest path reconstruction, "
            "stable camera, high contrast arrows, no text"
        ),
        "attention": (
            "2D course infographic of self-attention: query key similarity weighting, "
            "normalized attention links, weighted sum of values, and contextual token output, "
            "stable camera, high contrast arrows, no text"
        ),
        "general": (
            "2D course infographic of a machine learning concept shown step-by-step with causal arrows, "
            "stable camera, high contrast layout, no text"
        ),
    }
    return f"{prefix}{templates.get(topic_tag, templates['general'])}"


def _parse_split(split: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in split.split(",")]
    if len(parts) != 3:
        raise ValueError("Split must have 3 comma-separated values, e.g. 0.8,0.1,0.1")
    train, val, test = (float(parts[0]), float(parts[1]), float(parts[2]))
    total = train + val + test
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive number.")
    return train / total, val / total, test / total


def _split_name(index: int, total: int, split: tuple[float, float, float]) -> str:
    train_ratio, val_ratio, _ = split
    train_cut = int(total * train_ratio)
    val_cut = int(total * (train_ratio + val_ratio))
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LoRA fine-tuning manifest from generated course-style videos."
    )
    parser.add_argument(
        "--input-root",
        default="/workspace/Disertatie/outputs",
        help="Root folder scanned recursively for mp4 videos.",
    )
    parser.add_argument(
        "--glob",
        default="**/*.mp4",
        help="Glob pattern relative to --input-root.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/workspace/Disertatie/data/lora_course_manifest.jsonl",
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional CSV mirror path (defaults next to output JSONL).",
    )
    parser.add_argument(
        "--trigger-token",
        default="acad_course_v1",
        help="Optional trigger token used in captions.",
    )
    parser.add_argument(
        "--split",
        default="0.8,0.1,0.1",
        help="Split ratios train,val,test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for split assignment.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=20_000,
        help="Ignore very small video files under this size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input root not found: {root}")

    split = _parse_split(args.split)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv) if args.output_csv else output_jsonl.with_suffix(".csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(root.glob(args.glob)) if p.is_file() and p.stat().st_size >= args.min_bytes]
    if not files:
        raise RuntimeError(
            f"No videos found for pattern '{args.glob}' in {root} with min-bytes={args.min_bytes}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(files)

    rows: list[dict] = []
    for idx, path in enumerate(files):
        topic_tag = _infer_topic_tag(path)
        rows.append(
            {
                "video_path": str(path.resolve()),
                "caption": _build_caption(topic_tag, args.trigger_token.strip()),
                "split": _split_name(idx, len(files), split),
                "topic_tag": topic_tag,
                "source": "generated_pipeline",
                "weight": 1.0,
            }
        )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "caption", "split", "topic_tag", "source", "weight"],
        )
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {"train": 0, "val": 0, "test": 0}
    topic_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        topic_counts[row["topic_tag"]] = topic_counts.get(row["topic_tag"], 0) + 1

    print(f"Videos included: {len(rows)}")
    print(f"Output JSONL: {output_jsonl}")
    print(f"Output CSV:   {output_csv}")
    print(f"Split counts: {split_counts}")
    print(f"Topic counts: {topic_counts}")


if __name__ == "__main__":
    main()
