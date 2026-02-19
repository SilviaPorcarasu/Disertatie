from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _bootstrap_cache_env() -> None:
    cache_root = "/workspace/.cache"
    hf_root = f"{cache_root}/huggingface"
    hub_root = f"{hf_root}/hub"
    tmp_root = f"{cache_root}/tmp"
    os.makedirs(tmp_root, exist_ok=True)

    os.environ["XDG_CACHE_HOME"] = cache_root
    os.environ["HF_HOME"] = hf_root
    os.environ["TRANSFORMERS_CACHE"] = hf_root
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_root
    os.environ["HF_HUB_CACHE"] = hub_root
    os.environ["TMPDIR"] = tmp_root
    os.environ["HF_HUB_DISABLE_XET"] = "1"


_bootstrap_cache_env()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2v.rag.chunking import clean_chunks, load_document, make_chunks, save_chunks_jsonl  # noqa: E402
from t2v.rag.retrieve import DEFAULT_EMBEDDING_MODEL_ID, build_embedding_index  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one persistent RAG library (chunks + embeddings index) from many books."
    )
    parser.add_argument(
        "--books-list",
        required=True,
        help="Text file with one absolute/relative document path per line.",
    )
    parser.add_argument(
        "--output-chunks",
        default="/workspace/Disertatie/data/library_chunks.jsonl",
        help="Output JSONL for merged cleaned chunks.",
    )
    parser.add_argument(
        "--output-index",
        default="/workspace/Disertatie/data/library_chunks.embeddings.npz",
        help="Output embedding index path (.npz).",
    )
    parser.add_argument(
        "--embedding-model-id",
        default=os.getenv("T2V_EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID),
        help="Embedding model id.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.getenv("T2V_EMBEDDING_BATCH_SIZE", "32")),
        help="Embedding batch size.",
    )
    parser.add_argument("--min-words", type=int, default=45)
    parser.add_argument("--max-words", type=int, default=220)
    return parser.parse_args()


def _load_books_list(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Books list not found: {path}")
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(Path(line).expanduser())
    if not rows:
        raise ValueError(f"No book paths found in: {path}")
    return rows


def main() -> None:
    args = _parse_args()
    books_list = _load_books_list(Path(args.books_list))

    merged = []
    for doc_path in books_list:
        if not doc_path.exists():
            print(f"Skip missing: {doc_path}")
            continue
        text = load_document(doc_path)
        raw = make_chunks(text=text, source=str(doc_path))
        cleaned, _rejected = clean_chunks(
            raw,
            min_words=max(1, args.min_words),
            max_words=max(2, args.max_words),
        )
        merged.extend(cleaned)
        print(f"Loaded {doc_path}: raw={len(raw)} cleaned={len(cleaned)}")

    if not merged:
        raise RuntimeError("No chunks collected from provided books.")

    chunks_path = Path(args.output_chunks)
    count = save_chunks_jsonl(merged, chunks_path)
    print(f"Merged chunks: {count}")
    print(f"Chunks JSONL: {chunks_path}")

    index_path = build_embedding_index(
        chunks_path,
        model_id=args.embedding_model_id,
        batch_size=max(1, args.embedding_batch_size),
        index_path=Path(args.output_index),
    )
    print(f"Embedding model: {args.embedding_model_id}")
    print(f"Embedding index: {index_path}")
    print("Done.")


if __name__ == "__main__":
    main()
