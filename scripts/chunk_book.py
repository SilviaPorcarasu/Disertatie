from pathlib import Path
import argparse
import os
import sys


def _bootstrap_cache_env() -> None:
    cache_root = "/workspace/.cache"
    hf_root = f"{cache_root}/huggingface"
    hub_root = f"{hf_root}/hub"
    tmp_root = f"{cache_root}/tmp"
    os.makedirs(tmp_root, exist_ok=True)

    # Must be set before importing transformers/huggingface_hub.
    os.environ["XDG_CACHE_HOME"] = cache_root
    os.environ["HF_HOME"] = hf_root
    os.environ["TRANSFORMERS_CACHE"] = hf_root
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_root
    os.environ["HF_HUB_CACHE"] = hub_root
    os.environ["TMPDIR"] = tmp_root
    os.environ["HF_HUB_DISABLE_XET"] = "1"


_bootstrap_cache_env()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2v.rag.chunking import (
    clean_chunks,
    load_document,
    make_chunks,
    save_chunks_jsonl,
    save_rejected_jsonl,
)
from t2v.rag.retrieve import DEFAULT_EMBEDDING_MODEL_ID, build_embedding_index

DEFAULT_INPUT = "/workspace/t2v/data/2017_Book_AnIntroductionToMachineLearnin.pdf"
DEFAULT_OUTPUT = "/workspace/t2v/data/book_chunks.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk a book/document by paragraphs for RAG")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to .txt, .md, or .pdf file",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSONL with chunks",
    )
    parser.add_argument(
        "--build-embedding-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build semantic embedding index after chunking (default: enabled)",
    )
    parser.add_argument(
        "--clean-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run robust cleaning pass (default: enabled)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=45,
        help="Minimum words per cleaned chunk",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=220,
        help="Maximum words per cleaned chunk (long chunks are split)",
    )
    parser.add_argument(
        "--save-rejected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save rejected/noisy chunks to audit JSONL (default: enabled)",
    )
    parser.add_argument(
        "--rejected-output",
        default="",
        help="Optional path for rejected chunks JSONL",
    )
    parser.add_argument(
        "--embedding-model-id",
        default=os.getenv("T2V_EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID),
        help="Embedding model for semantic RAG index",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.getenv("T2V_EMBEDDING_BATCH_SIZE", "32")),
        help="Batch size used to compute chunk embeddings",
    )
    parser.add_argument(
        "--embedding-index-output",
        default="",
        help="Optional path for .npz embedding index (defaults next to output JSONL)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = load_document(input_path)
    raw_chunks = make_chunks(
        text=text,
        source=str(input_path),
    )
    rejected = []
    chunks = raw_chunks
    if args.clean_chunks:
        chunks, rejected = clean_chunks(
            raw_chunks,
            min_words=max(1, args.min_words),
            max_words=max(2, args.max_words),
        )

    output_path = Path(args.output)
    count = save_chunks_jsonl(chunks, output_path)

    print(f"Input: {input_path}")
    print("Chunking mode: paragraph")
    print(f"Raw chunks: {len(raw_chunks)}")
    print(f"Cleaned chunks: {len(chunks)}")
    print(f"Rejected chunks: {len(rejected)}")
    print(f"Chunks: {count}")
    print(f"Output: {output_path}")

    if args.clean_chunks and args.save_rejected:
        rejected_path = (
            Path(args.rejected_output)
            if args.rejected_output
            else output_path.with_name(f"{output_path.stem}.rejected.jsonl")
        )
        rejected_count = save_rejected_jsonl(rejected, rejected_path)
        print(f"Rejected output: {rejected_path} ({rejected_count} rows)")

    if args.build_embedding_index:
        index_path = Path(args.embedding_index_output) if args.embedding_index_output else None
        try:
            built_index = build_embedding_index(
                output_path,
                model_id=args.embedding_model_id,
                batch_size=max(1, args.embedding_batch_size),
                index_path=index_path,
            )
            print(f"Embedding index: {built_index}")
            print(f"Embedding model: {args.embedding_model_id}")
        except Exception as exc:
            print(f"Embedding index build failed: {exc}")


if __name__ == "__main__":
    main()
