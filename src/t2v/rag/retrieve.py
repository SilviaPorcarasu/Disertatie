from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

_ENCODER_CACHE: Dict[str, tuple] = {}
_EMBEDDING_CACHE: Dict[Tuple[str, str, int, int], np.ndarray] = {}

_NOISY_PATTERNS = (
    "contents ",
    "isbn",
    "library of congress",
    "solidify your knowledge",
    "what have you learned",
    "computer assignments",
    "historical remarks",
    "www.",
)

_OCR_NOISE_MARKERS = ("/nul", "/stx", "/esc", "/dc", "/nak", "c141")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _hf_cache_root() -> Path:
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.getenv(key, "").strip()
        if value:
            return Path(value)
    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _local_hf_snapshot(model_id: str) -> Path | None:
    cache_root = _hf_cache_root()
    model_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref_main = model_dir / "refs" / "main"
    if ref_main.exists():
        revision = ref_main.read_text(encoding="utf-8").strip()
        if revision:
            candidate = snapshots_dir / revision
            if (candidate / "config.json").exists():
                return candidate

    for candidate in sorted(snapshots_dir.iterdir(), reverse=True):
        if candidate.is_dir() and (candidate / "config.json").exists():
            return candidate
    return None


def _get_encoder(model_id: str):
    cached = _ENCODER_CACHE.get(model_id)
    if cached is not None:
        return cached

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Embedding retrieval needs `torch` and `transformers`.") from exc

    local_snapshot = _local_hf_snapshot(model_id)
    load_id = str(local_snapshot) if local_snapshot is not None else model_id
    local_only = bool(local_snapshot is not None) or _env_flag("T2V_HF_LOCAL_ONLY", default=False)

    tokenizer = AutoTokenizer.from_pretrained(load_id, local_files_only=local_only)
    model = AutoModel.from_pretrained(load_id, local_files_only=local_only)
    embed_device = os.getenv("T2V_EMBEDDING_DEVICE", "cpu").strip().lower()
    if embed_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif embed_device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    try:
        model = model.to(device)
    except Exception:
        if device == "cuda":
            device = "cpu"
            model = model.to(device)
        else:
            raise
    model.eval()

    cached = (tokenizer, model, device)
    _ENCODER_CACHE[model_id] = cached
    return cached


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def _embed_texts(texts: List[str], model_id: str, batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    import torch

    tokenizer, model, device = _get_encoder(model_id)
    vectors: List[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(vectors, axis=0)


def _load_chunks(chunks_path: Path) -> List[dict]:
    rows: List[dict] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _default_index_path(chunks_path: Path) -> Path:
    return chunks_path.with_name(f"{chunks_path.stem}.embeddings.npz")


def _save_embedding_index(
    index_path: Path,
    embeddings: np.ndarray,
    model_id: str,
    chunks_path: Path,
    chunk_count: int,
) -> None:
    stats = chunks_path.stat()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        index_path,
        embeddings=embeddings.astype(np.float32, copy=False),
        model_id=np.array([model_id]),
        chunks_path=np.array([str(chunks_path.resolve())]),
        chunks_mtime_ns=np.array([stats.st_mtime_ns], dtype=np.int64),
        chunks_size=np.array([stats.st_size], dtype=np.int64),
        chunk_count=np.array([chunk_count], dtype=np.int64),
    )


def _index_matches_chunks(
    index_data: np.lib.npyio.NpzFile,
    chunks_path: Path,
    expected_model_id: str,
    expected_chunk_count: int,
) -> bool:
    required = {
        "embeddings",
        "model_id",
        "chunks_mtime_ns",
        "chunks_size",
        "chunk_count",
    }
    if not required.issubset(set(index_data.files)):
        return False

    model_id = str(np.asarray(index_data["model_id"]).reshape(-1)[0])
    if model_id != expected_model_id:
        return False

    stats = chunks_path.stat()
    mtime_ns = int(np.asarray(index_data["chunks_mtime_ns"]).reshape(-1)[0])
    size = int(np.asarray(index_data["chunks_size"]).reshape(-1)[0])
    chunk_count = int(np.asarray(index_data["chunk_count"]).reshape(-1)[0])
    if mtime_ns != stats.st_mtime_ns or size != stats.st_size:
        return False
    if chunk_count != expected_chunk_count:
        return False

    embeddings = np.asarray(index_data["embeddings"])
    return embeddings.ndim == 2 and embeddings.shape[0] == expected_chunk_count


def build_embedding_index(
    chunks_path: Path,
    *,
    model_id: str | None = None,
    batch_size: int = 32,
    index_path: Path | None = None,
) -> Path:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    model_id = model_id or os.getenv("T2V_EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID)
    index_path = index_path or _default_index_path(chunks_path)
    batch_size = max(1, batch_size)

    rows = _load_chunks(chunks_path)
    texts = [row.get("text", "") for row in rows]
    embeddings = _embed_texts(texts, model_id=model_id, batch_size=batch_size)
    _save_embedding_index(index_path, embeddings, model_id, chunks_path, chunk_count=len(rows))

    stats = chunks_path.stat()
    cache_key = (str(chunks_path.resolve()), model_id, stats.st_mtime_ns, stats.st_size)
    _EMBEDDING_CACHE[cache_key] = embeddings

    return index_path


def _load_or_build_embeddings(
    chunks_path: Path,
    rows: List[dict],
    model_id: str,
    batch_size: int,
    index_path: Path,
) -> np.ndarray:
    stats = chunks_path.stat()
    cache_key = (str(chunks_path.resolve()), model_id, stats.st_mtime_ns, stats.st_size)
    cached = _EMBEDDING_CACHE.get(cache_key)
    if cached is not None and cached.shape[0] == len(rows):
        return cached

    embeddings: np.ndarray | None = None
    if index_path.exists():
        try:
            with np.load(index_path, allow_pickle=False) as data:
                if _index_matches_chunks(
                    data,
                    chunks_path=chunks_path,
                    expected_model_id=model_id,
                    expected_chunk_count=len(rows),
                ):
                    embeddings = np.asarray(data["embeddings"]).astype(np.float32, copy=False)
        except Exception:
            embeddings = None

    if embeddings is None:
        if not _env_flag("T2V_BUILD_INDEX_ON_DEMAND", default=True):
            raise RuntimeError(
                "Embedding index missing/stale and T2V_BUILD_INDEX_ON_DEMAND=0."
            )
        built_path = build_embedding_index(
            chunks_path,
            model_id=model_id,
            batch_size=batch_size,
            index_path=index_path,
        )
        with np.load(built_path, allow_pickle=False) as data:
            embeddings = np.asarray(data["embeddings"]).astype(np.float32, copy=False)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise RuntimeError(
            f"Invalid embedding index shape {embeddings.shape} for {len(rows)} chunks."
        )

    _EMBEDDING_CACHE[cache_key] = embeddings
    return embeddings


def _penalize_noisy_text(text: str, score: float) -> float:
    lowered = text.lower()
    penalty = 0.0

    for pattern in _NOISY_PATTERNS:
        if pattern in lowered:
            penalty += 0.16

    for marker in _OCR_NOISE_MARKERS:
        if marker in lowered:
            penalty += 0.10

    dense_ellipsis = lowered.count("...") + lowered.count("..")
    if dense_ellipsis >= 3:
        penalty += 0.08

    return score - penalty


def _tokenize_for_diversity(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _preview(text: str, max_chars: int = 140) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 3] + "..."


def _too_similar_to_selected(text: str, selected_texts: List[str], threshold: float = 0.85) -> bool:
    if not selected_texts:
        return False

    a = _tokenize_for_diversity(text)
    if not a:
        return False

    for prev in selected_texts:
        b = _tokenize_for_diversity(prev)
        if not b:
            continue
        inter = len(a.intersection(b))
        union = len(a.union(b))
        if union and (inter / union) >= threshold:
            return True
    return False


def retrieve_context(chunks_path: Path, query: str, top_k: int = 3) -> str:
    if not chunks_path.exists():
        return ""

    query = query.strip()
    if not query:
        return ""

    model_id = os.getenv("T2V_EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID)
    batch_size = max(1, int(os.getenv("T2V_EMBEDDING_BATCH_SIZE", "32")))
    top_k = max(1, top_k)
    min_score = float(os.getenv("T2V_RAG_MIN_SCORE", "0.12"))
    candidate_k = max(top_k, int(os.getenv("T2V_RAG_CANDIDATE_K", "48")))
    context_word_budget = max(40, int(os.getenv("T2V_RAG_MAX_CONTEXT_WORDS", "220")))
    min_chunk_words = max(10, int(os.getenv("T2V_RAG_MIN_CHUNK_WORDS", "25")))
    rag_debug = _env_flag("T2V_RAG_DEBUG", default=False)
    rag_debug_top_n = max(1, int(os.getenv("T2V_RAG_DEBUG_TOP_N", "8")))

    index_override = os.getenv("T2V_EMBEDDING_INDEX_PATH", "").strip()
    index_path = Path(index_override) if index_override else _default_index_path(chunks_path)

    try:
        rows = _load_chunks(chunks_path)
        if not rows:
            return ""
        chunk_embeddings = _load_or_build_embeddings(
            chunks_path=chunks_path,
            rows=rows,
            model_id=model_id,
            batch_size=batch_size,
            index_path=index_path,
        )
        query_embedding = _embed_texts([query], model_id=model_id, batch_size=1)
    except Exception as exc:  # pragma: no cover - runtime/network dependent
        print(f"Embedding RAG unavailable: {exc}")
        return ""

    if query_embedding.shape[0] == 0:
        return ""

    raw_scores = chunk_embeddings @ query_embedding[0]
    candidate_indices = np.argsort(raw_scores)[::-1][: min(candidate_k, len(rows))]
    scored_rows: List[Tuple[float, float, str, str]] = []
    for idx in candidate_indices:
        row = rows[int(idx)]
        raw = float(raw_scores[int(idx)])
        text = row.get("text", "").strip()
        if not text:
            continue
        score = _penalize_noisy_text(text, raw)
        if score < min_score:
            continue
        chunk_id = str(row.get("chunk_id", f"row-{int(idx)}"))
        scored_rows.append((score, raw, chunk_id, text))

    scored_rows.sort(key=lambda x: x[0], reverse=True)

    if rag_debug:
        print(
            "RAG debug:"
            f" query={query!r}"
            f" model={model_id}"
            f" rows={len(rows)}"
            f" candidates={len(candidate_indices)}"
            f" kept={len(scored_rows)}"
            f" min_score={min_score}"
        )
        for rank, (score, raw, chunk_id, text) in enumerate(scored_rows[:rag_debug_top_n], start=1):
            penalty = raw - score
            print(
                f"  [{rank}] final={score:.4f} raw={raw:.4f} penalty={penalty:.4f} "
                f"words={len(text.split())} id={chunk_id} text={_preview(text)}"
            )

    selected: List[str] = []
    seen = set()
    used_words = 0
    selected_meta: List[str] = []
    for score, raw, chunk_id, text in scored_rows:
        if text in seen:
            continue
        if _too_similar_to_selected(text, selected):
            continue

        words = text.split()
        if len(words) < min_chunk_words:
            continue
        remaining = context_word_budget - used_words
        if remaining <= 0:
            break
        if len(words) > remaining:
            if remaining < min_chunk_words:
                break
            text = " ".join(words[:remaining]).strip()
            words = text.split()

        selected.append(text)
        selected_meta.append(f"{chunk_id}:{score:.4f}/{raw:.4f}")
        seen.add(text)
        used_words += len(words)
        if len(selected) >= min(top_k, len(rows)):
            break

    if not selected:
        return ""

    if rag_debug:
        print("RAG debug selected:", ", ".join(selected_meta))

    return "\n\n".join(selected)
