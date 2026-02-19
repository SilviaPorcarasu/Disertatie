from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass
class Chunk:
    chunk_id: str
    source: str
    chunk_index: int
    start_word: int
    end_word: int
    approx_tokens: int
    text: str


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "PDF support needs `pypdf`. Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _normalize_text(_read_text_file(path))
    if suffix == ".pdf":
        return _normalize_text(_read_pdf_file(path))
    raise ValueError(f"Unsupported format: {suffix}. Use .txt, .md, or .pdf")


def _split_words(text: str) -> List[str]:
    return text.split()


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n+", text)
    cleaned: List[str] = []
    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        if normalized:
            cleaned.append(normalized)
    return cleaned


def make_chunks(
    text: str,
    source: str,
) -> List[Chunk]:
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[Chunk] = []
    cursor = 0

    for chunk_index, paragraph in enumerate(paragraphs):
        chunk_words = _split_words(paragraph)
        if not chunk_words:
            continue

        start = cursor
        end = start + len(chunk_words)
        cursor = end
        approx_tokens = int(len(chunk_words) * 1.3)

        chunks.append(
            Chunk(
                chunk_id=f"{Path(source).stem}-{chunk_index:05d}",
                source=source,
                chunk_index=chunk_index,
                start_word=start,
                end_word=end,
                approx_tokens=approx_tokens,
                text=paragraph,
            )
        )

    return chunks


_DROP_PATTERNS = (
    "contents ",
    "bibliography",
    "index ",
    "solidify your knowledge",
    "what have you learned",
    "computer assignments",
    "historical remarks",
    "library of congress",
    "isbn ",
    "doi ",
)

_OCR_MARKERS = ("/nul", "/stx", "/esc", "/dc", "/nak", "c141", "uni00a0")
_OCR_TOKEN_RE = re.compile(r"/(?:nul|stx|esc|dc|nak)\b", re.IGNORECASE)
_HEX_UNICODE_RE = re.compile(r"\buni[0-9a-f]{4}\b", re.IGNORECASE)
_RAW_CODE_RE = re.compile(r"\bc1[0-9]{2}\b", re.IGNORECASE)
_FIG_REF_RE = re.compile(
    r"\b(?:fig(?:ure)?|table|eq(?:uation)?|chap(?:ter)?)\.?\s*\(?\d+(?:\.\d+)*\)?",
    re.IGNORECASE,
)
_FIG_CAPTION_START_RE = re.compile(r"^\s*(?:fig(?:ure)?|table)\.?\s*\d", re.IGNORECASE)
_BRACKETED_REF_RE = re.compile(
    r"\(\s*(?:fig(?:ure)?|table|eq(?:uation)?|chap(?:ter)?)\.?\s*[\d.]+\s*\)",
    re.IGNORECASE,
)
_PAGE_HEADER_RE = re.compile(
    r"^\s*\d+\s+\d+(?:\.\d+)?\s+[A-Za-z][A-Za-z \-]{8,}\s+",
    re.IGNORECASE,
)


def _normalize_chunk_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_non_textual_references(text: str) -> str:
    cleaned = _PAGE_HEADER_RE.sub("", text)
    cleaned = _BRACKETED_REF_RE.sub(" ", cleaned)
    cleaned = _FIG_REF_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _looks_like_noise(text: str) -> Tuple[bool, str]:
    lowered = text.lower()

    for pattern in _DROP_PATTERNS:
        if pattern in lowered:
            return True, f"pattern:{pattern}"

    if lowered.count("...") >= 4 or re.search(r"\.{5,}", lowered):
        return True, "toc_dots"

    marker_hits = sum(1 for marker in _OCR_MARKERS if marker in lowered)
    regex_hits = 0
    regex_hits += len(_OCR_TOKEN_RE.findall(text))
    regex_hits += len(_HEX_UNICODE_RE.findall(text))
    regex_hits += len(_RAW_CODE_RE.findall(text))
    total_hits = marker_hits + regex_hits
    if total_hits >= 1:
        return True, "ocr_markers"

    return False, ""


def _split_long_text(text: str, max_words: int) -> List[str]:
    words = _split_words(text)
    if len(words) <= max_words:
        return [text]

    parts: List[str] = []
    for i in range(0, len(words), max_words):
        part = " ".join(words[i : i + max_words]).strip()
        if part:
            parts.append(part)
    return parts


def _text_quality_ok(text: str) -> Tuple[bool, str]:
    total = max(1, len(text))
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    spaces = sum(1 for c in text if c.isspace())
    symbols = total - letters - digits - spaces

    alpha_ratio = letters / total
    digit_ratio = digits / total
    symbol_ratio = symbols / total

    if alpha_ratio < 0.45:
        return False, "low_alpha_ratio"
    if digit_ratio > 0.22:
        return False, "high_digit_ratio"
    if symbol_ratio > 0.22:
        return False, "high_symbol_ratio"

    tokens = _split_words(text)
    if not tokens:
        return False, "empty_tokens"

    noisy_tokens = 0
    very_long_tokens = 0
    for tok in tokens:
        if len(tok) > 30:
            very_long_tokens += 1
        if "/" in tok or "\\" in tok:
            noisy_tokens += 1
            continue
        if len(re.sub(r"[A-Za-z0-9\-]", "", tok)) >= 2:
            noisy_tokens += 1
    if very_long_tokens / len(tokens) > 0.02:
        return False, "high_very_long_token_ratio"
    if noisy_tokens / len(tokens) > 0.12:
        return False, "high_noisy_token_ratio"

    return True, ""


def _fingerprint(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def clean_chunks(
    chunks: List[Chunk],
    *,
    min_words: int = 45,
    max_words: int = 220,
    split_long_chunks: bool = True,
) -> Tuple[List[Chunk], List[dict]]:
    if min_words <= 0:
        raise ValueError("min_words must be > 0")
    if max_words < min_words:
        raise ValueError("max_words must be >= min_words")

    accepted_rows: List[Tuple[str, str]] = []
    rejected: List[dict] = []
    seen = set()

    for chunk in chunks:
        normalized = _normalize_chunk_text(chunk.text)
        if not normalized:
            rejected.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "reason": "empty",
                    "text": "",
                }
            )
            continue

        is_noise, reason = _looks_like_noise(normalized)
        if is_noise:
            rejected.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "reason": reason,
                    "text": normalized[:400],
                }
            )
            continue

        parts = _split_long_text(normalized, max_words=max_words) if split_long_chunks else [normalized]
        for part in parts:
            part = _strip_non_textual_references(part)
            if not part:
                rejected.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "reason": "empty_after_reference_strip",
                        "text": "",
                    }
                )
                continue

            if _FIG_CAPTION_START_RE.match(part) and len(_split_words(part)) <= 100:
                rejected.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "reason": "figure_caption",
                        "text": part[:400],
                    }
                )
                continue

            words = _split_words(part)
            if len(words) < min_words:
                rejected.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "reason": "too_short",
                        "text": part[:400],
                    }
                )
                continue

            ok, qreason = _text_quality_ok(part)
            if not ok:
                rejected.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "reason": qreason,
                        "text": part[:400],
                    }
                )
                continue

            fp = _fingerprint(part)
            if fp in seen:
                rejected.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "reason": "duplicate",
                        "text": part[:400],
                    }
                )
                continue

            seen.add(fp)
            accepted_rows.append((chunk.source, part))

    cleaned: List[Chunk] = []
    cursor = 0
    for idx, (source, text) in enumerate(accepted_rows):
        words = _split_words(text)
        start = cursor
        end = start + len(words)
        cursor = end
        cleaned.append(
            Chunk(
                chunk_id=f"{Path(source).stem}-{idx:05d}",
                source=source,
                chunk_index=idx,
                start_word=start,
                end_word=end,
                approx_tokens=int(len(words) * 1.3),
                text=text,
            )
        )

    return cleaned, rejected


def save_chunks_jsonl(chunks: Iterable[Chunk], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                        "start_word": c.start_word,
                        "end_word": c.end_word,
                        "approx_tokens": c.approx_tokens,
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def save_rejected_jsonl(rows: Iterable[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
