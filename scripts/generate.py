import os
from pathlib import Path
import sys


def _bootstrap_cache_env() -> None:
    cache_root = (
        os.getenv("T2V_CACHE_ROOT", "").strip()
        or os.getenv("XDG_CACHE_HOME", "").strip()
        or "/workspace/.cache"
    )
    hf_root = os.getenv("HF_HOME", "").strip() or f"{cache_root}/huggingface"
    hub_root = (
        os.getenv("HUGGINGFACE_HUB_CACHE", "").strip()
        or os.getenv("HF_HUB_CACHE", "").strip()
        or f"{hf_root}/hub"
    )
    transformers_cache = os.getenv("TRANSFORMERS_CACHE", "").strip() or hf_root
    tmp_root = os.getenv("TMPDIR", "").strip() or f"{cache_root}/tmp"
    os.makedirs(tmp_root, exist_ok=True)

    # Must be set before importing diffusers/huggingface_hub.
    os.environ.setdefault("XDG_CACHE_HOME", cache_root)
    os.environ.setdefault("HF_HOME", hf_root)
    os.environ.setdefault("TRANSFORMERS_CACHE", transformers_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_root)
    os.environ.setdefault("HF_HUB_CACHE", hub_root)
    os.environ.setdefault("TMPDIR", tmp_root)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # Default GPU model to Wan 2.1 14B unless explicitly overridden.
    os.environ.setdefault("T2V_GPU_MODEL_ID", "Wan-AI/Wan2.1-T2V-14B-Diffusers")


_bootstrap_cache_env()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from t2v.cli.generate import main
except ModuleNotFoundError as exc:
    if exc.name == "diffusers":
        print(
            "Missing dependency: diffusers.\n"
            "Run with the project virtual environment:\n"
            "  source /workspace/.venv/bin/activate\n"
            "  python /workspace/Disertatie/scripts/generate.py --topic \"...\" --use-rag\n"
            "Or directly:\n"
            "  /workspace/.venv/bin/python /workspace/Disertatie/scripts/generate.py --topic \"...\" --use-rag"
        )
        raise SystemExit(1)
    raise


if __name__ == "__main__":
    main()
