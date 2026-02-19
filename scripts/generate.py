import os
from pathlib import Path
import sys


def _bootstrap_cache_env() -> None:
    cache_root = "/workspace/.cache"
    hf_root = f"{cache_root}/huggingface"
    hub_root = f"{hf_root}/hub"
    tmp_root = f"{cache_root}/tmp"
    os.makedirs(tmp_root, exist_ok=True)

    # Must be set before importing diffusers/huggingface_hub.
    os.environ["XDG_CACHE_HOME"] = cache_root
    os.environ["HF_HOME"] = hf_root
    os.environ["TRANSFORMERS_CACHE"] = hf_root
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_root
    os.environ["HF_HUB_CACHE"] = hub_root
    os.environ["TMPDIR"] = tmp_root
    os.environ["HF_HUB_DISABLE_XET"] = "1"


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
            "  python /workspace/t2v/scripts/generate.py --topic \"...\" --use-rag\n"
            "Or directly:\n"
            "  /workspace/.venv/bin/python /workspace/t2v/scripts/generate.py --topic \"...\" --use-rag"
        )
        raise SystemExit(1)
    raise


if __name__ == "__main__":
    main()
