from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleCheck:
    import_name: str
    package_name: str
    required: bool
    note: str = ""


MODULES = [
    ModuleCheck("numpy", "numpy", True),
    ModuleCheck("PIL", "pillow", True),
    ModuleCheck("torch", "torch", True),
    ModuleCheck("transformers", "transformers", True),
    ModuleCheck("diffusers", "diffusers", True),
    ModuleCheck("pypdf", "pypdf", True, "required for PDF chunking"),
    ModuleCheck("imageio.v3", "imageio", False, "fallback video export backend"),
    ModuleCheck("google.protobuf", "protobuf", False, "needed by some tokenizer conversions"),
    ModuleCheck("tiktoken", "tiktoken", False, "fallback tokenizer converter backend"),
    ModuleCheck(
        "imageio_ffmpeg",
        "imageio-ffmpeg",
        False,
        "bundled ffmpeg backend for imageio video export",
    ),
]

def _check_modules(strict_optional: bool) -> tuple[list[str], list[str]]:
    missing_required: list[str] = []
    missing_optional: list[str] = []

    print("Python module checks:")
    for item in MODULES:
        try:
            importlib.import_module(item.import_name)
            print(f"  [OK]   {item.import_name}")
        except Exception as exc:
            details = str(exc).splitlines()[0]
            if item.required:
                missing_required.append(item.package_name)
                print(
                    f"  [FAIL] {item.import_name} -> install `{item.package_name}`"
                    f"{f' ({item.note})' if item.note else ''}: {details}"
                )
            else:
                missing_optional.append(item.package_name)
                print(
                    f"  [WARN] {item.import_name} missing"
                    f"{f' ({item.note})' if item.note else ''}: {details}"
                )

    if strict_optional and missing_optional:
        missing_required.extend(missing_optional)

    return missing_required, missing_optional


def _check_video_export_backend(*, strict_optional: bool) -> list[str]:
    missing_required: list[str] = []
    print("Video export backend checks:")

    has_ffmpeg_binary = shutil.which("ffmpeg") is not None
    if has_ffmpeg_binary:
        print("  [OK]   ffmpeg binary")
    else:
        print("  [WARN] ffmpeg binary missing")

    has_imageio_ffmpeg = True
    try:
        importlib.import_module("imageio_ffmpeg")
        print("  [OK]   imageio_ffmpeg module")
    except Exception as exc:
        has_imageio_ffmpeg = False
        details = str(exc).splitlines()[0]
        print(f"  [WARN] imageio_ffmpeg missing: {details}")

    if not has_ffmpeg_binary and not has_imageio_ffmpeg:
        if strict_optional:
            print("  [FAIL] no compatible ffmpeg backend available for video export")
            missing_required.append("ffmpeg or imageio-ffmpeg")
        else:
            print("  [WARN] no ffmpeg backend; mp4 export may fail (GIF fallback will be used)")

    return missing_required


def _check_model_tokenizer_dependencies(model_id: str, *, strict_optional: bool) -> list[str]:
    missing_required: list[str] = []
    model = model_id.strip().lower()
    if not model:
        return missing_required

    # CogVideoX tokenizers can require protobuf; if unavailable, transformers
    # may fall back to tiktoken conversion.
    if "cogvideo" not in model:
        return missing_required

    print(f"Tokenizer checks for model: {model_id}")
    has_protobuf = True
    has_tiktoken = True

    try:
        importlib.import_module("google.protobuf")
        print("  [OK]   protobuf")
    except Exception as exc:
        has_protobuf = False
        details = str(exc).splitlines()[0]
        print(f"  [WARN] protobuf missing: {details}")

    try:
        importlib.import_module("tiktoken")
        print("  [OK]   tiktoken")
    except Exception as exc:
        has_tiktoken = False
        details = str(exc).splitlines()[0]
        print(f"  [WARN] tiktoken missing: {details}")

    if not has_protobuf and not has_tiktoken:
        msg = "protobuf or tiktoken"
        if strict_optional:
            print("  [FAIL] CogVideoX tokenizer path will fail (missing protobuf and tiktoken)")
            missing_required.append(msg)
        else:
            print("  [WARN] CogVideoX may fail at tokenizer load (missing protobuf and tiktoken)")

    return missing_required


def _check_venv() -> None:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        print(f"Virtual env: [OK] {sys.prefix}")
    else:
        print("Virtual env: [WARN] not active")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight environment checks for Disertatie/t2v."
    )
    parser.add_argument(
        "--strict-optional",
        action="store_true",
        help="Treat optional dependencies as required.",
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("T2V_MODEL_ID", "Wan-AI/Wan2.1-T2V-14B-Diffusers"),
        help="Model id used to validate model-specific tokenizer/runtime dependencies.",
    )
    args = parser.parse_args()

    _check_venv()
    missing_modules, _ = _check_modules(strict_optional=args.strict_optional)
    missing_video_backend = _check_video_export_backend(strict_optional=args.strict_optional)
    missing_tokenizer_deps = _check_model_tokenizer_dependencies(
        args.model_id,
        strict_optional=args.strict_optional,
    )

    missing = missing_modules + missing_video_backend + missing_tokenizer_deps
    if missing:
        unique = sorted(set(missing))
        print("\nMissing dependencies detected.")
        print("Install with:")
        print("  pip install -r /workspace/Disertatie/requirements.txt")
        print("Missing:", ", ".join(unique))
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
