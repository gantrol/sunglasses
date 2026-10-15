"""Bounded synchronous evidence I/O for offline commands, outside the UI thread."""

import hashlib
import json
from pathlib import Path


def checked(path):
    path = Path(path)
    if "trash" in str(path).lower() or "trash" in str(path.resolve()).lower():
        raise ValueError("Excluded evidence path")
    return path.resolve()


def digest(path):
    result = hashlib.sha256()
    with checked(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def new_directory(path, source=None):
    path = checked(path)
    if source is not None and path.is_relative_to(checked(source)):
        raise ValueError("Write derived evidence outside the original run directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    checked(path).write_text(text + "\n", encoding="utf-8")


def implementation_hashes():
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    paths = list(package.glob("*.py")) + [root / "sunglasses" / name for name in
        ("contracts.py", "contract_suite.py", "presentation_scenes.py", "presentation_sim.py", "demo_filter.py")]
    paths.append(root / "shaders/screen.hlsl")
    return {path.relative_to(root).as_posix(): digest(path) for path in sorted(paths)}
