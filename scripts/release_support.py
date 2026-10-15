"""Shared release checks. Synchronous I/O is confined to offline build commands."""

import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
README_FILES = ("README.md", "README.zh-CN.md")
ROOT_FILES = ("LICENSE", *README_FILES, "AGENTS.md", ".gitignore", ".gitattributes", "CMakeLists.txt", "CMakePresets.json", "dev.ps1",
              "build.cmd", "run.cmd", "debug.cmd", "package.cmd", "pyproject.toml", "requirements-build.txt",
              "ci.ps1", "ci.cmd", ".python-version", "uv.lock")
SOURCE_TREES = {
    "assets/demos": {".gif", ".json"},
    "native": {".cpp", ".h", ".in", ".rc", ".manifest", ".ico"},
    "shaders": {".hlsl"},
    "sunglasses": {".py", ".json", ".html", ".css", ".js", ".txt", ".md"},
    "scripts": {".py", ".spec", ".cmd"},
    "tests": {".py", ".cpp"},
}
EXECUTABLES = ("Sunglasses.exe", "SunglassesLab.exe", "sunglasses-compute.exe",
               "sunglasses-auto-compute.exe", "sunglasses-worker-probe.exe", "sunglasses-screen-probe.exe")
CONFIGURATIONS = ("Debug", "RelWithDebInfo", "Release")


def checked(path, root=ROOT):
    path = Path(path)
    if "trash" in str(path).lower() or path.is_symlink():
        raise ValueError("Excluded release path")
    resolved = path.resolve()
    if "trash" in str(resolved).lower() or not resolved.is_relative_to(root.resolve()):
        raise ValueError("Release path outside its declared root")
    return resolved


def source_files(root=ROOT):
    # Local documentation is optional in source-only GitHub checkouts.
    result = [checked(root / name, root) for name in ROOT_FILES
              if name not in (*README_FILES, "AGENTS.md") or (root / name).is_file()]
    for folder, extensions in SOURCE_TREES.items():
        for parent, directories, names in os.walk(checked(root / folder, root)):
            directories[:] = sorted(d for d in directories if "trash" not in d.lower()
                                    and d != "__pycache__" and not Path(parent, d).is_symlink())
            for name in sorted(names):
                path = Path(parent, name)
                if "trash" not in name.lower() and path.suffix in extensions:
                    result.append(checked(path, root))
    result.extend(checked(p, root) for p in (root / "docs").glob("*.md") if "trash" not in p.name.lower())
    return sorted(result)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes(root=ROOT):
    return {p.relative_to(root).as_posix(): sha256(p) for p in source_files(root)}


def documentation_files(root=ROOT):
    """Include public link targets so bundled documentation remains navigable."""
    public = set(source_files(root))
    result = {path for path in public if path.relative_to(root).parts[0] == "docs"
              or path.name in (*README_FILES, "AGENTS.md", "LICENSE")}
    pending = list(result)
    while pending:
        path = pending.pop()
        if path.suffix.lower() != ".md":
            continue
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            linked = checked(path.parent / target.split("#", 1)[0], root)
            if linked not in public:
                raise ValueError(f"Documentation links to a non-public file: {path.name}")
            if linked not in result:
                result.add(linked)
                pending.append(linked)
    return sorted(result)


def native_inputs(root=ROOT):
    return {name: digest for name, digest in source_hashes(root).items()
            if name.startswith(("native/", "shaders/", "sunglasses/locales/"))
            or name in ("CMakeLists.txt", "CMakePresets.json", "dev.ps1", "pyproject.toml", "scripts/generate_strings.py",
                        "build.cmd", "run.cmd", "debug.cmd", "package.cmd", "scripts/dev.cmd",
                        "scripts/dev.py", "scripts/build_native.py", "scripts/release_support.py",
                        "tests/native_screen_probe.cpp", "tests/native_worker_probe.cpp",
                        "sunglasses/data/apple_ufm_kernels_6e9604d.txt",
                        "sunglasses/data/APPLE_VIDEO_FLASHING_REDUCTION_NOTICE.md")}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_build_record(manifest, *, for_package=True, allow_legacy=False):
    """Development builds cannot be mistaken for clean Release package inputs.

    Legacy records are accepted only when inspecting already-created archives.
    New package inputs must record their build mode explicitly.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Invalid native build record")
    schema = manifest.get("schema")
    configuration = manifest.get("configuration")
    if schema == 1 and allow_legacy:
        if configuration != "Release":
            raise ValueError("Legacy package build must be Release")
        return
    if schema != 2 or configuration not in CONFIGURATIONS:
        raise ValueError("Invalid native build schema or configuration; rebuild using scripts/build_native.py")
    toolchain = manifest.get("toolchain")
    if (manifest.get("generator") != "Ninja" or not isinstance(toolchain, dict)
            or toolchain.get("architecture") != "x64"
            or manifest.get("build_mode") not in ("clean", "incremental")):
        raise ValueError("Invalid native build toolchain or mode")
    if for_package and (configuration != "Release" or manifest["build_mode"] != "clean"):
        raise ValueError("Packaging requires a clean Release build")


def verify_native(build, root=ROOT, *, for_package=True):
    build = Path(build).resolve()
    if "trash" in str(build).lower():
        raise ValueError("Excluded build path")
    manifest = json.loads((build / "native-build.json").read_text(encoding="utf-8"))
    validate_build_record(manifest, for_package=for_package)
    if manifest.get("source_sha256") != native_inputs(root):
        raise ValueError("Native sources changed; run scripts/build_native.py again")
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict) or set(artifacts) != set(EXECUTABLES):
        raise ValueError("Incomplete native build record")
    for name, digest in artifacts.items():
        if sha256(checked(build / name, build)) != digest:
            raise ValueError(f"Native binary changed: {name}")
    return manifest


def check_links(root):
    for path in [*(root / name for name in README_FILES), *(root / "docs").glob("*.md")]:
        if "trash" in str(path).lower() or not path.is_file():
            continue
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            target = target.split("#", 1)[0]
            if target and not checked(path.parent / target, root).is_file():
                raise ValueError(f"Broken documentation link: {path.name}: {target}")
