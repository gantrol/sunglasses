"""Package checked binaries and corresponding sources; never publish or start a UI.

All blocking I/O runs in this explicit offline build command, outside the app.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import uuid
import zipfile

from release_support import ROOT, check_links, documentation_files, sha256, source_files, source_hashes, verify_native, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-only", action="store_true", help="Omit the offline laboratory and Python runtime")
    parser.add_argument("--native-build", type=Path, default=ROOT / "build/native")
    parser.add_argument("--result-json", type=Path, help="Write package paths for local automation")
    args = parser.parse_args()
    if args.result_json:
        result_path = args.result_json.resolve()
        if "trash" in str(result_path).lower() or result_path.exists() or not result_path.parent.is_dir():
            raise ValueError("Result JSON must be a new file in an existing, permitted directory")
    # Validate before creating output or freezing anything.
    native_build = args.native_build.resolve()
    native = verify_native(native_build)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project["license"] != "GPL-3.0-only":
        raise ValueError("Unexpected project license")
    before = source_hashes()
    check_links(ROOT)
    if not args.screen_only:
        configuration = Path(sys.prefix) / "pyvenv.cfg"
        if (sys.prefix == sys.base_prefix or not configuration.is_file()
                or "include-system-site-packages = false" not in configuration.read_text(encoding="utf-8").lower()):
            raise ValueError("Full packages require a venv without system-site-packages; see docs/USAGE.md")
    token = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f") + "-" + uuid.uuid4().hex[:8]
    work = ROOT / "build/packages" / token
    output = ROOT / "dist" / f"Sunglasses-{'screen' if args.screen_only else 'portable'}-{token}"
    work.mkdir(parents=True, exist_ok=False)
    python_hashes = {name.removeprefix("sunglasses/"): value for name, value in before.items()
                     if name.startswith("sunglasses/")}
    write_json(work / "source_hashes.json", python_hashes)
    versions = {}
    if not args.screen_only:
        environment = dict(os.environ, SUNGLASSES_FREEZE_MANIFEST=str(work / "source_hashes.json"))
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        # DLL resolution must not pull unrelated libraries from the user's PATH.
        windows = Path(os.environ["SystemRoot"])
        environment["PATH"] = os.pathsep.join(str(path) for path in (
            Path(sys.prefix) / "Scripts", Path(sys.base_prefix), Path(sys.base_prefix) / "DLLs",
            windows / "System32", windows))
        environment["PYTHONNOUSERSITE"] = "1"
        subprocess.run([sys.executable, "-m", "PyInstaller", "--distpath", str(work / "frozen"),
                        "--workpath", str(work / "cache"), str(ROOT / "scripts/worker.spec")],
                       cwd=ROOT, env=environment, check=True)
        versions = json.loads((work / "licenses/dependencies.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(native_build / "Sunglasses.exe", output / "Sunglasses.exe")
    if not args.screen_only:
        shutil.copy2(native_build / "SunglassesLab.exe", output / "SunglassesLab.exe")
        shutil.copytree(work / "frozen/sunglasses-worker", output / "runtime")
        shutil.copytree(work / "licenses", output / "licenses")
    # Public document links work in both the binary bundle and its source archive.
    (output / "docs").mkdir()
    for path in documentation_files():
        relative = path.relative_to(ROOT)
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output / relative)
    check_links(output)
    write_json(output / "BUILD.json", native)
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, path.relative_to(ROOT).as_posix())
    # Check the snapshot actually placed in the source archive, not only the live tree.
    with zipfile.ZipFile(output / "source.zip") as archive:
        archived = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    if archived != before or source_hashes() != before or verify_native(native_build) != native:
        raise ValueError("Inputs changed while packaging; no distributable archive was created")
    for name in ("Sunglasses.exe",) if args.screen_only else ("Sunglasses.exe", "SunglassesLab.exe"):
        if sha256(output / name) != native["artifacts"][name]:
            raise ValueError(f"Copied binary does not match its build record: {name}")
    write_json(output / "package.json", {
        "schema": 1, "name": "Sunglasses Screen Filter", "version": project["version"], "license": project["license"],
        "format": "windows-x64-portable", "entrypoint": "Sunglasses.exe", "screen_only": args.screen_only,
        "dependencies": versions, "source_archive": "source.zip", "source_sha256": before,
        "native_source_sha256": native["source_sha256"],
        "python_source_sha256": {} if args.screen_only else python_hashes,
        "capabilities": ["manual-screen-dimming", "wgc-monitor-capture", "borderless-capture-required",
                         "gpu-transient-highlight-overlay", "pixel-glasses-controller", "pixel-glasses-case",
                         "notification-area-controls", "recordable-display-overlay", "session-window-exclusion", "case-open-close-motion",
                         "graphite-palette"] + ([] if args.screen_only else ["offline-laboratory"]),
        "scope": "Windows 11 SDR; one selected display; recording-visible automatic mode requires session window exclusion; no exclusive fullscreen or HDR support claim",
        "automatic_policy": "sudden highlights; startup calibrates; static scenes transparent; 0.32 s event envelope",
        "automatic_strength": {"default": 0.5, "balanced": 0.5,
                               "balanced_anchor": "previous demonstrated full curve",
                               "maximum": "load-dependent stronger compression; knee can fall to Y=0.45"},
        "validation": "synthetic/headless only; real capture, composition, UI, OBS and games not verified",
    })
    hashes = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*"))
              if "trash" not in str(p).lower() and p.is_file()}
    write_json(output / "SHA256.json", hashes)
    archive_path = output.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if "trash" in str(path).lower():
                raise ValueError("Excluded package path")
            if path.is_file():
                bundle.write(path, Path("Sunglasses") / path.relative_to(output))
    result = {"folder": str(output), "archive": str(archive_path)}
    if args.result_json:
        if "trash" in str(args.result_json).lower():
            raise ValueError("Excluded result path")
        with args.result_json.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
