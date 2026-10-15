"""Build native targets and record inputs, configuration and clean/incremental mode.

File I/O and compilation are synchronous in this developer command, outside the UI.
The no-argument command preserves the legacy clean Release build in build/native.
"""

import argparse
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from release_support import CONFIGURATIONS, ROOT, EXECUTABLES, checked, native_inputs, sha256, write_json


def read_cache(build):
    entries = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([^#/][^:=]*):[^=]+=(.*)$", line)
        if match:
            entries[match[1]] = match[2]
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/native")
    parser.add_argument("--config", choices=CONFIGURATIONS, default="Release")
    parser.add_argument("--incremental", action="store_true", help="Skip the clean step; output is not eligible for packaging")
    args = parser.parse_args(argv)
    build = checked(args.build_dir, ROOT / "build")
    if build == (ROOT / "build").resolve():
        raise ValueError("Choose a subdirectory of build for native output")
    # Invalidate previous evidence before doing anything that can fail or alter output.
    record = checked(build / "native-build.json", build)
    record.unlink(missing_ok=True)
    before = native_inputs()
    subprocess.run(["cmake", "--preset", args.config.lower(), "-S", str(ROOT), "-B", str(build),
                    f"-DCMAKE_BUILD_TYPE={args.config}", f"-DCMAKE_TRY_COMPILE_CONFIGURATION={args.config}",
                    "-DSUNGLASSES_BUILD_DESKTOP=ON", "-DSUNGLASSES_BUILD_HEADLESS_CHECKS=ON",
                    f"-DPython3_EXECUTABLE={sys.executable}"], cwd=ROOT, check=True)
    cache = read_cache(build)
    expected = {"CMAKE_BUILD_TYPE": args.config, "CMAKE_GENERATOR": "Ninja",
                "SUNGLASSES_BUILD_DESKTOP": "ON", "SUNGLASSES_BUILD_HEADLESS_CHECKS": "ON"}
    if any(cache.get(key) != value for key, value in expected.items()):
        raise ValueError("Configured native build does not match the requested configuration")
    if Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != ROOT:
        raise ValueError("CMake cache belongs to a different source directory")
    command = ["cmake", "--build", str(build)]
    if not args.incremental:
        command.append("--clean-first")
    subprocess.run(command, cwd=ROOT, check=True)
    if before != native_inputs():
        raise ValueError("Native inputs changed during compilation; rebuild")
    compiler_files = list((build / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    compiler = max(compiler_files, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8")
    def value(name):
        match = re.search(r'set\(' + re.escape(name) + r' "?([^"\n)]*)"?\)', compiler)
        return match.group(1) if match else "unknown"
    if value("CMAKE_CXX_COMPILER_ARCHITECTURE_ID") != "x64":
        raise ValueError("Portable packages require an x64 native toolchain")
    manifest = {"schema": 2, "configuration": args.config, "generator": "Ninja",
                "build_mode": "incremental" if args.incremental else "clean",
                "toolchain": {"compiler": value("CMAKE_CXX_COMPILER_ID"),
                              "compiler_version": value("CMAKE_CXX_COMPILER_VERSION"),
                              "architecture": value("CMAKE_CXX_COMPILER_ARCHITECTURE_ID"),
                              "windows_sdk": os.environ.get("WindowsSDKVersion", "unknown").rstrip("\\/"),
                              "cmake": subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0],
                              "python": platform.python_version()},
                "source_sha256": before,
                "artifacts": {name: sha256(build / name) for name in EXECUTABLES}}
    temporary = checked(build / "native-build.json.tmp", build)
    write_json(temporary, manifest)
    temporary.replace(record)
    print(f"Native {args.config} {manifest['build_mode']} build record verified and saved: {build}")


if __name__ == "__main__":
    main()
