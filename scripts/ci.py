"""Local CI orchestration. No application UI, installation or publishing.

Synchronous file I/O is limited to manifests and stage logs in this offline CLI.
Commands run sequentially: fail before expensive stages and never race packaging.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import unittest
import uuid

from release_support import ROOT, checked, sha256, source_hashes, verify_native, write_json

NATIVE_ENV = {
    "SUNGLASSES_COMPUTE": "sunglasses-compute.exe",
    "SUNGLASSES_AUTO_COMPUTE": "sunglasses-auto-compute.exe",
    "SUNGLASSES_SCREEN_COMPUTE": "sunglasses-screen-probe.exe",
    "SUNGLASSES_WORKER_HOST": "sunglasses-worker-probe.exe",
}
NATIVE_TESTS = ("test_native", "test_auto_native", "test_screen_pipeline", "test_apple_vfm_screen")
STORE_OPTIONS = ("identity_name", "publisher", "publisher_display_name", "package_version")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", default="check", choices=("check", "native", "release", "store"))
    parser.add_argument("--dry-run", action="store_true", help="Print stages without syncing, building or testing")
    parser.add_argument("--no-build", action="store_true", help="For release/store: verify an existing clean Release")
    parser.add_argument("--build-dir", type=Path, help="Native output inside build/ (default: build/ci-native)")
    parser.add_argument("--sdk-bin", type=Path)
    for name in STORE_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"))
    parser.add_argument("--test-stage", choices=("offline", "native", "worker", "msix"), help=argparse.SUPPRESS)
    parser.add_argument("--plan-json", action="store_true", help=argparse.SUPPRESS)
    return parser


def run_tests(stage: str) -> int:
    """Select existing suites explicitly; an unexpected skip fails the CI stage."""
    sys.path.insert(0, str(ROOT))
    loader = unittest.TestLoader()
    if stage == "offline":
        names = [f"tests.{path.stem}" for path in sorted((ROOT / "tests").glob("test_*.py"))
                 if "trash" not in path.name.lower() and path.stem not in (*NATIVE_TESTS, "test_msix")]
        names.append("tests.test_msix.MsixConfigurationTests")
        names.append("tests.test_apple_vfm_screen.AppleVFMScreenTests")
    elif stage == "native":
        names = [f"tests.{name}" for name in NATIVE_TESTS]
    else:
        names = ["tests.test_service" if stage == "worker" else "tests.test_msix.MsixArchiveTests"]
    result = unittest.TextTestRunner(verbosity=2).run(loader.loadTestsFromNames(names))
    if result.skipped:
        print("Unexpected skipped checks; install/configure the required tools.", file=sys.stderr)
    return 0 if result.wasSuccessful() and result.testsRun > 0 and not result.skipped else 1


def stage_names(profile: str) -> list[str]:
    stages = ["dependencies", "resources", "lint", "types", "offline-tests"]
    if profile != "check":
        if profile != "native":
            stages.append("clean-release-build")
        stages.extend(["native-provenance", "warp-tests"])
    if profile == "release":
        stages.extend(["portable-screen", "portable-full", "worker-tests"])
    if profile in ("release", "store"):
        stages.extend(["msix-preview" if profile == "release" else "msix-store", "msix-tests"])
    if profile == "release":
        stages.extend(["wheel", "wheel-check"])
    return stages


class Pipeline:
    def __init__(self, profile: str, environment: dict[str, str]) -> None:
        token = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.directory = checked(ROOT / "docs/records/ci" / token)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.environment = environment
        self.steps: list[dict] = []
        self.artifacts: list[dict] = []
        self.report = {"schema": 1, "profile": profile, "status": "running", "steps": self.steps,
                       "artifacts": self.artifacts, "python": platform.python_version(),
                       "validation_scope": "offline/static/WARP only; no installation, WACK, UI or Store submission"}
        self.save()
        print(f"CI report: {self.directory / 'summary.json'}", flush=True)

    def save(self) -> None:
        temporary = self.directory / "summary.tmp"
        write_json(temporary, self.report)
        temporary.replace(self.directory / "summary.json")

    def run(self, name: str, command: list[str | Path], timeout: int = 600) -> None:
        arguments = [str(argument) for argument in command]
        log = self.directory / f"{len(self.steps) + 1:02d}-{name}.log"
        step = {"name": name, "command": arguments, "log": str(log), "status": "running"}
        self.steps.append(step)
        self.save()
        print(f"[{len(self.steps)}] {name}", flush=True)
        started = time.monotonic()
        try:
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(arguments, cwd=ROOT, env=self.environment, stdout=stream,
                                           stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                try:
                    exit_code = process.wait(timeout=timeout)
                except BaseException:
                    # Stop the tree before releasing the workspace lock, including
                    # CMake/compiler, SDK or frozen-worker children of this stage.
                    if process.poll() is None:
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       stdout=stream, stderr=subprocess.STDOUT, check=False, timeout=30)
                    process.wait(timeout=30)
                    raise
            step["exit_code"] = exit_code
            if exit_code:
                print(log.read_text(encoding="utf-8", errors="replace")[-8000:], file=sys.stderr)
                raise RuntimeError(f"{name} failed ({exit_code}); see {log}")
            step["status"] = "passed"
        except BaseException:
            step["status"] = "failed"
            raise
        finally:
            step["seconds"] = round(time.monotonic() - started, 3)
            self.save()
        print(f"    passed ({step['seconds']} s)", flush=True)

    def python(self, name: str, *arguments: str | Path, timeout: int = 600) -> None:
        self.run(name, [sys.executable, "-X", "utf8", *arguments], timeout)

    def package(self, name: str, script: str, arguments: list[str | Path]) -> dict:
        result_path = self.directory / f"{name}.json"
        self.python(name, ROOT / "scripts" / script, *arguments, "--result-json", result_path, timeout=900)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        archive = checked(result["archive"], ROOT / "dist")
        self.python(f"{name}-check", ROOT / "scripts/check_release.py", archive)
        self.artifacts.append({"kind": name, "path": str(archive), "sha256": sha256(archive)})
        self.save()
        return result


def check_environment() -> None:
    if sys.platform != "win32" or sys.version_info[:2] != (3, 12) or platform.machine().lower() not in ("amd64", "x86_64"):
        raise ValueError("CI requires Windows x64 / Python 3.12")
    config = Path(sys.prefix) / "pyvenv.cfg"
    if sys.prefix == sys.base_prefix or not config.is_file() or "include-system-site-packages = false" not in config.read_text().lower():
        raise ValueError("CI requires an isolated environment; use ci.ps1")
    for name in ("uv", "node", "ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise ValueError(f"Missing {name} on PATH; see docs/USAGE.md")


def execute(args: argparse.Namespace, pipeline: Pipeline, build: Path) -> None:
    check_environment()
    if args.profile in ("release", "store"):
        from package_msix import sdk_tools
        sdk_tools(args.sdk_bin)
    before = source_hashes()
    pipeline.report["source_sha256"] = before
    pipeline.run("lock", ["uv", "lock", "--check", "--offline"])
    exported = pipeline.directory / "requirements-build.txt"
    pipeline.run("dependency-export", ["uv", "export", "--locked", "--offline", "--only-group", "build",
                                      "--no-emit-project", "--no-header", "--no-hashes", "--output-file", exported])
    if exported.read_text(encoding="utf-8") != (ROOT / "requirements-build.txt").read_text(encoding="utf-8"):
        raise ValueError("requirements-build.txt is stale; export the build group from uv.lock")
    pipeline.run("dependencies", ["uv", "pip", "check", "--python", sys.executable])
    pipeline.python("resources", ROOT / "scripts/check_project.py")
    # Feed the existing public allowlist to tools, excluding private/local paths.
    python_files = [ROOT / name for name in before if name.endswith(".py")]
    pipeline.python("lint", "-m", "ruff", "check", *python_files)
    pipeline.python("types", "-m", "ty", "check")
    pipeline.python("offline-tests", Path(__file__), "--test-stage", "offline")
    if args.profile != "check":
        if args.profile != "native" and not args.no_build:
            pipeline.python("clean-release-build", ROOT / "scripts/dev.py", "build", "--config", "Release",
                            "--clean", "--build-dir", build, timeout=1200)
        native = verify_native(build, for_package=args.profile != "native")
        pipeline.report["native_build"] = native
        for name, binary in NATIVE_ENV.items():
            pipeline.environment[name] = str(build / binary)
        pipeline.python("warp-tests", Path(__file__), "--test-stage", "native")
    common: list[str | Path] = ["--native-build", build]
    if args.profile == "release":
        pipeline.package("portable-screen", "package_windows.py", [*common, "--screen-only"])
        full = pipeline.package("portable-full", "package_windows.py", common)
        pipeline.environment["SUNGLASSES_WORKER"] = str(checked(Path(full["folder"]) / "runtime/sunglasses-worker.exe"))
        pipeline.python("worker-tests", Path(__file__), "--test-stage", "worker")
    if args.profile in ("release", "store"):
        options = common.copy()
        if args.sdk_bin:
            options.extend(["--sdk-bin", args.sdk_bin])
        if args.profile == "release":
            options.append("--preview")
        else:
            for name in STORE_OPTIONS:
                options.extend(["--" + name.replace("_", "-"), getattr(args, name)])
        msix = pipeline.package("msix-" + ("preview" if args.profile == "release" else "store"), "package_msix.py", options)
        pipeline.environment["SUNGLASSES_MSIX"] = str(checked(msix["archive"]))
        pipeline.python("msix-tests", Path(__file__), "--test-stage", "msix")
    if args.profile == "release":
        wheels = checked(ROOT / "dist" / f"wheels-{pipeline.directory.name}")
        pipeline.run("wheel", ["uv", "build", "--wheel", "--no-build-isolation", "--python", sys.executable,
                               "--out-dir", wheels])
        archives = list(wheels.glob("*.whl"))
        if len(archives) != 1:
            raise ValueError("Expected exactly one wheel from this run")
        pipeline.python("wheel-check", ROOT / "scripts/check_release.py", archives[0])
        pipeline.artifacts.append({"kind": "wheel", "path": str(archives[0]), "sha256": sha256(archives[0])})
    if before != source_hashes():
        raise ValueError("Public inputs changed during CI; rerun before using the artifacts")


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.test_stage:
        if args.dry_run or args.plan_json:
            parser.error("Internal test stages cannot be combined with plan options")
        return run_tests(args.test_stage)
    if args.plan_json and not args.dry_run:
        parser.error("--plan-json requires --dry-run")
    if args.no_build and args.profile not in ("release", "store"):
        parser.error("--no-build applies only to release/store; check/native never compile")
    if args.profile == "store":
        from package_msix import package_identity
        package_identity(argparse.Namespace(preview=False, **{name: getattr(args, name) for name in STORE_OPTIONS}))
    elif any(getattr(args, name) for name in STORE_OPTIONS):
        parser.error("Store identity/version arguments require the store profile")
    if args.sdk_bin and args.profile not in ("release", "store"):
        parser.error("--sdk-bin applies only to release/store")
    build = checked(args.build_dir or ROOT / "build/ci-native", ROOT / "build")
    if build == (ROOT / "build").resolve():
        parser.error("Choose a subdirectory of build for native output")
    if args.dry_run:
        if args.plan_json:
            print(json.dumps({"profile": args.profile, "build": args.profile in ("release", "store") and not args.no_build}))
            return 0
        print("Profile:", args.profile)
        print("Native directory:", build)
        for name in stage_names(args.profile):
            print(" -", "verify-existing-clean-release" if name == "clean-release-build" and args.no_build else name)
        print("No commands executed; no validation result.")
        return 0
    # Ignore ambient paths so a developer shell cannot silently enable hardware,
    # use an old package, or accidentally hide required coverage.
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith("SUNGLASSES_") and key.upper() not in ("PYTHONPATH", "PYTHONHOME")}
    environment.update(PYTHONUTF8="1", PYTHONNOUSERSITE="1", SUNGLASSES_TEST_HARDWARE="0",
                       UV_PROJECT_ENVIRONMENT=sys.prefix, VIRTUAL_ENV=sys.prefix)
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    pipeline = Pipeline(args.profile, environment)
    try:
        execute(args, pipeline, build)
        pipeline.report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        pipeline.report.update(status="failed", error=str(error) or "Interrupted")
        print(f"CI failed: {error}", file=sys.stderr)
        return 1
    finally:
        pipeline.save()
    print(f"CI passed ({args.profile}). Report: {pipeline.directory / 'summary.json'}")
    for artifact in pipeline.artifacts:
        print(f"{artifact['kind']}: {artifact['path']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
