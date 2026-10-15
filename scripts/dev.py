"""Build, run, debug, or package the native Windows application.

Development commands build incrementally; packages require a clean Release build.
All file checks and child processes run in this explicit command-line entry point.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from release_support import CONFIGURATIONS, ROOT, checked, verify_native


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "build": "Build native targets incrementally",
        "run": "Build and run Sunglasses.exe",
        "debug": "Build and open Sunglasses.exe in Visual Studio; press F5 to start debugging",
        "package": "Clean-build Release and create a local package",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description)
        choices = ("Release",) if name == "package" else CONFIGURATIONS[:2] if name == "debug" else CONFIGURATIONS
        default = "Release" if name == "package" else "Debug" if name == "debug" else "RelWithDebInfo"
        command.add_argument("--config", choices=choices, default=default,
                             help=f"Native build configuration (default: {default})")
        command.add_argument("--build-dir", type=Path,
                             help="Output directory inside the project's build directory")
        command.add_argument("--dry-run", action="store_true",
                             help="Print commands without building, validating artifacts, or starting programs")
        if name == "build":
            command.add_argument("--clean", action="store_true", help="Rebuild every native target")
        else:
            command.add_argument("--no-build", action="store_true",
                                 help="Use existing binaries only after checking configuration, sources, and hashes")
        if name == "debug":
            command.add_argument("--debugger", type=Path,
                                 help="Path to Visual Studio devenv.exe (otherwise discovered with vswhere)")
        if name in ("run", "debug"):
            command.add_argument("app_arguments", nargs=argparse.REMAINDER,
                                 help="Arguments for Sunglasses.exe, after --")
        if name == "package":
            command.add_argument("--format", choices=("portable", "msix"), default="portable")
            command.add_argument("--full", action="store_true",
                                 help="Include the offline laboratory and Python runtime in a portable package")
            command.add_argument("--preview", action="store_true",
                                 help="Use the local preview identity; required for MSIX")
            command.add_argument("--sdk-bin", type=Path, help="Windows SDK x64 tools directory for MSIX")
            command.add_argument("--result-json", type=Path, help="Save package paths for local automation")
    return parser


def build_directory(directory, configuration):
    base = (ROOT / "build").resolve()
    build = checked(directory if directory is not None else base / configuration.lower(), base)
    if build == base:
        raise ValueError("Choose a subdirectory of the project's build directory")
    return build


def find_debugger(explicit=None, *, dry_run=False):
    if explicit:
        candidate = Path(explicit).resolve()
        if not dry_run and not candidate.is_file():
            raise ValueError(f"Visual Studio debugger not found: {candidate}")
        return str(candidate)
    located = shutil.which("devenv.exe")
    if located:
        return located
    if dry_run:
        return "devenv.exe"
    installer = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer"
    vswhere = shutil.which("vswhere.exe") or str(installer / "vswhere.exe")
    if Path(vswhere).is_file():
        # Search all installations: the newest one may be Build Tools without the IDE.
        result = subprocess.run([vswhere, "-products", "*", "-requires",
                                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                 "-find", r"Common7\IDE\devenv.exe"],
                                capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if Path(line.strip()).is_file():
                return line.strip()
    raise ValueError("Visual Studio C++ debugger not found; install the C++ desktop workload or use --debugger PATH")


def execute(command, *, cwd=ROOT, dry_run=False):
    command = [str(argument) for argument in command]
    print(subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=cwd, check=False).returncode


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.command == "package":
        if args.format == "msix" and (args.full or not args.preview):
            parser.error("MSIX requires --preview and does not support --full; Store identities use scripts/package_msix.py")
        if args.format == "portable" and (args.preview or args.sdk_bin):
            parser.error("--preview and --sdk-bin apply only to --format msix")
    build = build_directory(args.build_dir, args.config)
    debugger = find_debugger(args.debugger, dry_run=args.dry_run) if args.command == "debug" else None
    if not getattr(args, "no_build", False):
        command = [sys.executable, ROOT / "scripts/build_native.py", "--config", args.config,
                   "--build-dir", build]
        if args.command != "package" and not getattr(args, "clean", False):
            command.append("--incremental")
        result = execute(command, dry_run=args.dry_run)
        if result or args.command == "build":
            return result
    if args.dry_run:
        print(f"Verify {build / 'native-build.json'} ({args.config}"
              + (", clean package build)." if args.command == "package" else ")."))
    else:
        manifest = verify_native(build, for_package=args.command == "package")
        if manifest.get("configuration") != args.config:
            raise ValueError(f"Expected {args.config} binaries; rebuild with --config {args.config}")
    if args.command == "package":
        script = "package_msix.py" if args.format == "msix" else "package_windows.py"
        command = [sys.executable, ROOT / "scripts" / script, "--native-build", build]
        if args.format == "msix":
            command.append("--preview")
            if args.sdk_bin:
                command.extend(["--sdk-bin", args.sdk_bin.resolve()])
        elif not args.full:
            command.append("--screen-only")
        if args.result_json:
            command.extend(["--result-json", args.result_json.resolve()])
        return execute(command, dry_run=args.dry_run)
    arguments = args.app_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    command = [build / "Sunglasses.exe", *arguments]
    if args.command == "debug":
        print("Visual Studio opens the executable; press F5 to start debugging.", flush=True)
        command = [debugger, "/DebugExe", *command]
    return execute(command, cwd=build, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(error.returncode if isinstance(error, subprocess.CalledProcessError) else 1)
