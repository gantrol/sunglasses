"""Command-line research workflow. No network, window capture or media playback."""

import argparse
from pathlib import Path
import sys

from .filter import FilterConfig
from .fixtures import CASES, AUTO_CASES, SequenceSpec
from .automatic import AutoConfig
from .service import ExperimentRequest, ImageRequest, process_image, run_experiment
from .i18n import LANGUAGES, ValidationError, translator


def parser(language: str = "en") -> argparse.ArgumentParser:
    t = translator(language)
    root = argparse.ArgumentParser(prog="sunglasses", description=t("cli.description"))
    root.add_argument("--language", choices=LANGUAGES, default=language, help=t("cli.language"))
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("cases", help=t("cli.cases"))
    suite = commands.add_parser("suite", help=t("cli.suite"))
    automatic = commands.add_parser("auto-suite", help=t("cli.auto_suite"))
    for command, cases, duration in ((suite, CASES, 2), (automatic, AUTO_CASES, 4)):
        command.add_argument("--output", type=Path, required=True, help=t("cli.output"))
        command.add_argument("--case", action="append", choices=[c.id for c in cases])
        command.add_argument("--width", type=int, default=320)
        command.add_argument("--height", type=int, default=180)
        command.add_argument("--fps", type=int, default=60)
        command.add_argument("--duration", type=float, default=duration)
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--threshold", type=float, default=.5)
    automatic.add_argument("--strength", type=float, default=1)
    automatic.add_argument("--hold", type=float, default=.2, help=t("cli.hold"))
    automatic.add_argument("--release", type=float, default=.5, help=t("cli.release"))
    automatic.add_argument("--no-repeat", action="store_true", help=t("cli.no_repeat"))
    automatic.add_argument("--policy", choices=("highlight", "legacy"), default="highlight", help=t("cli.policy"))
    process = commands.add_parser("process", help=t("cli.process"))
    process.add_argument("input", type=Path)
    process.add_argument("output", type=Path)
    process.add_argument("--assume-srgb", action="store_true", help=t("cli.assume_srgb"))
    for command in (suite, process):
        command.add_argument("--knee", type=float, default=.20)
        command.add_argument("--ceiling", type=float, default=.55)
        command.add_argument("--strength", type=float, default=1)
    return root


def main(argv=None) -> int:
    # Windows redirected consoles may default to a code page without Chinese.
    # Keep CLI output UTF-8 without changing in-memory streams used by callers.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    initial = argparse.ArgumentParser(add_help=False)
    initial.add_argument("--language", choices=LANGUAGES, default="en")
    language = initial.parse_known_args(argv)[0].language
    t = translator(language)
    args = parser(language).parse_args(argv)
    try:
        if args.command == "cases":
            for case in CASES:
                print(f"{case.id}: {case.description}")
            return 0
        if args.command == "auto-suite":
            config = AutoConfig(args.strength, args.hold, args.release, not args.no_repeat, args.policy)
            spec = SequenceSpec(args.width, args.height, args.fps, args.duration, args.seed)
            result = run_experiment(ExperimentRequest(args.output, sequence=spec, cases=tuple(args.case or ()),
                                                     automatic=config, threshold=args.threshold))
            print(t("cli.completed", count=result.case_count, path=result.report))
            return 0
        config = FilterConfig(args.knee, args.ceiling, args.strength)
        if args.command == "process":
            process_image(ImageRequest(args.input, args.output, config, args.assume_srgb))
            print(t("cli.saved", path=args.output))
        else:
            spec = SequenceSpec(args.width, args.height, args.fps, args.duration, args.seed)
            result = run_experiment(ExperimentRequest(args.output, mode="static", sequence=spec, cases=tuple(args.case or ()),
                                                     static=config, threshold=args.threshold))
            print(t("cli.completed", count=result.case_count, path=result.report))
        return 0
    except ValidationError as error:
        print(t("cli.error", detail=t(f"error.{error.code}")), file=sys.stderr)
        return 2
    except (ValueError, OSError) as error:
        print(t("cli.error", detail=error), file=sys.stderr)
        return 2
