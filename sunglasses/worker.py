"""Desktop worker protocol v1: bounded UTF-8 lines on stdout; cancellation on stdin."""

import argparse
import os
from pathlib import Path
import sys
import threading
import time

from .filter import FilterConfig
from .i18n import LANGUAGES, ValidationError, translator
from .jobs import JobCancelled, JobContext
from .service import ImageRequest, experiment_preset, process_image, run_experiment


def emit(kind, value=""):
    # One event per line; OS messages cannot inject additional protocol events.
    clean = str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")[:2048]
    print(f"{kind}\t{clean}", flush=True)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=LANGUAGES, default="en")
    parser.add_argument("--strength", type=float, default=1)
    commands = parser.add_subparsers(dest="operation", required=True)
    image = commands.add_parser("image")
    image.add_argument("source", type=Path)
    image.add_argument("output", type=Path)
    image.add_argument("--assume-srgb", action="store_true")
    experiment = commands.add_parser("experiment")
    experiment.add_argument("output", type=Path)
    experiment.add_argument("--preset", choices=("quick", "full"), default="quick")
    args = parser.parse_args(argv)
    t = translator(args.language)
    cancelled = threading.Event()

    def listen():
        # A single bounded command; EOF also means the controlling host went away.
        command = os.read(sys.stdin.fileno(), 32)
        if not command or command.strip() == b"cancel":
            cancelled.set()

    threading.Thread(target=listen, daemon=True).start()
    last_update = 0.0

    def progress(value):
        nonlocal last_update
        now = time.monotonic()
        if now - last_update >= .1:
            emit("progress", min(99, value.completed * 100 // max(1, value.total)))
            last_update = now

    emit("ready", 1)
    job = JobContext(cancelled.is_set, progress)
    try:
        if args.operation == "image":
            config = FilterConfig(strength=args.strength)
            job.check()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result = process_image(ImageRequest(args.source, args.output, config, args.assume_srgb), job)
        else:
            result = run_experiment(experiment_preset(args.output, args.preset, strength=args.strength), job)
        emit("done", result.report or result.output)
        return 0
    except JobCancelled:
        emit("cancelled")
        return 3
    except ValidationError as error:
        emit("error", t(f"error.{error.code}"))
    except FileExistsError:
        emit("error", t("error.output_exists"))
    except (OSError, ValueError) as error:
        emit("error", t("cli.error", detail=error))
    except Exception:
        emit("error", t("error.worker_failed"))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
