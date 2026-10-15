"""Offline measurements: evaluator controls, synthetic comparisons, recorded runs."""

import argparse
from pathlib import Path
import subprocess

from .recording import analyze_run
from .suite import DEFAULT_SCENES, TIMING_OPTIONS, run_checks, run_suite
from ..presentation_scenes import PROFILES, build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List deterministic scenes and timing models; no execution")
    check = commands.add_parser("check", help="Verify known positive/negative evaluator controls")
    check.add_argument("--out", type=Path, required=True, help="New evidence directory")
    synthetic = commands.add_parser("synthetic", help="Compare off/50%/100% in the same presentation model")
    synthetic.add_argument("--out", type=Path, required=True)
    synthetic.add_argument("--profiles", nargs="+", choices=PROFILES, default=["portrait"])
    synthetic.add_argument("--timings", nargs="+", choices=TIMING_OPTIONS, default=["clean-lag2"])
    synthetic.add_argument("--scenes", nargs="+", help="Scene IDs; default is seven representative cases")
    synthetic.add_argument("--all-scenes", action="store_true", help="All 21 cases at their full durations")
    recording = commands.add_parser("recording", help="Analyze an existing run_check folder; never records")
    recording.add_argument("run", type=Path)
    recording.add_argument("--out", type=Path, required=True, help="New directory outside the original run")
    recording.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        for scene in build(PROFILES["portrait"]):
            print(f"{scene.id:28} {scene.duration:5.1f}s  {','.join(scene.contracts)}")
        print("Profiles:", ", ".join(PROFILES))
        print("Timings:", ", ".join(TIMING_OPTIONS))
        return 0
    try:
        if args.command == "check":
            result = run_checks(args.out)
            print(f"Evaluator controls: {sum(row['ok'] for row in result['checks'])}/{len(result['checks'])}")
            return 0 if result["passed"] else 1
        if args.command == "synthetic":
            if args.all_scenes and args.scenes:
                parser.error("Choose --all-scenes or --scenes")
            result = run_suite(args.out, args.profiles, args.timings,
                               None if args.all_scenes else args.scenes or DEFAULT_SCENES)
            print(f"Measured {len(result['reports'])} runs. Contract failures are retained; no overall effectiveness pass.")
        else:
            result = analyze_run(args.run, args.out, args.metadata_only)
            for key, value in result["findings"].items():
                print(f"{key}: {value}")
        print(f"Evidence: {args.out.resolve()}")
        return 0
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
