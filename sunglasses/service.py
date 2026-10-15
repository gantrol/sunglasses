"""Public, synchronous offline operations. Hosts run these away from their UI thread."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .automatic import AutoConfig
from .auto_lab import run_auto_suite
from .filter import FilterConfig, compress_highlights
from .fixtures import AUTO_CASES, CASES, SequenceSpec
from .image_io import read_sdr_png, save_sdr_png
from .i18n import ValidationError
from .jobs import JobContext
from .lab import run_suite


@dataclass(frozen=True)
class ImageRequest:
    source: Path
    output: Path
    config: FilterConfig = FilterConfig()
    assume_srgb: bool = False


@dataclass(frozen=True)
class ExperimentRequest:
    output: Path
    mode: Literal["static", "automatic"] = "automatic"
    sequence: SequenceSpec = SequenceSpec(duration=4)
    cases: tuple[str, ...] = ()
    static: FilterConfig = FilterConfig()
    automatic: AutoConfig = AutoConfig()
    threshold: float = .5

    def __post_init__(self):
        if self.mode not in ("static", "automatic"):
            raise ValidationError("experiment_mode")
        object.__setattr__(self, "cases", tuple(self.cases))


@dataclass(frozen=True)
class JobResult:
    output: Path
    report: Path | None = None
    case_count: int = 0


def process_image(request: ImageRequest, job: JobContext = JobContext()) -> JobResult:
    job.advance(0, 3)
    image = read_sdr_png(Path(request.source), assume_srgb=request.assume_srgb)
    job.advance(1, 3)
    filtered = compress_highlights(image, request.config)
    job.advance(2, 3)
    output = Path(request.output)
    save_sdr_png(output, filtered)
    # Once the exclusive write succeeds, report completion even if cancellation arrives late.
    return JobResult(output)


def run_experiment(request: ExperimentRequest, job: JobContext = JobContext()) -> JobResult:
    job.check()
    automatic = request.mode == "automatic"
    cases = list(request.cases) or [case.id for case in (AUTO_CASES if automatic else CASES)]
    output = Path(request.output)
    runner = run_auto_suite if automatic else run_suite
    report = runner(output, cases, request.sequence,
                    request.automatic if automatic else request.static, request.threshold, job=job)
    return JobResult(output, output / "report.md", len(report["cases"]))


def experiment_preset(output: Path, preset="quick", *, strength=1.0) -> ExperimentRequest:
    """Named, immutable presets. A quick run still includes known failure controls."""
    if preset == "quick":
        return ExperimentRequest(output, sequence=SequenceSpec(96, 54, 30, 4),
                                 cases=("single_white_flash", "midtone_recovery", "slow_ramp", "moving_pattern"),
                                 automatic=AutoConfig(strength=strength))
    if preset == "full":
        return ExperimentRequest(output, automatic=AutoConfig(strength=strength))
    raise ValidationError("experiment_preset")
