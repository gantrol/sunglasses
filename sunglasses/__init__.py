"""Offline reference implementation; not a real-time protection layer."""

from .filter import FilterConfig, compress_highlights
from .automatic import AutoConfig, AutoFilter
from .jobs import JobCancelled, JobContext, Progress
from .service import ExperimentRequest, ImageRequest, JobResult, experiment_preset, process_image, run_experiment

__all__ = ["FilterConfig", "compress_highlights", "AutoConfig", "AutoFilter", "JobCancelled", "JobContext", "Progress",
           "ExperimentRequest", "ImageRequest", "JobResult", "experiment_preset", "process_image", "run_experiment"]
