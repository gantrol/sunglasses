"""Cooperative task hooks shared by the CLI, desktop worker and offline library."""

from dataclasses import dataclass
from typing import Callable


class JobCancelled(Exception):
    """The caller requested cancellation at a safe boundary."""


@dataclass(frozen=True)
class Progress:
    completed: int
    total: int
    case_id: str = ""


@dataclass(frozen=True)
class JobContext:
    cancelled: Callable[[], bool] = lambda: False
    progress: Callable[[Progress], None] = lambda _: None

    def check(self):
        if self.cancelled():
            raise JobCancelled()

    def advance(self, completed: int, total: int, case_id=""):
        self.check()
        self.progress(Progress(completed, total, case_id))
