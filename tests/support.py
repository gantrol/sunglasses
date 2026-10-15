from contextlib import contextmanager
from pathlib import Path
import tempfile


@contextmanager
def workspace_temp():
    root = (Path(__file__).resolve().parents[1] / "docs/records/test-temp").resolve()
    root.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="case-", dir=root)
    try:
        yield temporary.name
    finally:
        # TemporaryDirectory recursively cleans only the checked generated workspace path.
        target = Path(temporary.name).resolve()
        if target.parent != root or not target.name.startswith("case-"):
            raise RuntimeError("Refusing to clean a temporary directory outside the test workspace")
        temporary.cleanup()
