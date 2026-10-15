"""Collect notices for distributions actually included by PyInstaller Analysis."""

import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import sys

from release_support import ROOT, write_json


def collect_licenses(entries, destination):
    owners = {}
    distributions = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        distributions[name] = distribution
        for entry in distribution.files or ():
            if "trash" in str(entry).lower():
                continue
            path = Path(distribution.locate_file(entry)).resolve()
            if "trash" not in str(path).lower():
                owners[os.path.normcase(str(path))] = name
    used = set()
    for _, source, _ in entries:
        if not source:
            continue
        if "trash" in str(source).lower():
            raise ValueError("Excluded frozen input")
        path = Path(source).resolve()
        name = owners.get(os.path.normcase(str(path)))
        if path.is_relative_to(ROOT / "sunglasses") or path == ROOT / "scripts/worker_entry.py":
            continue
        elif name:
            used.add(name)
        elif path.is_relative_to(Path(sys.base_prefix).resolve()) and "site-packages" not in path.parts:
            continue  # Python stdlib/runtime notices are copied separately below.
        elif path.is_relative_to(Path(os.environ["SystemRoot"]).resolve()):
            continue  # Windows system libraries.
        elif path.is_relative_to(destination.parent.resolve()):
            continue  # Generated provenance data and PyInstaller intermediates.
        else:
            raise ValueError(f"Frozen dependency has no distribution metadata: {path.name}")
    versions = {}
    for name in sorted(used):
        distribution = distributions[name]
        copied = 0
        declared = set(distribution.metadata.get_all("License-File", []))
        for entry in distribution.files or ():
            path = Path(str(entry))
            if "trash" in str(path).lower() or path.is_absolute() or ".." in path.parts:
                continue
            metadata_index = next((i for i, p in enumerate(path.parts) if p.endswith(".dist-info")), None)
            if metadata_index is None:
                continue
            # Preserve subdirectories, including notices with names other than LICENSE.
            relative = Path(*path.parts[metadata_index + 1:])
            if (not any(word in str(relative).lower() for word in ("license", "copying", "notice"))
                    and relative.as_posix() not in declared):
                continue
            source = Path(distribution.locate_file(entry)).resolve()
            if "trash" in str(source).lower() or not source.is_file():
                continue
            target = destination / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
        if not copied:
            raise ValueError(f"Missing license text for frozen dependency: {name}")
        versions[name] = distribution.version
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.is_file():
        raise ValueError("Missing Python runtime license")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(python_license, destination / "Python-LICENSE.txt")
    write_json(destination / "dependencies.json", versions)
