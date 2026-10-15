"""Validate release archives without extracting files or starting executables."""

import argparse
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tomllib
import zipfile

from release_support import README_FILES, ROOT, ROOT_FILES, SOURCE_TREES, validate_build_record


def public_source(name):
    path = PurePosixPath(name)
    if name in ROOT_FILES:
        return True
    if len(path.parts) == 2 and path.parts[0] == "docs" and path.suffix == ".md":
        return True
    return any(path.is_relative_to(folder) and path.suffix in extensions
               for folder, extensions in SOURCE_TREES.items())


def safe_names(archive):
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive entries")
    for name in names:
        path = PurePosixPath(name)
        if ("trash" in name.lower() or "\\" in name or path.is_absolute() or ".." in path.parts
                or "records" in path.parts or "__pycache__" in path.parts):
            raise ValueError("Unexpected archive path")
    return names


def digest(archive, name):
    with archive.open(name) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path):
    if "trash" in str(path).lower():
        raise ValueError("Excluded archive path")
    if path.suffix.lower() == ".msix":
        from check_msix import verify as verify_msix
        verify_msix(path)
        return
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = safe_names(archive)
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise ValueError("Expected one wheel metadata record")
            metadata_name = metadata_names[0]
            metadata_root = metadata_name.rsplit("/", 1)[0] + "/"
            if any(not name.startswith(("sunglasses/", metadata_root)) for name in names):
                raise ValueError("Wheel exceeded its package allowlist")
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
            license_name = metadata_name.rsplit("/", 1)[0] + "/licenses/LICENSE"
            if (metadata["License-Expression"] != "GPL-3.0-only"
                    or archive.read(license_name) != (ROOT / "LICENSE").read_bytes()):
                raise ValueError("Wheel license metadata/text mismatch")
        print(f"Verified {path.name}: GPLv3 metadata, license text and file scope")
        return
    with zipfile.ZipFile(path) as bundle:
        names = safe_names(bundle)
        if not all(name.startswith("Sunglasses/") for name in names):
            raise ValueError("Unexpected archive root")
        prefix = "Sunglasses/"
        def read(name):
            return bundle.read(prefix + name)
        hashes = json.loads(read("SHA256.json"))
        actual = {name.removeprefix(prefix): digest(bundle, name) for name in names
                  if name != prefix + "SHA256.json"}
        if hashes != actual:
            raise ValueError("Package hash mismatch")
        manifest = json.loads(read("package.json"))
        build = json.loads(read("BUILD.json"))
        validate_build_record(build, allow_legacy=True)
        if manifest["license"] != "GPL-3.0-only":
            raise ValueError("Missing GPLv3 declaration")
        if bundle.getinfo(prefix + "source.zip").file_size > 64 * 1024 * 1024:
            raise ValueError("Unexpectedly large source archive")
        with zipfile.ZipFile(io.BytesIO(read("source.zip"))) as source:
            source_names = safe_names(source)
            if not all(public_source(name) for name in source_names):
                raise ValueError("Source archive exceeded its public allowlist")
            sources = {name: digest(source, name) for name in source_names}
            if sources != manifest["source_sha256"]:
                raise ValueError("Source archive hash mismatch")
            if not all(sources.get(name) == value for name, value in build["source_sha256"].items()):
                raise ValueError("Build does not correspond to the enclosed sources")
            project = tomllib.loads(source.read("pyproject.toml").decode("utf-8"))["project"]
            if (project["license"] != manifest["license"] or project["version"] != manifest["version"]
                    or source.read("LICENSE") != read("LICENSE")):
                raise ValueError("Source/package metadata mismatch")
        expected = ("Sunglasses.exe",) if manifest["screen_only"] else ("Sunglasses.exe", "SunglassesLab.exe")
        for name in expected:
            if hashes[name] != build["artifacts"][name]:
                raise ValueError("Binary does not match its build record")
        runtime = any(name.startswith(prefix + "runtime/") for name in names)
        if runtime == manifest["screen_only"]:
            raise ValueError("Incorrect runtime in package variant")
        if not manifest["screen_only"]:
            if json.loads(read("licenses/dependencies.json")) != manifest["dependencies"]:
                raise ValueError("Dependency inventory mismatch")
            for name in manifest["dependencies"]:
                if not any(n.startswith(prefix + "licenses/" + name + "/") for n in names):
                    raise ValueError(f"Missing dependency notices: {name}")
            read("licenses/Python-LICENSE.txt")
        for name in names:
            relative = name.removeprefix(prefix)
            if relative not in README_FILES and not (relative.startswith("docs/") and relative.endswith(".md")):
                continue
            for link in re.findall(r"\]\(([^)]+)\)", bundle.read(name).decode("utf-8")):
                if link.startswith(("http://", "https://", "#")):
                    continue
                # Normalize '..' from ordinary documentation links without extraction.
                parts = list(PurePosixPath(name).parent.parts)
                for part in PurePosixPath(link.split("#", 1)[0]).parts:
                    if part == "..":
                        parts.pop()
                    elif part != ".":
                        parts.append(part)
                if "/".join(parts) not in names:
                    raise ValueError(f"Broken packaged link: {relative}: {link}")
    print(f"Verified {path.name}: GPLv3, sources, build correspondence, notices, hashes and documentation links")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    for archive in parser.parse_args().archives:
        verify(archive)
