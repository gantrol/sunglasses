"""Source provenance for both source installs and frozen worker bundles."""

import hashlib
import json
from importlib.resources import files


def source_hashes(*names: str) -> dict[str, str]:
    package = files("sunglasses")
    manifest = None
    result = {}
    for name in names:
        source = package.joinpath(name)
        if source.is_file():
            result[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        else:
            if manifest is None:
                manifest = json.loads(package.joinpath("source_hashes.json").read_text(encoding="utf-8"))
            result[name] = manifest[name]
    return result
