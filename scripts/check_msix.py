"""Check an MSIX payload without installing it or running its executable.

Checks complement MakeAppx validation; they do not certify runtime behavior or signatures.
"""

import argparse
import io
import json
import posixpath
from pathlib import Path, PurePosixPath
import re
import struct
import tomllib
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

from check_release import digest, public_source, safe_names
from package_msix import LANGUAGES, NS, package_identity
from msix_assets import asset_sizes
from icon_resources import verify_embedded_icon
from release_support import validate_build_record

ENVELOPE = {"[Content_Types].xml", "AppxBlockMap.xml", "AppxSignature.p7x"}


def verify(path):
    path = Path(path)
    if "trash" in str(path).lower() or "trash" in str(path.resolve()).lower():
        raise ValueError("Excluded archive path")
    with zipfile.ZipFile(path) as bundle:
        names = safe_names(bundle)
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Case-colliding archive paths")
        hashes = json.loads(bundle.read("SHA256.json"))
        actual = {name: digest(bundle, name) for name in names if name not in ENVELOPE | {"SHA256.json"}}
        if hashes != actual:
            raise ValueError("MSIX payload hash mismatch")
        metadata = json.loads(bundle.read("package.json"))
        build = json.loads(bundle.read("BUILD.json"))
        validate_build_record(build, allow_legacy=True)
        if (metadata.get("format") != "windows-x64-msix" or metadata.get("license") != "GPL-3.0-only"
                or metadata.get("screen_only") is not True or metadata.get("entrypoint") != "Sunglasses.exe"
                or metadata.get("languages") != list(LANGUAGES) or not isinstance(metadata.get("preview"), bool)):
            raise ValueError("Unexpected MSIX product metadata")
        manifest = ET.fromstring(bundle.read("AppxManifest.xml"))
        identity = manifest.find("Identity", NS).attrib
        if identity != metadata["package_identity"] or identity.get("ProcessorArchitecture") != "x64":
            raise ValueError("MSIX identity mismatch")
        expected_identity, expected_publisher = package_identity(argparse.Namespace(
            preview=metadata["preview"], identity_name=None if metadata["preview"] else identity["Name"],
            publisher=None if metadata["preview"] else identity["Publisher"],
            publisher_display_name=None if metadata["preview"] else metadata["publisher_display_name"],
            package_version=None if metadata["preview"] else identity["Version"]))
        if identity != expected_identity or metadata["publisher_display_name"] != expected_publisher:
            raise ValueError("Preview/Store identity mismatch")
        if manifest.find("Properties/PublisherDisplayName", NS).text != metadata["publisher_display_name"]:
            raise ValueError("Publisher display name mismatch")
        applications = manifest.findall("Applications/Application", NS)
        if (len(applications) != 1 or applications[0].get("Executable") != "Sunglasses.exe"
                or applications[0].get("EntryPoint") != "Windows.FullTrustApplication"
                or applications[0].get(f"{{{NS['uap10']}}}TrustLevel") != "mediumIL"):
            raise ValueError("Package must contain only the full-trust screen app")
        visual = applications[0].find("uap:VisualElements", NS)
        if (manifest.findtext("Properties/Logo", namespaces=NS) != "Assets\\StoreLogo.png"
                or visual.get("Square44x44Logo") != "Assets\\Square44x44Logo.png"
                or visual.get("Square150x150Logo") != "Assets\\Square150x150Logo.png"
                or visual.get("BackgroundColor") != "transparent"):
            raise ValueError("MSIX icon references or background differ")
        capabilities = list(manifest.find("Capabilities", NS))
        expected_caps = {(f"{{{NS['rescap']}}}Capability", "runFullTrust"),
                         (f"{{{NS['uap11']}}}Capability", "graphicsCaptureWithoutBorder")}
        if len(capabilities) != 2 or {(node.tag, node.get("Name")) for node in capabilities} != expected_caps:
            raise ValueError("Unexpected MSIX capabilities")
        family = manifest.find("Dependencies/TargetDeviceFamily", NS)
        if family.get("Name") != "Windows.Desktop" or family.get("MinVersion") != "10.0.22000.0":
            raise ValueError("MSIX requires Windows 11 desktop")
        if [node.get("Language") for node in manifest.findall("Resources/Resource", NS)] != list(LANGUAGES):
            raise ValueError("MSIX language declarations differ")
        if bundle.getinfo("source.zip").file_size > 64 * 1024 * 1024:
            raise ValueError("Unexpectedly large source archive")
        with zipfile.ZipFile(io.BytesIO(bundle.read("source.zip"))) as source:
            source_names = safe_names(source)
            if not all(public_source(name) for name in source_names):
                raise ValueError("MSIX source archive exceeded its public allowlist")
            sources = {name: digest(source, name) for name in source_names}
            if sources != metadata["source_sha256"]:
                raise ValueError("MSIX source archive hash mismatch")
            if not all(sources.get(name) == value for name, value in build["source_sha256"].items()):
                raise ValueError("Build does not correspond to MSIX sources")
            project = tomllib.loads(source.read("pyproject.toml").decode("utf-8"))["project"]
            source_icon = source.read("native/screen/assets/Sunglasses.ico")
            if (project["license"] != metadata["license"] or project["version"] != metadata["version"]
                    or source.read("LICENSE") != bundle.read("LICENSE")):
                raise ValueError("MSIX source/package license or version mismatch")
            docs = {name for name in source_names if len(PurePosixPath(name).parts) == 2 and name.startswith("docs/")}
            documents = docs | {"README.md", "AGENTS.md", "LICENSE"}
            pending = list(documents)
            while pending:
                name = pending.pop()
                if not name.endswith(".md"):
                    continue
                for target in re.findall(r"\]\(([^)]+)\)", source.read(name).decode("utf-8")):
                    if target.startswith(("http://", "https://", "#")):
                        continue
                    linked = posixpath.normpath(posixpath.join(posixpath.dirname(name), target.split("#", 1)[0]))
                    if linked not in source_names:
                        raise ValueError("MSIX documentation link is outside the public source archive")
                    if linked not in documents:
                        documents.add(linked)
                        pending.append(linked)
            for name in documents:
                if source.read(name) != bundle.read(name):
                    raise ValueError("MSIX documentation differs from corresponding sources")
            for language in LANGUAGES:
                locale = json.loads(source.read(f"sunglasses/locales/{language}.json"))
                resources = ET.fromstring(bundle.read(f"Strings/{language}/Resources.resw"))
                expected = {"DisplayName": locale["msix.preview_name" if metadata["preview"] else "desktop.widget_title"],
                            "Description": locale["msix.description"]}
                if {node.get("name"): node.findtext("value") for node in resources} != expected:
                    raise ValueError("MSIX localized strings differ from the shared language resources")
        allowed = documents | ENVELOPE | {"AppxManifest.xml", "resources.pri", "Sunglasses.exe", "LICENSE", "README.md",
                                    "AGENTS.md", "BUILD.json", "package.json", "SHA256.json", "source.zip"}
        allowed |= {f"Strings/{language}/Resources.resw" for language in LANGUAGES}
        logos = asset_sizes()
        allowed |= {f"Assets/{name}" for name in logos}
        if set(names) - allowed:
            raise ValueError("MSIX exceeded its screen-only payload allowlist")
        if actual["Sunglasses.exe"] != build["artifacts"]["Sunglasses.exe"]:
            raise ValueError("MSIX binary differs from its build record")
        if bundle.getinfo("resources.pri").file_size == 0:
            raise ValueError("Empty MSIX resource index")
        if not {f"Assets/{name}" for name in logos} <= set(names):
            raise ValueError("Missing MSIX scale, target-size or theme icon")
        for name, size in logos.items():
            payload = bundle.read(f"Assets/{name}")
            header = payload[:24]
            if header[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", header[16:24]) != (size, size):
                raise ValueError("Invalid MSIX icon size")
            with Image.open(io.BytesIO(payload)) as icon:
                if (icon.mode != "RGBA" or icon.getchannel("A").getextrema() != (0, 255)
                        or any(icon.getpixel(point)[3] != 0 for point in ((0, 0), (size-1, 0), (0, size-1), (size-1, size-1)))):
                    raise ValueError("MSIX icon must contain artwork on a transparent background")
        verify_embedded_icon(bundle.read("Sunglasses.exe"), source_icon)
    print(f"Verified {path.name}: MSIX identity, capabilities, languages, sources, binary, file scope and hashes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    for archive in parser.parse_args().archives:
        verify(archive)
