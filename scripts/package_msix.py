"""Build an unsigned, screen-only MSIX. Never install, sign, capture, or publish.

All synchronous file I/O and SDK processes run in this explicit offline build command.
Use --preview before obtaining an application identity from Partner Center.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
import uuid
import xml.etree.ElementTree as ET
import zipfile

from release_support import ROOT, checked, check_links, documentation_files, sha256, source_files, source_hashes, verify_native, write_json

NS = {
    "": "http://schemas.microsoft.com/appx/manifest/foundation/windows10",
    "uap": "http://schemas.microsoft.com/appx/manifest/uap/windows10",
    "uap10": "http://schemas.microsoft.com/appx/manifest/uap/windows10/10",
    "uap11": "http://schemas.microsoft.com/appx/manifest/uap/windows10/11",
    "rescap": "http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities",
}
PREVIEW_NAME = "Sunglasses.LocalPreview"
PREVIEW_PUBLISHER = "CN=Sunglasses Local Preview"
LANGUAGES = ("en", "zh-CN")


def package_identity(args):
    if args.preview:
        if any((args.identity_name, args.publisher, args.publisher_display_name, args.package_version)):
            raise ValueError("--preview cannot be combined with Store identity/version arguments")
        return {"Name": PREVIEW_NAME, "Publisher": PREVIEW_PUBLISHER, "Version": "0.1.0.0",
                "ProcessorArchitecture": "x64"}, "Sunglasses Local Preview"
    if not all((args.identity_name, args.publisher, args.publisher_display_name, args.package_version)):
        raise ValueError("Use --preview, or supply --identity-name, --publisher, --publisher-display-name and --package-version")
    if not re.fullmatch(r"[A-Za-z0-9.-]{3,50}", args.identity_name):
        raise ValueError("Identity name must contain 3..50 ASCII letters, digits, dots or hyphens")
    if args.identity_name.casefold() == PREVIEW_NAME.casefold() or args.publisher.casefold() == PREVIEW_PUBLISHER.casefold():
        raise ValueError("Local preview identity cannot be used for a Store candidate")
    if not args.publisher.startswith("CN=") or len(args.publisher) <= 3:
        raise ValueError("Publisher must be copied exactly from Partner Center (CN=...)")
    if not args.publisher_display_name.strip():
        raise ValueError("Publisher display name cannot be blank")
    if not re.fullmatch(r"(?:0|[1-9]\d{0,4})(?:\.(?:0|[1-9]\d{0,4})){3}", args.package_version):
        raise ValueError("MSIX version must have four decimal components")
    version = tuple(map(int, args.package_version.split(".")))
    if any(part > 65535 for part in version) or version[0] == 0 or version[-1] != 0:
        raise ValueError("Store MSIX version components must be 0..65535, major must be positive and revision must be 0")
    return {"Name": args.identity_name, "Publisher": args.publisher, "Version": args.package_version,
            "ProcessorArchitecture": "x64"}, args.publisher_display_name


def sdk_tools(explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    else:
        base = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Windows Kits/10/bin"
        if "trash" in str(base).lower():
            raise ValueError("Excluded SDK path")
        candidates = sorted((p / "x64" for p in base.glob("10.*") if re.fullmatch(r"\d+(?:\.\d+){3}", p.name)),
                            key=lambda p: tuple(map(int, p.parent.name.split("."))), reverse=True)
    for candidate in candidates:
        if "trash" in str(candidate).lower():
            raise ValueError("Excluded SDK path")
        candidate = candidate.resolve()
        if "trash" in str(candidate).lower():
            raise ValueError("Excluded SDK path")
        paths = {name: candidate / f"{name}.exe" for name in ("makeappx", "makepri")}
        if all(path.is_file() for path in paths.values()):
            return paths
    raise ValueError("Windows SDK MakeAppx/MakePri not found; use --sdk-bin with an SDK x64 directory")


def write_xml(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_manifest(stage, identity, publisher_display_name):
    for prefix, uri in NS.items():
        ET.register_namespace(prefix, uri)
    root = ET.parse(ROOT / "native/screen/msix/AppxManifest.xml.in").getroot()
    root.find("Identity", NS).attrib = identity
    root.find("Properties/PublisherDisplayName", NS).text = publisher_display_name
    write_xml(root, stage / "AppxManifest.xml")


def write_strings(stage, preview):
    for language in LANGUAGES:
        locale = json.loads((ROOT / "sunglasses/locales" / f"{language}.json").read_text(encoding="utf-8"))
        resources = ET.Element("root")
        for name, key in {"DisplayName": "msix.preview_name" if preview else "desktop.widget_title",
                          "Description": "msix.description"}.items():
            data = ET.SubElement(resources, "data", {"name": name, "{http://www.w3.org/XML/1998/namespace}space": "preserve"})
            ET.SubElement(data, "value").text = locale[key]
        folder = stage / "Strings" / language
        folder.mkdir(parents=True)
        write_xml(resources, folder / "Resources.resw")


def run_sdk(executable, arguments, log):
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(executable), *map(str, arguments)], stdout=stream, stderr=subprocess.STDOUT,
                                cwd=log.parent, timeout=180, check=False)
    if result.returncode:
        raise ValueError(f"{executable.name} failed ({result.returncode}); see {log}")


def verify_pri_dump(path, stage, identity_name):
    root = ET.parse(path).getroot()
    resource_map = root.find("ResourceMap")
    if resource_map is None or resource_map.get("name") != identity_name:
        raise ValueError("PRI resource map does not match the package identity")
    subtree = resource_map.find("ResourceMapSubtree[@name='Resources']")
    if subtree is None:
        raise ValueError("PRI is missing localized resources")
    actual = {node.get("name"): {candidate.get("qualifiers"): candidate.findtext("Value")
                                for candidate in node.findall("Candidate")} for node in subtree}
    expected = {}
    for language in LANGUAGES:
        for node in ET.parse(stage / "Strings" / language / "Resources.resw").getroot():
            expected.setdefault(node.get("name"), {})[f"Language-{language}"] = node.findtext("value")
    if actual != expected:
        raise ValueError("PRI is missing or has changed localized string values")
    from msix_assets import asset_sizes
    paths = {node.findtext("Value") for node in root.iter("Candidate") if node.get("type") == "Path"}
    if not {"Assets\\" + name for name in asset_sizes()} <= paths:
        raise ValueError("PRI is missing scaled or target-size application icons")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Use a distinct local-only identity and fixed 0.1.0.0 version")
    parser.add_argument("--identity-name")
    parser.add_argument("--publisher")
    parser.add_argument("--publisher-display-name")
    parser.add_argument("--package-version", help="Explicit four-part Store version; revision must be 0")
    parser.add_argument("--native-build", type=Path, default=ROOT / "build/native")
    parser.add_argument("--sdk-bin", type=Path)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    identity, publisher = package_identity(args)
    sdk = sdk_tools(args.sdk_bin)
    native_build = checked(args.native_build, ROOT / "build")
    native = verify_native(native_build)
    result_path = checked(args.result_json) if args.result_json else None
    if result_path and (result_path.exists() or not result_path.parent.is_dir()):
        raise ValueError("Result JSON must be a new file in an existing workspace directory")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project["license"] != "GPL-3.0-only":
        raise ValueError("Unexpected project license")
    before = source_hashes()
    check_links(ROOT)
    from msix_assets import generate_assets
    from check_msix import verify

    token = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f") + "-" + uuid.uuid4().hex[:8]
    work = checked(ROOT / "build/msix" / token)
    stage = work / "package"
    stage.mkdir(parents=True, exist_ok=False)
    # Index only the manifest, icons and strings, before adding source/docs to the package.
    write_manifest(stage, identity, publisher)
    write_strings(stage, args.preview)
    generate_assets(stage / "Assets")
    config = work / "priconfig.xml"
    run_sdk(sdk["makepri"], ["createconfig", "/cf", config, "/dq", "en", "/pv", "10.0.0"], work / "pri-config.log")
    config_tree = ET.parse(config)
    packaging = config_tree.getroot().find("packaging")
    if packaging is not None:
        config_tree.getroot().remove(packaging)  # Keep both languages in a single self-contained package.
    write_xml(config_tree.getroot(), config)
    run_sdk(sdk["makepri"], ["new", "/pr", stage, "/cf", config, "/mn", stage / "AppxManifest.xml",
                            "/of", stage / "resources.pri"], work / "pri-new.log")
    run_sdk(sdk["makepri"], ["dump", "/if", stage / "resources.pri", "/of", work / "resources.xml"], work / "pri-dump.log")
    verify_pri_dump(work / "resources.xml", stage, identity["Name"])
    shutil.copy2(native_build / "Sunglasses.exe", stage / "Sunglasses.exe")
    (stage / "docs").mkdir()
    for path in documentation_files():
        relative = path.relative_to(ROOT)
        (stage / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, stage / relative)
    check_links(stage)
    with zipfile.ZipFile(stage / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(stage / "source.zip") as archive:
        archived = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    if archived != before or source_hashes() != before or verify_native(native_build) != native:
        raise ValueError("Inputs changed while staging the MSIX; rebuild")
    write_json(stage / "BUILD.json", native)
    write_json(stage / "package.json", {
        "schema": 1, "format": "windows-x64-msix", "version": project["version"],
        "package_identity": identity, "publisher_display_name": publisher, "preview": args.preview,
        "license": project["license"], "screen_only": True, "entrypoint": "Sunglasses.exe",
        "source_archive": "source.zip", "source_sha256": before, "languages": list(LANGUAGES),
        "validation": "Package structure only; installation, WGC consent, UI, composition and games not verified",
    })
    write_json(stage / "SHA256.json", {p.relative_to(stage).as_posix(): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()})
    candidate = work / "Sunglasses.msix"
    run_sdk(sdk["makeappx"], ["pack", "/d", stage, "/p", candidate], work / "makeappx.log")
    verify(candidate)
    if source_hashes() != before or verify_native(native_build) != native:
        raise ValueError("Inputs changed while creating the MSIX; rebuild")
    variant = "preview" if args.preview else "store-candidate"
    output = checked(ROOT / "dist" / f"Sunglasses-{variant}-{identity['Version']}-{token}.msix")
    output.parent.mkdir(parents=True, exist_ok=True)
    with candidate.open("rb") as source, output.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    result = {"archive": str(output), "folder": str(stage), "work": str(work), "preview": args.preview,
              "signed": False, "package_identity": identity, "sha256": sha256(output)}
    if result_path:
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
