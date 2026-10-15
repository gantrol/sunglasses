"""Offline packaging checks; no installation, signing, window, or capture."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_msix import ENVELOPE, verify
from package_msix import NS, package_identity, write_manifest, write_strings
from tests.support import workspace_temp


class MsixConfigurationTests(unittest.TestCase):
    def arguments(self, **values):
        defaults = dict(preview=False, identity_name=None, publisher=None, publisher_display_name=None, package_version=None)
        return argparse.Namespace(**(defaults | values))

    def test_preview_is_explicit_and_cannot_mix_with_store_identity(self):
        with self.assertRaises(ValueError):
            package_identity(self.arguments())
        identity, _ = package_identity(self.arguments(preview=True))
        self.assertEqual(identity["Name"], "Sunglasses.LocalPreview")
        self.assertEqual(identity["Version"], "0.1.0.0")
        with self.assertRaises(ValueError):
            package_identity(self.arguments(preview=True, identity_name="Example.Sunglasses"))

    def test_store_identity_and_version_reject_invalid_inputs(self):
        valid = dict(identity_name="Example.Sunglasses", publisher="CN=12345678", publisher_display_name="Example", package_version="1.0.0.0")
        package_identity(self.arguments(**valid))
        for change in ({"identity_name": "../Name"}, {"identity_name": "Sunglasses.LocalPreview"},
                       {"publisher": ""}, {"publisher": "CN=Sunglasses Local Preview"}, {"publisher_display_name": " "},
                       *({"package_version": version} for version in ("0.1.0.0", "1.2.3.1", "1.0.0", "1.0.0.dev0", "65536.0.0.0", "01.0.0.0"))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                package_identity(self.arguments(**(valid | change)))

    def test_publisher_is_xml_escaped_and_both_languages_are_generated(self):
        identity, publisher = package_identity(self.arguments(identity_name="Example.Sunglasses", publisher="CN=Example & Co",
                                                             publisher_display_name="Example & Co", package_version="1.0.0.0"))
        with workspace_temp() as folder:
            stage = Path(folder)
            write_manifest(stage, identity, publisher)
            root = ET.parse(stage / "AppxManifest.xml").getroot()
            self.assertEqual(root.find("Identity", NS).get("Publisher"), "CN=Example & Co")
            write_strings(stage, True)
            for language, expected in (("en", "Sunglasses (Preview)"), ("zh-CN", "墨镜（预览）")):
                root = ET.parse(stage / "Strings" / language / "Resources.resw").getroot()
                self.assertEqual(root.find("data[@name='DisplayName']/value").text, expected)


@unittest.skipUnless(os.environ.get("SUNGLASSES_MSIX"), "MSIX package not configured")
class MsixArchiveTests(unittest.TestCase):
    def setUp(self):
        self.package = Path(os.environ["SUNGLASSES_MSIX"])
        with zipfile.ZipFile(self.package) as archive:
            self.files = {name: archive.read(name) for name in archive.namelist()}

    def check_mutation(self, mutate, message, refresh_hashes=False):
        mutate(self.files)
        if refresh_hashes:
            self.files["SHA256.json"] = json.dumps({name: hashlib.sha256(data).hexdigest() for name, data in self.files.items()
                                                   if name not in ENVELOPE | {"SHA256.json"}}).encode()
        with workspace_temp() as folder:
            path = Path(folder) / "changed.msix"
            with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as archive:
                for name, data in self.files.items():
                    archive.writestr(name, data)
            with self.assertRaisesRegex(ValueError, message):
                verify(path)

    def test_generated_package_passes(self):
        verify(self.package)

    def test_changed_executable_is_rejected_even_with_updated_payload_hash(self):
        self.check_mutation(lambda files: files.update({"Sunglasses.exe": b"replaced"}), "binary differs", True)

    def test_private_files_are_rejected(self):
        self.check_mutation(lambda files: files.update({"docs/records/private.txt": b"private"}), "Unexpected archive path", True)

    def test_extra_executable_is_rejected_even_with_updated_hash(self):
        self.check_mutation(lambda files: files.update({"Extra.exe": b"MZ"}), "payload allowlist", True)

    def test_unexpected_capture_capability_is_rejected(self):
        self.check_mutation(lambda files: files.update({"AppxManifest.xml": files["AppxManifest.xml"].replace(
            b"graphicsCaptureWithoutBorder", b"graphicsCaptureProgrammatic")}), "capabilities", True)

    def test_corrupted_resources_fail_payload_integrity(self):
        self.check_mutation(lambda files: files.update({"resources.pri": b"invalid"}), "hash mismatch")

    def test_missing_small_dark_theme_icon_is_rejected(self):
        self.check_mutation(lambda files: files.pop("Assets/Square44x44Logo.targetsize-16_altform-unplated.png"),
                            "Missing MSIX", True)

    def test_opaque_icon_is_rejected_even_at_the_correct_size(self):
        stream = io.BytesIO()
        Image.new("RGBA", (16, 16), (255, 255, 255, 255)).save(stream, format="PNG")
        self.check_mutation(lambda files: files.update({"Assets/Square44x44Logo.targetsize-16.png": stream.getvalue()}),
                            "transparent background", True)

    def test_source_and_build_must_correspond(self):
        def mutate(files):
            build = json.loads(files["BUILD.json"])
            build["source_sha256"]["native/screen/capture.cpp"] = "0" * 64
            files["BUILD.json"] = json.dumps(build).encode()
        self.check_mutation(mutate, "does not correspond", True)


if __name__ == "__main__":
    unittest.main()
