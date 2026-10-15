"""Headless development/build provenance checks; never compile or start an app."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import release_support
import build_native
import dev
from check_release import public_source
from tests.support import workspace_temp


def build_record(configuration="Release", mode="clean"):
    return {
        "schema": 2,
        "configuration": configuration,
        "generator": "Ninja",
        "build_mode": mode,
        "toolchain": {"architecture": "x64"},
        "source_sha256": {"native/example.cpp": "1" * 64},
        "artifacts": {},
    }


class BuildRecordTests(unittest.TestCase):
    def test_development_accepts_all_configurations_and_build_modes(self):
        for configuration in ("Debug", "RelWithDebInfo", "Release"):
            for mode in ("clean", "incremental"):
                with self.subTest(configuration=configuration, mode=mode):
                    release_support.validate_build_record(build_record(configuration, mode), for_package=False)

    def test_packaging_requires_clean_release(self):
        release_support.validate_build_record(build_record())
        for configuration, mode in (("Debug", "clean"), ("RelWithDebInfo", "clean"),
                                    ("Release", "incremental")):
            with self.subTest(configuration=configuration, mode=mode), self.assertRaisesRegex(ValueError, "clean Release"):
                release_support.validate_build_record(build_record(configuration, mode))

    def test_malformed_records_cannot_pass_development_validation(self):
        for change in ({"schema": 3}, {"configuration": "MinSizeRel"}, {"configuration": None},
                       {"generator": "Visual Studio"}, {"build_mode": None}, {"build_mode": "unknown"},
                       {"toolchain": {}}, {"toolchain": {"architecture": "x86"}}, {"toolchain": None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                release_support.validate_build_record(build_record() | change, for_package=False)
        for value in (None, [], "Release"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                release_support.validate_build_record(value, for_package=False)

    def test_legacy_records_are_only_accepted_for_explicit_archive_inspection(self):
        legacy = build_record() | {"schema": 1}
        legacy.pop("build_mode")
        for for_package in (False, True):
            with self.subTest(for_package=for_package), self.assertRaises(ValueError):
                release_support.validate_build_record(legacy, for_package=for_package)
        release_support.validate_build_record(legacy, allow_legacy=True)
        with self.assertRaises(ValueError):
            release_support.validate_build_record(legacy | {"configuration": "Debug"}, allow_legacy=True)


class NativeVerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(workspace_temp())
        self.build = Path(self.directory)
        self.record = build_record()
        for name in release_support.EXECUTABLES:
            binary = self.build / name
            binary.write_bytes(b"headless fixture: " + name.encode("ascii"))
            self.record["artifacts"][name] = release_support.sha256(binary)
        self.save()
        self.inputs = self.enterContext(patch.object(release_support, "native_inputs",
                                                    return_value=dict(self.record["source_sha256"])))

    def save(self):
        release_support.write_json(self.build / "native-build.json", self.record)

    def test_complete_matching_release_record_verifies(self):
        self.assertEqual(release_support.verify_native(self.build), self.record)

    def test_development_record_can_run_but_cannot_be_packaged(self):
        self.record.update(configuration="Debug", build_mode="incremental")
        self.save()
        self.assertEqual(release_support.verify_native(self.build, for_package=False), self.record)
        with self.assertRaisesRegex(ValueError, "clean Release"):
            release_support.verify_native(self.build)

    def test_changed_native_inputs_invalidate_an_existing_build(self):
        self.inputs.return_value = {"native/example.cpp": "2" * 64}
        with self.assertRaisesRegex(ValueError, "sources changed"):
            release_support.verify_native(self.build)

    def test_changed_binary_is_rejected(self):
        (self.build / "Sunglasses.exe").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "binary changed"):
            release_support.verify_native(self.build)

    def test_missing_binary_is_rejected(self):
        (self.build / "Sunglasses.exe").unlink()
        with self.assertRaises((ValueError, OSError)):
            release_support.verify_native(self.build)

    def test_incomplete_or_extra_artifact_record_is_rejected(self):
        original = dict(self.record["artifacts"])
        for artifacts in ({key: value for key, value in original.items() if key != "Sunglasses.exe"},
                          original | {"Unexpected.exe": "3" * 64}):
            with self.subTest(artifacts=tuple(artifacts)):
                self.record["artifacts"] = artifacts
                self.save()
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    release_support.verify_native(self.build)


class PublicBuildSourceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(workspace_temp()))
        for name in release_support.ROOT_FILES:
            (self.root / name).write_text("public fixture\n", encoding="utf-8")
        for name in ("scripts/dev.py", "scripts/build_native.py", "native/example.cpp",
                     "docs/USAGE.md", "CMakeUserPresets.json", ".vscode/launch.json",
                     "docs/records/private.md", "build/local.json"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")

    def test_public_entrypoints_are_archived_but_local_configuration_is_excluded(self):
        sources = release_support.source_hashes(self.root)
        for name in ("CMakePresets.json", "dev.ps1", "scripts/dev.py"):
            with self.subTest(name=name):
                self.assertIn(name, sources)
                self.assertTrue(public_source(name))
        for name in ("CMakeUserPresets.json", ".vscode/launch.json", "docs/records/private.md", "build/local.json"):
            with self.subTest(name=name):
                self.assertNotIn(name, sources)
                self.assertFalse(public_source(name))

    def test_preset_and_launcher_changes_invalidate_native_provenance(self):
        for name in ("CMakePresets.json", "dev.ps1", "scripts/dev.py"):
            with self.subTest(name=name):
                before = release_support.native_inputs(self.root)
                path = self.root / name
                path.write_text(path.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
                after = release_support.native_inputs(self.root)
                self.assertNotEqual(before[name], after[name])


class NativeBuildFailureTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(workspace_temp())).resolve()
        self.build = self.root / "build/debug"
        self.build.mkdir(parents=True)
        self.record = self.build / "native-build.json"
        release_support.write_json(self.record, build_record())
        self.enterContext(patch.object(build_native, "ROOT", self.root))
        self.inputs = self.enterContext(patch.object(build_native, "native_inputs", return_value={"native.cpp": "original"}))
        self.arguments = ["--config", "Debug", "--build-dir", str(self.build), "--incremental"]

    def test_failed_configure_removes_previous_success_record(self):
        with patch.object(build_native.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "cmake")):
            with self.assertRaises(subprocess.CalledProcessError):
                build_native.main(self.arguments)
        self.assertFalse(self.record.exists())

    def test_wrong_cache_configuration_is_rejected_before_compilation(self):
        with patch.object(build_native.subprocess, "run") as run, patch.object(build_native, "read_cache", return_value={
                "CMAKE_BUILD_TYPE": "Release", "CMAKE_GENERATOR": "Ninja",
                "SUNGLASSES_BUILD_DESKTOP": "ON", "SUNGLASSES_BUILD_HEADLESS_CHECKS": "ON",
                "CMAKE_HOME_DIRECTORY": str(self.root)}):
            with self.assertRaisesRegex(ValueError, "requested configuration"):
                build_native.main(self.arguments)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(self.record.exists())

    def test_changed_inputs_during_build_do_not_leave_success_record(self):
        self.inputs.side_effect = [{"native.cpp": "before"}, {"native.cpp": "after"}]
        with patch.object(build_native.subprocess, "run") as run, patch.object(build_native, "read_cache", return_value={
                "CMAKE_BUILD_TYPE": "Debug", "CMAKE_GENERATOR": "Ninja",
                "SUNGLASSES_BUILD_DESKTOP": "ON", "SUNGLASSES_BUILD_HEADLESS_CHECKS": "ON",
                "CMAKE_HOME_DIRECTORY": str(self.root)}):
            with self.assertRaisesRegex(ValueError, "changed during compilation"):
                build_native.main(self.arguments)
        self.assertEqual(run.call_count, 2)
        self.assertFalse(self.record.exists())


class DevelopmentCommandTests(unittest.TestCase):
    def setUp(self):
        self.output = self.enterContext(redirect_stdout(io.StringIO()))

    def test_launcher_and_cmake_presets_use_separate_matching_configuration_directories(self):
        presets = json.loads((release_support.ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        configurations = {preset["name"]: preset for preset in presets["configurePresets"]}
        directories = []
        for configuration in ("Debug", "RelWithDebInfo", "Release"):
            with self.subTest(configuration=configuration):
                name = configuration.lower()
                preset = configurations[name]
                parent = configurations[preset["inherits"]]
                directory = preset.get("binaryDir", parent["binaryDir"])
                directory = Path(directory.replace("${sourceDir}", str(release_support.ROOT))
                                 .replace("${presetName}", name)).resolve()
                directories.append(directory)
                self.assertEqual(dev.build_directory(None, configuration), directory)
                self.assertEqual(preset["cacheVariables"]["CMAKE_BUILD_TYPE"], configuration)
                self.assertEqual(preset["cacheVariables"]["CMAKE_TRY_COMPILE_CONFIGURATION"], configuration)
                self.assertTrue(any(item["name"] == name and item["configurePreset"] == name
                                    for item in presets["buildPresets"]))
        self.assertEqual(len(set(directories)), 3)

    def test_dry_run_neither_starts_processes_nor_requires_built_artifacts(self):
        with patch.object(dev.subprocess, "run") as run, patch.object(dev.subprocess, "check_output") as output, \
                patch.object(dev, "verify_native") as verify, patch.object(dev.shutil, "which", return_value=None):
            for command in ("build", "run", "debug", "package"):
                with self.subTest(command=command):
                    self.assertEqual(dev.main([command, "--dry-run"]), 0)
            run.assert_not_called()
            output.assert_not_called()
            verify.assert_not_called()

    def test_no_build_rejects_different_configuration_without_launching(self):
        with patch.object(dev, "verify_native", return_value=build_record("RelWithDebInfo")) as verify, \
                patch.object(dev.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "Expected Debug"):
                dev.main(["run", "--config", "Debug", "--no-build"])
            verify.assert_called_once_with(dev.build_directory(None, "Debug"), for_package=False)
            run.assert_not_called()

    def test_invalid_existing_binaries_cannot_launch(self):
        with patch.object(dev, "verify_native", side_effect=ValueError("binary changed")), \
                patch.object(dev.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "binary changed"):
                dev.main(["run", "--no-build"])
            run.assert_not_called()

    def test_run_preserves_app_argument_boundaries_and_child_exit_status(self):
        with patch.object(dev, "verify_native", return_value=build_record("RelWithDebInfo")), \
                patch.object(dev.subprocess, "run", return_value=SimpleNamespace(returncode=17)) as run:
            self.assertEqual(dev.main(["run", "--no-build", "--", "--example", "two words"]), 17)
        self.assertEqual(run.call_args.args[0], [str(dev.build_directory(None, "RelWithDebInfo") / "Sunglasses.exe"),
                                               "--example", "two words"])

    def test_package_builds_clean_release_then_checks_before_packaging(self):
        with patch.object(dev, "verify_native", return_value=build_record()) as verify, \
                patch.object(dev.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(dev.main(["package"]), 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 2)
        self.assertIn("--config", commands[0])
        self.assertEqual(commands[0][commands[0].index("--config") + 1], "Release")
        self.assertNotIn("--incremental", commands[0])
        self.assertEqual(Path(commands[1][1]).name, "package_windows.py")
        self.assertIn("--screen-only", commands[1])
        verify.assert_called_once_with(dev.build_directory(None, "Release"), for_package=True)

    def test_failed_build_prevents_verification_and_packaging(self):
        with patch.object(dev, "verify_native") as verify, \
                patch.object(dev.subprocess, "run", return_value=SimpleNamespace(returncode=2)) as run:
            self.assertEqual(dev.main(["package"]), 2)
        self.assertEqual(run.call_count, 1)
        verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
