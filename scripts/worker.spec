# The CLI packager supplies unique output directories and a provenance file.
import os
from pathlib import Path
import sys

sys.path.insert(0, SPECPATH)
from freeze_support import collect_licenses
from PyInstaller.utils.hooks import collect_data_files

root = Path(SPECPATH).parent
manifest = Path(os.environ["SUNGLASSES_FREEZE_MANIFEST"])
a = Analysis([str(root / "scripts/worker_entry.py")], pathex=[str(root)],
             datas=collect_data_files("sunglasses") + [(str(manifest), "sunglasses")])
# Windows 11 supplies the UCRT; do not copy a host tool's app-local runtime.
# https://learn.microsoft.com/en-us/cpp/windows/universal-crt-deployment
a.binaries = [entry for entry in a.binaries
              if not Path(entry[0]).name.lower().startswith("api-ms-win-crt-")
              and Path(entry[0]).name.lower() != "ucrtbase.dll"]
collect_licenses([*a.pure, *a.binaries, *a.datas, *a.scripts], manifest.parent / "licenses")
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="sunglasses-worker",
          console=True, strip=False, upx=False)
COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="sunglasses-worker")
