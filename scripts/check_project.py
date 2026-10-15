"""Check public resources and source syntax without opening UI or capturing pixels."""

import ast
import json
import re
import string
import subprocess

from release_support import ROOT, check_links, source_files


def main():
    check_links(ROOT)
    en = json.loads((ROOT / "sunglasses/locales/en.json").read_text(encoding="utf-8"))
    zh = json.loads((ROOT / "sunglasses/locales/zh-CN.json").read_text(encoding="utf-8"))
    if en.keys() != zh.keys():
        raise ValueError("Locale keys differ")
    formatter = string.Formatter()
    for key in en:
        fields = lambda value: {field for _, field, _, _ in formatter.parse(value) if field is not None}
        if fields(en[key]) != fields(zh[key]):
            raise ValueError(f"Locale placeholders differ: {key}")
    for path in source_files():
        if path.suffix in (".py", ".spec"):
            ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        if path.suffix in (".cpp", ".h"):
            keys = re.findall(r'L"(desktop\.[a-z_]+)"', path.read_text(encoding="utf-8"))
            if set(keys) - en.keys():
                raise ValueError(f"Unknown native language key in {path.name}")
    subprocess.run(["node", "--check", str(ROOT / "sunglasses/comparison_ui/viewer.js")], check=True)
    print("Public document links, locale keys/placeholders, native strings, Python and JavaScript syntax verified.")


if __name__ == "__main__":
    main()
