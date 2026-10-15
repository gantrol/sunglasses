"""Generate native UTF-16 resources from the same JSON used by Python."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
resources = [json.loads((root / "sunglasses/locales" / f"{language}.json").read_text(encoding="utf-8"))
             for language in ("en", "zh-CN")]
assert resources[0].keys() == resources[1].keys(), "Locale keys must match"
keys = sorted(key for key in resources[0] if key.startswith("desktop."))
def literal(value):
    return 'L"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'
lines = ['#pragma once', '#include <string_view>', 'namespace sunglasses::strings {',
         'struct Entry { std::wstring_view id, english, chinese; };', 'inline constexpr Entry entries[] = {']
for key in keys:
    lines.append('    {' + ', '.join(literal(value) for value in (key, resources[0][key], resources[1][key])) + '},')
lines += ['};', '}']
target = Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text('\n'.join(lines)+'\n', encoding='utf-8')
