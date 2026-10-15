"""Centralized CLI strings; core validation returns stable identifiers."""

from functools import lru_cache
from importlib.resources import files
import json

LANGUAGES = ("en", "zh-CN")


class ValidationError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@lru_cache(maxsize=2)
def messages(language: str = "en") -> dict[str, str]:
    if language not in LANGUAGES:
        raise ValueError("Unsupported language")
    return json.loads(files("sunglasses").joinpath("locales", f"{language}.json").read_text(encoding="utf-8"))


def translator(language: str):
    resource = messages(language)
    return lambda key, **parameters: resource[key].format(**parameters)
