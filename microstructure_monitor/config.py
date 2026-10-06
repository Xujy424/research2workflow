"""Small config loader for monitor yaml-like files."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "output"


def read_simple_yaml(path: str | Path) -> dict[str, object]:
    """Read the flat key-value yaml files used by this project.

    The project intentionally avoids a PyYAML dependency; nested values that
    matter operationally can be passed as dictionaries from Python.
    """

    values: dict[str, object] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value == "[]":
            parsed: object = []
        else:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
        values[key.strip()] = parsed
    return values
