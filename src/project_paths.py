"""Resolve all bundled assets relative to this checkout."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
