"""Synthetic protected attribute assignment."""

from __future__ import annotations

import json
from pathlib import Path

_NAMES_PATH = Path("data/synthetic/bertrand_mullainathan_names.json")
_DEFAULT_NAMES = {
    "group_a": ["Brad", "Greg", "Todd", "Matthew", "Jay"],
    "group_b": ["Aisha", "Keisha", "Tamika", "Latoya", "Ebony"],
}


def _load_names() -> dict[str, list[str]]:
    if _NAMES_PATH.exists():
        return json.loads(_NAMES_PATH.read_text())
    return _DEFAULT_NAMES


def assign(item: dict[str, object]) -> str:
    """Assign synthetic protected group label from Bertrand & Mullainathan names.

    Args:
        item: Document dict with meta flag synthetic=True.

    Returns:
        Group label 'A' or 'B'.

    Raises:
        ValueError: If item is not flagged synthetic.
    """
    meta = item.get("meta", {})
    if not isinstance(meta, dict) or meta.get("synthetic") is not True:
        msg = "synthetic_protected.assign may only be called on synthetic items"
        raise ValueError(msg)

    names = _load_names()
    text = str(item.get("text", ""))
    group_a = names.get("group_a", _DEFAULT_NAMES["group_a"])
    group_b = names.get("group_b", _DEFAULT_NAMES["group_b"])

    for name in group_b:
        if name.lower() in text.lower():
            return "B"
    for name in group_a:
        if name.lower() in text.lower():
            return "A"
    return "A"
