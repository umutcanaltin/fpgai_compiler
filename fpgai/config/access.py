from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def get_path(
    data: Any,
    path: str,
    default: Any = None,
) -> Any:
    """Read a dotted configuration path through mappings and list indices."""
    current = data

    for component in path.split("."):
        if isinstance(current, Mapping):
            if component not in current:
                return default

            current = current[component]
            continue

        if (
            isinstance(current, Sequence)
            and not isinstance(current, (str, bytes, bytearray))
        ):
            try:
                index = int(component)
            except ValueError:
                return default

            if index < 0 or index >= len(current):
                return default

            current = current[index]
            continue

        return default

    return current