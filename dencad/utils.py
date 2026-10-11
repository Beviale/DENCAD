"""Shared helpers"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import yaml


PARAMS_PATH = Path("params.yaml")


class Params:
    """A nested YAML tree with attribute, item and dotted-path access."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(cls, path: Path | str = PARAMS_PATH) -> "Params":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; it holds the pipeline config")
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def _wrap(self, value: Any) -> Any:
        return Params(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        try:
            return self._wrap(self._data[name])
        except KeyError:
            raise AttributeError(
                f"params has no '{name}'; it holds {', '.join(sorted(self._data))}"
            ) from None

    def __getitem__(self, key: str) -> Any:
        return self._wrap(self._data[key])

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Params({', '.join(sorted(self._data))})"

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for step in path.split("."):
            if not isinstance(node, dict) or step not in node:
                return default
            node = node[step]
        return self._wrap(node)

    def to_dict(self) -> dict[str, Any]:
        return self._data


CATEGORICAL_PATH = Path("data/raw/categorical_columns.yaml")


def declared_categorical(dataset: str,
                         path: Path = CATEGORICAL_PATH) -> list[str]:
    """The columns a dataset's own documentation calls categorical."""

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if dataset not in data:
        raise KeyError(f"{path} names no categorical columns for {dataset!r}; "
                       f"it covers {', '.join(sorted(data))}")
    return list(data[dataset])


__all__ = ["Params", "PARAMS_PATH", "CATEGORICAL_PATH", "declared_categorical"]
