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


def section(name: str) -> dict:
    """One top-level section of params.yaml as a plain dict."""
    try:
        got = Params.load(PARAMS_PATH).get(name)
    except Exception:
        return {}
    return got.to_dict() if hasattr(got, "to_dict") else dict(got or {})


def grid_axes(model: str, dataset: str | None = None) -> dict[str, list]:
    """A model's grid: one list of values per axis, the dataset's entry resolved."""
    out: dict[str, list] = {}
    for name, values in (section(model).get("grid") or {}).items():
        if isinstance(values, dict):          # written per dataset
            if dataset is None or dataset not in values:
                continue
            values = values[dataset]
        out[name] = list(values) if isinstance(values, (list, tuple)) else [values]
    return out


def model_config(model: str, dataset: str | None = None) -> dict:
    """A model's settings: the shared data block, the head of every grid axis as
    its default, then the model's own keys on top."""
    cfg = dict(section("data"))
    cfg.pop("min_similarity", None)           # an axis, not a setting
    cfg.update({k: v[0] for k, v in grid_axes(model, dataset).items() if v})
    cfg.update({k: v for k, v in section(model).items() if k != "grid"})
    if dataset is not None:
        cfg["window_steps"] = (cfg.get("window_steps") or {}).get(dataset)
    return cfg


def k_sigma() -> list[float]:
    """The sigma multipliers to try, shared by the two models that use the rule."""
    try:
        got = Params.load(PARAMS_PATH).get("k_sigma")
    except Exception:
        got = None
    return [float(x) for x in (got or [3.0])]


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


__all__ = ["Params", "PARAMS_PATH", "CATEGORICAL_PATH", "declared_categorical",
           "section", "grid_axes", "model_config", "k_sigma"]
