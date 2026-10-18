"""Hyperparameter selection for DENCAD: a random draw from the grid, on validation."""

from __future__ import annotations


import argparse
import itertools
import json
from pathlib import Path
from typing import Optional

from loguru import logger
import random

from dencad.modeling.train_dencad import dencad_config, train_dencad
from dencad.modeling.train_plain import train_plain
from dencad.modeling.train_sigma import train_sigma
from dencad.modeling.windowing import discover

PER_DATASET = ("min_similarity",)
MODELS = {
    "dencad": (train_dencad, None),
    "plain": (train_plain, ("hidden", "latent", "layers", "learning_rate",
                            "patience", "batch_size", "max_epochs")),
    "sigma": (train_sigma, ("bits", "permutations", "band", "min_similarity",
                            "min_pts", "label_change_rate", "attach_to")),
}


def axes(dataset: str, model: str = "dencad") -> dict[str, list]:
    """The grid as one list of values per parameter, this dataset's entries resolved."""

    section = dencad_config().get("grid") or {}
    if hasattr(section, "to_dict"):
        section = section.to_dict()
    keep = MODELS[model][1]
    out: dict[str, list] = {}
    for name, values in section.items():
        if keep is not None and name not in keep:
            continue
        if name in PER_DATASET:
            values = (values.to_dict() if hasattr(values, "to_dict")
                      else values).get(dataset)
            if values is None:
                continue
        if values is None or not isinstance(values, (list, tuple)):
            values = [values]
        out[name] = list(values)
    return out


def plan(grid: dict[str, list], runs: int, seed: int = 42) -> list[dict]:
    names = list(grid)
    total = 1
    for n in names:
        total *= len(grid[n])
    if total <= runs:
        return [dict(zip(names, values))
                for values in itertools.product(*(grid[n] for n in names))]

    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < runs:
        draw = {n: grid[n][rng.randrange(len(grid[n]))] for n in names}
        key = tuple(str(draw[n]) for n in names)
        if key not in seen:
            seen.add(key)
            out.append(draw)
    return out


def _write(reports_dir: Path, split_dir: Path, dataset: str,
           grid: dict[str, list], rows: list[dict], model: str = "dencad") -> Path:
    """The table and the current winner, rewritten after every run."""

    done = [r for r in rows if "failed" not in r]
    best = (max(done, key=lambda r: r["validation"]["average_precision"])
            if done else None)
    out = reports_dir / f"search_{model}_{split_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": model, "dataset": dataset,
         "selected_on": "validation average precision",
         "grid": grid, "runs": rows, "best": best}, indent=2, default=str),
        encoding="utf-8")
    return out


def search(split_dir: Path | str, reports_dir: Path | str = Path("reports"),
           window_steps: Optional[int] = None, runs: int = 15,
           seed: int = 42, model: str = "dencad") -> dict:
    """Draw 'runs' configurations, run them, write the table and the winner."""

    split_dir = Path(split_dir)
    reports_dir = Path(reports_dir)
    dataset, _ = discover(split_dir)
    trainer = MODELS[model][0]
    grid = axes(dataset, model)
    if not grid:
        raise ValueError("the 'grid' section of params.yaml is empty")
    drawn = plan(grid, runs, seed)
    total = 1
    for v in grid.values():
        total *= len(v)
    for name, values in grid.items():
        logger.info("  {:<18} {}", name, values)

    rows: list[dict] = []
    for i, config in enumerate(drawn, start=1):
        if model == "dencad" and "layers" in config and "max_hops" not in grid:
            config = {**config, "max_hops": int(config["layers"])}
        logger.info("[{}/{}] {}", i, len(drawn),
                    ", ".join(f"{k}={v}" for k, v in config.items()))
        try:
            report = trainer(split_dir, reports_dir / "select",
                             window_steps=window_steps, stage="select", **config)
        except Exception as exc:
            logger.warning("  failed: {}: {}", type(exc).__name__, exc)
            rows.append({"config": config, "failed": f"{type(exc).__name__}: {exc}"})
            continue
        val = report["metrics"]["validation"]
        rows.append({
            "config": config,
            "epoch": int(report.get("epoch", 0)),
            "threshold": float(report["threshold"]),
            "validation": {k: val[k] for k in ("roc_auc", "average_precision", "f1",
                                               "segments_caught", "segments")},
            "edges": int(report.get("graph", {}).get("edges", 0)),
        })
        logger.success("  validation AP {:.4f} ROC-AUC {:.4f} at epoch {}",
                       val["average_precision"], val["roc_auc"],
                       report.get("epoch", 0))
        out = _write(reports_dir, split_dir, dataset, grid, rows, model)

    done = [r for r in rows if "failed" not in r]
    if not done:
        raise RuntimeError("every configuration failed")
    best = max(done, key=lambda r: r["validation"]["average_precision"])
    logger.success("winner: validation AP {:.4f} | {}",
                   best["validation"]["average_precision"],
                   ", ".join(f"{k}={v}" for k, v in best["config"].items()))
    logger.success("search written to {}", out)
    return {"best": best, "runs": rows, "report": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run DENCAD's grid on validation. The test split is never read.")
    ap.add_argument("--split_dir", type=Path, default=Path("data/processed/swat"))
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--window-steps", type=int, default=None)
    ap.add_argument("--runs", type=int, default=15,
                    help="how many configurations to draw")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=tuple(MODELS), default="dencad")
    a = ap.parse_args()
    search(a.split_dir, a.reports_dir, a.window_steps, a.runs, a.seed, a.model)


if __name__ == "__main__":
    main()


__all__ = ["search", "plan", "axes"]
