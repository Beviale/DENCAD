"""Sigma rule on DENCAST's placement distance: the baseline with no autoencoder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
from pyspark.sql import functions as F
import json

from dencad.modeling import clustering as C
from dencad.modeling import windowing as W
from dencad.modeling.evaluate import (in_sigmas, metrics, save_report,
                                      sigma_scale)
from dencad.modeling.lsh import ID, NeighbourhoodGraph
from dencad.modeling.train_dencad import DEFAULTS, _place, _pool_frame
from dencad.utils import declared_categorical, k_sigma, model_config

PARTS = ("train", "validation", "test")


def _scores(placed, n: int) -> np.ndarray:
    """Placement distance per window, ordered by row id."""
    got = (placed.select("_row", F.col("_dist").alias("_z"))
           .orderBy("_row").toPandas())
    out = np.full(n, np.nan)
    out[got["_row"].to_numpy().astype("int64")] = got["_z"].to_numpy()
    if np.isnan(out).any():
        logger.warning("  {:,} of {:,} windows without a score", int(np.isnan(out).sum()), n)
        out = np.where(np.isnan(out), np.nanmax(out) * 1.01, out)
    return out


def train_sigma(split_dir: Path | str, reports_dir: Path | str = Path("reports"),
                window_steps: Optional[int] = None,
                eps: Optional[float] = None,
                stage: str = "select", selected: Optional[dict] = None,
                **overrides) -> dict:
    """One phase of the protocol. "select" never reads the test split."""
    if stage not in ("select", "final"):
        raise ValueError(f"stage must be 'select' or 'final', not {stage!r}")
    if stage == "final":
        if not selected:
            raise ValueError("stage='final' needs the entry a search chose")
        overrides = {**selected.get("config", {}), **overrides}

    split_dir, reports_dir = Path(split_dir), Path(reports_dir)
    dataset, files = W.discover(split_dir)
    cfg = {**DEFAULTS, **model_config("sigma", dataset, split_dir.name),
           **{k: v for k, v in overrides.items() if v is not None}}
    if eps is None:
        eps = cfg.get("eps")
    if eps is None or cfg.get("window_steps") is None:
        raise ValueError(f"params.yaml has no entry for {dataset!r} under "
                         "data.window_steps or data.eps")
    eps = float(eps)
    width = float(cfg.get("width_factor", 1.0)) * eps
    window_steps = int(window_steps or cfg["window_steps"])
    min_coverage = float(cfg["min_coverage"])

    logger.info("sigma on {} from {} | window {} readings | eps {} | minPts {}",
                dataset, split_dir, window_steps, eps, cfg["min_pts"])
    spark = W.session(f"sigma-{dataset}", int(cfg["workers"]))
    graph = NeighbourhoodGraph(int(cfg["bits"]), int(cfg["permutations"]),
                               int(cfg["band"]), eps=eps, width=width,
                               seed=int(cfg["seed"]), workers=int(cfg["workers"]))
    try:
        parts = PARTS if stage == "final" else ("train", "validation")
        win, labels, features, label = W.windowed_parts(
            spark, files, parts, window_steps, declared_categorical(dataset),
            min_coverage=min_coverage, scale=cfg["scale_windows"])

        if stage == "final":
            extra = (win["validation"] if cfg.get("contaminated_refit")
                     else win["validation"].filter(F.col(label) == 0))
            n_extra = extra.count()
            win["train"] = W.number_rows(
                win["train"].drop(ID).unionByName(extra.drop(ID)))
            got = win["train"].select(ID, label).orderBy(ID).toPandas()
            labels["train"] = (got[label].to_numpy() != 0).astype("uint8")
            logger.info("refit: {:,} normal windows of validation join the graph, "
                        "{:,} rows in all", n_extra, len(labels["train"]))

        signed = graph._signatures(win["train"], features).select(
            ID, "_sig", "_vec", "_norm").cache()
        candidates = graph._candidates(signed)
        n_candidates = candidates.count()
        e_df = (graph._distances(candidates, signed)
                .localCheckpoint(eager=True))
        n_edges = e_df.count()
        signed.unpersist()
        vertices = win["train"].select(F.col(ID).alias(C.NODE))
        cores_df = C.core_objects(e_df, cfg["min_pts"]).localCheckpoint(eager=True)
        logger.info("graph: {:,} edges of {:,} candidates, {:,} core objects",
                    n_edges, n_candidates, cores_df.count())
        state = (C.find_clusters(vertices, e_df, cores_df,
                                 cfg["label_change_rate"]).localCheckpoint(eager=True))
        summary = C.summarise(state, dataset)
        clustered = state.filter(F.col(C.CLUSTER) != F.lit(C.NOISE))

        if cfg["attach_to"] == "core":
            pool_nodes = clustered.join(cores_df, C.NODE).select(C.NODE)
        else:
            pool_nodes = clustered.select(C.NODE)
        pool = _pool_frame(win["train"], pool_nodes,
                           features).localCheckpoint(eager=True)

        placed = {p: _place(spark, win[p], pool, features,
                            exclude_self=(p == "train")) for p in parts}

        scores = {p: _scores(placed[p], len(labels[p])) for p in parts}
    finally:
        spark.stop()

    mu, sd = sigma_scale(scores["train"])
    z = {p: in_sigmas(s, mu, sd) for p, s in scores.items()}

    if stage == "final":
        carried = float(selected["threshold"])
    else:
        by_k = {k: metrics(labels["validation"], z["validation"], threshold=k)
                for k in k_sigma()}
        for k, m in by_k.items():
            logger.info("  k = {:<4} precision {:.4f}  recall {:.4f}  F1 {:.4f}  "
                        "flagged {:.2%}", k, m["precision"], m["recall"], m["f1"],
                        m["flagged_share"])
        carried = float(max(by_k, key=lambda k: by_k[k]["f1"]))
    logger.info("k = {:.3f} sigma | scale {:.6f} +/- {:.6f}", carried, mu, sd)

    results = {}
    for part in parts[1:]:
        m = metrics(labels[part], z[part], threshold=carried)
        results[part] = m
        logger.success("  {:<11} ROC-AUC {:.4f}  AP {:.4f}  F1 {:.4f}  "
                       "segments {}/{}", part, m["roc_auc"],
                       m["average_precision"], m["f1"], m["segments_caught"],
                       m["segments"])

    report = save_report(
        reports_dir / f"sigma_{split_dir.name}_{window_steps}steps_{stage}.json",
        "DENCAST + sigma", dataset, results, carried,
        {"window_steps": int(window_steps), "eps": float(eps), "width": float(width), **{k: cfg[k] for k in ("bits", "permutations", "band", "min_pts", "label_change_rate", "attach_to", "seed")}},
        {"stage": stage, "k_sigma": float(carried),
         "cut_distance": mu + carried * sd,
         "spread": {"mean": mu, "deviation": sd},
         "windows": {p: int(len(labels[p])) for p in parts},
         "graph": {"edges": int(n_edges), "clusters": summary.get("clusters")}},
        write=(stage == "final"))
    return report


def _carry(path):
    """The threshold and epoch to carry: from a search report or from a select one."""

    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "best" in d:
        return d["best"]
    return {"config": d.get("config", {}), "threshold": d["threshold"],
            "epoch": d.get("epoch", 0),
            "spread": d.get("validation_distance")}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sigma rule on DENCAST's placement distance.")
    ap.add_argument("--split_dir", type=Path, required=True)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--window-steps", type=int, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--min-pts", type=int, default=None)
    ap.add_argument("--stage", choices=("select", "final"), default="select")
    ap.add_argument("--selected", type=Path, default=None)
    ap.add_argument("--contaminated-refit", action="store_true", default=None,
                    help="keep the positives of validation in the final refit")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    train_sigma(a.split_dir, a.reports_dir, a.window_steps, a.eps,
                stage=a.stage, min_pts=a.min_pts, workers=a.workers,
                contaminated_refit=a.contaminated_refit,
                selected=(_carry(a.selected)
                          if a.selected else None))


if __name__ == "__main__":
    main()


__all__ = ["train_sigma"]
