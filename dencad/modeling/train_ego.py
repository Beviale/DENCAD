"""A graph autoencoder without the clustering: every window is scored in its hop ball."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
from pyspark.sql import functions as F

from dencad.modeling import windowing as W
from dencad.modeling.ego import EgoGraph
from dencad.modeling.evaluate import (in_sigmas, metrics, save_report,
                                      sigma_scale)
from dencad.modeling.lsh import ID, NeighbourhoodGraph
from dencad.modeling.train_dencad import DEFAULTS, GraphAutoEncoder, _score
from dencad.modeling.train_plain import _carry, _fit
from dencad.utils import declared_categorical, k_sigma, model_config

PARTS = ("train", "validation", "test")


def _matrix(win, features) -> np.ndarray:
    """Window features of one part, in id order."""
    got = win.select(ID, *features).orderBy(ID).toPandas()
    return got[list(features)].to_numpy("float32")


def _edges(graph, win, features) -> np.ndarray:
    """The neighbourhood graph over one part, as undirected pairs."""
    signed = graph._signatures(win, features).select(
        ID, "_sig", "_vec", "_norm").cache()
    got = graph._distances(graph._candidates(signed), signed).select(
        "_src", "_dst").toPandas()
    signed.unpersist()
    return got.to_numpy("int64").reshape(-1, 2)


def _attached(graph, train, part, features, n_train: int, n_part: int) -> list:
    """For each window of 'part', the training nodes the same hashing joins it to."""
    shifted = part.select((F.col(ID) + F.lit(n_train)).alias(ID), *features)
    both = train.select(ID, *features).unionByName(shifted)
    signed = graph._signatures(both, features).select(
        ID, "_sig", "_vec", "_norm").cache()
    cross = (graph._distances(graph._candidates(signed), signed)
             .filter((F.col("_src") < n_train) & (F.col("_dst") >= n_train))
             .select(F.col("_dst").alias("_q"), F.col("_src").alias("_n"))
             .toPandas())
    signed.unpersist()
    out = [np.zeros(0, dtype="int64") for _ in range(n_part)]
    for q, grp in cross.groupby("_q"):
        out[int(q) - n_train] = grp["_n"].to_numpy("int64")
    alone = sum(1 for nb in out if not len(nb))
    logger.info("  attached {:,} windows, {:.1f} neighbours on average, "
                "{:,} with none", n_part,
                float(np.mean([len(nb) for nb in out])), alone)
    return out


def train_ego(split_dir: Path | str, reports_dir: Path | str = Path("reports"),
              window_steps: Optional[int] = None, eps: Optional[float] = None,
              stage: str = "select", selected: Optional[dict] = None,
              **overrides) -> dict:
    if stage not in ("select", "final"):
        raise ValueError(f"stage must be 'select' or 'final', not {stage!r}")
    if stage == "final" and not selected:
        raise ValueError("stage='final' needs the entry a search chose")
    if stage == "final":
        overrides = {**selected.get("config", {}), **overrides}

    split_dir, reports_dir = Path(split_dir), Path(reports_dir)
    dataset, files = W.discover(split_dir)
    cfg = {**DEFAULTS, **model_config("ego", dataset, split_dir.name),
           **{k: v for k, v in overrides.items() if v is not None}}
    eps = float(eps if eps is not None else cfg["eps"])
    width = float(cfg["width_factor"]) * eps
    window_steps = int(window_steps or cfg["window_steps"])
    hops = int(cfg["layers"])

    logger.info("ego autoencoder on {} | window {} readings | eps {} | {} hops | "
                "{}x{} latent {}", dataset, window_steps, eps, hops,
                cfg["layers"], cfg["hidden"], cfg["latent"])
    spark = W.session(f"ego-{dataset}", int(cfg["workers"]))
    graph = NeighbourhoodGraph(cfg["bits"], cfg["permutations"], cfg["band"],
                               eps=eps, width=width, seed=cfg["seed"],
                               workers=cfg["workers"])
    try:
        parts = PARTS if stage == "final" else ("train", "validation")
        win, labels, features, label = W.windowed_parts(
            spark, files, parts, window_steps, declared_categorical(dataset),
            min_coverage=float(cfg["min_coverage"]), scale=cfg["scale_windows"])

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

        edges = _edges(graph, win["train"], features)
        n_train = len(labels["train"])
        joined = {p: _attached(graph, win["train"], win[p], features,
                               n_train, len(labels[p]))
                  for p in parts if p != "train"}
        X = {p: _matrix(win[p], features) for p in parts}
    finally:
        spark.stop()

    ball = EgoGraph(X["train"], edges, hops)
    sizes = ball.sizes()
    logger.info("graph: {:,} edges over {:,} windows | ball of {:.1f} nodes on "
                "average, {} at most, {:.2f} GB of examples", len(edges), n_train,
                sizes.mean(), int(sizes.max()),
                float(sizes.sum()) * len(features) * 4 / 1e9)

    egos = {"train": [ball.of_node(i) for i in range(n_train)]}
    for part, nbrs in joined.items():
        egos[part] = [ball.of_query(X[part][i], nbrs[i]) for i in range(len(nbrs))]

    model = GraphAutoEncoder(len(features), cfg)
    watch = (egos["validation"], labels["validation"]) if stage == "select" else None
    best, epoch = _fit(model, egos["train"], cfg, watch,
                       fixed=int(selected["epoch"]) if stage == "final" else None)
    logger.success("fitted, best validation AP {:.5f} at epoch {}", best, epoch)
    scores = {p: _score(model, egos[p], cfg) for p in parts}

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
    for part in [p for p in parts if p != "train"]:
        m = metrics(labels[part], z[part], threshold=carried)
        results[part] = m
        logger.success("  {:<11} ROC-AUC {:.4f}  AP {:.4f}  F1 {:.4f}  "
                       "segments {}/{}", part, m["roc_auc"],
                       m["average_precision"], m["f1"], m["segments_caught"],
                       m["segments"])

    report = save_report(
        reports_dir / f"ego_{split_dir.name}_{window_steps}steps_{stage}.json",
        "ego autoencoder", dataset, results, carried,
        {"window_steps": int(window_steps), "eps": float(eps), "width": float(width),
         **{k: cfg[k] for k in ("bits", "permutations", "band", "width_factor",
                                "hidden", "latent", "layers", "learning_rate",
                                "batch_size", "patience", "mask_centre", "seed")}},
        {"stage": stage, "epoch": int(epoch),
         "train_loss": float(np.mean(scores["train"])),
         "spread": {"mean": mu, "deviation": sd},
         "graph": {"edges": int(len(edges)), "ball_mean": float(sizes.mean()),
                   "ball_max": int(sizes.max())},
         "windows": {p: int(len(labels[p])) for p in parts}},
        write=(stage == "final"))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Graph autoencoder on hop balls, without clustering.")
    ap.add_argument("--split_dir", type=Path, required=True)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--window-steps", type=int, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--stage", choices=("select", "final"), default="select")
    ap.add_argument("--selected", type=Path, default=None)
    ap.add_argument("--contaminated-refit", action="store_true", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    train_ego(a.split_dir, a.reports_dir, a.window_steps, a.eps, stage=a.stage,
              contaminated_refit=a.contaminated_refit, workers=a.workers,
              selected=(_carry(a.selected) if a.selected else None))


if __name__ == "__main__":
    main()


__all__ = ["train_ego"]
