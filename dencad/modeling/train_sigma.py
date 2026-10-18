"""Sigma rule on DENCAST's placement cosine: the baseline with no autoencoder."""

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
from dencad.modeling.evaluate import metrics, save_report
from dencad.modeling.lsh import ID, NeighbourhoodGraph
from dencad.modeling.train_dencad import _place, _pool_frame, dencad_config
from dencad.utils import Params, PARAMS_PATH, declared_categorical

PARTS = ("train", "validation", "test")
EPS = 1e-6


def _scores(placed, mu: float, sd: float, n: int) -> np.ndarray:
    """z per window, ordered by row id."""
    got = (placed.select("_row", ((F.lit(mu) - F.col("_sim"))
                                  / F.lit(max(sd, EPS))).alias("_z"))
           .orderBy("_row").toPandas())
    out = np.full(n, np.nan)
    out[got["_row"].to_numpy().astype("int64")] = got["_z"].to_numpy()
    if np.isnan(out).any():
        logger.warning("  {:,} of {:,} windows without a score", int(np.isnan(out).sum()), n)
        out = np.where(np.isnan(out), np.nanmax(out) * 1.01, out)
    return out


def train_sigma(split_dir: Path | str, reports_dir: Path | str = Path("reports"),
                window_steps: Optional[int] = None,
                min_similarity: Optional[float] = None,
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
    cfg = {**dencad_config(), **{k: v for k, v in overrides.items() if v is not None}}
    dataset, files = W.discover(split_dir)
    kmeans = Params.load(PARAMS_PATH)["kmeans"]
    window_steps = window_steps or (kmeans.get("window_steps") or {}).get(dataset, 60)
    min_coverage = float(kmeans.get("min_coverage", 0.0))
    if min_similarity is None:
        by_dataset = cfg["min_similarity"]
        min_similarity = float(by_dataset[dataset] if hasattr(by_dataset, "__getitem__")
                               and not isinstance(by_dataset, float) else by_dataset)

    logger.info("sigma on {} from {} | window {} readings | cosine >= {} | minPts {}",
                dataset, split_dir, window_steps, min_similarity, cfg["min_pts"])
    spark = W.session(f"sigma-{dataset}", int(cfg["workers"]))
    graph = NeighbourhoodGraph(int(cfg["bits"]), int(cfg["permutations"]),
                               int(cfg["band"]), min_similarity=-1.0,
                               seed=int(cfg["seed"]), workers=int(cfg["workers"]))
    try:
        parts = PARTS if stage == "final" else ("train", "validation")
        win, labels, features, label, scaling, period = W.windowed_parts(
            spark, files, parts, window_steps, declared_categorical(dataset),
            min_coverage=min_coverage, scale=cfg["scale_windows"])

        if stage == "final":
            # Only the normal windows, except on the contaminated ESA case, which
            # exists to ask what happens when they are not filtered out.
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
        e_df = (graph._similarities(candidates, signed)
                .filter(F.col("similarity") >= F.lit(min_similarity))
                .localCheckpoint(eager=True))
        n_edges = e_df.count()
        signed.unpersist()
        logger.info("graph: {:,} edges of {:,} candidates", n_edges, n_candidates)

        vertices = win["train"].select(F.col(ID).alias(C.NODE))
        cores_df = C.core_objects(e_df, cfg["min_pts"]).localCheckpoint(eager=True)
        n_cores = cores_df.count()
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

        placed = {p: _place(spark, win[p], pool, features, exclude_self=(p == "train"))
                  for p in parts}

        # mu and sd per cluster, from the training cosines
        spread = placed["train"].agg(F.avg("_sim"),
                                      F.stddev_pop("_sim")).collect()[0]
        mu, sd = float(spread[0]), float(spread[1] or 0.0)
        logger.info("training cosine: mean {:.4f}, deviation {:.4f}", mu, sd)

        scores = {p: _scores(placed[p], mu, sd, len(labels[p])) for p in parts}
    finally:
        spark.stop()

    own = Params.load(PARAMS_PATH).get("sigma") or {}
    own = own.to_dict() if hasattr(own, "to_dict") else dict(own)
    k_values = [float(k) for k in (own.get("k_sigma") or [3.0])]

    by_k = {k: {p: metrics(labels[p], scores[p], threshold=k) for p in parts}
            for k in k_values}
    for k in k_values:
        v = by_k[k]["validation"]
        logger.info("  k = {:<4} precision {:.4f}  recall {:.4f}  F1 {:.4f}  "
                    "flagged {:.2%}", k, v["precision"], v["recall"], v["f1"],
                    v["flagged_share"])

    if stage == "final":
        carried = float(selected["threshold"])
        logger.info("threshold carried from the selection: k = {:.3f} sigma", carried)
    else:
        carried = max(k_values, key=lambda k: by_k[k]["validation"]["f1"])
        fine = metrics(labels["validation"], scores["validation"])
        logger.info("k = {:.3f} sigma chosen on validation (F1 {:.4f}); a free search "
                    "over 400 quantiles would reach {:.4f} at {:.3f}", carried,
                    by_k[carried]["validation"]["f1"], fine["f1"], fine["threshold"])

    results = {}
    for part in parts:
        m = metrics(labels[part], scores[part], threshold=carried)
        results[part] = m
        if m["roc_auc"] is not None:
            logger.success("  {:<11} ROC-AUC {:.4f}  AP {:.4f}  F1 {:.4f}  "
                           "segments {}/{}", part, m["roc_auc"],
                           m["average_precision"], m["f1"], m["segments_caught"],
                           m["segments"])

    report = save_report(
        reports_dir / f"sigma_{split_dir.name}_{window_steps}steps_{stage}.json",
        "DENCAST + sigma", dataset, results, carried,
        {"window_steps": int(window_steps), "min_similarity": float(min_similarity), **{k: cfg[k] for k in ("bits", "permutations", "band", "min_pts", "label_change_rate", "attach_to", "seed")}},
        {"stage": stage, "k_sigma": float(carried),
         "windows": {p: int(len(labels[p])) for p in parts},
         "graph": {"edges": int(n_edges), "clusters": summary.get("clusters")}})
    return report


def _carry(path):
    """The threshold and epoch to carry: from a search report or from a select one."""

    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "best" in d:
        return d["best"]
    return {"config": d.get("config", {}), "threshold": d["threshold"],
            "epoch": d.get("epoch", 0)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sigma rule on DENCAST's placement cosine.")
    ap.add_argument("--split_dir", type=Path, required=True)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--window-steps", type=int, default=None)
    ap.add_argument("--min-similarity", type=float, default=None)
    ap.add_argument("--min-pts", type=int, default=None)
    ap.add_argument("--stage", choices=("select", "final"), default="select")
    ap.add_argument("--selected", type=Path, default=None)
    ap.add_argument("--contaminated-refit", action="store_true", default=None,
                    help="keep the positives of validation in the final refit")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    train_sigma(a.split_dir, a.reports_dir, a.window_steps, a.min_similarity,
                stage=a.stage, min_pts=a.min_pts, workers=a.workers,
                contaminated_refit=a.contaminated_refit,
                selected=(_carry(a.selected)
                          if a.selected else None))


if __name__ == "__main__":
    main()


__all__ = ["train_sigma"]
