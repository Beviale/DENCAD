"""A plain autoencoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
import torch

from dencad.modeling import windowing as W
from dencad.modeling.evaluate import (in_sigmas, metrics, save_report,
                                      sigma_scale)
from dencad.modeling.lsh import ID
from dencad.modeling.subgraphs import ContextualSubgraph
from dencad.modeling.train_dencad import (DEFAULTS, GraphAutoEncoder, batches,
                                          _score)
from dencad.utils import declared_categorical, k_sigma, model_config

PARTS = ("train", "validation", "test")


def _alone(x: np.ndarray) -> ContextualSubgraph:
    """One window as its own subgraph: no neighbours, so A_hat is the identity."""
    return ContextualSubgraph(features=np.asarray(x, dtype="float32")[None, :],
                              edges=np.zeros((0, 2), dtype="int32"), centre=0)


def _egos(win, features) -> tuple[np.ndarray, list]:
    """Row ids and subgraphs of one part, collected in id order."""
    got = win.select(ID, *features).orderBy(ID).toPandas()
    X = got[list(features)].to_numpy("float32")
    return got[ID].to_numpy(), [_alone(x) for x in X]


def _score_rows(spark, win, features, model, cfg, n: int) -> np.ndarray:
    """One score per window, computed on the tasks that hold the rows."""
    import torch

    d, conf = model.d, dict(cfg)
    state = spark.sparkContext.broadcast(
        {k: v.detach().numpy() for k, v in model.net.state_dict().items()})
    cols = list(features)

    def work(blocks):
        local = GraphAutoEncoder(d, conf)
        local.net.load_state_dict({k: torch.from_numpy(v)
                                   for k, v in state.value.items()})
        for pdf in blocks:
            if pdf.empty:
                continue
            egos = [_alone(x) for x in pdf[cols].to_numpy("float32")]
            yield pd.DataFrame({"_row": pdf[ID].to_numpy(),
                                "_s": _score(local, egos, conf)})

    got = (win.select(ID, *cols)
           .mapInPandas(work, schema="_row long, _s double").toPandas())
    state.unpersist()
    out = np.full(n, np.nan)
    out[got["_row"].to_numpy().astype("int64")] = got["_s"].to_numpy()
    if np.isnan(out).any():
        raise RuntimeError(f"{int(np.isnan(out).sum())} windows without a score")
    return out


def _fit(model, egos, cfg, watch, fixed: Optional[int] = None) -> tuple[float, int]:
    """Train on 'egos'; stop on the validation AP of 'watch', or run 'fixed' epochs."""

    opt = torch.optim.Adam(model.net.parameters(), lr=cfg["learning_rate"])
    fit_batches = batches(egos, list(range(len(egos))), model.d, cfg["batch_size"],
                          cfg["mask_centre"])
    order = np.arange(len(fit_batches))
    shuffler = np.random.default_rng(cfg["seed"])
    best, waited, state, best_epoch = -np.inf, 0, None, 0
    epochs = int(fixed or cfg["max_epochs"])
    logger.info("  {:,} examples, {} batches, {} epochs{}", len(egos),
                len(fit_batches), epochs, "" if fixed else ", stopping on validation AP")
    for epoch in range(1, epochs + 1):
        model.net.train()
        shuffler.shuffle(order)
        for j in order:
            X, A_hat, centre, target, _ = fit_batches[j]
            opt.zero_grad()
            ((model.forward(X, A_hat, centre) - target) ** 2).mean().backward()
            opt.step()
        if fixed:
            state, best_epoch = {k: v.clone() for k, v in
                                 model.net.state_dict().items()}, epoch
            continue
        seen = float(metrics(watch[1], _score(model, watch[0], cfg))
                     ["average_precision"])
        if epoch % 5 == 0 or epoch == 1:
            logger.info("  epoch {:>3}: validation AP {:.5f}", epoch, seen)
        if seen > best + 1e-5:
            best, waited, best_epoch = seen, 0, epoch
            state = {k: v.clone() for k, v in model.net.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg["patience"]:
                logger.info("  stopped at epoch {}", epoch)
                break
    model.net.load_state_dict(state)
    return (float(best) if not fixed else float("nan")), best_epoch


def train_plain(split_dir: Path | str, reports_dir: Path | str = Path("reports"),
                window_steps: Optional[int] = None, stage: str = "select",
                selected: Optional[dict] = None, **overrides) -> dict:
    """One phase of the protocol. "select" never reads the test split."""
    if stage not in ("select", "final"):
        raise ValueError(f"stage must be 'select' or 'final', not {stage!r}")
    if stage == "final" and not selected:
        raise ValueError("stage='final' needs the entry a search chose")
    if stage == "final":
        overrides = {**selected.get("config", {}), **overrides}

    split_dir, reports_dir = Path(split_dir), Path(reports_dir)
    dataset, files = W.discover(split_dir)
    cfg = {**DEFAULTS, **model_config("plain", dataset, split_dir.name),
           **{k: v for k, v in overrides.items() if v is not None}}
    window_steps = int(window_steps or cfg["window_steps"])
    min_coverage = float(cfg["min_coverage"])

    logger.info("plain autoencoder on {} | window {} readings | {}x{} latent {}",
                dataset, window_steps, cfg["layers"], cfg["hidden"], cfg["latent"])
    spark = W.session(f"plain-{dataset}", int(cfg["workers"]))
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
            logger.info("refit: {:,} normal windows of validation join the fit, "
                        "{:,} rows in all", n_extra, len(labels["train"]))

        model = GraphAutoEncoder(len(features), cfg)
        _, train_egos = _egos(win["train"], features)
        watch = None
        if stage == "select":
            _, val_egos = _egos(win["validation"], features)
            watch = (val_egos, labels["validation"])
        best, epoch = _fit(model, train_egos, cfg, watch,
                           fixed=int(selected["epoch"]) if stage == "final" else None)
        logger.success("fitted, best validation AP {:.5f} at epoch {}", best, epoch)

        scores = {p: _score_rows(spark, win[p], features, model, cfg, len(labels[p]))
                  for p in parts}
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

    train_loss = float(np.mean(scores["train"]))
    logger.info("mean reconstruction error on train {:.6f}", train_loss)
    results = {}
    for part in [p for p in parts if p != "train"]:
        m = metrics(labels[part], z[part], threshold=carried)
        results[part] = m
        logger.success("  {:<11} ROC-AUC {:.4f}  AP {:.4f}  F1 {:.4f}  "
                       "segments {}/{}", part, m["roc_auc"],
                       m["average_precision"], m["f1"], m["segments_caught"],
                       m["segments"])

    report = save_report(
        reports_dir / f"plain_{split_dir.name}_{window_steps}steps_{stage}.json",
        "plain autoencoder", dataset, results, carried,
        {"window_steps": int(window_steps), **{k: cfg[k] for k in ("hidden", "latent", "layers", "learning_rate", "batch_size", "patience", "mask_centre", "seed")}},
        {"stage": stage, "epoch": int(epoch), "train_loss": train_loss,
         "spread": {"mean": mu, "deviation": sd},
         "windows": {p: int(len(labels[p])) for p in parts}},
        write=(stage == "final"))
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
        description="Plain autoencoder on the windows: the control for DENCAD.")
    ap.add_argument("--split_dir", type=Path, required=True)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--window-steps", type=int, default=None)
    ap.add_argument("--stage", choices=("select", "final"), default="select")
    ap.add_argument("--selected", type=Path, default=None)
    ap.add_argument("--contaminated-refit", action="store_true", default=None,
                    help="keep the positives of validation in the final refit")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    train_plain(a.split_dir, a.reports_dir, a.window_steps, stage=a.stage,
                workers=a.workers,
                contaminated_refit=a.contaminated_refit,
                selected=(_carry(a.selected)
                          if a.selected else None))


if __name__ == "__main__":
    main()


__all__ = ["train_plain"]
