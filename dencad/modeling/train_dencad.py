"""DENCAD: anomaly detection on DENCAST's clusters, with a graph autoencoder."""

from __future__ import annotations

import json
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


from pyspark.sql.types import DoubleType, LongType, StructField, StructType
from pyspark.sql import functions as F



from dencad.modeling import clustering as C
from dencad.modeling import windowing as W
from dencad.modeling.evaluate import in_sigmas, metrics, save_report
from dencad.modeling.lsh import ID, NeighbourhoodGraph
from dencad.modeling.placement import nearest_grid_frames
from dencad.utils import declared_categorical, k_sigma, model_config
from dencad.modeling.shards import query_frame
from dencad.modeling.subgraphs import egos_from_rows
from dencad.modeling.parallel import fit_shards
from dencad.modeling import shards as S

PARTS = ("train", "validation", "test")


DEFAULTS = {
    "bits": 16, "permutations": 10, "band": 20, "min_pts": 10,
    "label_change_rate": 0.05, "attach_to": "clustered", "max_members": None,
    "mask_centre": False, "hidden": 64, "latent": 16, "layers": 2,
    "learning_rate": 1e-3, "batch_size": 128, "max_epochs": 60, "patience": 8,
    "train_examples": None, "scale_windows": True,
    "early_stopping": "validation_ap",
    "replicas": 4,
    # Adam's rate times the replica count, so the fit is not penalised for the larger
    # effective batch N replicas produce.
    "scale_learning_rate": True,
    "seed": 42, "workers": 4,
}


_NET = None


def _net_class():
    """The autoencoder as a real 'nn.Module', with the convolution in its 'forward'."""
    global _NET
    if _NET is not None:
        return _NET

    class _Net(nn.Module):
        def __init__(self, d: int, cfg: dict):
            super().__init__()
            widths = [d] + [cfg["hidden"]] * (cfg["layers"] - 1) + [cfg["latent"]]
            self.conv = nn.ModuleList(
                [nn.Linear(widths[i], widths[i + 1]) for i in range(cfg["layers"])])
            self.decode = nn.Sequential(
                nn.Linear(cfg["latent"], cfg["hidden"]), nn.ReLU(),
                nn.Linear(cfg["hidden"], d))

        def forward(self, X, A_hat, centre):
            h = X
            for i, layer in enumerate(self.conv):
                h = A_hat @ layer(h)
                if i < len(self.conv) - 1:
                    h = torch.relu(h)
            return self.decode(h[torch.arange(len(centre)), centre])

    _NET = _Net
    return _NET


class GraphAutoEncoder:
    """A stack of graph convolutions over a dense adjacency, decoding the centre."""

    def __init__(self, d: int, cfg: dict):

        torch.manual_seed(cfg["seed"])

        self.d = d
        self.cfg = cfg
        self.net = _net_class()(d, cfg)

    def forward(self, X, A_hat, centre):
        return self.net(X, A_hat, centre)


def _normalise(A, mask, degrees=None):
    """Symmetric normalisation of A + I over the real nodes only."""

    eye = torch.eye(A.shape[1]).unsqueeze(0) * mask.unsqueeze(2)
    A = A + eye
    raw = A.sum(2) if degrees is None else degrees
    deg = raw.clamp(min=1e-6).pow(-0.5) * mask
    return A * deg.unsqueeze(2) * deg.unsqueeze(1)


CHUNK_FLOATS = 16 * 1024 * 1024


def _chunks(egos, indices, size: int):
    chunk: list = []
    for i in indices:
        m = len(egos[i].features)
        if chunk and ((len(chunk) + 1) * m * m > CHUNK_FLOATS or len(chunk) >= size):
            yield chunk
            chunk = []
        chunk.append(i)
    if chunk:
        yield chunk


def batches(egos, indices, d: int, size: int, mask_centre: bool) -> list:
    """Padded batches of similar-sized contextual subgraphs, built once."""

    out = []
    indices = sorted(indices, key=lambda i: len(egos[i].features))
    for chunk in _chunks(egos, indices, size):
        b = len(chunk)
        k = np.fromiter((len(egos[i].features) for i in chunk), dtype="int64",
                        count=b)
        m = int(k.max())

      
        X = np.zeros((b, m, d), dtype="float32")
        A = np.zeros((b, m, m), dtype="float32")
       
        has_deg = any(egos[i].degrees is not None for i in chunk)
        D = np.zeros((b, m), dtype="float32") if has_deg else None
        for j, i in enumerate(chunk):
            e = egos[i]
            kk = int(k[j])
            X[j, :kk] = e.features
            u, v = e.edges[:, 0], e.edges[:, 1]
            A[j, u, v] = A[j, v, u] = 1.0
            if has_deg:
                D[j, :kk] = (e.degrees if e.degrees is not None else
                             np.bincount(np.concatenate([u, v]),
                                         minlength=kk) + 1.0)

       
        mask = (np.arange(m)[None, :] < k[:, None]).astype("float32")
        centre = np.fromiter((egos[i].centre for i in chunk), dtype="int64", count=b)
        target = X[np.arange(b), centre].copy()
        if mask_centre:
            X[np.arange(b), centre] = 0.0

        tA = torch.from_numpy(A)
        tmask = torch.from_numpy(mask)
        tdeg = None if D is None else torch.from_numpy(D)
        out.append((torch.from_numpy(X), _normalise(tA, tmask, tdeg),
                    torch.from_numpy(centre), torch.from_numpy(target), chunk))
    return out


def _score(model, egos, cfg) -> np.ndarray:
    """Distance between the centre and its reconstruction, per contextual subgraph."""

    out = np.full(len(egos), np.nan)
    present = [i for i, e in enumerate(egos) if e is not None]
    model.net.eval()
    with torch.no_grad():
        for X, A_hat, centre, target, chunk in batches(
                egos, present, model.d, cfg["batch_size"], cfg["mask_centre"]):
            err = ((model.forward(X, A_hat, centre) - target) ** 2).mean(1)
            out[chunk] = err.sqrt().numpy()
    return out


def _pool_frame(train, pool_nodes, features):
    """The eligible training objects, with their features, ready to be placed against."""

    return (pool_nodes.join(train, pool_nodes[C.NODE] == train[ID])
            .select(F.col(C.NODE).alias("_id"), *features))


AUDIT_QUERIES = 64


def _audit_placement(queries, pool, placed, features, sample: int = AUDIT_QUERIES):
    chosen = queries.orderBy(F.rand(42)).limit(sample).alias("q")
    gap = sum(((F.col(f"q.`{c}`") - F.col(f"p.`{c}`")) ** 2 for c in features),
              F.lit(0.0))
    truth = (chosen.crossJoin(pool.alias("p"))
             .filter(F.col("q._ex") != F.col("p._id"))
             .groupBy(F.col("q._row").alias("_qrow"))
             .agg(F.sqrt(F.min(gap)).alias("_true")))
    checked = (truth.join(placed.alias("g"), F.col("_qrow") == F.col("g._row"))
               .select("_qrow", "_true", F.col("g._dist").alias("_claimed")))
    wrong = checked.filter(
        F.abs(F.col("_true") - F.col("_claimed"))
        > F.lit(1e-4) * F.greatest(F.col("_true"), F.lit(1.0))).count()
    if wrong:
        worst = checked.orderBy(
            (F.col("_claimed") - F.col("_true")).desc()).first()
        raise RuntimeError(
            f"{wrong} of {sample} audited queries were not placed on their closest "
            f"object: row {worst['_qrow']} claims {worst['_claimed']:.6f} "
            f"where the pool offers {worst['_true']:.6f}")


def _place(spark, win, pool, features, exclude_self: bool = False):
    """Every window of 'win' on its closest object in 'pool', without collecting."""

    queries = win.select(
        F.col(ID).alias("_row"),
        (F.col(ID) if exclude_self else F.lit(-1)).cast("long").alias("_ex"),
        *features)
    placed = (nearest_grid_frames(spark, queries, pool, features)
              .withColumnRenamed("_id", "_core")
              .localCheckpoint(eager=True))
    _audit_placement(queries, pool, placed, features)
    stats = placed.agg(F.count(F.lit(1)).alias("n"),
                       F.max("_dist").alias("hi"),
                       F.avg("_dist").alias("mid")).collect()[0]
    logger.info("placed {:,} queries over the eligible objects, blocked on both sides "
                "| distance to the chosen object: mean {:.4f}, max {:.4f}",
                int(stats["n"]), float(stats["mid"]), float(stats["hi"]))
    return placed


def _score_frames(spark, win, placed, clustered, members, model, cfg, features,
                  eps) -> np.ndarray:
    """Score every window of one part on the task that owns the cluster it landed in."""

    d, conf = model.d, dict(cfg)
    cap, cut = cfg["max_members"], float(eps)
    hops = int(cfg["layers"])     # with L convolutions the centre reads its L-hop ball
    b_state = spark.sparkContext.broadcast(
        {k: v.detach().numpy() for k, v in model.net.state_dict().items()})
    queries = query_frame(placed, clustered, win, features)
    schema = StructType([StructField("_row", LongType(), False),
                         StructField("_s", DoubleType(), True)])

    def work(qpd: pd.DataFrame, mpd: pd.DataFrame) -> pd.DataFrame:
        rows, egos = egos_from_rows(mpd, qpd, cut, cap, max_hops=hops)
        if not len(egos):
            return pd.DataFrame({"_row": [], "_s": []})
        local = GraphAutoEncoder(d, conf)
        local.net.load_state_dict({k: torch.from_numpy(v)
                                   for k, v in b_state.value.items()})
        return pd.DataFrame({"_row": rows,
                             "_s": _score(local, egos, conf)})

    got = (queries.groupBy("_cluster").cogroup(members.groupBy("_cluster"))
           .applyInPandas(work, schema=schema).toPandas())
    b_state.unpersist()

    n = win.count()
    out = np.full(n, np.nan)
    out[got["_row"].to_numpy().astype("int64")] = got["_s"].to_numpy()
    missing = int(np.isnan(out).sum())
    if missing:
        logger.warning("  {:,} of {:,} windows came back without a score", missing, n)
        out = np.where(np.isnan(out), np.nanmax(out) * 1.01, out)
    logger.info("scored {:,} windows on the tasks that own their clusters", n)
    return out


def train_dencad(split_dir: Path | str,
                 reports_dir: Path | str = Path("reports"),
                 window_steps: Optional[int] = None,
                 eps: Optional[float] = None,
                 stage: str = "select",
                 selected: Optional[dict] = None,
                 **overrides) -> dict:
   
    if stage not in ("select", "final"):
        raise ValueError(f"stage must be 'select' or 'final', not {stage!r}")
    if stage == "final":
        if not selected:
            raise ValueError("stage='final' needs the entry a search chose: its "
                             "configuration, its epoch count and its threshold")
        overrides = {**selected.get("config", {}), **overrides}

    split_dir, reports_dir = Path(split_dir), Path(reports_dir)
    dataset, files = W.discover(split_dir)
    cfg = {**DEFAULTS, **model_config("dencad", dataset, split_dir.name),
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

    if int(cfg["replicas"]) < 2:
        raise ValueError(
            f"replicas={cfg['replicas']}: the fit is data-parallel only. The single"
            "-process path, the driver-side builder and the broadcast scoring were "
            "removed, so there is nothing left to run in one process -- the shards a "
            "replica reads are written by Spark and a lone replica would still read "
            "one of them")

    logger.info("dataset {} from {} | window {} readings | eps {} | minPts {}",
                dataset, split_dir, window_steps, eps, cfg["min_pts"])
    logger.info("contextual subgraphs: {} members, {} hops, centre {} | "
                "autoencoder {}x{} latent {}",
                cfg["max_members"] or "whole cluster", cfg["layers"],
                "masked" if cfg["mask_centre"] else "kept in the input",
                cfg["layers"], cfg["hidden"], cfg["latent"])

    spark = W.session(f"dencad-{dataset}", cfg["workers"])
    graph = NeighbourhoodGraph(cfg["bits"], cfg["permutations"], cfg["band"],
                               eps=eps, width=width, seed=cfg["seed"],
                               workers=cfg["workers"])
    try:
        # ---------------------------------------------------------- windowing
        declared = declared_categorical(dataset)
        parts = PARTS if stage == "final" else ("train", "validation")
        win, labels, features, label = W.windowed_parts(
            spark, files, parts, window_steps, declared,
            min_coverage=min_coverage, scale=cfg["scale_windows"])

        replicas = int(cfg["replicas"])
        on_ap = stage == "select"
        if stage == "final":
            cfg["fixed_epochs"] = int(selected["epoch"])
            extra = (win["validation"] if cfg.get("contaminated_refit")
                     else win["validation"].filter(F.col(label) == 0))
            n_extra = extra.count()
            win["train"] = W.number_rows(
                win["train"].drop(ID).unionByName(extra.drop(ID)))
            got = win["train"].select(ID, label).orderBy(ID).toPandas()
            labels["train"] = (got[label].to_numpy() != 0).astype("uint8")
            logger.info("refit: {:,} normal windows of validation join the graph, "
                        "{:,} rows in all, {} epochs, no stopping",
                        n_extra, len(labels["train"]), cfg["fixed_epochs"])

        # ------------------------------------------------ graph over the train

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

        # ------------------------------------------------ contextual subgraphs
      
        take = cfg["train_examples"]
        model = GraphAutoEncoder(len(features), cfg)

        if cfg["attach_to"] == "core":
            pool_nodes = clustered.join(cores_df, C.NODE).select(C.NODE)
        elif cfg["attach_to"] == "clustered":
            pool_nodes = clustered.select(C.NODE)
        else:
            raise ValueError("attach_to='any' has no distributed form: a noise "
                             "object has no cluster to be the context")

        members = S.member_frame(win["train"], clustered, e_df,
                                 features).localCheckpoint(eager=True)
        pool = _pool_frame(win["train"], pool_nodes,
                           features).localCheckpoint(eager=True)
        placed = _place(spark, win["train"], pool, features, exclude_self=True)
        queries = S.query_frame(placed, clustered, win["train"], features,
                                leave_out=True)
        if take:
            queries = queries.sample(False, min(1.0, take / len(labels["train"])),
                                     seed=cfg["seed"])
        shard_dir = Path(tempfile.mkdtemp(prefix="dencad_shards_"))
        vqueries = None
        if on_ap:
            vqueries = S.query_frame(
                _place(spark, win["validation"], pool, features),
                clustered, win["validation"], features, label=label)
        S.write_shards(shard_dir, replicas, queries, members, features,
                       validation=vqueries,
                       meta={"dataset": dataset,
                             "eps": float(eps)})
        state_dict, best_ap, best_epoch, train_loss, spread = fit_shards(
            shard_dir, cfg, len(features), replicas,
            eps=eps, on_ap=on_ap)
        model.net.load_state_dict(state_dict)
       
        shutil.rmtree(shard_dir, ignore_errors=True)
        logger.success("autoencoder fitted, best validation AP {:.5f}", best_ap)

       
        scores = {part: _score_frames(
            spark, win[part], _place(spark, win[part], pool, features),
            clustered, members, model, cfg, features, eps) for part in parts[1:]}
    finally:
        spark.stop()

    mu, sd = spread
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


    logger.info("training loss at the chosen epoch {:.6f}", train_loss)
    results = {}
    for part in [p for p in parts if p != "train"]:
        m = metrics(labels[part], z[part], threshold=carried)
        results[part] = m
        logger.success("  {:<11} ROC-AUC {:.4f}  AP {:.4f}  F1 {:.4f}  "
                       "segments {}/{}", part, m["roc_auc"],
                       m["average_precision"], m["f1"], m["segments_caught"],
                       m["segments"])

    report = save_report(
        reports_dir / f"dencad_{split_dir.name}_{window_steps}steps_{stage}.json",
        "DENCAD", dataset, results, carried,
        {"window_steps": int(window_steps), "eps": float(eps), "width": float(width), **{k: cfg[k] for k in ("bits", "permutations", "band", "min_pts", "label_change_rate", "attach_to", "max_members", "mask_centre", "hidden", "latent", "layers", "learning_rate", "batch_size", "patience", "replicas", "seed")}},
        {"stage": stage, "epoch": int(best_epoch), "train_loss": train_loss,
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
            "epoch": d.get("epoch", 0)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DENCAD: anomaly detection over DENCAST clusters, on windowed telemetry.")
    ap.add_argument("--split_dir", type=Path, default=Path("data/processed/swat"),
                    help="directory holding *_train/_validation/_test.parquet")
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--window-steps", type=int, default=None,
                    help="readings per window; the interval is that many sampling "
                         "periods, measured from the data")
    ap.add_argument("--eps", type=float, default=None,
                    help="overrides the per-dataset value in params.yaml")
    ap.add_argument("--min-pts", type=int, default=None)
    ap.add_argument("--max-members", type=int, default=None,
                    help="cap on the cluster; omitted the whole cluster is used")
    ap.add_argument("--mask-centre", action="store_true", default=None,
                    help="hide the centre's attributes from the input")
    ap.add_argument("--latent", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--train-examples", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None,
                    help="epochs without improvement before the fit stops")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--replicas", type=int, default=None)
    ap.add_argument("--stage", choices=("select", "final"), default="final",
                    help="select: validation only, the test is never read. "
                         "final: refit on train + validation's normals, score the test")
    ap.add_argument("--selected", type=Path, default=None,
                    help="for --stage final: the search report to take the winning "
                         "configuration, epoch count and threshold from")
    ap.add_argument("--contaminated-refit", action="store_true", default=None,
                    help="keep the positives of validation in the final refit")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    train_dencad(a.split_dir, a.reports_dir, a.window_steps, a.eps,
                  min_pts=a.min_pts, max_members=a.max_members,
                  mask_centre=a.mask_centre, latent=a.latent, layers=a.layers,
                  train_examples=a.train_examples, workers=a.workers,
                contaminated_refit=a.contaminated_refit,
                  patience=a.patience, max_epochs=a.max_epochs,
                  replicas=a.replicas, stage=a.stage,
                  selected=(_carry(a.selected)
                            if a.selected else None))


if __name__ == "__main__":
    main()


__all__ = ["train_dencad", "GraphAutoEncoder", "DEFAULTS"]
