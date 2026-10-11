"""DENCAD: anomaly detection on DENCAST's clusters, with a graph autoencoder.

Two names, deliberately. DENCAST is the published algorithm -- the LSH neighbourhood
graph, the core-object test and the max-id propagation, which live in `lsh` and
`clustering` and are reimplemented faithfully. DENCAD is what this module adds and is
not in the paper: each window is scored by how badly the cluster it lands in
reconstructs it, so the clustering becomes a detector. Keeping the names apart keeps it
clear which results belong to the paper and which to the extension.

The whole chain on one split directory, with every setting read from the `dencad`
section of params.yaml so that a variant is a change to that file rather than to this
one.

**The steps.**

1. Window the rows. A numeric column contributes its mean and standard deviation, a
   categorical one its mode, entropy, number of states and dominant share.
2. Build the neighbourhood graph over the training windows with LSH -- random
   hyperplanes, permuted signatures, a sliding band, and an exact cosine filter.
3. Identify core objects: degree at least `min_pts`, which is DBSCAN's test with the
   cosine threshold standing in for `eps`.
4. Cluster by propagation, Algorithm 2: every core starts with an id of its own, the
   largest id received wins, and the loop stops below `label_change_rate` of the edges.
5. Place every window -- training ones too -- on its most similar core or border
   object, by `argmax` and with no threshold, and cut out that object's cluster as the
   contextual subgraph.
6. Train a graph autoencoder to rebuild the centre's attributes, and score a window by
   how badly it is rebuilt.

**Training examples are built exactly like inference examples.** A training node sits
inside its cluster with all its edges while a new window arrives attached by one, and a
model trained on the first and asked about the second reads that difference for every
new window, anomalous or not. So each training node is detached -- every edge removed,
itself excluded from the pool of candidate cores -- and re-placed as an unseen window
would be. The core it then finds may belong to another cluster, which is what would
happen to a new window too.

**The centre is left in the input by default.** With `mask_centre: true` it is hidden
and the error measures whether the window is predictable from its neighbours; left
visible, the latent is what stops the model copying it and the error measures distance
from the learned manifold -- the same quantity a plain autoencoder measures, with the
neighbourhood as extra evidence. Both were measured and they are close; the second has
the better average precision.

Spark does the windowing, the graph, the clustering and the placement. The autoencoder
is PyTorch on one machine: the contextual subgraphs are independent of each other, so this step is
data-parallel in principle and would distribute with `TorchDistributor`, but at these
window counts it takes minutes and the complication would buy nothing.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd

from dencast.modeling import clustering as C
from dencast.modeling import windowing as W
from dencast.modeling.subgraphs import ContextualSubgraphBuilder
from dencast.modeling.evaluate import metrics
from dencast.modeling.lsh import ID, NeighbourhoodGraph
from dencast.utils import Params, PARAMS_PATH, declared_categorical

PARTS = ("train", "validation", "test")


def dencad_config() -> dict:
    """The `dencad` section of params.yaml, or an empty dict if it cannot be read."""
    try:
        section = Params.load(PARAMS_PATH).get("dencad")
        return section.to_dict() if hasattr(section, "to_dict") else (section or {})
    except Exception as exc:
        logger.warning("params.yaml unreadable ({}): falling back to the built-in "
                       "defaults", exc)
        return {}


DEFAULTS = {
    "bits": 16, "permutations": 10, "band": 20, "min_pts": 10,
    "label_change_rate": 0.05, "attach_to": "clustered", "max_members": None,
    "max_hops": None,
    "mask_centre": False, "hidden": 64, "latent": 16, "layers": 2,
    "learning_rate": 1e-3, "batch_size": 128, "max_epochs": 60, "patience": 8,
    "holdout": 0.1, "train_examples": None, "scale_windows": True,
    # "validation_ap" or "held_out_loss". The second optimises against the task past a
    # point: see `_fit`.
    "early_stopping": "validation_ap",
    # Validation and test scored on the task that owns each query's cluster, instead
    # of one subgraph at a time in this process.
    "scoring_by_cluster": True,
    # Place every query over Spark tasks rather than in this process. Same answers.
    "placement_in_spark": True,
    # True splits the data across tasks; False copies it to every executor. The answers
    # are the same, the broadcast is faster on one machine and cannot outgrow it.
    "partition_data": True,
    # 1 fits in this process. Above that, `parallel` spawns that many gloo replicas and
    # averages their gradients, which is what Horovod does for the same reason and does
    # not reproduce the single-process weights -- see that module.
    "replicas": 4,
    # Adam's rate times the replica count, so the fit is not penalised for the larger
    # effective batch N replicas produce.
    "scale_learning_rate": True,
    "seed": 42, "workers": 4,
}


_NET = None


def _net_class():
    """The autoencoder as a real `nn.Module`, with the convolution in its `forward`.

    Defined on first call rather than at import, because torch is imported lazily
    throughout this module. It carries a `forward` of its own for one reason:
    `DistributedDataParallel` arms the hooks that average the gradients inside the
    forward of the wrapper, so a backward whose graph started at `replica.module` is
    never synchronised. The layers used to hang off a bare `nn.Module`, which has no
    forward to wrap, and `parallel` had no choice but to reach for `.module`.
    """
    global _NET
    if _NET is not None:
        return _NET

    import torch
    import torch.nn as nn

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
    """A stack of graph convolutions over a dense adjacency, decoding the centre.

    Dense rather than sparse because an contextual subgraph is one cluster: a few tens of nodes
    typically and a few hundred at worst, so `A_hat @ X @ W` is the whole convolution
    and PyTorch Geometric is not needed. `layers` beyond three over-smooths -- the node
    representations converge towards each other and the centre stops standing out,
    which is precisely what this has to detect.
    """

    def __init__(self, d: int, cfg: dict):
        import torch
        import torch.nn as nn

        # Seeded here and not in `_fit`, which is where it used to be: `nn.Linear`
        # draws its weights the moment it is constructed, so seeding afterwards left
        # the initialisation to whatever entropy torch had picked up at import. Two
        # runs over an identical graph gave held-out 0.35330 and 0.36637, and 11 attack
        # segments against 10, with nothing else changed.
        torch.manual_seed(cfg["seed"])

        self.d = d
        self.cfg = cfg
        self.net = _net_class()(d, cfg)

    def forward(self, X, A_hat, centre):
        return self.net(X, A_hat, centre)


def _normalise(A, mask, degrees=None):
    """Symmetric normalisation of A + I over the real nodes only.

    `degrees` overrides the row sums, and is how hop pruning stays exact: a node cut at
    the edge of the ball has lost edges it really has, so its row sum is too small and
    every weight on its edges would come out too large. Passed in, the degrees are the
    ones the whole cluster gives, and the centre's output is the same number it would
    have been without pruning.
    """
    import torch

    eye = torch.eye(A.shape[1]).unsqueeze(0) * mask.unsqueeze(2)
    A = A + eye
    raw = A.sum(2) if degrees is None else degrees
    deg = raw.clamp(min=1e-6).pow(-0.5) * mask
    return A * deg.unsqueeze(2) * deg.unsqueeze(1)


def _batches(egos, indices, d: int, size: int, mask_centre: bool) -> list:
    """Padded batches of similar-sized contextual subgraphs, built once.

    Sorted by node count so padding tracks the batch rather than the global maximum:
    the median contextual subgraph is a few dozen nodes and the largest a few hundred.

    A list and not a generator, and the normalisation is folded in here. Both for the
    same reason: none of this depends on the model. Rebuilding it per epoch meant the
    padding loop ran 973,000 times over a 51-epoch fit on HAI and `A + I` was symmetric
    normalised 51 times to the same numbers, which is where three of the pipeline's
    eleven minutes went -- not in `A_hat @ X @ W`, which on a median of 17 nodes is
    nothing. The order is unchanged, so the fit is the same to the last bit.
    """
    import torch

    out = []
    indices = sorted(indices, key=lambda i: len(egos[i].features))
    for s in range(0, len(indices), size):
        chunk = indices[s:s + size]
        b = len(chunk)
        k = np.fromiter((len(egos[i].features) for i in chunk), dtype="int64",
                        count=b)
        m = int(k.max())

        # Padded in numpy and converted once per batch. The per-example version issued
        # five torch calls each -- two slice assignments, a scalar store and two
        # `from_numpy` -- which is 95,000 of them over HAI's 19,082 examples and cost
        # 37 s, more than the distributed clustering and the whole LSH together. A
        # numpy slice assignment is the same copy without the dispatch.
        X = np.zeros((b, m, d), dtype="float32")
        A = np.zeros((b, m, m), dtype="float32")
        # Carried only when a subgraph was pruned by hops, which is when its row sums
        # no longer give the right degrees.
        has_deg = any(egos[i].degrees is not None for i in chunk)
        D = np.zeros((b, m), dtype="float32") if has_deg else None
        for j, i in enumerate(chunk):
            e = egos[i]
            kk = int(k[j])
            X[j, :kk] = e.features
            A[j, :kk, :kk] = e.adjacency
            if has_deg:
                D[j, :kk] = (e.degrees if e.degrees is not None
                             else e.adjacency.sum(1) + 1.0)

        # The three that needed no per-example work at all: the mask is a comparison,
        # the centre is read straight off the subgraphs, and the target is a row of X
        # that is already there -- taken before the centre is optionally blanked.
        mask = (np.arange(m)[None, :] < k[:, None]).astype("float32")
        centre = np.fromiter((egos[i].centre for i in chunk), dtype="int64", count=b)
        target = X[np.arange(b), centre].copy()
        if mask_centre:
            X[np.arange(b), centre] = 0.0

        tA = torch.from_numpy(A)
        tmask = torch.from_numpy(mask)
        tdeg = None if D is None else torch.from_numpy(D)
        out.append((torch.from_numpy(X), _normalise(tA, tmask, tdeg), tmask,
                    torch.from_numpy(centre), torch.from_numpy(target), chunk))
    return out


def _fit(model, egos, cfg, watch=None) -> float:
    """Train on the contextual subgraphs, stopping on whichever criterion `cfg` names.

    `watch` is `(validation subgraphs, validation labels)`, and when `early_stopping`
    asks for it the fit stops on validation average precision instead of on the
    reconstruction loss of a slice held out of train.

    The reason is measured, not stylistic. Across four configurations the held-out loss
    ran from 0.306 to 0.411 and the test average precision from 0.525 to 0.569, and the
    two did not move together: the best-converged model was the worst detector. That is
    not a paradox. The reconstruction error only separates an anomaly if the manifold
    the model learned is tight enough that a point outside it reconstructs badly; a
    better-converged autoencoder of the same capacity generalises further, and starts
    reconstructing the attacks too. Minimising the held-out loss optimises against the
    task after a point, and nothing in the loss says where that point is.

    Watching validation does make the fit weakly supervised, which the threshold and
    `min_similarity` already were. It is worth saying plainly rather than letting
    "one-class" imply the labels touched nothing before the threshold.
    """
    import torch

    opt = torch.optim.Adam(model.net.parameters(), lr=cfg["learning_rate"])
    idx = np.arange(len(egos))
    np.random.default_rng(cfg["seed"]).shuffle(idx)
    cut = int((1 - cfg["holdout"]) * len(idx))
    fit, dev = idx[:cut].tolist(), idx[cut:].tolist()

    fit_batches = _batches(egos, fit, model.d, cfg["batch_size"], cfg["mask_centre"])
    dev_batches = _batches(egos, dev, model.d, cfg["batch_size"], cfg["mask_centre"])
    logger.info("  {:,} fitting batches, {:,} held out, padded and normalised once",
                len(fit_batches), len(dev_batches))

    on_ap = cfg["early_stopping"] == "validation_ap" and watch is not None
    if cfg["early_stopping"] == "validation_ap" and watch is None:
        raise ValueError("early_stopping is 'validation_ap' but no validation "
                         "subgraphs were handed to the fit")
    order = np.arange(len(fit_batches))
    shuffler = np.random.default_rng(cfg["seed"])
    label = "validation AP" if on_ap else "held-out"
    logger.info("  stopping on {}", label)

    # A better score is larger for AP and smaller for a loss, so one sign carries the
    # difference and the loop below stays single.
    sign = 1.0 if on_ap else -1.0
    best, waited, state = -np.inf, 0, None
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.net.train()
        shuffler.shuffle(order)
        for j in order:
            X, A_hat, mask, centre, target, _ = fit_batches[j]
            opt.zero_grad()
            ((model.forward(X, A_hat, centre) - target) ** 2).mean().backward()
            opt.step()

        model.net.eval()
        if on_ap:
            seen = float(metrics(watch[1], _score(model, watch[0], cfg))[
                "average_precision"])
        else:
            with torch.no_grad():
                seen = sum(((model.forward(X, A_hat, centre) - target) ** 2
                            ).mean().item() * len(centre)
                           for X, A_hat, mask, centre, target, _ in dev_batches)
            seen /= max(len(dev), 1)

        if epoch % 5 == 0 or epoch == 1:
            logger.info("  epoch {:>3}: {} {:.5f}", epoch, label, seen)
        if sign * seen > best + 1e-5:
            best, waited = sign * seen, 0
            state = {k: v.clone() for k, v in model.net.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg["patience"]:
                logger.info("  stopped at epoch {}", epoch)
                break
    model.net.load_state_dict(state)
    return float(sign * best)


def _score(model, egos, cfg, fill: bool = True) -> np.ndarray:
    """Reconstruction error of the centre, one number per contextual subgraph.

    `fill=False` leaves a query that could not be placed as NaN. The replacement value
    is the largest error seen, which is a statistic over every query, so a worker
    holding one cluster's share must not compute it: that is the driver's job once the
    shares are back.
    """
    import torch

    out = np.full(len(egos), np.nan)
    present = [i for i, e in enumerate(egos) if e is not None]
    model.net.eval()
    with torch.no_grad():
        for X, A_hat, mask, centre, target, chunk in _batches(
                egos, present, model.d, cfg["batch_size"], cfg["mask_centre"]):
            err = ((model.forward(X, A_hat, centre) - target) ** 2).mean(1)
            out[chunk] = err.numpy()
    if fill and np.isnan(out).any():
        out = np.where(np.isnan(out), np.nanmax(out) * 1.01, out)
    return out


def _score_by_cluster(spark, builder, queries: np.ndarray, model, cfg) -> np.ndarray:
    """Build and score every query on the task that owns its cluster.

    The structure of the problem is already partitioned: a query is scored inside one
    cluster and inside nothing else, so grouping the queries by the cluster they landed
    in puts each task's work entirely in front of it. No shuffle of the subgraphs, and
    each task touches only the clusters it was given.

    What travels is the builder, by broadcast -- 8.7 MB of training features on HAI,
    45 MB on ESA -- rather than per-cluster slices of it. That is deliberate: the worker
    then calls the same `build` the driver would, so the scores are the same numbers
    instead of nearly the same. Cutting the features down per cluster means renumbering
    the member indices, and an error there would produce plausible scores that are
    quietly wrong. The parallelism is in the work, which is where the time is; the
    memory per executor is a few tens of megabytes.
    """
    import torch
    from pyspark.sql.types import (DoubleType, LongType, StructField, StructType)

    # Placed on the driver: exact, tiled, and under a second, and every query's cluster
    # has to be known before the queries can be grouped by it.
    cores, sims = builder.nearest_core(queries)
    clusters = builder.cluster_of[cores]

    sc = spark.sparkContext
    held = [sc.broadcast(builder.without_session()), sc.broadcast(queries),
            sc.broadcast({k: v.detach().numpy()
                          for k, v in model.net.state_dict().items()})]
    b_builder, b_queries, b_state = held
    d, conf = model.d, dict(cfg)

    schema = StructType([StructField("_row", LongType(), False),
                         StructField("_s", DoubleType(), True)])

    def work(pdf: pd.DataFrame) -> pd.DataFrame:
        bld, Q = b_builder.value, b_queries.value
        rows = pdf["_row"].to_numpy()
        egos = [bld.build(Q[int(r)], int(c), float(v))
                for r, c, v in zip(rows, pdf["_core"], pdf["_sim"])]
        local = GraphAutoEncoder(d, conf)
        local.net.load_state_dict({k: torch.from_numpy(v)
                                   for k, v in b_state.value.items()})
        return pd.DataFrame({"_row": rows,
                             "_s": _score(local, egos, conf, fill=False)})

    plan = pd.DataFrame({"_row": np.arange(len(queries), dtype="int64"),
                         "_cluster": clusters.astype("int64"),
                         "_core": cores.astype("int64"), "_sim": sims})
    got = (spark.createDataFrame(plan)
           .groupBy("_cluster").applyInPandas(work, schema=schema)
           .toPandas().sort_values("_row", kind="stable"))
    for b in held:
        b.unpersist()

    out = np.full(len(queries), np.nan)
    out[got["_row"].to_numpy()] = got["_s"].to_numpy()
    if np.isnan(out).any():
        out = np.where(np.isnan(out), np.nanmax(out) * 1.01, out)
    logger.info("scored {:,} queries over {:,} clusters, partitioned by cluster",
                len(out), int(len(np.unique(clusters))))
    return out


def _score_by_cluster_shuffled(spark, builder, queries: np.ndarray, model,
                               cfg) -> np.ndarray:
    """The same scores, with the cluster data shuffled to the tasks instead of copied.

    `_score_by_cluster` broadcasts the builder, so every executor holds the whole
    training matrix. That partitions the work but not the data, and it has a ceiling:
    the day the matrix does not fit in an executor the design stops rather than slows.

    Here a cluster's members travel as rows, keyed by the cluster, and the queries that
    landed in that cluster travel with the same key. A task is handed one cluster's
    members, their neighbour lists and its queries, and nothing else -- so the memory a
    task needs follows the size of the largest cluster, not of the dataset.

    The worry that kept the broadcast was renumbering: a worker holding one cluster
    cannot use the global row indices `build` expects. Rather than reimplement the
    construction with local indices, which would be a second version to keep in step,
    the worker assembles a one-cluster builder -- its members are the whole world, all
    of them core, all in cluster 1 -- and calls the same `build`. The only translation
    is a map from global member id to position, applied to the chosen core and to the
    edges.

    The model's weights are still broadcast, and that is not the same compromise: a
    model is small by definition, and every task needs all of it.
    """
    import torch
    from pyspark.sql.types import (ArrayType, DoubleType, FloatType, LongType,
                                   StructField, StructType)

    from dencast.modeling.subgraphs import ContextualSubgraphBuilder

    cores, sims = builder.nearest_core(queries)
    clusters = builder.cluster_of[cores]
    if (clusters == 0).any():
        raise RuntimeError(f"{int((clusters == 0).sum()):,} queries chose an object "
                           "with no cluster; this path has no context to give them")

    d, conf = model.d, dict(cfg)
    # Resolved here, not read from `cfg` in the worker: `cfg["min_similarity"]` is a
    # dict keyed by dataset, and the builder is the thing that already holds the float
    # this run settled on.
    min_sim = float(builder.min_similarity)
    cap = builder.max_members
    b_state = spark.sparkContext.broadcast(
        {k: v.detach().numpy() for k, v in model.net.state_dict().items()})

    # Neighbour lists per member, taken from the per-cluster edges the builder already
    # holds, once on the driver. Only clustered objects travel: 3,859 of HAI's 19,082.
    member_rows = []
    for c, idx in builder.members.items():
        neighbours: dict[int, list[int]] = {}
        for u, v in builder.cluster_edges[c]:
            neighbours.setdefault(int(u), []).append(int(v))
            neighbours.setdefault(int(v), []).append(int(u))
        for node in idx:
            member_rows.append((int(c), int(node),
                                builder.X[node].astype("float32").tolist(),
                                neighbours.get(int(node), [])))
    members = spark.createDataFrame(
        pd.DataFrame(member_rows, columns=["_cluster", "_node", "_f", "_nb"]),
        schema=StructType([StructField("_cluster", LongType(), False),
                           StructField("_node", LongType(), False),
                           StructField("_f", ArrayType(FloatType()), False),
                           StructField("_nb", ArrayType(LongType()), False)]))

    plan = spark.createDataFrame(
        pd.DataFrame({"_cluster": clusters.astype("int64"),
                      "_row": np.arange(len(queries), dtype="int64"),
                      "_core": cores.astype("int64"), "_sim": sims,
                      "_f": [v.astype("float32").tolist() for v in queries]}),
        schema=StructType([StructField("_cluster", LongType(), False),
                           StructField("_row", LongType(), False),
                           StructField("_core", LongType(), False),
                           StructField("_sim", DoubleType(), False),
                           StructField("_f", ArrayType(FloatType()), False)]))

    schema = StructType([StructField("_row", LongType(), False),
                         StructField("_s", DoubleType(), True)])

    def work(qpd: pd.DataFrame, mpd: pd.DataFrame) -> pd.DataFrame:
        if qpd.empty or mpd.empty:
            return pd.DataFrame({"_row": [], "_s": []})
        mpd = mpd.sort_values("_node", kind="stable")
        nodes = mpd["_node"].to_numpy()
        where = {int(n): i for i, n in enumerate(nodes)}
        X = np.stack([np.asarray(f, dtype="float32") for f in mpd["_f"]])

        pairs = [(where[int(u)], where[int(v)])
                 for u, nb in zip(nodes, mpd["_nb"]) for v in nb
                 if int(v) in where and int(u) < int(v)]
        edges = (np.asarray(pairs, dtype="int64") if pairs
                 else np.zeros((0, 2), dtype="int64"))

        # One cluster is the worker's whole world: every member is core and in it.
        mini = ContextualSubgraphBuilder(
            X, np.ones(len(X), dtype="int64"), np.ones(len(X), dtype=bool), edges,
            min_similarity=min_sim, max_members=cap, attach_to="clustered",
            max_hops=conf["max_hops"])

        egos = [mini.build(np.asarray(f, dtype="float32"), where[int(c)], float(v))
                for f, c, v in zip(qpd["_f"], qpd["_core"], qpd["_sim"])]
        local = GraphAutoEncoder(d, conf)
        local.net.load_state_dict({k: torch.from_numpy(v)
                                   for k, v in b_state.value.items()})
        return pd.DataFrame({"_row": qpd["_row"].to_numpy(),
                             "_s": _score(local, egos, conf, fill=False)})

    got = (plan.groupBy("_cluster").cogroup(members.groupBy("_cluster"))
           .applyInPandas(work, schema=schema)
           .toPandas().sort_values("_row", kind="stable"))
    b_state.unpersist()

    if len(got) != len(queries):
        raise RuntimeError(f"{len(got)} scores came back for {len(queries)} queries")
    out = np.full(len(queries), np.nan)
    out[got["_row"].to_numpy()] = got["_s"].to_numpy()
    if np.isnan(out).any():
        out = np.where(np.isnan(out), np.nanmax(out) * 1.01, out)
    logger.info("scored {:,} queries over {:,} clusters, the members shuffled to the "
                "tasks rather than copied to every executor", len(out),
                int(len(np.unique(clusters))))
    return out


def _unit_columns(df, features, keep):
    """`df` reduced to `keep` plus its features divided by the row's norm.

    Placement compares cosines, so the normalisation has to happen before the blocks
    meet: a task holding a block of each side must be able to take a dot product and
    call it a similarity. Done here in Spark, column by column, rather than by bringing
    the rows down to numpy for a `/ norm`.
    """
    from pyspark.sql import functions as F

    norm = F.sqrt(sum((F.col(c) * F.col(c) for c in features), F.lit(0.0)))
    safe = F.greatest(norm, F.lit(1e-12))
    return df.select(*keep, *[(F.col(c) / safe).alias(c) for c in features])


def _pool_frame(train, pool_nodes, features):
    """The eligible training objects, with their features, ready to be placed against.

    Built once from the training windows and reused for every part, because that is what
    it is: the objects already in the graph. It used to be derived inside `_place` from
    whatever frame was being placed, so validation and test were matched against rows of
    their own part whose ids happened to equal training node ids -- a pool of the right
    size and the wrong objects, which placed every validation window on a cosine higher
    than the true maximum and was caught by checking the argmax against the pool itself.
    """
    from pyspark.sql import functions as F

    return _unit_columns(pool_nodes.join(train, pool_nodes[C.NODE] == train[ID]),
                         features, [F.col(C.NODE).alias("_id")])


AUDIT_QUERIES = 64


def _audit_placement(queries, pool, placed, features, sample: int = AUDIT_QUERIES):
    """Re-derive the argmax for a handful of queries and insist it is the one claimed.

    The blocked placement is a distributed argmax, and a distributed argmax can be
    wrong in ways that look like a worse model rather than like a failure: a cogroup key
    that never matches leaves queries unplaced (it did, once, for 9,541 of 19,082), and
    a pool built from the wrong frame places every query on an object that is not in the
    graph (it did, for every validation window, until it was caught by exactly this
    comparison run by hand).

    So it runs every time, on `sample` queries chosen at random: their cosine against
    the whole pool, reduced in Spark, against the cosine the placement recorded. Costs
    one small job -- 64 queries against a few thousand objects -- and nothing is
    collected but a count.
    """
    from pyspark.sql import functions as F

    chosen = queries.orderBy(F.rand(42)).limit(sample).alias("q")
    dot = sum((F.col(f"q.`{c}`") * F.col(f"p.`{c}`") for c in features), F.lit(0.0))
    truth = (chosen.crossJoin(pool.alias("p"))
             .filter(F.col("q._ex") != F.col("p._id"))
             .groupBy(F.col("q._row").alias("_qrow"))
             .agg(F.max(dot).alias("_true")))
    checked = (truth.join(placed.alias("g"), F.col("_qrow") == F.col("g._row"))
               .select("_qrow", "_true", F.col("g._sim").alias("_claimed")))
    wrong = checked.filter(
        F.abs(F.col("_true") - F.col("_claimed")) > F.lit(1e-5)).count()
    if wrong:
        worst = checked.orderBy(
            (F.col("_true") - F.col("_claimed")).desc()).first()
        raise RuntimeError(
            f"{wrong} of {sample} audited queries were not placed on their most "
            f"similar object: row {worst['_qrow']} claims {worst['_claimed']:.6f} "
            f"where the pool offers {worst['_true']:.6f}")


def _place(spark, win, pool, features, exclude_self: bool = False):
    """Every window of `win` on its most similar object in `pool`, without collecting.

    The DataFrame counterpart of `builder.nearest_core`: the queries are the windows of
    one part, the pool is `_pool_frame`'s eligible training objects, and the answer
    stays a DataFrame of `_row`, `_core` and `_sim` that the next step joins against.
    Nothing here has a row per window on the driver.

    `exclude_self` carries each query's own id as the object it may not choose, which is
    what makes a training window look like an unseen one -- it travels in the query's
    own row, so whichever task holds that query applies it to whichever block of the
    pool it was given.
    """
    from pyspark.sql import functions as F

    from dencast.modeling.placement import nearest_grid_frames

    queries = _unit_columns(
        win, features,
        [F.col(ID).alias("_row"),
         (F.col(ID) if exclude_self else F.lit(-1)).cast("long").alias("_ex")])
    placed = (nearest_grid_frames(spark, queries, pool, features)
              .withColumnRenamed("_id", "_core")
              .localCheckpoint(eager=True))
    _audit_placement(queries, pool, placed, features)
    stats = placed.agg(F.count(F.lit(1)).alias("n"),
                       F.min("_sim").alias("lo"),
                       F.avg("_sim").alias("mid")).collect()[0]
    logger.info("placed {:,} queries over the eligible objects, blocked on both sides "
                "| cosine with the chosen object: mean {:.4f}, min {:.4f}",
                int(stats["n"]), float(stats["mid"]), float(stats["lo"]))
    return placed


def _score_frames(spark, win, placed, clustered, members, model, cfg, features,
                  min_similarity) -> np.ndarray:
    """Score every window of one part on the task that owns the cluster it landed in.

    The same arrangement as `_score_by_cluster_shuffled`, with the rows coming from
    DataFrames instead of from a builder on the driver: the queries are grouped by the
    cluster they were placed in, the members of that cluster are grouped the same way,
    and the task holding both builds the subgraphs and scores them. The model's weights
    are broadcast, which is the one copy worth making -- a model is small by definition
    and every task needs all of it.
    """
    import torch
    from pyspark.sql.types import DoubleType, LongType, StructField, StructType

    from dencast.modeling.shards import query_frame
    from dencast.modeling.subgraphs import egos_from_rows

    d, conf = model.d, dict(cfg)
    cap, min_sim = cfg["max_members"], float(min_similarity)
    hops = cfg["max_hops"]
    b_state = spark.sparkContext.broadcast(
        {k: v.detach().numpy() for k, v in model.net.state_dict().items()})
    queries = query_frame(placed, clustered, win, features)
    schema = StructType([StructField("_row", LongType(), False),
                         StructField("_s", DoubleType(), True)])

    def work(qpd: pd.DataFrame, mpd: pd.DataFrame) -> pd.DataFrame:
        rows, egos = egos_from_rows(mpd, qpd, min_sim, cap, max_hops=hops)
        if not len(egos):
            return pd.DataFrame({"_row": [], "_s": []})
        local = GraphAutoEncoder(d, conf)
        local.net.load_state_dict({k: torch.from_numpy(v)
                                   for k, v in b_state.value.items()})
        return pd.DataFrame({"_row": rows,
                             "_s": _score(local, egos, conf, fill=False)})

    got = (queries.groupBy("_cluster").cogroup(members.groupBy("_cluster"))
           .applyInPandas(work, schema=schema).toPandas())
    b_state.unpersist()

    n = win.count()
    out = np.full(n, np.nan)
    out[got["_row"].to_numpy().astype("int64")] = got["_s"].to_numpy()
    missing = int(np.isnan(out).sum())
    if missing:
        # A window whose cluster was lost on the way would be scored as the worst thing
        # seen, which is a silent false positive; it has never happened, so it is logged
        # rather than swallowed.
        logger.warning("  {:,} of {:,} windows came back without a score", missing, n)
        out = np.where(np.isnan(out), np.nanmax(out) * 1.01, out)
    logger.info("scored {:,} windows on the tasks that own their clusters", n)
    return out


def train_dencad(split_dir: Path | str,
                  reports_dir: Path | str = Path("reports"),
                  window_steps: Optional[int] = None,
                  min_similarity: Optional[float] = None,
                  **overrides) -> dict:
    """Run the chain end to end and write one report. Returns it."""
    from pyspark.sql import functions as F

    cfg = {**DEFAULTS, **dencad_config(), **{k: v for k, v in overrides.items()
                                              if v is not None}}
    split_dir, reports_dir = Path(split_dir), Path(reports_dir)
    dataset, files = W.discover(split_dir)

    if min_similarity is None:
        by_dataset = cfg.get("min_similarity")
        if isinstance(by_dataset, dict):
            min_similarity = by_dataset.get(dataset)
        elif by_dataset is not None:
            min_similarity = float(by_dataset)
    if min_similarity is None:
        raise ValueError(f"no min_similarity for {dataset!r}: params.yaml lists "
                         f"{sorted((cfg.get('min_similarity') or {}))}")
    kmeans = Params.load(PARAMS_PATH).get("kmeans") or {}
    if window_steps is None:
        window_steps = (kmeans.get("window_steps") or {}).get(dataset, 60)
    # Lives beside `window_steps`, because both describe the windowing rather than
    # either model, and both entry points have to agree on them.
    min_coverage = float(kmeans.get("min_coverage", 0.0))

    if cfg["max_hops"] is not None and int(cfg["max_hops"]) < int(cfg["layers"]):
        raise ValueError(
            f"max_hops={cfg['max_hops']} with layers={cfg['layers']} would throw away "
            "nodes the centre's output depends on. The pruning is exact only at or "
            "above the layer count: with L convolutions the centre reads its L-hop "
            "ball, so cutting at L costs nothing and cutting below it changes the "
            "model without saying so")

    logger.info("dataset {} from {} | window {} readings | cosine >= {} | minPts {}",
                dataset, split_dir, window_steps, min_similarity, cfg["min_pts"])
    logger.info("contextual subgraphs: {} members, {} hops, centre {} | "
                "autoencoder {}x{} latent {}",
                cfg["max_members"] or "whole cluster",
                cfg["max_hops"] or "all",
                "masked" if cfg["mask_centre"] else "kept in the input",
                cfg["layers"], cfg["hidden"], cfg["latent"])

    spark = W.session(f"dencad-{dataset}", cfg["workers"])
    graph = NeighbourhoodGraph(cfg["bits"], cfg["permutations"], cfg["band"],
                               min_similarity=-1.0, seed=cfg["seed"],
                               workers=cfg["workers"])
    try:
        # ---------------------------------------------------------- windowing
        declared = declared_categorical(dataset)
        win, labels, features, label = {}, {}, None, None
        # Fitted on the training windows and reused for the other two parts, like any
        # other statistic taken from train.
        scaling, period = None, None
        for part in PARTS:
            raw = spark.read.parquet(str(files[part].resolve()))
            columns = W.feature_columns(raw.columns)
            label = W.label_column(raw.columns)
            cat = [c for c in columns if c in set(declared)]
            if period is None:
                # Measured on train and reused, like the scaling beside it.
                period = W.sampling_period(raw)
            windowed, features = W.window(raw, columns, window_steps, label, cat,
                                          min_coverage=min_coverage, period=period)
            if cfg["scale_windows"]:
                if part == "train":
                    windowed = windowed.cache()
                    scaling = W.window_statistics(windowed, features)
                windowed = W.apply_scaling(windowed, scaling)
            # The node id of a window is its position in time, and it is assigned in
            # Spark. It used to be its position in a pandas frame, which meant the whole
            # windowed matrix had to reach the driver before the graph could be built --
            # 19,080 x 114 on HAI, and the step that decides the run on ESA.
            win[part] = W.number_rows(windowed.select(W.INDEX, *features, label))
            # The labels do come down, and they are the one thing that should: a label
            # is one number per window, the metrics are computed here, and nothing
            # downstream of the fit can be verified without them.
            got = win[part].select(ID, label).orderBy(ID).toPandas()
            labels[part] = (got[label].to_numpy() != 0).astype("uint8")
            logger.info("  {:<11} {:>7,} windows, {:>6,} positive ({:.2%})",
                        part, len(labels[part]), int(labels[part].sum()),
                        labels[part].mean())

        by_cluster = bool(cfg["scoring_by_cluster"])
        replicas = int(cfg["replicas"])
        on_ap = cfg["early_stopping"] == "validation_ap"
        # The distributed layout: the rows stay in Spark, the shards are written as
        # parquet partitioned by replica, and each replica reads its own. The other path
        # assembles everything from one matrix on the driver, which is what every
        # measurement before this was taken on and what a single-process fit still
        # needs -- `_fit` cannot read a shard it does not have.
        distributed = bool(cfg["partition_data"]) and replicas > 1 and by_cluster

        def collected(part):
            """The windows as one matrix here, for the paths that cannot avoid it."""
            got = win[part].orderBy(ID).toPandas()
            return got[features].to_numpy("float64")

        # ------------------------------------------------ graph over the train
        # Read straight off the windowed frame by column name: the id is already there
        # and the features are already columns, so the detour through pandas and
        # `createDataFrame` is gone.
        signed = graph._signatures(win["train"], features).select(
            ID, "_sig", "_vec", "_norm").cache()
        candidates = graph._candidates(signed)
        n_candidates = candidates.count()
        # Thresholded in Spark. Every candidate used to be collected and then filtered
        # here, which on HAI moved 2,777,541 rows to keep 27,787 -- a hundredfold for
        # nothing, now that placement no longer reads the unthresholded list.
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

        # ------------------------------------------------ contextual subgraphs
        # The LSH built the graph; it does not place objects into it. Placement asks
        # for one answer per query, which is a reduction and therefore affordable
        # exactly: the LSH `attach` cost about 210 s per part here and found 30% of
        # the true edges, while the exact argmax over the eligible objects takes under
        # a second. Training examples go through the same call, because a training node
        # placed by one procedure and a test node by another would hand the model a
        # difference it can read.
        take = cfg["train_examples"]
        model = GraphAutoEncoder(len(features), cfg)
        egos, builder = {}, None

        if distributed:
            from dencast.modeling import shards as S
            from dencast.modeling.parallel import fit_shards

            # Which objects an arriving one may attach to, the same rule the driver-side
            # builder applies: border objects carry a cluster id too, so "clustered"
            # keeps them and "core" throws them away. "any" admits noise objects, whose
            # cluster is 0 and which therefore have no context to give -- the by-cluster
            # scoring has always refused them, and here it cannot even be expressed.
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
                                 "min_similarity": float(min_similarity)})
            state_dict, held_out = fit_shards(
                shard_dir, cfg, len(features), replicas,
                min_similarity=min_similarity, on_ap=on_ap)
            model.net.load_state_dict(state_dict)
            # Only the fit reads the shards; the scoring works off the frames that are
            # still in Spark. Removed on the way out and kept on the way out through an
            # exception, which is when there is something in them worth reading -- and
            # a grid search would otherwise leave one directory per run behind.
            shutil.rmtree(shard_dir, ignore_errors=True)
        else:
            T = collected("train")
            pick = np.arange(len(T)) if not take else np.random.default_rng(
                cfg["seed"]).choice(len(T), size=min(take, len(T)), replace=False)
            core_mask = np.zeros(len(T), dtype=bool)
            core_mask[[r[C.NODE] for r in cores_df.select(C.NODE).collect()]] = True
            cluster_of = (state.toPandas().set_index(C.NODE)[C.CLUSTER]
                          .reindex(range(len(T))).fillna(C.NOISE)
                          .to_numpy().astype("int64"))
            edge_pairs = e_df.select("_src", "_dst").toPandas().to_numpy()
            builder = ContextualSubgraphBuilder(
                T, cluster_of, core_mask, edge_pairs,
                min_similarity=min_similarity, max_members=cfg["max_members"],
                attach_to=cfg["attach_to"], max_hops=cfg["max_hops"])
            if cfg["placement_in_spark"]:
                builder.spark = spark
                builder.partition_pool = bool(cfg["partition_data"])
            frames = {p: collected(p) for p in ("validation", "test")}

            if replicas > 1:
                from dencast.modeling.parallel import fit_data_parallel

                state_dict, held_out = fit_data_parallel(
                    builder, T, pick, cfg, len(features), replicas,
                    validation=((frames["validation"], labels["validation"])
                                if on_ap else None))
                model.net.load_state_dict(state_dict)
            else:
                egos["train"] = [e for e in builder.examples(T[pick], drop=pick)
                                 if e is not None]
                sizes = np.array([len(e.features) for e in egos["train"]])
                logger.info("contextual subgraphs: {:,} for training, nodes median {}, "
                            "largest {}", len(egos["train"]), int(np.median(sizes)),
                            int(sizes.max()))
                watch = None
                if on_ap:
                    # Built here only because this branch already holds everything; the
                    # sharded paths keep them split.
                    watch = (list(builder.examples(frames["validation"])),
                             labels["validation"])
                held_out = _fit(model, egos["train"], cfg, watch=watch)
        logger.success("autoencoder fitted, best held-out {:.5f}", held_out)

        # The session stays open through the fit so the inference parts can be scored
        # on the tasks that own their clusters, which is also why their subgraphs were
        # never built here.
        if distributed:
            scores = {"train": _score_frames(spark, win["train"], placed, clustered,
                                             members, model, cfg, features,
                                             min_similarity)}
            for part in ("validation", "test"):
                scores[part] = _score_frames(
                    spark, win[part], _place(spark, win[part], pool, features),
                    clustered, members, model, cfg, features, min_similarity)
        elif by_cluster:
            score_part = (_score_by_cluster_shuffled if cfg["partition_data"]
                          else _score_by_cluster)
            scores = {p: score_part(spark, builder, f, model, cfg)
                      for p, f in (("train", T[pick]),
                                   ("validation", frames["validation"]),
                                   ("test", frames["test"]))}
        else:
            if "train" not in egos:
                egos["train"] = [e for e in builder.examples(T[pick], drop=pick)
                                 if e is not None]
            for part in ("validation", "test"):
                egos[part] = list(builder.examples(frames[part]))
            scores = {p: _score(model, egos[p], cfg) for p in PARTS}
    finally:
        spark.stop()

    carried = metrics(labels["validation"], scores["validation"])["threshold"]
    logger.info("threshold fixed on validation at {:.6f}", carried)

    results = {}
    for part in PARTS:
        m = metrics(labels[part], scores[part],
                    threshold=carried if part != "validation" else None)
        results[part] = m
        if m["roc_auc"] is not None:
            logger.success("  {:<11} ROC-AUC {:.4f}  AP {:.4f}  F1 {:.4f}  "
                           "segments {}/{}", part, m["roc_auc"],
                           m["average_precision"], m["f1"], m["segments_caught"],
                           m["segments"])

    report = {
        "model": {
            "name": "DENCAD: a graph autoencoder over DENCAST clusters",
            "score": "reconstruction error of the centre of its contextual subgraph",
            "threshold": float(carried),
            "threshold_chosen_on": "validation, maximising F1",
            "stopping_score": held_out,
            **{k: cfg[k] for k in ("bits", "permutations", "band", "min_pts",
                                   "label_change_rate", "attach_to", "max_members",
                                   "max_hops",
                                   "mask_centre", "hidden", "latent", "layers",
                                   "learning_rate", "batch_size", "scale_windows",
                                   "replicas", "scale_learning_rate",
                                   "scoring_by_cluster", "placement_in_spark",
                                   "partition_data", "early_stopping", "seed")},
            "min_similarity": float(min_similarity),
        },
        "dataset": {
            "name": dataset, "split_dir": str(split_dir),
            "window_steps": int(window_steps),
            "sampling_period_seconds": int(period),
            "window_seconds": int(window_steps) * int(period),
            "features": len(features), "feature_columns": features,
            "label_column": label, "window_scaling": scaling,
            "windows": {p: int(len(labels[p])) for p in PARTS},
            "base_rate": {p: float(labels[p].mean()) for p in PARTS},
        },
        "graph": {"edges": int(n_edges), "candidates": int(n_candidates),
                  "cores": int(n_cores), **summary},
        "metrics": results,
        "run": {"finished_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds")},
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"dencad_{split_dir.name}_{window_steps}steps.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.success("report written to {}", out)
    return report


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="DENCAD: anomaly detection over DENCAST clusters, on windowed telemetry.")
    ap.add_argument("split_dir", type=Path,
                    help="directory holding *_train/_validation/_test.parquet")
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--window-steps", type=int, default=None,
                    help="readings per window; the interval is that many sampling "
                         "periods, measured from the data")
    ap.add_argument("--min-similarity", type=float, default=None,
                    help="overrides the per-dataset value in params.yaml")
    ap.add_argument("--min-pts", type=int, default=None)
    ap.add_argument("--max-members", type=int, default=None,
                    help="cap on the cluster; omitted the whole cluster is used")
    ap.add_argument("--mask-centre", action="store_true", default=None,
                    help="hide the centre's attributes from the input")
    ap.add_argument("--latent", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--train-examples", type=int, default=None)
    ap.add_argument("--max-hops", type=int, default=None,
                    help="keep only the nodes within this many hops of the centre; "
                         "exact when it equals `layers`")
    ap.add_argument("--patience", type=int, default=None,
                    help="epochs without improvement before the fit stops")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--replicas", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    train_dencad(a.split_dir, a.reports_dir, a.window_steps, a.min_similarity,
                  min_pts=a.min_pts, max_members=a.max_members,
                  mask_centre=a.mask_centre, latent=a.latent, layers=a.layers,
                  train_examples=a.train_examples, workers=a.workers,
                  patience=a.patience, max_epochs=a.max_epochs,
                  max_hops=a.max_hops,
                  replicas=a.replicas)


if __name__ == "__main__":
    main()


__all__ = ["train_dencad", "dencad_config", "GraphAutoEncoder", "DEFAULTS"]
