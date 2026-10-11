"""Data-parallel training of the autoencoder, simulated on one machine.

The paper's second instalment reaches for Horovod, and for a reason that has nothing to
do with its modelling choices: training **one** network across workers needs the
gradients summed between every backward pass and every step, and Spark cannot do that.
Its tasks are independent by construction -- immutable partitions, no channel between
them -- so an all-reduce has no place to happen. Horovod, `torch.distributed` and NCCL
exist to supply the channel Spark does not have.

Horovod does not run on Windows: it builds against MPI and is supported on Linux and
macOS. `torch.distributed` with the **gloo** backend does, and it performs the same
synchronous SGD: each rank holds a replica, takes a shard of the examples, and the
gradients are averaged across ranks before the optimiser steps. Several processes on one
machine simulate several machines faithfully in everything except the network.

**This does not reproduce the single-process fit.** Averaging gradients over `world_size`
shards is a different optimisation from stepping through the same examples in one
sequence: the loss landscape is walked differently and the weights land somewhere else,
close but not equal. The single-process fit stays the reference, and this is here to
show the step scales, not to replace it.

**And on one machine it is slower.** Measured on HAI, 19,082 subgraphs: 1.15 s per epoch
across four replicas against 1.05 s in one process. Each replica holds a quarter of the
examples, so its share of the arithmetic is about 0.26 s; the rest is the all-reduce,
roughly 26 ms a batch on a model of two linear layers of 64 and a 16-dimensional latent.
There is nothing to amortise. The communication pays for itself when a replica has its
own node and its own memory bandwidth, not when four processes contend for four cores.

Convergence is also worse at equal epochs, for a separate and expected reason: with
`batch_size` unchanged, four replicas make the effective batch four times larger and the
optimiser takes 34 steps an epoch instead of 135. Held-out 0.41293 against 0.34641 at
epoch 60. Scaling the learning rate with the replica count is the usual answer, and is
tuning rather than a property of this code.

So this is here to show the step scales and to say what it costs, not because it is the
faster path on one machine.
"""

from __future__ import annotations

import os
import pickle
import socket
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np


def _score_local(replica, forward, egos, cfg, d) -> np.ndarray:
    """Reconstruction error of each subgraph, with the replica's own weights."""
    import torch

    from dencast.modeling.train_dencad import _batches

    out = np.full(len(egos), np.nan)
    if not len(egos):
        return out
    with torch.no_grad():
        for X, A_hat, mask, centre, target, chunk in _batches(
                egos, list(range(len(egos))), d, cfg["batch_size"],
                cfg["mask_centre"]):
            out[chunk] = ((forward(X, A_hat, centre) - target) ** 2
                          ).mean(1).numpy()
    return out


def _restrict(builder, cluster_ids):
    """The slice of the graph a replica needs: its clusters and nothing else.

    Returns the members' features, their cluster ids and core flags, the intra-cluster
    edges renumbered to local positions, and the map from global row to local position.
    A replica that owns 23 of HAI's 94 clusters receives 965 of the 3,859 clustered
    rows instead of all of them.
    """
    nodes = np.sort(np.concatenate([builder.members[int(c)] for c in cluster_ids]))
    where = {int(n): i for i, n in enumerate(nodes)}
    pairs = [(where[int(u)], where[int(v)])
             for c in cluster_ids for u, v in builder.cluster_edges[int(c)]]
    edges = (np.asarray(pairs, dtype="int64") if pairs
             else np.zeros((0, 2), dtype="int64"))
    return (builder.X[nodes], builder.cluster_of[nodes], builder.core_mask[nodes],
            edges, where)


def _assign(clusters: np.ndarray, world_size: int) -> dict[int, list[int]]:
    """Clusters to replicas, the largest first and always to the lightest replica.

    Not a stride: HAI's clusters run from 17 members to 351, so handing out every
    fourth one would leave a replica with several times another's work and the
    all-reduce waiting on it at every step. Greedy on the example count balances to
    within a tenth here -- 4,369 against 5,233 of 19,082.
    """
    from collections import Counter

    counts = Counter(int(c) for c in clusters)
    load: dict[int, list[int]] = {k: [] for k in range(world_size)}
    tally = {k: 0 for k in range(world_size)}
    for c in sorted(counts, key=lambda x: -counts[x]):
        k = min(tally, key=lambda j: tally[j])
        load[k].append(c)
        tally[k] += counts[c]
    return load


def _load_shard(rank: int, source: dict, cfg: dict) -> dict:
    """The replica's examples, from parquet it reads itself or from the driver's pickle.

    Both forms end in the same dictionary, so the fit below does not know which one it
    was given. The parquet form is the one that scales: the replica opens
    `queries/_replica=k` and `members/_replica=k` and builds its subgraphs cluster by
    cluster, and the clusters it was not given were never in this process. The pickle
    form needs a driver that holds every window to cut the shards out of.
    """
    if source["kind"] == "parquet":
        from dencast.modeling.shards import read_shard

        got = read_shard(Path(source["dir"]), rank,
                         min_similarity=float(source["min_similarity"]),
                         max_members=source["max_members"],
                         seed=int(cfg["seed"]), holdout=float(cfg["holdout"]),
                         max_hops=cfg["max_hops"])
        # Uniform across the replicas, and deliberately not conditioned on this
        # replica having validation queries: the epoch ends in an `all_gather`, so a
        # replica that decided on its own not to take part would leave the others
        # waiting for it. One with nothing to score contributes an empty piece.
        got["on_ap"] = bool(source["on_ap"])
        got["all_examples"] = len(got["egos"])
        return got

    from dencast.modeling.subgraphs import ContextualSubgraphBuilder

    shard = pickle.load(open(f"{source['bundle']}.{rank}", "rb"))
    # The replica's own slice of the graph: the members of the clusters it was given,
    # and nothing else. It rebuilds a builder over that local world, where the cluster
    # ids are the original ones so `build` finds its members, and the core and drop
    # indices have already been translated to local positions on the driver.
    local = ContextualSubgraphBuilder(
        shard["X"], shard["cluster_of"], shard["core_mask"], shard["edges"],
        min_similarity=shard["min_similarity"],
        max_members=shard["max_members"], attach_to=shard["attach_to"],
        max_hops=cfg["max_hops"])
    # Built here, so the subgraphs are never all in one process: on HAI they come to
    # 3.1 GB for 19,082 of them, nearly two thirds of it dense adjacency, and that is
    # the figure that decides whether ESA's 78,308 fit anywhere at all.
    egos = [local.build(q, int(c), float(v), drop=(None if dp < 0 else int(dp)))
            for q, c, v, dp in zip(shard["queries"], shard["cores"],
                                   shard["sims"], shard["drops"])]
    on_ap = "val_queries" in shard
    return {
        "egos": egos,
        "val_egos": ([local.build(q, int(c), float(v))
                      for q, c, v in zip(shard["val_queries"], shard["val_cores"],
                                         shard["val_sims"])] if on_ap else []),
        "val_rows": shard.get("val_rows", np.zeros(0, dtype="int64")),
        "val_y": shard.get("val_y", np.zeros(0, dtype="uint8")),
        "fit": shard["fit"], "dev": shard["dev"], "on_ap": on_ap,
        "all_examples": shard["all_examples"], "n_clusters": shard["n_clusters"],
        "member_rows": len(shard["X"]),
        "nodes_median": int(np.median([len(e.features) for e in egos])) if egos else 0,
        "nodes_max": int(max((len(e.features) for e in egos), default=0)),
    }


def _worker(rank: int, world_size: int, source: dict, out: str, cfg: dict,
            d: int) -> None:
    """One replica: its shard of the examples, gradients averaged with the others.

    `source` says where the shard comes from. "parquet" is the distributed layout: the
    replica reads its own partition and builds its own subgraphs, so no process ever
    holds the whole windowed matrix. "pickle" is the earlier layout, kept for the paths
    that still assemble the shards from a driver-side builder.
    """
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from dencast.modeling.evaluate import metrics
    from dencast.modeling.train_dencad import GraphAutoEncoder, _batches

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        shard = _load_shard(rank, source, cfg)
        egos, val_egos = shard["egos"], shard["val_egos"]
        on_ap = shard["on_ap"]
        mine, mine_dev = shard["fit"], shard["dev"]
        fit = shard["all_examples"]
        if rank == 0:
            logger.info("  rank 0: {:,} examples over {} clusters, {:,} member rows "
                        "in memory | subgraph nodes median {}, largest {}",
                        len(egos), shard["n_clusters"], shard["member_rows"],
                        shard.get("nodes_median", "?"), shard.get("nodes_max", "?"))

        torch.manual_seed(cfg["seed"])
        model = GraphAutoEncoder(d, cfg)
        replica = DistributedDataParallel(model.net)
        # Linear scaling. `batch_size` is per replica, so N of them make the effective
        # batch N times larger and the optimiser takes N times fewer steps an epoch --
        # 34 instead of 135 at four replicas, which is why an unscaled run reached
        # held-out 0.41293 where one process reached 0.34641. Multiplying the rate by
        # the replica count is the standard correction (Goyal et al., 2017) and keeps
        # the comparison about where the work runs rather than about the batch size.
        lr = cfg["learning_rate"] * (world_size if cfg["scale_learning_rate"] else 1)
        opt = torch.optim.Adam(replica.parameters(), lr=lr)
        if rank == 0:
            logger.info("  learning rate {:g}{}", lr,
                        f" ({cfg['learning_rate']:g} x {world_size})"
                        if cfg["scale_learning_rate"] else " (unscaled)")

        fit_batches = _batches(egos, mine, d, cfg["batch_size"], cfg["mask_centre"])
        dev_batches = _batches(egos, mine_dev, d, cfg["batch_size"],
                               cfg["mask_centre"])
        # Every replica must take the same number of optimiser steps. DDP averages the
        # gradients inside `backward`, so a replica with more batches than the others
        # runs its surplus **alone**: those steps are never averaged, the replicas stop
        # being copies of one model, and what rank 0 saves is its own drifted weights.
        # With rows sharded evenly the counts matched by luck and the hazard was
        # invisible; sharding by cluster made them 35 against 41 and it became real.
        steps = torch.tensor([len(fit_batches)], dtype=torch.long)
        dist.all_reduce(steps, op=dist.ReduceOp.MIN)
        steps = int(steps.item())
        if rank == 0:
            logger.info("  rank 0 of {}: {:,} of {:,} examples, {:,} batches, {:,} "
                        "steps an epoch (the smallest replica's count)",
                        world_size, len(mine), fit, len(fit_batches), steps)

        def forward(X, A_hat, centre):
            # Through the wrapper, never through `replica.module`. DDP arms the hooks
            # that all-reduce the gradients inside its own forward, so a backward whose
            # graph starts at the inner module is never synchronised: two gloo ranks on
            # deliberately different data kept gradients 1.589 apart and stepped into
            # two different models, while the same step through the wrapper left them
            # bit-identical. Until this call went through `replica`, every
            # multi-replica fit trained `world_size` independent models on disjoint
            # shards and saved rank 0's -- which is why the AP the ranks agreed on
            # while stopping (0.18049, gathered from four different models) did not
            # match the AP of the saved weights (0.1322).
            return replica(X, A_hat, centre)

        best, waited, state = -np.inf, 0, None
        sign = 1.0 if on_ap else -1.0
        label = "validation AP" if on_ap else "held-out"
        if rank == 0:
            logger.info("  stopping on {}", label)
        order = np.arange(len(fit_batches))
        shuffler = np.random.default_rng(cfg["seed"] + rank)
        for epoch in range(1, cfg["max_epochs"] + 1):
            replica.train()
            # Reshuffled every epoch, and not only to break the truncation's bias.
            # `_batches` sorts the examples by node count and the list is built once,
            # so without this a replica walks the same batches in the same order every
            # epoch. Sharded by row that was tolerable, since a replica's examples came
            # from 93 of 94 clusters and each batch was already a mixture; sharded by
            # cluster the first batch is always the same small cluster and consecutive
            # steps stay correlated for the whole fit.
            shuffler.shuffle(order)
            for j in order[:steps]:
                X, A_hat, mask, centre, target, _ = fit_batches[j]
                opt.zero_grad()
                ((forward(X, A_hat, centre) - target) ** 2).mean().backward()
                opt.step()

            if epoch == 1:
                # One all-reduce, once, against the failure that hid for three runs:
                # a forward that skips the wrapper trains the replicas apart and
                # nothing says so -- the loss falls, the fit converges, and what rank 0
                # saves is a model fitted on a quarter of the data. Compared against
                # rank 0 because DDP leaves the parameters bit-identical, so any gap at
                # all means the gradients are not being averaged.
                flat = torch.cat([p.detach().flatten()
                                  for p in replica.module.parameters()])
                theirs = flat.clone()
                dist.broadcast(theirs, src=0)
                drift = float((flat - theirs).abs().max())
                if drift > 1e-6:
                    raise RuntimeError(
                        f"rank {rank} drifted {drift:.3e} from rank 0 after one epoch: "
                        "the gradients are not being averaged")

            replica.eval()
            if on_ap:
                # Each replica scores its own validation queries; the pieces are
                # gathered so every rank sees the same whole and reaches the same stop
                # decision without anyone broadcasting it. A few thousand numbers.
                mine_scores = _score_local(replica, forward, val_egos, cfg, d)
                pieces = [None] * world_size
                dist.all_gather_object(
                    pieces, (shard["val_rows"], mine_scores, shard["val_y"]))
                rows = np.concatenate([p[0] for p in pieces])
                scores = np.concatenate([p[1] for p in pieces])
                ys = np.concatenate([p[2] for p in pieces])
                held = float(metrics(ys, scores)["average_precision"])
            else:
                with torch.no_grad():
                    total = sum(((forward(X, A_hat, centre) - target) ** 2
                                 ).mean().item() * len(centre)
                                for X, A_hat, mask, centre, target, _ in dev_batches)
                    seen = torch.tensor([total, float(len(mine_dev))])
                dist.all_reduce(seen, op=dist.ReduceOp.SUM)
                held = float(seen[0] / max(seen[1].item(), 1.0))

            if rank == 0 and (epoch % 5 == 0 or epoch == 1):
                logger.info("  epoch {:>3}: {} {:.5f}", epoch, label, held)
            if sign * held > best + 1e-5:
                best, waited = sign * held, 0
                if rank == 0:
                    state = {k: v.clone()
                             for k, v in replica.module.state_dict().items()}
            else:
                waited += 1
                if waited >= cfg["patience"]:
                    if rank == 0:
                        logger.info("  stopped at epoch {}", epoch)
                    break

        if rank == 0:
            torch.save({"state": state, "best": float(sign * best)}, out)
    finally:
        dist.destroy_process_group()


def fit_shards(shards_dir, cfg: dict, d: int, world_size: int,
               min_similarity: float, on_ap: bool,
               out: Optional[Path] = None) -> tuple[dict, float]:
    """Fit the replicas from parquet shards, with nothing passing through the driver.

    The counterpart of `fit_data_parallel` for the distributed layout: Spark has already
    written each replica's queries and member rows, so this only spawns the processes
    and collects the weights. What crosses back is one state dictionary.
    """
    import torch
    import torch.multiprocessing as mp

    shards_dir = Path(shards_dir)
    out = Path(out) if out else shards_dir / "fitted.pt"
    source = {"kind": "parquet", "dir": str(shards_dir),
              "min_similarity": float(min_similarity),
              "max_members": cfg["max_members"], "on_ap": bool(on_ap)}

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(probe.getsockname()[1])
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    mp.spawn(_worker, args=(world_size, source, str(out), cfg, d),
             nprocs=world_size, join=True)
    got = torch.load(out, weights_only=False)
    return got["state"], float(got["best"])


def fit_data_parallel(builder, T: np.ndarray, pick: np.ndarray, cfg: dict, d: int,
                      world_size: int, validation=None,
                      scratch: Optional[Path] = None) -> tuple[dict, float]:
    """Spawn `world_size` replicas and return the best weights and held-out loss.

    Takes the builder and the training rows rather than the subgraphs: each replica
    builds the share it will train on, so nothing holds all of them.
    """
    import torch
    import torch.multiprocessing as mp

    pick = np.asarray(pick)
    # Placed once, on the driver, with the full pool and the full leave-one-out: which
    # cluster a row lands in has to be known before the rows can be split by it.
    cores, sims = builder.nearest_core(T[pick], exclude=pick)
    clusters = builder.cluster_of[cores]
    load = _assign(clusters, world_size)

    # The validation queries are placed once too, and travel to whichever replica owns
    # the cluster they landed in, so watching validation costs no extra centralisation.
    if validation is not None:
        V, y_val = validation
        v_cores, v_sims = builder.nearest_core(V)
        v_clusters = builder.cluster_of[v_cores]

    scratch = Path(scratch or tempfile.gettempdir())
    bundle = scratch / "dencad_shard.pkl"
    out = scratch / "dencad_fitted.pt"
    rng = np.random.default_rng(cfg["seed"])
    sizes = []
    for k in range(world_size):
        mine = np.flatnonzero(np.isin(clusters, load[k]))
        rng.shuffle(mine)
        cut = int((1 - cfg["holdout"]) * len(mine))
        X, cl, cm, edges, where = _restrict(builder, load[k])
        shard = {
            "X": X, "cluster_of": cl, "core_mask": cm, "edges": edges,
            "min_similarity": float(builder.min_similarity),
            "max_members": builder.max_members, "attach_to": builder.attach_to,
            "queries": T[pick][mine],
            "cores": np.array([where[int(c)] for c in cores[mine]], dtype="int64"),
            "sims": sims[mine],
            # A training row that is itself a member of its cluster must leave its own
            # subgraph; -1 where it is not one, which is most of them.
            "drops": np.array([where.get(int(pick[i]), -1) for i in mine],
                              dtype="int64"),
            "fit": list(range(cut)), "dev": list(range(cut, len(mine))),
            "all_examples": len(mine), "n_clusters": len(load[k]),
        }
        if validation is not None:
            vm = np.flatnonzero(np.isin(v_clusters, load[k]))
            shard["val_rows"] = vm.astype("int64")
            shard["val_queries"] = V[vm]
            shard["val_cores"] = np.array([where[int(c)] for c in v_cores[vm]],
                                          dtype="int64")
            shard["val_sims"] = v_sims[vm]
            shard["val_y"] = np.asarray(y_val)[vm]
            shard["val_total"] = len(V)
        with open(f"{bundle}.{k}", "wb") as fh:
            pickle.dump(shard, fh, protocol=pickle.HIGHEST_PROTOCOL)
        sizes.append(Path(f"{bundle}.{k}").stat().st_size / 1e6)

    logger.info("data-parallel fit: {} gloo replicas sharded by cluster | examples "
                "{} | member rows {} | shards {:.1f}-{:.1f} MB", world_size,
                "/".join(f"{int((clusters[np.isin(clusters, load[k])]).size):,}"
                         for k in range(world_size)),
                "/".join(f"{sum(len(builder.members[int(c)]) for c in load[k]):,}"
                         for k in range(world_size)),
                min(sizes), max(sizes))

    # Chosen free rather than fixed: the children inherit this environment, and a
    # second fit on the same machine -- a grid search runs them back to back -- would
    # otherwise fail to bind and report it as a decode error in a localised Windows
    # message rather than as a busy port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(probe.getsockname()[1])
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    mp.spawn(_worker,
             args=(world_size, {"kind": "pickle", "bundle": str(bundle)}, str(out),
                   cfg, d),
             nprocs=world_size, join=True)

    got = torch.load(out, weights_only=False)
    for k in range(world_size):
        Path(f"{bundle}.{k}").unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return got["state"], got["best"]


__all__ = ["fit_data_parallel", "fit_shards"]
