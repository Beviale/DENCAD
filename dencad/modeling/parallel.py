"""Data-parallel training of the autoencoder, simulated on one machine."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from dencad.modeling.evaluate import metrics
from dencad.modeling.train_dencad import GraphAutoEncoder, batches
from dencad.modeling.shards import read_shard


def _score_local(replica, forward, egos, cfg, d) -> np.ndarray:
    """Reconstruction error of each subgraph, with the replica's own weights."""

    out = np.full(len(egos), np.nan)
    if not len(egos):
        return out
    with torch.no_grad():
        for X, A_hat, mask, centre, target, chunk in batches(
                egos, list(range(len(egos))), d, cfg["batch_size"],
                cfg["mask_centre"]):
            out[chunk] = ((forward(X, A_hat, centre) - target) ** 2
                          ).mean(1).numpy()
    return out


def _read_shard(rank: int, source: dict, cfg: dict) -> dict:
    """The replica's own parquet partition, as subgraphs it builds itself."""

    got = read_shard(Path(source["dir"]), rank,
                     min_similarity=float(source["min_similarity"]),
                     max_members=source["max_members"],
                     seed=int(cfg["seed"]), holdout=float(cfg["holdout"]),
                     max_hops=cfg["max_hops"])
    got["on_ap"] = bool(source["on_ap"])
    got["all_examples"] = len(got["egos"])
    return got


def _worker(rank: int, world_size: int, source: dict, out: str, cfg: dict,
            d: int) -> None:
    """One replica: its shard of the examples, gradients averaged with the others."""

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        shard = _read_shard(rank, source, cfg)
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
       
        lr = cfg["learning_rate"] * (world_size if cfg["scale_learning_rate"] else 1)
        opt = torch.optim.Adam(replica.parameters(), lr=lr)
        if rank == 0:
            logger.info("  learning rate {:g}{}", lr,
                        f" ({cfg['learning_rate']:g} x {world_size})"
                        if cfg["scale_learning_rate"] else " (unscaled)")

        fitbatches = batches(egos, mine, d, cfg["batch_size"], cfg["mask_centre"])
        devbatches = batches(egos, mine_dev, d, cfg["batch_size"],
                               cfg["mask_centre"])
     
        steps = torch.tensor([len(fitbatches)], dtype=torch.long)
        dist.all_reduce(steps, op=dist.ReduceOp.MIN)
        steps = int(steps.item())
        if rank == 0:
            logger.info("  rank 0 of {}: {:,} of {:,} examples, {:,} batches, {:,} "
                        "steps an epoch (the smallest replica's count)",
                        world_size, len(mine), fit, len(fitbatches), steps)

        def forward(X, A_hat, centre):
            return replica(X, A_hat, centre)

        best, waited, state = -np.inf, 0, None
        sign = 1.0 if on_ap else -1.0
        label = "validation AP" if on_ap else "held-out"
        if rank == 0:
            logger.info("  stopping on {}", label)
        order = np.arange(len(fitbatches))
        shuffler = np.random.default_rng(cfg["seed"] + rank)
        for epoch in range(1, cfg["max_epochs"] + 1):
            replica.train()
            shuffler.shuffle(order)
            for j in order[:steps]:
                X, A_hat, mask, centre, target, _ = fitbatches[j]
                opt.zero_grad()
                ((forward(X, A_hat, centre) - target) ** 2).mean().backward()
                opt.step()

            if epoch == 1:
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
                                for X, A_hat, mask, centre, target, _ in devbatches)
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
    """Fit the replicas from parquet shards"""

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


__all__ = ["fit_data_parallel", "fit_shards"]
