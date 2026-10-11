"""The data layout for the distributed fit."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from loguru import logger
import numpy as np


def assign_by_count(counts: dict[int, int], world_size: int) -> dict[int, int]:
    """Clusters to replicas, the largest first and always to the lightest replica."""
    tally = {k: 0 for k in range(world_size)}
    where: dict[int, int] = {}
    for c in sorted(counts, key=lambda x: -counts[x]):
        k = min(tally, key=lambda j: tally[j])
        where[int(c)] = k
        tally[k] += counts[c]
    return where


def member_frame(win, state, edges, cols: Sequence[str], node: str = "_id"):
    """One row per clustered window: its cluster, its features, its neighbours.

    The neighbour list is the cluster's adjacency in the only form that survives a
    shuffle cheaply -- an edge becomes two directed rows, grouped per node. Edges that
    leave the cluster come along and are dropped by the task, which knows which nodes
    it holds; filtering them here would need a second join for nothing.
    """
    from pyspark.sql import functions as F

    both = (edges.select(F.col("_src").alias("_node"), F.col("_dst").alias("_other"))
            .union(edges.select(F.col("_dst").alias("_node"),
                                F.col("_src").alias("_other"))))
    nb = both.groupBy("_node").agg(F.collect_list("_other").alias("_nb"))
    return (state.join(win, state["_node"] == win[node])
            .join(nb, "_node", "left")
            .select("_cluster", "_node", F.array(*cols).alias("_f"),
                    F.coalesce(F.col("_nb"),
                               F.array().cast("array<bigint>")).alias("_nb")))


def query_frame(placed, state, win, cols: Sequence[str], node: str = "_id",
                label: Optional[str] = None, leave_out: bool = False):
    """One row per query: the cluster it landed in, its features, its core.

    `leave_out` carries the query's own node id as `_drop`, which is how a training
    window is detached from its own subgraph. A validation or test window was never in
    the graph, so it has nothing to leave out.
    """
    from pyspark.sql import functions as F

    cores = state.select(F.col("_node").alias("_core"), "_cluster")
    return (placed.join(cores, "_core")
            .join(win, placed["_row"] == win[node])
            .select("_cluster", "_row", "_core", "_sim",
                    F.array(*cols).alias("_f"),
                    *([F.col(label).cast("long").alias("_y")] if label else []),
                    *([F.col("_row").alias("_drop")] if leave_out else [])))


def _write_by_replica(df, out_dir: Path, name: str, world_size: int) -> int:
    """Each task writes its own rows, as parquet, into the directory of their replica.

    Spark's own writer is not used, and not by preference: on Windows it goes through
    Hadoop to set permissions on the output directory, which needs `winutils.exe` and
    fails without it -- the rest of the pipeline writes its parquet with pandas for the
    same reason. Writing from the tasks is also the closer analogue of a cluster, where
    an executor writes the partition it holds rather than sending it to the driver to be
    written.

    The frame is repartitioned by replica first, so a task normally holds one replica's
    rows and writes one file; where a partition holds more than one, it writes one file
    each. A replica reads every file in its directory, so the count per directory does
    not matter.
    """
    import pandas as pd

    root = str(out_dir.resolve())

    def write(batches):
        from pathlib import Path as _Path

        import uuid

        for pdf in batches:
            if pdf.empty:
                continue
            for replica, part in pdf.groupby("_replica"):
                where = _Path(root) / name / f"_replica={int(replica)}"
                where.mkdir(parents=True, exist_ok=True)
                (part.drop(columns=["_replica"])
                 .to_parquet(where / f"part-{uuid.uuid4().hex}.parquet",
                             index=False, compression="snappy"))
                yield pd.DataFrame({"_rows": [len(part)]})

    for sub in (out_dir / name,):
        if sub.exists():
            import shutil

            shutil.rmtree(sub)
    counted = (df.repartition(world_size, "_replica")
               .mapInPandas(write, schema="_rows long").collect())
    return sum(int(r["_rows"]) for r in counted)


def write_shards(out_dir: Path, world_size: int, queries, members,
                 cols: Sequence[str], validation=None,
                 meta: Optional[dict] = None) -> Path:
    """Queries, members and validation queries as parquet, partitioned by replica.

    Which replica a cluster belongs to is decided from the cluster sizes, and that is
    the only thing reaching the driver here: one row per cluster, 96 of them on HAI.
    """
    import json

    out_dir = Path(out_dir)
    counts = {int(r["_cluster"]): int(r["count"])
              for r in queries.groupBy("_cluster").count().collect()}
    if not counts:
        raise RuntimeError("no query landed in any cluster: there is nothing to shard")
    where = assign_by_count(counts, world_size)

    rep = queries.sparkSession.createDataFrame(
        [(int(c), int(k)) for c, k in where.items()], "_cluster long, _replica int")

    for name, df in (("queries", queries), ("members", members),
                     ("validation", validation)):
        if df is None:
            continue
        tagged = df.join(rep, "_cluster").localCheckpoint(eager=True)
        expected = tagged.count()
        written = _write_by_replica(tagged, out_dir, name, world_size)
        # A row that reached no task would be a training example or a validation window
        # silently missing from the fit, which is the kind of loss that looks like a
        # worse model rather than like a bug.
        if written != expected:
            raise RuntimeError(f"{name}: {written:,} rows written of {expected:,}")

    share = {k: sum(v for c, v in counts.items() if where[c] == k)
             for k in range(world_size)}
    logger.info("shards written under {} | examples {} | {} clusters over {} replicas",
                out_dir.name, "/".join(f"{share[k]:,}" for k in range(world_size)),
                len(counts), world_size)
    (out_dir / "meta.json").write_text(json.dumps(
        {"features": list(cols), "world_size": world_size, **(meta or {})}))
    return out_dir


def read_shard(out_dir: Path, rank: int, min_similarity: float,
               max_members: Optional[int], seed: int, holdout: float,
               max_hops: Optional[int] = None) -> dict:
    """A replica's own partition, as subgraphs: its queries, its validation, its split.

    Read with pandas and built cluster by cluster, so the features of the clusters this
    replica was not given are never in this process -- which is the point of writing the
    shards partitioned rather than pickling them whole.
    """
    import pandas as pd

    from dencad.modeling.subgraphs import egos_from_rows

    out_dir = Path(out_dir)

    def part(name):
        path = out_dir / name / f"_replica={rank}"
        return pd.read_parquet(path) if path.exists() else None

    members, queries = part("members"), part("queries")
    if members is None or queries is None:
        raise RuntimeError(f"replica {rank} has no shard under {out_dir}")
    by_cluster = {int(c): g for c, g in members.groupby("_cluster")}

    def build(frame):
        rows, egos = [], []
        for cluster, group in frame.groupby("_cluster"):
            got_rows, got_egos = egos_from_rows(
                by_cluster[int(cluster)], group, min_similarity, max_members,
                max_hops=max_hops)
            rows.append(got_rows)
            egos.extend(got_egos)
        return (np.concatenate(rows) if rows else np.zeros(0, dtype="int64")), egos

    _, egos = build(queries)
    # The held-out slice, drawn here rather than on the driver: the driver no longer has
    # the rows to draw it from. Only read when the fit stops on the reconstruction loss.
    order = np.arange(len(egos))
    np.random.default_rng(seed + rank).shuffle(order)
    cut = int((1 - holdout) * len(order))

    val = part("validation")
    val_rows, val_egos = np.zeros(0, dtype="int64"), []
    val_y = np.zeros(0, dtype="uint8")
    if val is not None and len(val):
        val_rows, val_egos = build(val)
        val_y = (val.set_index("_row")["_y"].reindex(val_rows)
                 .to_numpy().astype("uint8"))

    sizes = np.array([len(e.features) for e in egos]) if egos else np.zeros(1)
    return {"egos": egos, "fit": order[:cut].tolist(), "dev": order[cut:].tolist(),
            "val_egos": val_egos, "val_rows": val_rows, "val_y": val_y,
            "n_clusters": len(by_cluster), "member_rows": len(members),
            "nodes_median": int(np.median(sizes)), "nodes_max": int(sizes.max())}


__all__ = ["assign_by_count", "member_frame", "query_frame", "write_shards",
           "read_shard"]
