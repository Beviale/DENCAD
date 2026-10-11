"""Exact placement of an object into the clustered graph."""

from __future__ import annotations

from typing import Optional, Sequence

from loguru import logger
import numpy as np
import pandas as pd

from pyspark.sql.types import (DoubleType, LongType, StructField, StructType)
from pyspark.sql import functions as F

BUDGET_BYTES = 256 * 1024 * 1024


def _unit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype="float32")
    return a / np.maximum(np.linalg.norm(a, axis=1), 1e-12)[:, None]


def nearest(queries: np.ndarray, pool: np.ndarray, pool_ids: np.ndarray,
            exclude: Optional[np.ndarray] = None,
            budget_bytes: int = BUDGET_BYTES) -> tuple[np.ndarray, np.ndarray]:
    """For each query, the most similar object in 'pool', and that cosine."""

    qu = _unit(queries)
    pu = _unit(pool)
    if len(pool_ids) != len(pu):
        raise ValueError("pool_ids and pool disagree in length")
    if not len(pu):
        raise ValueError("the pool is empty: nothing to place against")

    tile = max(1, budget_bytes // (4 * len(pu)))
    best = np.empty(len(qu), dtype="int64")
    score = np.empty(len(qu), dtype="float32")
    if exclude is not None:
        exclude = np.asarray(exclude, dtype="int64")
        at = np.searchsorted(pool_ids, exclude)
        inside = (at < len(pool_ids)) & (pool_ids[np.minimum(at, len(pool_ids) - 1)]
                                         == exclude)

    for start in range(0, len(qu), tile):
        stop = min(start + tile, len(qu))
        block = qu[start:stop] @ pu.T
        if exclude is not None:
            rows = np.flatnonzero(inside[start:stop])
            if len(rows):
                block[rows, at[start:stop][rows]] = -np.inf
        idx = block.argmax(1)
        best[start:stop] = pool_ids[idx]
        score[start:stop] = block[np.arange(stop - start), idx]

    logger.info("placed {:,} queries against {:,} eligible objects exactly, {:,} per "
                "tile | cosine with the chosen object: median {:.4f}, min {:.4f}",
                len(qu), len(pu), min(tile, len(qu)), float(np.median(score)),
                float(score.min()))
    return best, score.astype("float64")


def nearest_grid_frames(spark, queries, pool, cols: Sequence[str],
                        query_blocks: Optional[int] = None,
                        pool_blocks: Optional[int] = None):
    """The grid argmax over DataFrames."""

    par = max(1, int(spark.sparkContext.defaultParallelism))
    query_blocks = query_blocks or par
    pool_blocks = pool_blocks or par

    block = lambda n: F.explode(F.array(*[F.lit(i).cast("long") for i in range(n)]))
    qdf = (queries.withColumn("_qb", (F.col("_row") % F.lit(query_blocks)).cast("long"))
           .withColumn("_pb", block(pool_blocks)))
    pdf = (pool.withColumn("_pb", (F.col("_id") % F.lit(pool_blocks)).cast("long"))
           .withColumn("_qb", block(query_blocks)))

    schema = StructType([StructField("_row", LongType(), False),
                         StructField("_id", LongType(), True),
                         StructField("_sim", DoubleType(), True)])
    columns = list(cols)

    def partial(qpd, ppd):
        if qpd.empty or ppd.empty:
            return pd.DataFrame({"_row": [], "_id": [], "_sim": []})
        ids = ppd["_id"].to_numpy()
        order = np.argsort(ids)
        ids = ids[order]
        P = ppd[columns].to_numpy("float32")[order]
        Q = qpd[columns].to_numpy("float32")
        ex = qpd["_ex"].to_numpy()
        ex = None if (ex < 0).all() else ex
        best, sim = nearest(Q, P, ids, exclude=ex)
        return pd.DataFrame({"_row": qpd["_row"].to_numpy(), "_id": best, "_sim": sim})

    partials = (qdf.groupBy("_qb", "_pb")
                .cogroup(pdf.groupBy("_qb", "_pb"))
                .applyInPandas(partial, schema=schema))

    return (partials.groupBy("_row")
            .agg(F.max(F.struct(F.col("_sim").alias("s"),
                                F.col("_id").alias("i"))).alias("_b"))
            .select("_row", F.col("_b.i").alias("_id"), F.col("_b.s").alias("_sim")))


__all__ = ["BUDGET_BYTES", "nearest", "nearest_grid",
           "nearest_grid_frames"]
