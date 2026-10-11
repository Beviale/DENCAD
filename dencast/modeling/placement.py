"""Exact placement of an object into the clustered graph.

The neighbourhood graph is built with the LSH, which is what DENCAST is about and what
an `n^2` pass could not do at scale. Placing an object is a different problem: it asks
for **one answer per query**, the most similar eligible training object, so it is a
reduction rather than a selection. A reduction never has to hold more than the running
best, which is why the exact answer is affordable here and the approximation is not
worth its cost.

Measured on HAI's 19,082 scaled training windows: the global argmax for all 3,841 test
windows takes 0.4 s, against roughly 210 s for the LSH `attach` and the million-row
transfer it ends with. The LSH also found only 30.3% of the true edges at the same
cosine, so the approximation was losing placements as well as time.

`budget_bytes` is what keeps this usable on ESA. The naive form materialises
`n_queries x n_eligible` -- 65,376 x 78,308 is 20 GB, and the mask built to exclude an
object from its own pool doubled it. Tiling bounds the footprint by one block whatever
the query count costs.
"""

from __future__ import annotations

from typing import Optional, Sequence

from loguru import logger
import numpy as np

BUDGET_BYTES = 256 * 1024 * 1024


def _unit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype="float32")
    return a / np.maximum(np.linalg.norm(a, axis=1), 1e-12)[:, None]


def nearest(queries: np.ndarray, pool: np.ndarray, pool_ids: np.ndarray,
            exclude: Optional[np.ndarray] = None,
            budget_bytes: int = BUDGET_BYTES) -> tuple[np.ndarray, np.ndarray]:
    """For each query, the most similar object in `pool`, and that cosine.

    `pool_ids` names the pool's members in the caller's own numbering and must be
    ascending. `exclude` gives, per query, one id to remove from its pool, which is what
    makes a training object look like one the graph has never seen.
    """
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
        # Position of each excluded id in the pool, where it is in the pool at all.
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


def nearest_spark(spark, queries: np.ndarray, pool: np.ndarray, pool_ids: np.ndarray,
                  exclude: Optional[np.ndarray] = None, blocks: Optional[int] = None,
                  budget_bytes: int = BUDGET_BYTES
                  ) -> tuple[np.ndarray, np.ndarray]:
    """The same argmax, with each task reducing its own block of queries.

    The textbook distributed argmax: the pool is broadcast, the queries are split, and
    every task returns one answer per query it owns. Nothing is shuffled, because no
    task needs to see another's queries, and what comes back is two numbers a query
    rather than a similarity matrix.

    It is the same answer as `nearest`, verified on HAI: 19,082 of 19,082 queries chose
    the identical object. It is also slower on one machine -- 1.69 s against 0.11 s --
    and the reason is worth knowing rather than working around. `queries @ pool.T` is a
    BLAS `sgemm`, already multithreaded: 142.6 Gflop/s across the cores against 77.6 on
    one. So the driver version is not serial, and splitting it across processes trades
    thread parallelism with shared memory for process parallelism with serialisation.
    What is more, at 1.8x from 20 cores the operation is bound by memory bandwidth, not
    by arithmetic -- and bandwidth is shared by every process on the machine, so another
    process cannot add any. Only another machine can.

    Which is exactly when this version earns its place: when the pool no longer fits in
    one machine's memory, or when the tasks sit on separate hardware with separate
    bandwidth.
    """
    import pandas as pd
    from pyspark.sql.types import (DoubleType, LongType, StructField, StructType)

    if blocks is None:
        blocks = max(1, int(spark.sparkContext.defaultParallelism))
    sc = spark.sparkContext
    held = [sc.broadcast(np.asarray(pool)), sc.broadcast(np.asarray(pool_ids)),
            sc.broadcast(np.asarray(queries))]
    b_pool, b_ids, b_queries = held
    b_exclude = None if exclude is None else sc.broadcast(np.asarray(exclude))

    schema = StructType([StructField("_row", LongType(), False),
                         StructField("_id", LongType(), True),
                         StructField("_sim", DoubleType(), True)])

    def work(pdf):
        rows = pdf["_row"].to_numpy()
        ex = None if b_exclude is None else b_exclude.value[rows]
        got, sims = nearest(b_queries.value[rows], b_pool.value, b_ids.value,
                            exclude=ex, budget_bytes=budget_bytes)
        return pd.DataFrame({"_row": rows, "_id": got, "_sim": sims})

    plan = pd.DataFrame({"_row": np.arange(len(queries), dtype="int64")})
    plan["_blk"] = plan["_row"] % blocks
    got = (spark.createDataFrame(plan).groupBy("_blk")
           .applyInPandas(work, schema=schema)
           .toPandas().sort_values("_row", kind="stable"))
    for b in held:
        b.unpersist()
    if b_exclude is not None:
        b_exclude.unpersist()

    if len(got) != len(queries):
        raise RuntimeError(f"{len(got)} placements came back for "
                           f"{len(queries)} queries")
    logger.info("placed {:,} queries over {} blocks in Spark, against {:,} eligible "
                "objects", len(queries), blocks, len(pool))
    return got["_id"].to_numpy(), got["_sim"].to_numpy()


def nearest_grid(spark, queries: np.ndarray, pool: np.ndarray, pool_ids: np.ndarray,
                 exclude: Optional[np.ndarray] = None,
                 query_blocks: Optional[int] = None,
                 pool_blocks: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """The same argmax with neither side broadcast: both are cut into blocks.

    `nearest_spark` sends the whole pool to every executor. That partitions the work but
    not the data, and it has a ceiling: the day the eligible objects no longer fit in
    one executor's memory the design does not slow down, it stops. For a pipeline whose
    point is that the data is too big for one machine, that is the wrong failure mode.

    Here the pool is cut into `pool_blocks` pieces and the queries into `query_blocks`,
    and the task holding (i, j) sees query block i against pool block j -- a fraction of
    each. It reports, for every query it holds, the best candidate **in its piece of the
    pool**. A second pass takes the best of those partial answers per query:

        query 7 against pool block 1 -> candidate A, 0.81
        query 7 against pool block 2 -> candidate B, 0.93   <- wins
        query 7 against pool block 3 -> candidate C, 0.76

    This is block matrix multiplication followed by a reduction, which is what Spark is
    for. What crosses the network is each side replicated across the other's blocks and
    then `pool_blocks` partial answers a query -- a few numbers, never the similarity
    matrix, and never the whole pool to one place.

    The leave-one-out survives untouched: a query's excluded id travels in its own row,
    so whichever task holds that query applies it to whichever block of the pool it is
    looking at. The cost against `nearest_spark` is a shuffle, and on one machine that
    makes it slower; the gain is that nothing has a size it cannot exceed.
    """
    import pandas as pd

    cols = [f"f{i}" for i in range(queries.shape[1])]
    qu, pu = _unit(queries), _unit(pool)

    q = pd.DataFrame(qu, columns=cols)
    q.insert(0, "_row", np.arange(len(qu), dtype="int64"))
    q.insert(1, "_ex", (np.full(len(qu), -1, dtype="int64") if exclude is None
                        else np.asarray(exclude, dtype="int64")))
    p = pd.DataFrame(pu, columns=cols)
    p.insert(0, "_id", np.asarray(pool_ids).astype("int64"))

    won = (nearest_grid_frames(spark, spark.createDataFrame(q),
                               spark.createDataFrame(p), cols,
                               query_blocks=query_blocks, pool_blocks=pool_blocks)
           .toPandas().sort_values("_row", kind="stable"))
    if len(won) != len(queries):
        raise RuntimeError(f"{len(won)} placements came back for {len(queries)} "
                           f"queries: a query reached no task")
    return won["_id"].to_numpy(), won["_sim"].to_numpy()


def nearest_grid_frames(spark, queries, pool, cols: Sequence[str],
                        query_blocks: Optional[int] = None,
                        pool_blocks: Optional[int] = None):
    """The grid argmax over DataFrames, in and out, with nothing collected.

    `queries` carries `_row`, `_ex` and the feature columns already unit-normalised;
    `pool` carries `_id` and the same columns. The answer is a DataFrame of `_row`,
    `_id` and `_sim`, one row a query, which the caller can join against without ever
    bringing either side down. `nearest_grid` is this function with a pandas frame at
    each end, kept for the callers that still hold numpy.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (DoubleType, LongType, StructField, StructType)

    par = max(1, int(spark.sparkContext.defaultParallelism))
    query_blocks = query_blocks or par
    pool_blocks = pool_blocks or par

    # Each side is replicated across the other's blocks, so task (i, j) can be given
    # query block i and pool block j without either being sent whole to anyone. The
    # block of a row is its own id modulo the block count, which needs no numbering
    # pass of its own.
    #
    # `.cast("long")` on the literals is not decoration. A block id that comes from a
    # column arrives as bigint while `F.lit(i)` is an int, so without the cast the
    # cogroup key is (bigint, int) on one side and (int, bigint) on the other, the two
    # do not match, and most pairs of blocks simply never meet: 11 of 16 silently
    # produced nothing and 9,541 of 19,082 queries came back unplaced.
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
        import pandas as pd

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

    # The reduction: the best of the partial bests, one row a query. `max` over a struct
    # compares the similarity first, so the id that comes back is the winner's.
    return (partials.groupBy("_row")
            .agg(F.max(F.struct(F.col("_sim").alias("s"),
                                F.col("_id").alias("i"))).alias("_b"))
            .select("_row", F.col("_b.i").alias("_id"), F.col("_b.s").alias("_sim")))


__all__ = ["BUDGET_BYTES", "nearest", "nearest_spark", "nearest_grid",
           "nearest_grid_frames"]
