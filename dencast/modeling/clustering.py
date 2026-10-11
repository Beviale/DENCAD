"""Core objects and the density-based clustering of DENCAST, on the neighbourhood graph.

Steps 3 to 5 of Algorithm 1. From here the feature vectors are gone: an object is a
node id, its density is its degree, and a cluster is what a propagation along the edges
reaches. That is the point of the design -- the graph is an approximate stand-in for
the geometry, and everything after it costs the same whatever the original
dimensionality was.

**Core objects** are a degree test, `|N(p)| >= min_pts`, which is DBSCAN's definition
with the similarity threshold of the graph playing the part of `eps`.

**The clustering is Algorithm 2.** Every core object starts with a cluster id of its
own and every other node with 0. Then, repeatedly: each core object sends its id along
its edges, each node keeps the largest id it receives, and the loop stops when the
number of propagations falls below `label_change_rate * |E|`. Taking the maximum makes
the result independent of the order the messages arrive in, so neighbouring clusters
collapse into the one with the highest id and no final merge on a single machine is
needed -- which is the paper's central claim.

**Two deliberate departures.**

*Vertices that receive nothing are kept.* Algorithm 2 writes the map phase as
`clusters = E.map{...}`, which replaces the whole collection with the result of mapping
the edges, so a node with no incoming edge vanishes after the first iteration. The
Scala avoids this by using GraphX's `joinVertices`, which leaves unmatched vertices
alone, and Algorithm 3 -- the sequential description of the same procedure -- keeps
them too. Here they are kept: a node that never receives a message is exactly the noise
this is meant to find, so losing it would lose the answer.

*The stopping criterion counts propagations.* The Scala counts `messages.count()`,
which after `aggregateMessages` is the number of *vertices that received something*,
the reduce having already collapsed multiple messages per vertex. That is smaller than
the number of propagating edges, so it stops earlier than Algorithm 2 prescribes. This
counts edges, as the paper does.

PySpark has no GraphX, so the propagation is written as the join-and-reduce the paper's
Algorithm 2 actually describes rather than as the GraphX calls the Scala uses. The two
express the same map/reduce step.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

SRC, DST, SIM = "_src", "_dst", "similarity"
NODE, CLUSTER = "_node", "_cluster"
NOISE = 0


def core_objects(edges, min_pts: int):
    """Nodes whose degree in the neighbourhood graph reaches `min_pts`.

    The graph is undirected and `edges` holds each pair once, so both endpoints are
    counted before the degree test.
    """
    from pyspark.sql import functions as F

    degree = (edges.select(F.col(SRC).alias(NODE))
              .union(edges.select(F.col(DST).alias(NODE)))
              .groupBy(NODE).agg(F.count(F.lit(1)).alias("degree")))
    return degree.filter(F.col("degree") >= min_pts).select(NODE, "degree")


def find_clusters(vertices, edges, cores, label_change_rate: float = 0.05,
                  max_iterations: int = 50):
    """Algorithm 2: propagate cluster ids from the core objects until it settles.

    `vertices` is a one-column frame of every node, `edges` holds each undirected pair
    once, `cores` is the frame `core_objects` returned. Returns one row per node with
    its cluster id, 0 meaning the propagation never reached it.
    """
    from pyspark.sql import functions as F

    # A distinct id per core object. The value only has to be unique and positive:
    # the maximum is what decides, so any order gives the same partition.
    # `row_number()` over a window with no `partitionBy` moves every row into one
    # partition, which is the anti-pattern `lsh._positions` exists to avoid -- and here
    # it buys nothing. A cluster id only has to be a distinct positive integer, and
    # `NODE` already is one, so `NODE + 1` is distinct, clears NOISE at 0, keeps the
    # same relative order the ranking gave, and needs no shuffle at all.
    ranked = cores.select(NODE).withColumn(CLUSTER, F.col(NODE) + F.lit(1))
    state = (vertices.join(ranked, NODE, "left")
             .withColumn(CLUSTER, F.coalesce(F.col(CLUSTER), F.lit(NOISE)))
             .select(NODE, CLUSTER).cache())

    # Both directions: the graph is undirected, and a core has to be able to send
    # along an edge however that edge happens to be stored.
    both = (edges.select(F.col(SRC).alias("u"), F.col(DST).alias("v"))
            .union(edges.select(F.col(DST).alias("u"), F.col(SRC).alias("v")))
            .cache())
    n_edges = both.count()
    threshold = label_change_rate * n_edges
    core_ids = cores.select(F.col(NODE).alias("u")).cache()
    logger.info("propagating over {:,} directed edges, stopping below {:,.0f} "
                "propagations ({:.0%} of them)", n_edges, threshold,
                label_change_rate)

    for step in range(1, max_iterations + 1):
        sender = state.select(F.col(NODE).alias("u"), F.col(CLUSTER).alias("u_id"))
        receiver = state.select(F.col(NODE).alias("v"), F.col(CLUSTER).alias("v_id"))
        # A core object sends its id along an edge when the id is larger than the one
        # the other end currently holds. Only those count as propagations.
        messages = (both.join(core_ids, "u")
                    .join(sender, "u").join(receiver, "v")
                    .filter(F.col("u_id") > F.col("v_id"))
                    .select(F.col("v").alias(NODE), F.col("u_id").alias(CLUSTER)))
        propagations = messages.count()
        if propagations == 0:
            logger.info("  iteration {}: nothing left to propagate", step)
            break
        best = messages.groupBy(NODE).agg(F.max(CLUSTER).alias("_new"))
        previous = state
        state = (state.join(best, NODE, "left")
                 .withColumn(CLUSTER, F.greatest(F.col(CLUSTER),
                                                 F.coalesce(F.col("_new"),
                                                            F.lit(NOISE))))
                 .select(NODE, CLUSTER))
        # Cut the lineage. Each iteration joins onto the previous state, so without
        # this the logical plan gains a level per iteration and the driver runs out of
        # heap building the plan string, not the data. Caching alone does not help: it
        # avoids recomputation, not the growth of the plan. GraphX's Pregel loop, which
        # the Scala uses, truncates for the same reason.
        state = state.localCheckpoint(eager=True)
        previous.unpersist()
        moved = state.filter(F.col(CLUSTER) != F.lit(NOISE)).count()
        logger.info("  iteration {}: {:,} propagations, {:,} nodes now in a cluster",
                    step, propagations, moved)
        if propagations < threshold:
            logger.info("  below the threshold, stopping")
            break
    else:
        logger.warning("  stopped at the {} iteration cap without falling below the "
                       "threshold", max_iterations)

    both.unpersist()
    core_ids.unpersist()
    return state


def summarise(state, label: str = "clusters") -> dict:
    """Sizes of what came out, and how much of it is noise."""
    from pyspark.sql import functions as F

    total = state.count()
    noise = state.filter(F.col(CLUSTER) == F.lit(NOISE)).count()
    sizes = (state.filter(F.col(CLUSTER) != F.lit(NOISE))
             .groupBy(CLUSTER).agg(F.count(F.lit(1)).alias("size")))
    n_clusters = sizes.count()
    stats = sizes.agg(F.max("size").alias("largest"),
                      F.expr("percentile_approx(size, 0.5)").alias("median"),
                      F.sum("size").alias("clustered")).first() if n_clusters else None

    out = {"nodes": int(total), "clusters": int(n_clusters),
           "noise": int(noise), "noise_share": noise / total if total else 0.0}
    if stats is not None:
        out.update({"largest_cluster": int(stats["largest"]),
                    "median_cluster": int(stats["median"]),
                    "largest_share": int(stats["largest"]) / total})
    logger.info("{}: {:,} clusters over {:,} nodes | noise {:,} ({:.2%})",
                label, out["clusters"], out["nodes"], out["noise"], out["noise_share"])
    if stats is not None:
        logger.info("  largest {:,} ({:.1%} of all nodes), median {:,}",
                    out["largest_cluster"], out["largest_share"],
                    out["median_cluster"])
    return out


__all__ = ["core_objects", "find_clusters", "summarise",
           "SRC", "DST", "SIM", "NODE", "CLUSTER", "NOISE"]
