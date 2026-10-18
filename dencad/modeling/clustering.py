"""Core objects and the density-based clustering of DENCAST, on the neighbourhood graph."""

from __future__ import annotations


from loguru import logger

from pyspark.sql import functions as F


SRC, DST, DIST = "_src", "_dst", "distance"
NODE, CLUSTER = "_node", "_cluster"
NOISE = 0


def core_objects(edges, min_pts: int):
    """Nodes whose degree in the neighbourhood graph reaches 'min_pts'.

    The graph is undirected and edges holds each pair once.
    """
    degree = (edges.select(F.col(SRC).alias(NODE))
              .union(edges.select(F.col(DST).alias(NODE)))
              .groupBy(NODE).agg(F.count(F.lit(1)).alias("degree")))
    return degree.filter(F.col("degree") >= min_pts).select(NODE, "degree")


def find_clusters(vertices, edges, cores, label_change_rate: float = 0.05,
                  max_iterations: int = 50):
    """Propagate cluster ids from the core objects until it settles."""

    ranked = cores.select(NODE).withColumn(CLUSTER, F.col(NODE) + F.lit(1))
    state = (vertices.join(ranked, NODE, "left")
             .withColumn(CLUSTER, F.coalesce(F.col(CLUSTER), F.lit(NOISE)))
             .select(NODE, CLUSTER).cache())

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
           "SRC", "DST", "DIST", "NODE", "CLUSTER", "NOISE"]
