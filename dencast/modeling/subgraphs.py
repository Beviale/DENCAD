"""Contextual subgraphs: the subgraph an object is scored inside.

An object is placed in the graph the way DENCAST places an unlabelled one -- it takes
the **most similar core object**, by exact cosine and with no threshold, and inherits
that core's cluster. The subgraph is that cluster, with the object attached to the core
it chose and to any other member of the cluster it is similar enough to.

**Not an ego-graph**, though an earlier version of this was. An ego-network is the
subgraph induced by a node and its neighbours within n hops: what decides membership is
distance from the centre. Here membership is decided by the cluster, so a member can sit
many hops from the centre, and the centre itself usually has a single edge -- the one to
the core it chose. The centre defines nothing about the set; the set is the cluster and
the centre is attached to it. A one-hop ego-graph would also be useless, two nodes with
nothing to aggregate.

`argmax` and not a threshold, deliberately. A threshold would leave the most anomalous
objects with no attachment at all, and those are the ones the whole exercise is about;
`argmax` is always defined, so every object gets a subgraph and the score comes from how
badly it fits it rather than from whether it was placed.

**Training examples are built the same way, which is the point.** A training node sits
inside its cluster with all its edges, while a new object arrives attached by one. A
model trained on the first and asked about the second reads that difference for every
new object, anomalous or not, and scores novelty instead of anomaly. So a training
example detaches its node first -- every edge removed, itself excluded from the pool of
candidate cores -- and re-attaches it exactly as inference would. The core it then finds
may belong to a different cluster than the one it came from, which is realistic rather
than a defect.

Exact cosine over the core objects is what Algorithm 4 does: its prediction step is a
cartesian product, not an approximate search. On these window counts that is one matrix
product; on ESA it would be four billion pairs and the LSH would have to come back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from loguru import logger
import numpy as np

from dencast.modeling import placement


@dataclass
class ContextualSubgraph:
    """One example: node features, a dense adjacency, and which row is the centre."""

    features: np.ndarray          # (m, d), the centre last
    adjacency: np.ndarray         # (m, m), symmetric, no self-loops
    centre: int                   # index of the centre, always m - 1
    cluster: int                  # cluster it was attached to, 0 when noise
    core: int                     # index in the training set of the core it chose
    similarity: float             # cosine with that core
    # Degrees of `A + I` in the subgraph **before** any hop pruning, one per kept node.
    # Set only when the subgraph was pruned: the normalisation has to use the degree a
    # node really has, not the one left after its far edges were cut away, or the
    # pruning stops being exact. None means "read them off the adjacency", as before.
    degrees: Optional[np.ndarray] = None


class ContextualSubgraphBuilder:
    """Places objects into a clustered neighbourhood graph and cuts out their context.

    `features` are the training objects, `cluster_of` their cluster ids with 0 meaning
    noise, `core_mask` which of them are core objects, and `edges` the undirected pairs
    of the training graph. `min_similarity` decides the *extra* edges an attached object
    gets beyond the one to its chosen core.
    """

    ATTACH_TO = ("core", "clustered", "any")

    def __init__(self, features: np.ndarray, cluster_of: np.ndarray,
                 core_mask: np.ndarray, edges: np.ndarray,
                 min_similarity: float = 0.99, max_members: Optional[int] = None,
                 attach_to: str = "core", max_hops: Optional[int] = None):
        if not (len(features) == len(cluster_of) == len(core_mask)):
            raise ValueError("features, cluster_of and core_mask disagree in length")
        self.X = np.asarray(features, dtype="float32")
        self.cluster_of = np.asarray(cluster_of, dtype="int64")
        self.core_mask = np.asarray(core_mask, dtype=bool)
        self.min_similarity = min_similarity
        self.max_members = max_members
        # Keep only the nodes that can reach the centre in this many hops. With L graph
        # convolutions the centre's output is a function of its L-hop ball and of
        # nothing else, so cutting at L is exact rather than approximate -- unlike
        # `max_members`, which cuts by cosine and rewrites which paths reach the centre.
        # None keeps the whole cluster.
        self.max_hops = max_hops
        # Set to a live session to place every query over Spark tasks instead of in
        # this process. The answers are the same either way -- verified to the object
        # on all 19,082 of HAI's training queries -- so this decides only where the
        # argmax runs.
        self.spark = None
        # With `spark` set, True splits the eligible pool across tasks and False sends
        # a copy of it to every executor. Same answers either way, verified to the
        # object on all 19,082 of HAI's training queries; only the broadcast has a size
        # it cannot exceed.
        self.partition_pool = True

        norms = np.linalg.norm(self.X, axis=1)
        self.unit = self.X / np.maximum(norms, 1e-12)[:, None]

        # Which training objects an arriving object may attach to. Algorithm 4 uses
        # every labelled object, "any" here; restricting to core objects guarantees a
        # cluster but throws away the border ones, which carry a cluster id too and may
        # well be the most similar thing there is.
        if attach_to not in self.ATTACH_TO:
            raise ValueError(f"attach_to must be one of {self.ATTACH_TO}")
        self.attach_to = attach_to
        if attach_to == "core":
            eligible = self.core_mask
        elif attach_to == "clustered":
            eligible = self.core_mask | (self.cluster_of != 0)
        else:
            eligible = np.ones(len(self.X), dtype=bool)
        self.eligible = eligible
        self.core_ids = np.flatnonzero(eligible)
        if not len(self.core_ids):
            raise ValueError(f"attach_to={attach_to!r} leaves nothing to attach to")

        # One structure per cluster rather than one per example: there are a few
        # hundred clusters and tens of thousands of examples, and the cluster part of
        # every contextual subgraph is the same for all objects that land in it.
        self.members: dict[int, np.ndarray] = {}
        for c in np.unique(self.cluster_of):
            if c == 0:
                continue
            self.members[int(c)] = np.flatnonzero(self.cluster_of == c)
        # Edges kept as pairs per cluster, not as a dense matrix. A dense adjacency is
        # quadratic in the cluster size, and a propagation that merges most of the
        # graph into one component -- which a low similarity threshold does -- then
        # asks for an 18,533 x 18,533 array, 1.4 GB, before `max_members` has had any
        # chance to cut it down. The dense form is built per example instead, over the
        # members that example actually keeps.
        edges = np.asarray(edges)
        self.cluster_edges: dict[int, np.ndarray] = {}
        owner = np.full(len(self.X), 0, dtype="int64")
        for c, idx in self.members.items():
            owner[idx] = c
        same = (owner[edges[:, 0]] == owner[edges[:, 1]]) & (owner[edges[:, 0]] != 0)
        inside = edges[same]
        for c in self.members:
            self.cluster_edges[c] = inside[owner[inside[:, 0]] == c]
        sizes = np.array([len(v) for v in self.members.values()])
        logger.info("{} clusters | members: median {}, largest {} | attach_to={!r}: "
                    "{:,} of {:,} objects eligible ({:,} core, {:,} border, {:,} noise)",
                    len(self.members), int(np.median(sizes)), int(sizes.max()),
                    attach_to, len(self.core_ids), len(self.X),
                    int(self.core_mask.sum()),
                    int((~self.core_mask & (self.cluster_of != 0)).sum()),
                    int((~self.core_mask & (self.cluster_of == 0)).sum()))

    def without_session(self) -> "ContextualSubgraphBuilder":
        """A copy safe to ship to a worker.

        A builder that places over Spark holds the session, and a session cannot be
        serialised: broadcasting one raises CONTEXT_ONLY_VALID_ON_DRIVER. A worker has
        no use for it either, being a task already. The copy is shallow, so the feature
        matrix and the per-cluster structures are shared rather than duplicated.
        """
        import copy

        shipped = copy.copy(self)
        shipped.spark = None
        return shipped

    # ------------------------------------------------------------------ placing

    def nearest_core(self, vectors: np.ndarray,
                     exclude: Optional[np.ndarray] = None) -> tuple[np.ndarray,
                                                                    np.ndarray]:
        """The most similar core object for each row, and that similarity.

        `exclude` removes a training object from its own candidate pool, which is what
        makes a training example look like an unseen one.

        Tiled in `placement`, so the footprint is one block rather than the whole
        `n_queries x n_eligible` matrix. Over Spark tasks when `spark` is set.
        """
        pool = self.X[self.core_ids]
        if self.spark is not None:
            run = (placement.nearest_grid if self.partition_pool
                   else placement.nearest_spark)
            return run(self.spark, vectors, pool, self.core_ids, exclude=exclude)
        return placement.nearest(vectors, pool, self.core_ids, exclude=exclude)

    def build(self, vector: np.ndarray, core: int, similarity: float,
              drop: Optional[int] = None) -> ContextualSubgraph:
        """The contextual subgraph of one object attached to `core`."""
        cluster = int(self.cluster_of[core])
        if cluster == 0:
            # The chosen core carries no cluster id only if the propagation never ran;
            # fall back to the core alone so the example still exists.
            idx = np.array([core])
        else:
            idx = self.members[cluster]
        if drop is not None and drop in idx:
            idx = idx[idx != drop]
        if self.max_members is not None and len(idx) > self.max_members:
            # The most similar members, so a very large cluster still gives the centre
            # the context closest to it. Done before the adjacency is materialised.
            unit = vector / max(np.linalg.norm(vector), 1e-12)
            order = np.argsort(-(self.unit[idx] @ unit.astype("float32")))
            idx = np.sort(idx[order[:self.max_members]])

        m = len(idx)
        position = {int(v): i for i, v in enumerate(idx)}
        unit = vector / max(np.linalg.norm(vector), 1e-12)
        sims = self.unit[idx] @ unit.astype("float32")
        attach = sims >= self.min_similarity
        attach |= idx == core                     # the chosen core, whatever its cosine

        # The member-to-member edges as pairs of local positions. Read off the edge
        # list rather than a matrix: the degrees below need every edge of the cluster,
        # and materialising an m x m array to count them is what hop pruning exists to
        # avoid in the first place.
        pairs = []
        if cluster != 0 and m > 1:
            for u, v in self.cluster_edges[cluster]:
                iu = position.get(int(u))
                iv = position.get(int(v))
                if iu is not None and iv is not None:
                    pairs.append((iu, iv))

        # Degrees of `A + I` over the whole cluster plus the query: one for the self
        # loop, one per member edge, one for the query's own attachment. These are the
        # numbers the normalisation must use whether or not the subgraph is then cut.
        degrees = np.ones(m + 1, dtype="float32")
        for iu, iv in pairs:
            degrees[iu] += 1.0
            degrees[iv] += 1.0
        degrees[:m] += attach.astype("float32")
        degrees[m] += float(attach.sum())

        keep = np.arange(m)
        pruned = False
        if self.max_hops is not None and m > 1:
            # Breadth-first from the centre: hop 1 is what the query attached to, and
            # each further hop follows the member edges. A node further than `max_hops`
            # contributes to no term of the centre's output.
            adjacency: dict[int, list[int]] = {}
            for iu, iv in pairs:
                adjacency.setdefault(iu, []).append(iv)
                adjacency.setdefault(iv, []).append(iu)
            reached = set(np.flatnonzero(attach).tolist())
            frontier = reached
            for _ in range(max(0, self.max_hops - 1)):
                nxt = {w for v in frontier for w in adjacency.get(v, [])} - reached
                if not nxt:
                    break
                reached |= nxt
                frontier = nxt
            if len(reached) < m:
                keep = np.sort(np.fromiter(reached, dtype="int64", count=len(reached)))
                pruned = True

        k = len(keep)
        local = {int(v): i for i, v in enumerate(keep)}
        big = np.zeros((k + 1, k + 1), dtype="float32")
        for iu, iv in pairs:
            a, b = local.get(iu), local.get(iv)
            if a is not None and b is not None:
                big[a, b] = big[b, a] = 1.0
        kept_attach = attach[keep].astype("float32")
        big[k, :k] = big[:k, k] = kept_attach

        features = np.vstack([self.X[idx[keep]],
                              vector.astype("float32")[None, :]])
        return ContextualSubgraph(
            features=features, adjacency=big, centre=k, cluster=cluster,
            core=int(core), similarity=float(similarity),
            degrees=(np.concatenate([degrees[keep], degrees[m:]])
                     if pruned else None))

    # ----------------------------------------------------------------- datasets

    def place_from_edges(self, pairs: np.ndarray, n_queries: int,
                         vectors: Optional[np.ndarray] = None,
                         exclude: Optional[np.ndarray] = None,
                         fallback: bool = True
                         ) -> tuple[np.ndarray, np.ndarray]:
        """The core object each query attaches to, taken from LSH candidate edges.

        `pairs` is (query index, training index, cosine). Among the candidates the LSH
        proposed for a query, the most similar **core** one wins -- `argmax` over those,
        with no threshold, for the same reason the exact version uses one: a threshold
        would drop the objects the exercise is about.

        The promise is the nearest core object however far away it is, so a query whose
        LSH candidates happen to hold no core is not left unplaced: it falls back to an
        exact search over the core objects. The approximation decides how hard it is to
        look, never whether an object gets placed. Set `fallback=False` to see how often
        the band alone is enough, which is what the -1 values then count.
        """
        best = np.full(n_queries, -1, dtype="int64")
        score = np.full(n_queries, -np.inf, dtype="float64")
        if not len(pairs):
            return best, score
        q = pairs[:, 0].astype("int64")
        t = pairs[:, 1].astype("int64")
        s = pairs[:, 2].astype("float64")
        keep = self.eligible[t]
        q, t, s = q[keep], t[keep], s[keep]
        # Ascending, so the last write per query is the most similar one.
        order = np.argsort(s)
        best[q[order]] = t[order]
        score[q[order]] = s[order]
        found = int((best >= 0).sum())
        logger.info("LSH candidates held a core object for {:,} of {:,} queries "
                    "({:.1%})", found, n_queries, found / max(n_queries, 1))
        missing = np.flatnonzero(best < 0)
        if len(missing) and fallback and vectors is not None:
            ex = None if exclude is None else np.asarray(exclude)[missing]
            cores, sims = self.nearest_core(np.asarray(vectors)[missing], exclude=ex)
            best[missing], score[missing] = cores, sims
            logger.info("  the remaining {:,} placed by exact search, their cosine "
                        "with the core median {:.4f}", len(missing),
                        float(np.median(sims)))
        elif len(missing):
            logger.warning("  {:,} queries left unplaced", len(missing))
        return best, score

    def examples_from_edges(self, vectors: np.ndarray, pairs: np.ndarray,
                            drop: Optional[np.ndarray] = None
                            ) -> Iterator[Optional[ContextualSubgraph]]:
        """One contextual subgraph per row of `vectors`, or None where no core was found.

        `drop` names, for each row, a training object to leave out of its contextual subgraph --
        the row's own index when the rows are the training objects themselves, which is
        what makes a training example look like an unseen one.
        """
        cores, sims = self.place_from_edges(pairs, len(vectors), vectors=vectors,
                                            exclude=drop)
        for i in range(len(vectors)):
            if cores[i] < 0:
                yield None
                continue
            yield self.build(vectors[i], int(cores[i]), float(sims[i]),
                             drop=None if drop is None else int(drop[i]))

    def examples(self, vectors: np.ndarray, drop: Optional[np.ndarray] = None
                 ) -> Iterator[ContextualSubgraph]:
        """One subgraph per row, placed by exact argmax over the eligible objects.

        `drop` names, per row, a training object to leave out of its own pool and out of
        its subgraph -- the row's own index when the rows are the training objects. The
        same call serves inference with `drop` left out, which is the point: a training
        example and a new object are placed by the same procedure, so the model cannot
        read the difference and score novelty instead of anomaly.
        """
        vectors = np.asarray(vectors, dtype="float32")
        cores, sims = self.nearest_core(vectors, exclude=drop)
        for i in range(len(vectors)):
            yield self.build(vectors[i], int(cores[i]), float(sims[i]),
                             drop=None if drop is None else int(drop[i]))

    def training_examples(self) -> Iterator[ContextualSubgraph]:
        """Every training object, detached and re-placed as if it were new."""
        return self.examples(self.X, drop=np.arange(len(self.X)))

    def inference_examples(self, vectors: np.ndarray) -> Iterator[ContextualSubgraph]:
        """Objects the graph has never seen, placed by their most similar core."""
        return self.examples(vectors)


def egos_from_rows(members, queries, min_similarity: float,
                   max_members: Optional[int] = None,
                   max_hops: Optional[int] = None):
    """The subgraphs of one cluster, built from rows instead of a driver-side builder.

    `members` carries `_node`, `_f` and `_nb` -- a member's features and the node ids of
    its neighbours inside the cluster; `queries` carries `_row`, `_core`, `_sim`, `_f`
    and optionally `_drop`, the node a training query must leave out because it is
    itself a member. Both are pandas frames, which is what `applyInPandas` hands a task
    and what `pyarrow` hands a replica reading its parquet partition.

    One cluster is the task's whole world, so every member is in it and counts as core;
    the cluster id is a constant and the ids are local positions. `build` is the same
    method the driver-side builder calls, which is what keeps the scores identical
    rather than nearly so.

    Returns the query ids in the order they arrived and their subgraphs.
    """
    rows = np.asarray(queries["_row"].to_numpy(), dtype="int64")
    if not len(rows) or members.empty:
        return rows, []

    members = members.sort_values("_node", kind="stable")
    nodes = members["_node"].to_numpy()
    where = {int(n): i for i, n in enumerate(nodes)}
    X = np.stack([np.asarray(f, dtype="float32") for f in members["_f"]])

    pairs = [(where[int(u)], where[int(v)])
             for u, nb in zip(nodes, members["_nb"]) for v in nb
             if int(v) in where and int(u) < int(v)]
    edges = (np.asarray(pairs, dtype="int64") if pairs
             else np.zeros((0, 2), dtype="int64"))

    mini = ContextualSubgraphBuilder(
        X, np.ones(len(X), dtype="int64"), np.ones(len(X), dtype=bool), edges,
        min_similarity=min_similarity, max_members=max_members,
        attach_to="clustered", max_hops=max_hops)

    drops = (queries["_drop"].to_numpy() if "_drop" in queries
             else np.full(len(rows), -1, dtype="int64"))
    # `where.get` and not `where[...]`: a training query carries its own node id, and
    # most training queries land in a cluster they are not a member of -- there is then
    # nothing to leave out, which is what a miss means here.
    egos = [mini.build(np.asarray(f, dtype="float32"), where[int(c)], float(v),
                       drop=(where.get(int(dp)) if int(dp) >= 0 else None))
            for f, c, v, dp in zip(queries["_f"], queries["_core"], queries["_sim"],
                                   drops)]
    return rows, egos


__all__ = ["ContextualSubgraph", "ContextualSubgraphBuilder", "egos_from_rows"]
