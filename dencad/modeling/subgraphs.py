"""Contextual subgraphs: the subgraph an object is scored inside.

An object is placed in the graph the way DENCAST places an unlabelled one -- it takes
the **most similar core object**, by exact cosine and with no threshold, and inherits
that core's cluster. The subgraph is that cluster, with the object attached to the core
it chose and to any other member of the cluster it is similar enough to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np



@dataclass
class ContextualSubgraph:
    """One example: node features, a dense adjacency, and which row is the centre."""

    features: np.ndarray          # (m, d), the centre last
    adjacency: np.ndarray         # (m, m), symmetric, no self-loops
    centre: int                   # index of the centre, always m - 1
    degrees: Optional[np.ndarray] = None


class ContextualSubgraphBuilder:
    """Cuts the context out of a clustered graph: one cluster, plus the query."""

    def __init__(self, features: np.ndarray, cluster_of: np.ndarray,
                 edges: np.ndarray, min_similarity: float = 0.99,
                 max_members: Optional[int] = None,
                 max_hops: Optional[int] = None):
        if len(features) != len(cluster_of):
            raise ValueError("features and cluster_of disagree in length")
        self.X = np.asarray(features, dtype="float32")
        self.cluster_of = np.asarray(cluster_of, dtype="int64")
        self.min_similarity = min_similarity
        self.max_members = max_members
        # Keep only the nodes that can reach the centre in this many hops.
        self.max_hops = max_hops

        norms = np.linalg.norm(self.X, axis=1)
        self.unit = self.X / np.maximum(norms, 1e-12)[:, None]

        self.members: dict[int, np.ndarray] = {}
        for c in np.unique(self.cluster_of):
            if c == 0:
                continue
            self.members[int(c)] = np.flatnonzero(self.cluster_of == c)
        edges = np.asarray(edges)
        self.cluster_edges: dict[int, np.ndarray] = {}
        owner = np.full(len(self.X), 0, dtype="int64")
        for c, idx in self.members.items():
            owner[idx] = c
        same = (owner[edges[:, 0]] == owner[edges[:, 1]]) & (owner[edges[:, 0]] != 0)
        inside = edges[same]
        for c in self.members:
            self.cluster_edges[c] = inside[owner[inside[:, 0]] == c]

    # ------------------------------------------------------------------ placing

    def build(self, vector: np.ndarray, core: int,
              drop: Optional[int] = None) -> ContextualSubgraph:
        """The contextual subgraph of one object attached to `core`."""
        cluster = int(self.cluster_of[core])
        if cluster == 0:
            idx = np.array([core])
        else:
            idx = self.members[cluster]
        if drop is not None and drop in idx:
            idx = idx[idx != drop]
        if self.max_members is not None and len(idx) > self.max_members:
            unit = vector / max(np.linalg.norm(vector), 1e-12)
            order = np.argsort(-(self.unit[idx] @ unit.astype("float32")))
            idx = np.sort(idx[order[:self.max_members]])

        m = len(idx)
        position = {int(v): i for i, v in enumerate(idx)}
        unit = vector / max(np.linalg.norm(vector), 1e-12)
        sims = self.unit[idx] @ unit.astype("float32")
        attach = sims >= self.min_similarity
        attach |= idx == core                     # the chosen core, whatever its cosine

        pairs = []
        if cluster != 0 and m > 1:
            for u, v in self.cluster_edges[cluster]:
                iu = position.get(int(u))
                iv = position.get(int(v))
                if iu is not None and iv is not None:
                    pairs.append((iu, iv))

        degrees = np.ones(m + 1, dtype="float32")
        for iu, iv in pairs:
            degrees[iu] += 1.0
            degrees[iv] += 1.0
        degrees[:m] += attach.astype("float32")
        degrees[m] += float(attach.sum())

        keep = np.arange(m)
        pruned = False
        if self.max_hops is not None and m > 1:
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
            features=features, adjacency=big, centre=k,
            degrees=(np.concatenate([degrees[keep], degrees[m:]])
                     if pruned else None))

    # ----------------------------------------------------------------- datasets

def egos_from_rows(members, queries, min_similarity: float,
                   max_members: Optional[int] = None,
                   max_hops: Optional[int] = None):
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
        X, np.ones(len(X), dtype="int64"), edges, min_similarity=min_similarity,
        max_members=max_members, max_hops=max_hops)

    drops = (queries["_drop"].to_numpy() if "_drop" in queries
             else np.full(len(rows), -1, dtype="int64"))
    egos = [mini.build(np.asarray(f, dtype="float32"), where[int(c)],
                       drop=(where.get(int(dp)) if int(dp) >= 0 else None))
            for f, c, dp in zip(queries["_f"], queries["_core"], drops)]
    return rows, egos


__all__ = ["ContextualSubgraph", "ContextualSubgraphBuilder", "egos_from_rows"]
