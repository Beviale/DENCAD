"""Contextual subgraphs: the subgraph an object is scored inside.

An object is placed in the graph the way DENCAST places an unlabelled one -- it takes
the **closest core object**, by exact euclidean distance and with no cut, and inherits
that core's cluster. The subgraph is that cluster, with the object attached to the core
it chose and to any other member of the cluster it is similar enough to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np



@dataclass
class ContextualSubgraph:
    """One example: node features, its undirected edges, and which row is the centre."""

    features: np.ndarray          # (m, d), the centre last
    edges: np.ndarray             # (e, 2), one row per undirected edge, no self-loops
    centre: int                   # index of the centre, always m - 1
    degrees: Optional[np.ndarray] = None


class ContextualSubgraphBuilder:
    """Cuts the context out of a clustered graph: one cluster, plus the query."""

    def __init__(self, features: np.ndarray, cluster_of: np.ndarray,
                 edges: np.ndarray, eps: float = 1.0,
                 max_members: Optional[int] = None,
                 max_hops: Optional[int] = None,
                 node_degrees: Optional[np.ndarray] = None):
        if len(features) != len(cluster_of):
            raise ValueError("features and cluster_of disagree in length")
        self.X = np.asarray(features, dtype="float32")
        self.cluster_of = np.asarray(cluster_of, dtype="int64")
        self.eps = eps
        self.max_members = max_members
        self.max_hops = max_hops

        self.node_degrees = (None if node_degrees is None
                             else np.asarray(node_degrees, dtype="float32"))

        self.sq = (self.X ** 2).sum(1)
        self._pos = np.full(len(self.X), -1, dtype="int64")

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

    def _to(self, idx: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Euclidean distance from 'vector' to the members in 'idx'."""
        v = vector.astype("float32")
        d = self.sq[idx] + float(v @ v) - 2.0 * (self.X[idx] @ v)
        return np.sqrt(np.maximum(d, 0.0))

    def _context(self, core: int) -> tuple[np.ndarray, np.ndarray]:
        """Candidate members and the edges among them: the core's cluster."""
        cluster = int(self.cluster_of[core])
        if cluster == 0:
            return np.array([core]), np.zeros((0, 2), dtype="int64")
        return self.members[cluster], self.cluster_edges[cluster]

    def build(self, vector: np.ndarray, core: int,
              drop: Optional[int] = None) -> ContextualSubgraph:
        """The contextual subgraph of one object attached to `core`."""
        idx, context_edges = self._context(core)
        if drop is not None and drop in idx:
            idx = idx[idx != drop]
        gap = self._to(idx, vector)
        if self.max_members is not None and len(idx) > self.max_members:
            order = np.sort(np.argsort(gap)[:self.max_members])
            idx, gap = idx[order], gap[order]

        m = len(idx)
        attach = gap <= self.eps
        attach |= idx == core                     # the chosen core, however far

        pos = self._pos
        pos[idx] = np.arange(m)
        edges = context_edges if m > 1 else np.zeros((0, 2), dtype="int64")
        iu, iv = pos[edges[:, 0]], pos[edges[:, 1]]
        inside = (iu >= 0) & (iv >= 0)
        iu, iv = iu[inside], iv[inside]
        pos[idx] = -1

        degrees = np.ones(m + 1, dtype="float32")
        if self.node_degrees is None:
            degrees[:m] += np.bincount(np.concatenate([iu, iv]),
                                       minlength=m).astype("float32")
            degrees[:m] += attach
        else:
            degrees[:m] += self.node_degrees[idx]
            if drop is None:
    
                degrees[:m] += attach
        degrees[m] += float(attach.sum())

        keep = np.arange(m)
        pruned = False
        if self.max_hops is not None and m > 1:
            reached = attach.copy()
            for _ in range(max(0, self.max_hops - 1)):
                grown = reached.copy()
                grown[iv[reached[iu]]] = True
                grown[iu[reached[iv]]] = True
                if grown.sum() == reached.sum():
                    break
                reached = grown
            if reached.sum() < m:
                keep = np.flatnonzero(reached)
                pruned = True

        k = len(keep)
        if pruned:
            local = np.full(m, -1, dtype="int64")
            local[keep] = np.arange(k)
            a, b = local[iu], local[iv]
            both = (a >= 0) & (b >= 0)
            a, b = a[both], b[both]
        else:
            a, b = iu, iv
        hooked = np.flatnonzero(attach[keep])
        pairs = np.empty((len(a) + len(hooked), 2), dtype="int32")
        pairs[:len(a), 0], pairs[:len(a), 1] = a, b
        pairs[len(a):, 0], pairs[len(a):, 1] = k, hooked

        features = np.vstack([self.X[idx[keep]],
                              vector.astype("float32")[None, :]])
        return ContextualSubgraph(
            features=features, edges=pairs, centre=k,
            degrees=(np.concatenate([degrees[keep], degrees[m:]])
                     if pruned or self.node_degrees is not None else None))

    # ----------------------------------------------------------------- datasets

def egos_from_rows(members, queries, eps: float,
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

    
    degrees = np.fromiter((len(nb) for nb in members["_nb"]), dtype="float32",
                          count=len(nodes))
    mini = ContextualSubgraphBuilder(
        X, np.ones(len(X), dtype="int64"), edges, eps=eps,
        max_members=max_members, max_hops=max_hops, node_degrees=degrees)

    drops = (queries["_drop"].to_numpy() if "_drop" in queries
             else np.full(len(rows), -1, dtype="int64"))
    egos = [mini.build(np.asarray(f, dtype="float32"), where[int(c)],
                       drop=(where.get(int(dp)) if int(dp) >= 0 else None))
            for f, c, dp in zip(queries["_f"], queries["_core"], drops)]
    return rows, egos


__all__ = ["ContextualSubgraph", "ContextualSubgraphBuilder", "egos_from_rows"]
