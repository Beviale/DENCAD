"""Hop balls cut from a graph without clustering.

With L graph convolutions the output at a node depends only on the nodes within L hops,
so its ball gives the same value the whole graph would, provided the normalisation keeps
the degrees the graph gives.
"""

from __future__ import annotations

import numpy as np

from dencad.modeling.subgraphs import ContextualSubgraph


class EgoGraph:
    """The L-hop ball of a node, with the degrees of the whole graph."""

    def __init__(self, features: np.ndarray, edges: np.ndarray, hops: int):
        self.X = np.asarray(features, dtype="float32")
        self.hops = max(1, int(hops))
        n = len(self.X)
        e = np.asarray(edges, dtype="int64").reshape(-1, 2)
        both = np.concatenate([e, e[:, ::-1]]) if len(e) else e
        order = np.argsort(both[:, 0], kind="stable")
        self._nbr = both[order, 1] if len(both) else np.zeros(0, dtype="int64")
        heads = both[order, 0] if len(both) else np.zeros(0, dtype="int64")
        self._start = np.searchsorted(heads, np.arange(n + 1))
        self._pos = np.full(n, -1, dtype="int64")
        self.degree = np.bincount(e.ravel(), minlength=n).astype("float32")

    def _incident(self, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Endpoints of every edge touching 'nodes'."""
        counts = self._start[nodes + 1] - self._start[nodes]
        total = int(counts.sum())
        if total == 0:
            z = np.zeros(0, dtype="int64")
            return z, z
        shift = self._start[nodes] - np.concatenate([[0], np.cumsum(counts)[:-1]])
        return (np.repeat(nodes, counts),
                self._nbr[np.arange(total) + np.repeat(shift, counts)])

    def _grow(self, seed: np.ndarray, hops: int) -> np.ndarray:
        """Every node within 'hops' of 'seed', the seed included."""
        reached = np.zeros(len(self.X), dtype=bool)
        reached[seed] = True
        frontier = np.unique(seed)
        for _ in range(hops):
            nxt = np.unique(self._incident(frontier)[1])
            nxt = nxt[~reached[nxt]]
            if not len(nxt):
                break
            reached[nxt] = True
            frontier = nxt
        return np.flatnonzero(reached)

    def _inside(self, idx: np.ndarray) -> np.ndarray:
        """Edges with both ends in 'idx', once each, in local numbering."""
        pos = self._pos
        pos[idx] = np.arange(len(idx))
        u, v = self._incident(idx)
        keep = (u < v) & (pos[v] >= 0)
        out = np.stack([pos[u[keep]], pos[v[keep]]], axis=1)
        pos[idx] = -1
        return out

    def sizes(self) -> np.ndarray:
        """How many nodes each training example will hold."""
        return np.fromiter(
            (len(self._grow(np.array([i], dtype="int64"), self.hops))
             for i in range(len(self.X))), dtype="int64", count=len(self.X))

    def of_node(self, node: int) -> ContextualSubgraph:
        """A node of the graph, reconstructed inside its own ball."""
        idx = self._grow(np.array([node], dtype="int64"), self.hops)
        return ContextualSubgraph(features=self.X[idx], edges=self._inside(idx),
                                  centre=int(np.flatnonzero(idx == node)[0]),
                                  degrees=1.0 + self.degree[idx])

    def of_query(self, vector: np.ndarray,
                 neighbours: np.ndarray) -> ContextualSubgraph:
        """A window that is not in the graph, joined to the neighbours it was given."""
        v = np.asarray(vector, dtype="float32")
        nbr = np.unique(np.asarray(neighbours, dtype="int64"))
        if not len(nbr):
            return ContextualSubgraph(features=v[None, :],
                                      edges=np.zeros((0, 2), dtype="int64"),
                                      centre=0, degrees=np.ones(1, dtype="float32"))
        idx = self._grow(nbr, self.hops - 1)
        inside = self._inside(idx)
        self._pos[idx] = np.arange(len(idx))
        hooked = self._pos[nbr]
        self._pos[idx] = -1

        centre = len(idx)
        edges = np.vstack([inside, np.stack(
            [np.full(len(hooked), centre, dtype="int64"), hooked], axis=1)])
        degrees = np.empty(centre + 1, dtype="float32")
        degrees[:centre] = 1.0 + self.degree[idx]
        degrees[hooked] += 1.0                    # the edge the query brings
        degrees[centre] = 1.0 + len(nbr)
        return ContextualSubgraph(features=np.vstack([self.X[idx], v[None, :]]),
                                  edges=edges, centre=centre, degrees=degrees)


__all__ = ["EgoGraph"]
