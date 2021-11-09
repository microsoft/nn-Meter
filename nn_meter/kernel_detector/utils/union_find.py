# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
class UF:
    """
    UnionFind implemented with compression optimization
    """

    def __init__(self, N):
        self._parent = list(range(0, N))

    def find(self, p):
        while p != self._parent[p]:
            p = self._parent[p] = self._parent[self._parent[p]]
        return p

    def union(self, p, q):
        p = self.find(p)
        q = self.find(q)
        self._parent[q] = p

    def connected(self, p, q):
        return self.find(p) == self.find(q)
