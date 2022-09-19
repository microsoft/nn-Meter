# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import networkx as nx
from .union_find import UF
from nn_meter.utils.graph_tool import ModelGraph


class FusionAwareGraph:
    def __init__(self, model_graph: ModelGraph):
        self._model_graph = model_graph
        self._dag = list(nx.topological_sort(model_graph.get_networkx_graph()))
        self._uf = UF(len(self._dag))

        reverse = {}
        for index, name in enumerate(self._dag):
            reverse[name] = index
        outbounds = []
        inbounds = []
        for index, name in enumerate(self._dag):
            outbounds.append(
                {reverse[outbound] for outbound in self._model_graph.get_node_outbounds(name)}
            )
            inbounds.append(
                {reverse[inbound] for inbound in self._model_graph.get_node_inbounds(name)}
            )

        self._outbounds = outbounds
        self._inbounds = inbounds
        self._ready = [not inbounds[i] for i in range(0, len(self))]
        self._types = [model_graph.get_node_type(name) for name in self._dag]

    @property
    def nodes(self):
        return self._dag

    def __len__(self):
        return len(self._dag)

    def __getitem__(self, key):
        return self._dag[key]

    def fuse(self, node, outnode, update=False):
        """
        node should be root, outnode should be an unfused single node
        """
        self._uf.union(node, outnode)
        if not update:
            self._outbounds[node] = self._outbounds[outnode]
        else:
            self._outbounds[node].update(self._outbounds[outnode])

    def mark_ready(self, node):
        self._ready[node] = True

    def is_ready(self, node):
        for inbound in self._inbounds[node]:
            if not self.is_ready[inbound]:
                return False
        return True

    def is_visited(self, node):
        return self._ready[node]

    def get_outbounds(self, node):
        return self._outbounds[node]

    def get_inbounds(self, node):
        return self._inbounds[node]

    def get_type(self, node):
        return self._types[node]

    def get_basicblocks(self):
        bbs = []

        for _ in range(0, len(self)):
            bbs.append([])

        for i in range(0, len(self)):
            root = self._uf.find(i)
            bbs[root].append(self[i])

        bbs = [bb for bb in bbs if bb]
        return bbs

    def find_root(self, node):
        return self[self._uf.find(node)]

    def is_fused(self, node):
        return self._uf.find(node) != node

    def is_connected(self, p, q):
        return self._uf.connected(p, q)
