# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .rule_reader import RuleReader
from nn_meter.utils.graphe_tool import Graphe
from nn_meter.kerneldetection.utils.match_helper import MatchHelper
from nn_meter.kerneldetection.utils.fusion_aware_graph import FusionAwareGraph


class RuleSplitter:
    def __init__(self, rule_reader: RuleReader):
        self.rule_reader = rule_reader

    def fuse_multiop_blocks(self, graph: Graphe):
        for type, blocks in self.rule_reader.fusion_units.items():
            for block in blocks:
                subgraphs = graph.find_subgraphs(
                    block, MatchHelper.op_type_matcher)
                for subgraph in subgraphs:
                    graph.fuse(subgraph.keys(), type)

    def split(self, graph: Graphe):
        """
        Apply rules to graph
        """
        self.preprocess(graph)
        fusion_graph = FusionAwareGraph(graph)

        i = -1
        while i < len(fusion_graph) - 1:
            i += 1
            if fusion_graph.is_fused(i):
                continue
            fusion_graph.mark_ready(i)
            if not fusion_graph.get_outbounds(i):
                continue
            # MON
            mon = self.rule_reader.query_rule("MON")
            if mon == 0:  # can't fuse if having multiple out node
                if len(fusion_graph.get_outbounds(i)) > 1:
                    continue
            # FN: TODO: which one is the first node
            fused = False
            for j in fusion_graph.get_outbounds(i):
                if fusion_graph.is_fused(j):
                    continue
                outnode_type = fusion_graph.get_type(j)
                node_type = fusion_graph.get_type(i)
                if not self.rule_reader.is_fusible(node_type, outnode_type):
                    continue
                # RT
                if self.rule_reader.query_rule("RT"):
                    if not fusion_graph.is_ready(j):
                        continue
                # fuse node
                if mon == 0:
                    fusion_graph.fuse(i, j)
                else:
                    fusion_graph.fuse(i, j, True)
                fusion_graph.mark_ready(j)
                fused = True
                if mon == 1:  # only fused to first outnode
                    break
            if fused:
                i -= 1

        return fusion_graph.get_basicblocks()

    def preprocess(self, graph: Graphe):
        self.fuse_multiop_blocks(graph)
