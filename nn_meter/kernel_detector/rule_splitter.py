# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .rule_reader import RuleReader
from .utils.match_helper import MatchHelper
from .utils.fusion_aware_graph import FusionAwareGraph
from nn_meter.utils.graph_tool import ModelGraph


class RuleSplitter:
    def __init__(self, rule_reader: RuleReader):
        self.rule_reader = rule_reader

    def fuse_multiop_blocks(self, model_graph: ModelGraph):
        for type, blocks in self.rule_reader.fusion_units.items():
            for block in blocks:
                subgraphs = model_graph.find_subgraphs(block, MatchHelper.op_type_matcher)
                for subgraph in subgraphs:
                    model_graph.fuse(subgraph.keys(), type)

    def split(self, model_graph: ModelGraph):
        """
        Apply rules to graph
        """
        self.preprocess(model_graph)
        fusion_graph = FusionAwareGraph(model_graph)

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

        self._fusion_graph = fusion_graph
        return fusion_graph.get_basicblocks()

    def preprocess(self, model_graph: ModelGraph):
        self.fuse_multiop_blocks(model_graph)
