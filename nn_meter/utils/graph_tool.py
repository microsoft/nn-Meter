# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import copy
import json
import logging
from .utils import NumpyEncoder
logging = logging.getLogger("nn-Meter")


class ModelGraph:
    def __init__(self, filename=None, graph=None):
        if filename is not None:
            self.graph = json.load(open(filename, "r"))
        elif graph is not None:
            self.graph = copy.deepcopy(graph)
        else:
            self.graph = {}

    def node(self, name, inbound_nodes=None):
        self.graph[name] = {}
        if inbound_nodes is not None:
            self.graph[name]["inbounds"] = inbound_nodes
            for node in inbound_nodes:
                if node not in self.graph.keys():
                    self.graph[node] = {}
                if "outbounds" not in self.graph[node].keys():
                    self.graph[node]["outbounds"] = []
                self.graph[node]["outbounds"].append(name)

    def refresh(self):
        last_remove_nodes_cnt = -1
        while True:
            for name in self.graph.keys():
                self.graph[name]["outbounds"] = []

            for name in self.graph.keys():
                if "inbounds" in self.graph[name].keys():
                    for node in self.graph[name]["inbounds"]:
                        if node not in self.graph.keys():
                            while node in self.graph[name]["inbounds"]:
                                self.graph[name]["inbounds"].remove(node)
                        else:
                            if "outbounds" not in self.graph[node].keys():
                                self.graph[node]["outbounds"] = []

                            self.graph[node]["outbounds"].append(name)

            spare_nodes = []
            for name in self.graph.keys():
                if (
                    len(self.graph[name]["outbounds"]) == 0
                    and len(self.graph[name]["inbounds"]) == 0
                ):
                    spare_nodes.append(name)

            if last_remove_nodes_cnt == 0 and len(spare_nodes) == 0:
                break

            last_remove_nodes_cnt = len(spare_nodes)
            for removing_node_name in spare_nodes:
                del self.graph[removing_node_name]

    def get_graph(self):
        return self.graph

    def get_node_inbounds(self, name):
        if "inbounds" in self.graph[name]:
            return self.graph[name]["inbounds"]
        else:
            return []

    def get_node_outbounds(self, name):
        if "outbounds" in self.graph[name]:
            return self.graph[name]["outbounds"]
        else:
            return []

    def set_node_inbounds(self, name, inbounds):
        self.graph[name]["inbounds"] = inbounds

    def set_node_outbounds(self, name, outbounds):
        self.graph[name]["outbounds"] = outbounds

    def remove_node_inbounds(self, name, inbound):
        if inbound in self.graph[name]["inbounds"]:
            self.graph[name]["inbounds"].remove(inbound)

    def remove_node_outbounds(self, name, outbound):
        if outbound in self.graph[name]["outbounds"]:
            self.graph[name]["outbounds"].remove(outbound)

    def add_node_inbounds(self, name, inbound):
        self.graph[name]["inbounds"].append(inbound)

    def add_node_outbounds(self, name, outbound):
        self.graph[name]["outbounds"].append(outbound)

    def get_graph_head(self):
        self.heads = []
        for (key, value) in self.graph.items():
            if "inbounds" not in value.keys() or len(value["inbounds"]) == 0:
                self.heads.append(key)
        return self.heads

    def get_graph_tail(self):
        self.tails = []
        for (key, value) in self.graph.items():
            if "outbounds" not in value.keys() or len(value["outbounds"]) == 0:
                self.tails.append(key)
        return self.tails

    def add_node_attr(self, name, attr_key, attr_value):
        if name not in self.graph.keys():
            self.graph[name] = {}
        self.graph[name]["attr"]["attr"][attr_key] = attr_value

    def set_node_attr(self, name, attr):
        if name not in self.graph.keys():
            self.graph[name] = {}
        self.graph[name]["attr"] = attr

    def get_node_attr(self, name):
        if name in self.graph.keys():
            return self.graph[name]["attr"]
        else:
            return None

    def get_node_type(self, name):
        if name in self.graph.keys() and "attr" in self.graph[name].keys():
            return self.graph[name]["attr"]["type"]
        else:
            logging.info(name, self.graph[name])
            return None

    def get_root_node(self, subgraph):
        root = next(iter(subgraph))

        flag = True
        while flag:
            flag = False
            for inbound in self.get_node_inbounds(root):
                if inbound in subgraph:
                    flag = True
                    root = inbound
                    break

        return root

    def fuse(self, subgraph, type, name=None, attr=None, is_block=True):
        """
        subgraph: list of node name
        Nothing will be done if subgraph doesn't exist in self
        """
        for node in subgraph:
            if node not in self.graph:
                return False

        if name is None:
            name = ";".join(subgraph)

        if attr is None:
            root_node = self.get_root_node(subgraph)
            attr = self.get_node_attr(root_node)
        attr["type"] = type
        if is_block:
            attr["attr"]["primitive_nodes"] = list(subgraph)

        self.graph[name] = {
            "attr": attr,
            "inbounds": [],
            "outbounds": [],
        }

        for node in subgraph:
            for inbound in self.get_node_inbounds(node):
                if inbound not in subgraph:
                    if inbound not in self.get_node_inbounds(name):
                        self.add_node_inbounds(name, inbound)
                    self.remove_node_outbounds(inbound, node)
                    if name not in self.get_node_outbounds(inbound):
                        self.add_node_outbounds(inbound, name)
            for outbound in self.get_node_outbounds(node):
                if outbound not in subgraph:
                    if outbound not in self.get_node_outbounds(name):
                        self.add_node_outbounds(name, outbound)
                    self.remove_node_inbounds(outbound, node)
                    if name not in self.get_node_inbounds(outbound):
                        self.add_node_inbounds(outbound, name)

        for node in subgraph:
            del self.graph[node]

        return True

    def plot_graphs(self, comment="Network Graph View"):
        from graphviz import Digraph

        dot = Digraph(comment=comment)
        for (key, value) in self.graph.items():
            dot.node(key, key)
            if "inbounds" in value.keys():
                for node in value["inbounds"]:
                    dot.edge(
                        node,
                        key,
                        label=", ".join(str(x) for x in value["attr"]["output_shape"]),
                    )
        dot.render("graph.gv", view=False)

    def plot_networkx_graph(self):
        import matplotlib.pyplot as plt
        import networkx as nx

        plt.subplot(121)
        nx.draw(self.get_networkx_graph(), with_labels=True, font_weight="bold")
        plt.show()

    def get_networkx_graph(self):
        import networkx as nx

        G = nx.MultiDiGraph()
        for (key, value) in self.graph.items():
            G.add_node(key, type=value["attr"]["type"], **value["attr"]["attr"])
            if "inbounds" in value.keys():
                for node in value["inbounds"]:
                    G.add_edge(node, key)
        self.graphx = G
        return G

    def match_isomorph_vf2(self):
        pass

    def find_subgraphs(self, sub_graph, match_func):
        from networkx.algorithms import isomorphism as iso

        GM = iso.MultiDiGraphMatcher(
            self.get_networkx_graph(),
            sub_graph.get_networkx_graph(),
            node_match=match_func,
        )

        matches = []
        for match in GM.subgraph_isomorphisms_iter():
            matches.append(
                {
                    key: value
                    for key, value in match.items()
                    if sub_graph.get_node_type(value) != "dummy"
                }
            )
        return matches

    def find_weight_roots(self, layer_name):
        weight_roots = []
        weights_nodes = []
        for inbound in self.graph[layer_name]["inbounds"]:
            if (
                self.graph[inbound]["attr"]["type"] == "Identity"
                and len(self.graph[inbound]["inbounds"]) == 1
            ):
                if (
                    self.graph[self.graph[inbound]["inbounds"][0]]["attr"]["type"]
                    == "Const"
                ):
                    weight_roots.append(inbound)
                    weights_nodes.append(inbound)
                    weights_nodes.append(self.graph[inbound]["inbounds"][0])

            if (
                self.graph[inbound]["attr"]["type"] == "Const"
                and len(self.graph[inbound]["inbounds"]) == 0
            ):
                weight_roots.append(inbound)
                weights_nodes.append(inbound)

        return weight_roots, weights_nodes

    def get_subgraphs(self, sub_graph, match_func):
        import tensorflow as tf
        import copy

        fetched_subgraphs = self.find_subgraphs(sub_graph, match_func)
        tar_sub_graphs = []
        for sub_fetch_graph in fetched_subgraphs:
            tar_sub_graphs.append(tf.GraphDef())

            for op_entry in sub_fetch_graph.keys():
                # --- Repleace dummy op ---
                if (
                    sub_graph.get_graph()[sub_fetch_graph[op_entry]]["attr"]["type"]
                    == "dummy"
                ):
                    dummy_op = tar_sub_graphs[-1].node.add()
                    dummy_op.op = "Identity"
                    dummy_op.name = sub_fetch_graph[op_entry]
                    dummy_op.input.extend(
                        sub_graph.get_graph()[sub_fetch_graph[op_entry]]["inbounds"]
                    )
                    dummy_op.attr["T"].type = 1
                else:
                    # --- Fetch the main op ---
                    node = copy.deepcopy(self.graph[op_entry]["attr"]["node"])

                    node.name = sub_fetch_graph[op_entry]

                    del node.input[:]
                    node.input.extend(
                        sub_graph.get_graph()[sub_fetch_graph[op_entry]]["inbounds"]
                    )
                    # --- Fetch the constant op ---
                    roots, nodes = self.find_weight_roots(op_entry)
                    for weight_root in roots:
                        node.input.append(weight_root)

                    for weight_node in nodes:
                        tar_sub_graphs[-1].node.append(
                            self.graph[weight_node]["attr"]["node"]
                        )

                    tar_sub_graphs[-1].node.append(node)

            # tf.io.write_graph(tar_sub_graphs[-1], '', 'a.pb')
        return tar_sub_graphs

    def dump_json(self, filename):
        with open(filename, "w+") as fp:
            json.dump(
                self.graph,
                fp,
                indent=4,
                skipkeys=True,
                sort_keys=True,
                cls=NumpyEncoder,
            )
