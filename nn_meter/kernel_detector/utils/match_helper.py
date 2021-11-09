# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
class MatchHelper:
    @classmethod
    def op_type_matcher(cls, node_1, node_2):
        if "type" in node_1 and "type" in node_2:
            if "_tagged" in node_1 or "_tagged" in node_2:
                return False

            if node_1["type"] == "dummy" or node_2["type"] == "dummy":
                return True
            return node_1["type"] == node_2["type"]
        else:
            return False

    @staticmethod
    def strip_useless_nodes(model_graph):
        stripped_nodes = ["Const", "Identity"]

        graph = model_graph.get_graph()
        removed_node = []
        for key, value in graph.items():
            if value["attr"]["type"] in stripped_nodes:
                removed_node.append(key)

        for key in removed_node:
            del graph[key]

        model_graph.refresh()

    @staticmethod
    def tag_matched_nodes(model_graph, matched_subgraph):
        for matched_unit in matched_subgraph:
            for node_name in matched_unit.keys():
                model_graph.add_node_attr(node_name, "_tagged", "")

    @staticmethod
    def get_untagged_nodes(model_graph):
        untagged_node = []
        for node in model_graph.get_graph().keys():
            if "_tagged" not in model_graph.get_node_attr(node)["attr"]:
                untagged_node.append(node)
        return untagged_node
