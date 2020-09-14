from typing import Dict, Tuple

from preprocessing.data_workers.data_worker import DataWorker
from preprocessing.ast_builders.parser import Parser


class TreeSampler:
    @staticmethod
    def name() -> str:
        return "basic"

    def __init__(self, parser: Parser, node_map: Dict[str, int], batch_size:int,
                 max_tree_size: int, min_tree_size: int, max_depth: int, unknown_node: str):
        self.parser = parser
        self.node_map = node_map
        self.batch_size = batch_size
        self.max_tree_size = max_tree_size
        self.min_tree_size = min_tree_size
        self.max_depth = max_depth
        self.unknown_node = unknown_node
        self.STATEMENTS = parser.STATEMENTS

    def create(self, name: str):
        if name == 'basic':
            return self
        elif name == 'astnn':
            from preprocessing.samplers import tree_astnn_sampler
            return tree_astnn_sampler.TreeAstnnSampler(self.parser, self.node_map, self.max_tree_size,
                                                       self.min_tree_size, self.max_depth, self.unknown_node)
        else:
            raise ValueError("No such tree sampler")

    def sample_trees(self, source: DataWorker):
        new_asts = {}
        asts = source.get_data('code')
        for id in asts:
            sample, num_nodes, depth = self.sample(self.parser.get_root(asts[id]))
            if num_nodes > self.max_tree_size or num_nodes < self.min_tree_size or depth > self.max_depth:
                continue
            new_asts[id] = {'code': sample}
        source.update(new_asts)

    def sample(self, root) -> Tuple[Dict, int, int]:
        def namer(token):
            return self.node_map[token] if token in self.node_map else self.node_map[self.unknown_node]

        root_json = {
            "node": namer(self.parser.get_token(root)),
            "children": []
        }

        def constructor(node, node_json, depth):
            num_nodes = 1
            children = self.parser.get_children(node)
            d = depth
            for child in children:
                child_json = {
                    "node": namer(self.parser.get_token(child)),
                    "children": []
                }
                node_json['children'].append(child_json)
                tmp_nodes, tmp_depth = constructor(child, child_json, depth + 1)
                d = max(d, tmp_depth)
                num_nodes += tmp_nodes
            return num_nodes, d

        num_nodes, max_depth = constructor(root, root_json, 1)
        return root_json, num_nodes, max_depth