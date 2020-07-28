import pandas as pd


class TreeSampler:
    def __init__(self, parser, node_map, max_tree_size, min_tree_size, max_depth, unknown_node):
        self.parser = parser
        self.node_map = node_map
        self.max_tree_size = max_tree_size
        self.min_tree_size = min_tree_size
        self.max_depth = max_depth
        self.unknown_node = unknown_node
        self.STATEMENTS = parser.STATEMENTS

    @staticmethod
    def name():
        return "basic"

    def sample_trees(self, sources):
        labels = set()
        result = []

        for id, ast, label in zip(sources['id'], sources['code'], sources['label']):
            sample, num_nodes, depth = self.sample(self.parser.get_root(ast))
            if num_nodes > self.max_tree_size or num_nodes < self.min_tree_size or depth > self.max_depth:
                continue
            datum = [id, sample, label]
            labels.add(label)
            result.append(datum)

        result = pd.DataFrame(result, columns=['id', 'code', 'label'])
        labels = list(labels)
        return result, labels

    def sample(self, root):
        def namer(token):
            if token in self.node_map:
                return token
            return self.unknown_node

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