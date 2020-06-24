from models.tree_sampler import TreeSampler


class TbcnnSampler(TreeSampler):
    def __init__(self, parser):
        super().__init__(parser)

    def sample(self, root, node_map, unk_token):
        def namer(token):
            if token in node_map:
                return token
            return unk_token

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
