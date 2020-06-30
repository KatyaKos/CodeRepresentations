from models.tree_sampler import TreeSampler


class AstnnSampler(TreeSampler):
    def __init__(self, parser):
        super().__init__(parser)
        self.BASIC_STATEMENTS = [self.STATEMENTS['if'], self.STATEMENTS['for'],
                                 self.STATEMENTS['while'], self.STATEMENTS['do_while']]

    def sample(self, root, node_map, unk_token):
        def tree_to_index(node):
            token = self.parser.get_token(node)
            result = [node_map[token] if token in node_map else node_map[unk_token]]
            children = self.parser.get_non_block_children(node)
            for child in children:
                result.append(tree_to_index(child))
            return result

        blocks = []
        num_nodes, depth = self._get_blocks(root, blocks)
        tree = []
        for b in blocks:
            btree = tree_to_index(b)
            tree.append(btree)
        return tree, num_nodes, depth

    def _get_blocks(self, node, to, depth=1):
        d, num_nodes = depth, 1
        name = self.parser.get_token(node)
        if name in self.BASIC_STATEMENTS or name is self.STATEMENTS['func_def']:
            to.append(node)
            children = self.parser.get_block_children(node)
            for child in children:
                child_name = self.parser.get_token(child)
                if not (child_name in self.BASIC_STATEMENTS
                        or child_name is self.STATEMENTS['func_def'] or child_name is self.STATEMENTS['compound']):
                    to.append(child)
                new_nodes, new_d = self._get_blocks(child, to, depth + 1)
                num_nodes += new_nodes
                d = max(d, new_d)
        elif name is self.STATEMENTS['compound']:
            to.append(name)
            children = self.parser.get_children(node)
            for child in children:
                child_name = self.parser.get_token(child)
                if not (child_name in self.BASIC_STATEMENTS):
                    to.append(child)
                new_nodes, new_d = self._get_blocks(child, to, depth + 1)
                num_nodes += new_nodes
                d = max(d, new_d)
            to.append(self.STATEMENTS['end'])
        else:
            children = self.parser.get_children(node)
            for child in children:
                new_nodes, new_d = self._get_blocks(child, to, depth + 1)
                num_nodes += new_nodes
                d = max(d, new_d)
        return num_nodes, d


