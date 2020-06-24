from models.tree_sampler import TreeSampler


class AstnnSampler(TreeSampler):
    def __init__(self, parser):
        super().__init__(parser)

    def sample(self, root, node_map, unk_token):
        def tree_to_index(node):
            token = self.parser.get_token(node)
            result = [node_map[token] if token in node_map else node_map[unk_token]]
            children = self._get_block_children(node, token)
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

    def _get_blocks(self, node, to):
        if self.parser.__name__ == 'pycparser':
            return self._get_pycparser_blocks(node, to)

    def _get_block_children(self, node, token):
        if self.parser.__name__ == 'pycparser':
            return self._get_pycparser_block_children(node, token)

    def _get_pycparser_block_children(self, node, token):
        if isinstance(node, str):
            return []
        children = [child for _, child in node.children()]
        if token in ['FuncDef', 'If', 'While', 'DoWhile','Switch']:
            return [children[0]]
        elif token == 'For':
            return [children[c] for c in range(0, len(children)-1)]
        else:
            return children

    def _get_pycparser_blocks(self, node, to, depth=1):
        d, num_nodes = depth, 1
        children = [c[1] for c in node.children()]
        name = node.__class__.__name__
        if name in ['FuncDef', 'If', 'For', 'While', 'DoWhile']:
            to.append(node)
            if name is not 'For':
                skip = 1
            else:
                skip = len(children) - 1

            for i in range(skip, len(children)):
                child = children[i]
                if child.__class__.__name__ not in ['FuncDef', 'If', 'For', 'While', 'DoWhile', 'Compound']:
                    to.append(child)
                new_nodes, new_d = self._get_pycparser_blocks(child, to, depth + 1)
                num_nodes += new_nodes
                d = max(d, new_d)
        elif name is 'Compound':
            to.append(name)
            for child in children:
                if child.__class__.__name__ not in ['If', 'For', 'While', 'DoWhile']:
                    to.append(child)
                new_nodes, new_d = self._get_pycparser_blocks(child, to, depth + 1)
                num_nodes += new_nodes
                d = max(d, new_d)
            to.append('End')
        else:
            for _, child in node.children():
                new_nodes, new_d = self._get_pycparser_blocks(child, to, depth + 1)
                num_nodes += new_nodes
                d = max(d, new_d)
        return num_nodes, d

