"""Parse nodes from a given data source."""


def parse_nodes(root):
    new_samples = [
        [
            _name(root),  # node
            None,  # parent
            [_name(x[1]) for x in root.children()]  # children
        ]
    ]
    gen_samples = lambda x: new_samples.extend(_create_samples(x))
    _traverse_tree(root, gen_samples)
    return new_samples


def _create_samples(node):
    """Convert a node's children into a sample points."""
    samples = []
    for _, child in node.children():
        sample = [
            _name(child),
            _name(node),
            [_name(x[1]) for x in child.children()]
        ]
        samples.append(sample)

    return samples


def _traverse_tree(root, callback):
    """Traverse a tree and execute the callback on every node."""
    queue = [root]
    while queue:
        current_node = queue.pop(0)
        children = [x[1] for x in current_node.children()]
        queue.extend(children)
        callback(current_node)


def _name(node, lower=True):
    """Get the name of a node, copied from astnn"""
    if isinstance(node, str):
        return node
    name = node.__class__.__name__
    token = name
    is_name = False
    if _is_leaf(node):
        attr_names = node.attr_names
        if attr_names:
            if 'names' in attr_names:
                token = node.names[0]
            elif 'name' in attr_names:
                token = node.name
                is_name = True
            else:
                token = node.value
        else:
            token = name
    else:
        if name == 'TypeDecl':
            token = node.declname
        if node.attr_names:
            attr_names = node.attr_names
            if 'op' in attr_names:
                if node.op[0] == 'p':
                    token = node.op[1:]
                else:
                    token = node.op
    if token is None:
        token = name
    if lower and is_name:
        token = token.lower()
    return token


def _is_leaf(node):
    if isinstance(node, str):
        return True
    return len(node.children()) == 0