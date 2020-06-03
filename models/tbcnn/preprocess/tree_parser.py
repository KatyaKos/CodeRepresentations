def parse_tree(root):
    return _traverse_tree(root)


def _traverse_tree(root):
    from models.tbcnn.preprocess.node_parser import _name

    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),
        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        num_nodes += 1
        current_node_json = queue_json.pop(0)
        children = [x[1] for x in current_node.children()]
        queue.extend(children)
        for child in children:
            child_json = {
                "node": _name(child),
                "children": []
            }
            current_node_json['children'].append(child_json)
            queue_json.append(child_json)

    return root_json, num_nodes
