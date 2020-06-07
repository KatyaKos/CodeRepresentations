def parse_tree(root, node_map, unk_token, replaced_tokens=set()):
    from models.tbcnn.preprocess.node_parser import _name
    
    def namer(token):
        if token in node_map:
            return token
        return unk_token

    root_json = {
        "node": namer(_name(root)),
        "children": []
    }
    
    def constructor(node, node_json, depth):
        num_nodes = 0
        children = [x[1] for x in node.children()]
        d = depth
        for child in children:
            child_json = {
                "node": namer(_name(child)),
                "children": []
            }
            node_json['children'].append(child_json)
            tmp_nodes, tmp_depth = constructor(child, child_json, depth + 1)
            d = max(d, tmp_depth)
            num_nodes += tmp_nodes
        return num_nodes, d

    num_nodes, max_depth = constructor(root, root_json, 1)

    '''queue_json = [root_json]
    queue = [root]
    while queue:
        current_node = queue.pop(0)
        num_nodes += 1
        current_node_json = queue_json.pop(0)
        children = [x[1] for x in current_node.children()]
        queue.extend(children)
        for child in children:
            child_json = {
                "node": namer(_name(child)),
                "children": []
            }
            current_node_json['children'].append(child_json)
            queue_json.append(child_json)'''

    return root_json, num_nodes, max_depth

