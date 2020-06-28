from pycparser import c_parser


class PyCParser:
    def __init__(self):
        self.parser = c_parser.CParser()
        self.__name__ = 'pycparser'

    def parse_code(self, code):
        return self.parser.parse(code)

    def get_children(self, node):
        if isinstance(node, str):
            return []
        return [x[1] for x in node.children()]

    def get_root(self, ast):
        return ast

    def get_token(self, node, lower=True):
        if isinstance(node, str):
            return node
        name = node.__class__.__name__
        token = name
        is_name = False
        if self.is_leaf(node):
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

    def is_leaf(self, node):
        if isinstance(node, str):
            return True
        return len(node.children()) == 0
