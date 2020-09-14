from typing import List

from pycparser import c_parser, parse_file
from preprocessing.ast_builders.parser import Parser


class PyCParser(Parser):
    @staticmethod
    def name() -> str:
        return "pycparser"

    def __init__(self):
        super().__init__()
        self.parser = c_parser.CParser()
        self.__fill_statements__()

    def __fill_statements__(self):
        self.STATEMENTS["compound"] = "Compound"
        self.STATEMENTS["for"] = "For"
        self.STATEMENTS["if"] = "If"
        self.STATEMENTS["while"] = "While"
        self.STATEMENTS["func_def"] = "FuncDef"
        self.STATEMENTS["do_while"] = "DoWhile"

    def parse_code(self, code: str):
        #return self.parser.parse(code)
        parser = c_parser.CParser()
        asp = parser.parse(code)


    def parse_file(self, filename: str):
        return parse_file(filename, use_cpp=True)

    def get_children(self, node, mode: str='all') -> List:
        if isinstance(node, str):
            return []
        if mode == 'all':
            return [x[1] for x in node.children()]
        token = self.get_token(node)
        children = [child for _, child in node.children()]
        if token in ['FuncDef', 'If', 'While', 'DoWhile', 'Switch']:
            if mode == 'non_block':
                return [children[0]]
            elif mode == 'block':
                return children[1:]
        elif token == 'For':
            if mode == 'non_block':
                return children[:len(children) - 1]
            elif mode == 'block':
                return [children[-1]]
        else:
            return children

    def get_root(self, ast):
        return ast

    def get_token(self, node, lower: bool =True) -> str:
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

    def is_leaf(self, node) -> bool:
        if isinstance(node, str):
            return True
        return len(node.children()) == 0
