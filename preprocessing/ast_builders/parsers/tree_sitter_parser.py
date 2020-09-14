from typing import List

from tree_sitter import Language, Parser
from preprocessing.ast_builders import parser

'''
To use this parser, please, init $LANGUAGE_PATH and $LANGUAGE_NAME.
For example: LANGUAGE_PATH='/user/build/my-languages.so' and LANGUAGE_NAME='go'.
More info: https://pypi.org/project/tree-sitter/
'''
LANGUAGE_PATH = 'ast_builders/parsers/build/my-languages.so'
LANGUAGE_NAME = 'c'


class TreeSitter(parser.Parser):
    @staticmethod
    def name() -> str:
        return "tree_sitter"

    def __init__(self):
        super().__init__()
        LANGUAGE = Language(LANGUAGE_PATH, LANGUAGE_NAME)
        self.parser = Parser()
        self.parser.set_language(LANGUAGE)
        self.__fill_statements__()

    def __fill_statements__(self):
        self.STATEMENTS["compound"] = "compound_statement"
        self.STATEMENTS["for"] = "for_statement"
        self.STATEMENTS["if"] = "if_statement"
        self.STATEMENTS["while"] = "while_statement"
        self.STATEMENTS["func_def"] = "function_definition"
        self.STATEMENTS["do_while"] = "do_statement"

    def parse_code(self, code: str):
        return self.parser.parse(bytes(code, "utf8"))

    def parse_file(self, filename: str):
        with open(filename, 'rb') as fin:
            code = fin.read()
            return self.parser.parse(code)

    def get_root(self, ast):
        return ast.root_node

    def get_children(self, node, mode: str = 'all') -> List:
        if isinstance(node, str):
            return []
        if mode == 'all':
            return node.children
        children = node.children
        token = self.get_token(node)
        if token in ['function_definition', 'if_statement', 'while_statement', 'do_statement', 'switch_statement']:
            if mode == 'non_block':
                return [children[1]]
            elif mode == 'block':
                return children[2:]
        elif token == 'for_statement':
            if mode == 'non_block':
                return children[1:len(children) - 1]
            elif mode == 'block':
                return [children[-1]]
        else:
            return children

    def get_token(self, node) -> str:
        if isinstance(node, str):
            return node
        token = node.type
        if token == "\n":
            return "<NEW_LINE>"
        return token
