from typing import List


class Parser:
    @staticmethod
    def name() -> str:
        return ""

    def __init__(self):
        self.STATEMENTS = {"compound" : "", "end" : "End",
                           "for" : "", "if" : "", "while" : "", "func_def" : "", "do_while" : ""}
        self.__name__ = "abstract_parser"

    def create(self, name: str):
        from preprocessing.ast_builders.parsers import tree_sitter_parser
        from preprocessing.ast_builders.parsers import pycparser
        if name == 'pycparser':
            return pycparser.PyCParser()
        elif name == 'tree_sitter':
            return tree_sitter_parser.TreeSitter()
        else:
            raise ValueError('No such parser')

    # TODO сделать общий класс на то, что возвращается
    def parse_code(self, code: str):
        return None

    def parse_file(self, filename: str):
        return None

    def get_root(self, ast):
        return None

    def get_children(self, node, mode: str ='all') -> List:
        return []

    def get_non_block_children(self, node) -> List:
        return self._get_children(node, mode='non_block')

    def get_block_children(self, node) -> List:
        return self._get_children(node, mode='block')

    def get_token(self, node) -> str:
        return ""
