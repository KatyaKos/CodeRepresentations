from tree_sitter import Language, Parser

'''
To use this parser, please, init $LANGUAGE_PATH and $LANGUAGE_NAME.
For example: LANGUAGE_PATH='/user/build/my-languages.so' and LANGUAGE_NAME='go'.
More info: https://pypi.org/project/tree-sitter/
'''
LANGUAGE_PATH = ''
LANGUAGE_NAME = ''

class TreeSitter:
    def __init__(self):
        LANGUAGE = Language(LANGUAGE_PATH, LANGUAGE_NAME)
        self.parser = Parser()
        self.parser.set_language(LANGUAGE)
        self.__name__ = 'tree_sitter'

    def parse_code(self, code):
        return self.parser.parse(bytes(code), "utf8")

    def get_children(self, node):
        if isinstance(node, str):
            return []
        return node.children

    def get_token(self, node):
        if isinstance(node, str):
            return node
        token = node.type
        if token == "\n":
            return "<NEW_LINE>"
        return token
