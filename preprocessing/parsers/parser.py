class Parser:
    @staticmethod
    def name():
        return ""

    def __init__(self):
        self.STATEMENTS = {"compound" : "", "end" : "End",
                           "for" : "", "if" : "", "while" : "", "func_def" : "", "do_while" : ""}
        self.__name__ = "abstract_parser"

    def parse_code(self, code):
        return None

    def get_root(self, ast):
        return None

    def get_children(self, node, mode='all'):
        return []

    def get_non_block_children(self, node):
        return self.get_children(node, mode='non_block')

    def get_block_children(self, node):
        return self.get_children(node, mode='block')

    def get_token(self, node):
        return ""
