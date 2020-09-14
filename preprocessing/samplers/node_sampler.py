from typing import List

from preprocessing.ast_builders.parser import Parser


class NodeSampler:
    @staticmethod
    def name() -> str:
        return "basic"

    def __init__(self, parser: Parser):
        self.parser = parser

    def create(self, name: str):
        if name == 'basic':
            return self
        else:
            raise ValueError('No such node sampler')

    # TODO сделать класс AST деревьев общий для всех парсеров???
    def sample_node_sequences(self, ast) -> List[str]:
        sequence = []
        self._sample_seq(self.parser.get_root(ast), sequence)
        return sequence

    def _sample_seq(self, root, to: List[str]):
        current_token = self.parser.get_token(root)
        to.append(current_token)
        for child in self.parser.get_children(root):
            self._sample_seq(child, to)
        if current_token is self.parser.STATEMENTS['compound']:
            to.append(self.parser.STATEMENTS['end'])
