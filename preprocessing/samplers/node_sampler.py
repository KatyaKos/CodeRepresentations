class NodeSampler:
    @staticmethod
    def name():
        return "basic"

    def __init__(self, parser):
        self.parser = parser

    def sample_nodes(self, ast):
        sequence = []
        self.sample(self.parser.get_root(ast), sequence)
        return sequence

    def sample(self, root, to):
        current_token = self.parser.get_token(root)
        to.append(current_token)
        for child in self.parser.get_children(root):
            self.sample(child, to)
        if current_token is self.parser.STATEMENTS['compound']:
            to.append(self.parser.STATEMENTS['end'])