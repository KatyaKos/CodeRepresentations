import numpy as np


class Vectorizer:
    @staticmethod
    def name():
        return "basic"

    def __init__(self, embedding_dim, min_count, unknown_node):
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.unknown_node = unknown_node

    def vectorize(self, nodes):
        pass
