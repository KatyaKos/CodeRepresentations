from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


class Vectorizer:
    @staticmethod
    def name() -> str:
        return "basic"

    def __init__(self, embedding_dim: int, min_count: int, unknown_node: str):
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.unknown_node = unknown_node

    def create(self, name: str):
        from preprocessing.vectorizers import word2vec as w2v
        if name == 'word2vec':
            return w2v.W2vVectorizer(self.embedding_dim, self.min_count, self.unknown_node)
        else:
            raise ValueError('No such vectorizer')

    def vectorize(self, nodes: List[List[str]]) -> pd.DataFrame:
        embedding, node_map = self._get_embedding(nodes)
        result = []
        for node in node_map:
            id = node_map[node]
            emb = embedding[id]
            datum = [id, node, emb]
            result.append(datum)
        result.append([embedding.shape[0], self.unknown_node, np.zeros((embedding.shape[1]), dtype="float32")])
        df = pd.DataFrame(result, columns=['id', 'node', 'vector'])
        return df

    def _get_embedding(self, nodes: List[List[str]]) -> Tuple[np.ndarray, Dict[str, int]]:
        return np.zeros((0, 0),  dtype="float32"), {}
