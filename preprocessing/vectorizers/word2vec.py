from typing import Tuple, Dict, List

from gensim.models.word2vec import Word2Vec
import numpy as np
from preprocessing.vectorizers.vectorizer import Vectorizer


class W2vVectorizer(Vectorizer):
    @staticmethod
    def name() -> str:
        return "word2vec"

    def __init__(self, embedding_dim: int, min_count: int, unknown_node: str):
        super().__init__(embedding_dim, min_count, unknown_node)

    def _get_embedding(self, nodes: List[List[str]]) -> Tuple[np.ndarray, Dict[str, int]]:
        word2vec = Word2Vec(nodes, size=self.embedding_dim, workers=16, sg=1, min_count=self.min_count).wv
        embedding = np.zeros((word2vec.syn0.shape[0], word2vec.syn0.shape[1]), dtype="float32")
        embedding[:word2vec.syn0.shape[0]] = word2vec.syn0
        vocab = word2vec.vocab
        node_map = {t: vocab[t].index for t in vocab}
        return embedding, node_map