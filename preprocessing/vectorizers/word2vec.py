from gensim.models.word2vec import Word2Vec
import numpy as np
from preprocessing.vectorizers.vectorizer import Vectorizer


class W2vVectorizer(Vectorizer):
    @staticmethod
    def name():
        return "word2vec"

    def __init__(self, embedding_dim, min_count, unknown_node):
        super().__init__(embedding_dim, min_count, unknown_node)

    def vectorize(self, nodes):
        word2vec = Word2Vec(nodes, size=self.embedding_dim, workers=16, sg=1, min_count=self.min_count).wv
        embedding = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
        embedding[:word2vec.syn0.shape[0]] = word2vec.syn0
        vocab = word2vec.vocab
        node_map = {t: vocab[t].index for t in vocab}
        node_map[self.unknown_node] = word2vec.syn0.shape[0]
        return embedding, node_map