from gensim.models.word2vec import Word2Vec
import os
import numpy as np


class Config:
    @staticmethod
    def get_default_config(args):
        config = Config(args)
        config.SPLIT_RATIO = "3:1:1"
        config.HIDDEN_DIM = 100
        config.ENCODE_DIM = 128
        config.LABELS = 104
        config.EPOCHS = 15
        config.BATCH_SIZE = 64
        config.USE_GPU = False
        return config

    def __init__(self, args):
        self.SPLIT_RATIO = "3:1:1"
        self.HIDDEN_DIM = 100
        self.ENCODE_DIM = 128
        self.LABELS = 104
        self.EPOCHS = 15
        self.BATCH_SIZE = 64
        self.USE_GPU = False

        self.RAW_DATA_PATH = args.raw_data_path
        self.DATA_PATH = args.data_path
        self.SAVE_PATH = args.save_path if args.save_path is not None else args.data_path
        self.LOAD_PATH = args.load_path

        if self.LOAD_PATH is None:
            word2vec = Word2Vec.load(os.path.join(self.DATA_PATH, 'embedding/node_w2v_128')).wv
            self.MAX_TOKENS = word2vec.syn0.shape[0]
            self.EMBEDDING_DIM = word2vec.syn0.shape[1]
            # Embeddings can be None or any other 'pre-trained weight'
            self.embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
            self.embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
