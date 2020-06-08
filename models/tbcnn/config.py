from gensim.models.word2vec import Word2Vec
import os
import numpy as np


class Config:
    @staticmethod
    def get_config(args):
        config = Config(args)
        config.EPOCHS = 10
        config.LEARN_RATE = 0.001
        config.HIDDEN_SIZE = 128
        config.BATCH_SIZE = 30
        return config

    def __init__(self, args):
        self.BATCH_SIZE = 0
        self.HIDDEN_SIZE = 0
        self.LEARN_RATE = 0.
        self.EPOCHS = 0

        self.DATA_PATH = args.data_path
        self.EMBEDDING_PATH = args.embed_path
        self.SAVE_PATH = args.save_path
        self.LOAD_PATH = args.load_path
        self.PREDICT_PATH = args.predict_path
