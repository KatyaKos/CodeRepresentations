from gensim.models.word2vec import Word2Vec
import os
import numpy as np


class Config:
    @staticmethod
    def get_model_config(args):
        config = Config(args)
        config.EPOCHS = 10
        config.LEARN_RATE = 0.001
        config.CHECKPOINT_STEP = 100
        config.BATCH_SIZE = 64
        return config

    @staticmethod
    def get_preprocess_config(args):
        config = Config(args)
        # node sampling
        config.MAX_PER_NODE = -1  # Sample up to a maxmimum number for each node type
        config.MAX_NODES = -1  # Maximum number of samples to store.
        # vectorization
        config.NUM_FEATURES = 30
        config.BATCH_SIZE = 256
        config.HIDDEN_SIZE = 100
        config.LEARN_RATE = 0.01
        config.EPOCHS = 100
        config.CHECKPOINT_STEP = 1000
        # tree sampling
        config.SPLIT_RATIO = [7, 0, 3]
        config.MAX_TREE_SIZE = 30000  # Ignore trees with more than max_tree_size nodes
        config.MIN_TREE_SIZE = 0  # Ignore trees with less than min_tree_size nodes
        return config

    def __init__(self, args):
        self.MAX_PER_NODE = -1
        self.MAX_NODES = -1
        self.NUM_FEATURES = 0
        self.BATCH_SIZE = 0
        self.HIDDEN_SIZE = 0
        self.LEARN_RATE = 0.
        self.EPOCHS = 0
        self.CHECKPOINT_STEP = 0
        self.MAX_TREE_SIZE = 100000
        self.MIN_TREE_SIZE = 0
        self.SPLIT_RATIO = [1, 0, 0]

        self.AST_FILE = 'ast.pkl'
        self.SAMPLED_NODES_FILE = 'nodes.pkl'
        self.SAMPLED_TREES_FILE = 'trees.pkl'

        self.DATA_PATH = args.data_path
        self.EMBEDDING_PATH = args.embed_path
        self.SAVE_PATH = args.save_path if args.save_path is not None \
            else os.path.join(self.DATA_PATH, 'astnn_checkpoint.tar')
        self.LOAD_PATH = args.load_path
        self.RAW_DATA_PATH = args.raw_data_path
        self.LOGDIR = args.logdir
        self.PREDICT_PATH = args.predict_path