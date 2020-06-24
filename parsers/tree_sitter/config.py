
class Config:

    @staticmethod
    def get_config(args):
        config = Config(args)
        config.SPLIT_RATIO = [3, 1, 1]
        # vectorization
        config.EMBEDDING_DIM = 128
        config.MIN_COUNT = 3
        # tree sampling
        config.MAX_TREE_SIZE = 10000  # Ignore trees with more than max_tree_size nodes
        config.MIN_TREE_SIZE = 0  # Ignore trees with less than min_tree_size nodes
        config.MAX_DEPTH = 50
        return config

    def __init__(self, args):
        self.EMBEDDGIN_DIM = 0
        self.MIN_COUNT = 0
        self.MAX_TREE_SIZE = 100000
        self.MIN_TREE_SIZE = 0
        self.MAX_DEPTH = 100000
        self.SPLIT_RATIO = [1, 0, 0]

        self.AST_FILE = 'ast.pkl'
        self.SAMPLED_TREES_FILE = 'trees.pkl'
        self.UNK_NODE = '<UNKNOWN_NODE>'

        self.DATA_PATH = args.data_path
        self.EMBEDDING_PATH = args.embed_path
        self.RAW_DATA_PATH = args.raw_data_path
