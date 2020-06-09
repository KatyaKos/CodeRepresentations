import os


class Config:
    @staticmethod
    def get_config(args):
        config = Config(args)
        config.HIDDEN_DIM = 100
        config.ENCODE_DIM = 128
        config.LABELS = 104
        config.EPOCHS = 10
        config.BATCH_SIZE = 64
        config.USE_GPU = False
        return config

    def __init__(self, args):
        self.HIDDEN_DIM = 0
        self.ENCODE_DIM = 0
        self.LABELS = 0
        self.EPOCHS = 0
        self.BATCH_SIZE = 0
        self.USE_GPU = False

        self.DATA_PATH = args.data_path
        self.EMBEDDING_PATH = args.embed_path
        self.SAVE_PATH = args.save_path
        self.LOAD_PATH = args.load_path
        self.PREDICT_PATH = args.predict_path

