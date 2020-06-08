import pickle


class CodeRepresentationModel:
    def __init__(self, config):
        self.config = config
        self.labels = None
        self.labels_size = 0
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.embedding_size = 0
        self.embedding = None
        self.node_map = None

        self.read_data()

    def read_data(self):
        with open(self.config.DATA_PATH, 'rb') as fh:
            self.train_data, self.val_data, self.test_data, self.labels = pickle.load(fh)
            self.labels = [str(l) for l in self.labels]
            self.labels_size = len(self.labels)
            self.train_data['label'] = self.train_data['label'].apply(str)
            self.val_data['label'] = self.val_data['label'].apply(str)
            self.test_data['label'] = self.test_data['label'].apply(str)

        with open(self.config.EMBEDDING_PATH, 'rb') as fh:
            self.embedding, self.node_map = pickle.load(fh)
            self.embedding_size = len(self.embedding[0])

    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass