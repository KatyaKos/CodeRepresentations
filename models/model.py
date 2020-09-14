from typing import Dict

import pandas as pd
import numpy as np
import os


class CodeRepresentationModel:
    def __init__(self, params: Dict):
        self.params = params

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.embedding = None
        self.EMBEDDING_DIM = 0
        self.labels = None
        self.LABELS_SIZE = 0

        self._read_data()

    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

    def _read_data(self):
        from data_loaders import dataset
        loader_name = self.params['data_loader']
        self.train_data = dataset.CodeDataset(self.params['paths']['train_path']).create(loader_name)
        self.val_data = dataset.CodeDataset(self.params['paths']['val_path']).create(loader_name)
        self.test_data = dataset.CodeDataset(self.params['paths']['test_path']).create(loader_name)

        self._read_embedding(self.params['paths']['embedding_file'])
        self._read_labels(self.params['paths']['labels_file'])

    def _read_embedding(self, path: str):
        if os.path.exists(path):
            embedding = pd.read_pickle(path)
            self.EMBEDDING_DIM = len(embedding['vector'][0])
            self.embedding = np.zeros((len(embedding), self.EMBEDDING_DIM), dtype="float32")
            for id, vec in zip(embedding['id'], embedding['vector']):
                self.embedding[int(id)] = vec
        else:
            raise ValueError("No nodes-embeddings file found: " + path)

    def _read_labels(self, path: str):
        if os.path.exists(path):
            labels = pd.read_pickle(path)
            self.LABELS_SIZE = len(labels)
            self.labels = [""] * self.LABELS_SIZE
            for id, label in zip(labels['id'], labels['label']):
                self.labels[int(id)] = str(label)
        else:
            raise ValueError("No labels file found: " + path)