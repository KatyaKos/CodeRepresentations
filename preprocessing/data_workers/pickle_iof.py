import pickle
import pandas as pd
from preprocessing.data_workers.iofile import IOFile


class PickleReader(IOFile):
    @staticmethod
    def name():
        return "pickle"

    def __init__(self, parser):
        super().__init__(parser)

    def read_source(self, file):
        source = pd.read_pickle(file)
        source.columns = ['id', 'code', 'label']
        source['label'] = source['label'].apply(str)
        source['code'] = source['code'].apply(self.parser.parse_code)
        return source

    def read(self, path):
        with open(path, 'rb') as fin:
            return pickle.load(fin)

    def write(self, data, path):
        with open(path, 'wb') as fout:
            pickle.dump(data, fout)