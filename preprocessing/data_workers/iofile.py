import pandas as pd


class IOFile:
    @staticmethod
    def name():
        return "basic"

    def __init__(self, parser):
        self.parser = parser

    def read_source(self, file):
        d = {'id': [], 'code': [], 'label': []}
        df = pd.DataFrame(data=d)
        return df

    def read(self, path):
        pass

    def write(self, data, path):
        pass
