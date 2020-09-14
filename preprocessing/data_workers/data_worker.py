from typing import Dict

from preprocessing.ast_builders.parser import Parser


class DataWorker:
    @staticmethod
    def name() -> str:
        return "basic"

    def __init__(self, sources=None):
        self.source = sources

    def create(self, name: str):
        from preprocessing.data_workers import pandas_worker
        if name == 'pandas':
            return pandas_worker.PandasWorker(self.source)
        else:
            raise ValueError('No such DataWorker')

    def get_data(self, name: str) -> Dict:
        pass

    def __len__(self):
        pass

    def update(self, new_values: Dict):
        pass

    def parse(self, parser: Parser, labels_path: str):
        pass

    def write(self, output_path: str, filename: str):
        pass