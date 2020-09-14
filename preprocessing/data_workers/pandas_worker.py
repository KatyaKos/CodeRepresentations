from typing import Dict

import pandas as pd
import os

from preprocessing.data_workers.data_worker import DataWorker
from preprocessing.ast_builders.parser import Parser


class PandasWorker(DataWorker):
    @staticmethod
    def name() -> str:
        return "pandas"

    def __init__(self, sources: pd.DataFrame=None):
        super().__init__(sources)

    def __len__(self):
        return len(self.source)

    # assuming there is a pandas .csv file in the root: ['path', 'label']
    def parse(self, parser: Parser, data_path: str):
        label_files = filter(lambda _f: _f.endswith('.csv'), os.listdir(data_path))
        result = []
        for file in label_files:
            labels = pd.read_csv(os.path.join(data_path, file))
            for path, label in zip(labels['path'], labels['label']):
                path = os.path.join(data_path, path)
                ast = parser.parse_file(path)
                datum = [path, ast, str(label)]
                result.append(datum)
        self.source = pd.DataFrame(result, columns=['path', 'code', 'label'])

    def update(self, new_values: Dict):
        self.source = self.source.iloc[list(new_values.keys()), :]
        self.source = self.source.to_dict('records')
        for id in new_values:
            self.source[id].update(new_values[id])
        self.source = pd.DataFrame(self.source)

    def get_data(self, name: str) -> Dict:
        return {id: node for id, node in zip(list(self.source.index), self.source[name])}

    def write(self, output_path: str, filename: str):
        self.source.to_pickle(os.path.join(output_path, filename + '.pkl'))