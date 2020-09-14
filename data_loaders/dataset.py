import os

from torch.utils.data import IterableDataset


class CodeDataset(IterableDataset):

    def __init__(self, dataset_path: str):
        if not os.path.exists(dataset_path):
            raise ValueError("No such path found: " + dataset_path)
        self.dataset_path = dataset_path #пока считаем, что один файл

    def create(self, name: str):
        from data_loaders.poj import pandas_pckl_dataset, pandas_csv_dataset
        if name == "pandas_pickle":
            return pandas_pckl_dataset.PandasPickleDataset(self.dataset_path)
        elif name == "pandas_csv":
            return pandas_csv_dataset.PandasCsvDataset(self.dataset_path)
        else:
            raise ValueError("No such data loader...")

    def __iter__(self):
        pass
