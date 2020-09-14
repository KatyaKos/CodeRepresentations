import pandas as pd
import torch

from data_loaders.dataset import CodeDataset


class PandasPickleDataset(CodeDataset):

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        # TODO поменять на папку с поиском файлов
        self.file = pd.read_pickle(dataset_path)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            pass
        else:
            # TODO сделать разделение данных или просто убрать
            pass
        return zip(self.file['code'], self.file['label'])

