from typing import Tuple

import torch

from data_loaders.dataset import CodeDataset


class PandasCsvDataset(CodeDataset):

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

    def _line_processer(self, line: str):
        id, code, label = line.strip().split(",")
        # wtf делать с словарями/списками
        return code, label

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            pass
        else:
            # TODO сделать разделение данных
            pass
        file_itr = open(self.dataset_path)
        header = file_itr.readline()  # read header
        assert header.strip() == "path,code,label"

        mapped_itr = map(self._line_processer, file_itr)
        return mapped_itr
