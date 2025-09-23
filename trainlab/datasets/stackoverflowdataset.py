"""
Author: Kanna
Date: 2025-09-23
Version: 0.1
License: MIT
"""
import os
from .dataset_base import DatasetBase
from datasets import load_from_disk
from trainlab.builder import DATASETS

@DATASETS.register_module()
class StackOverflowDataset(DatasetBase):
    name = 'train-sample'

    def __init__(self, root, split='train', transform=None, cache_in_memory=False):
        super().__init__(transform, cache_in_memory)

        self.data_dir = os.path.join(root, self.name, split)
        self.data = self.load_data(self.data_dir)

    def load_data(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")

        dataset = load_from_disk(path)
        print(type(dataset))
        return dataset






if __name__=="__main__":
    dataset = StackOverflowDataset(
            root="/data/lj/task/trainlab/data",
            split='train'
                                    )
    item = dataset.__getitem__(1)
    