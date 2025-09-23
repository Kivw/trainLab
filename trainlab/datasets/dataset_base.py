"""
Author: Kanna
Date: 2025-09-23
Version: 0.1
License: MIT
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from datasets import load_from_disk

class DatasetBase(Dataset):
    """
    一次性加载所有数据到内存，适用于数据流较小的数据
    BaseDataset 提供：
    - transform 支持
    - 可选缓存
    - 可选 collate_fn（子类可重写）
    - 多卡训练逻辑由训练脚本处理
    """

    def __init__(self, transform=None, cache_in_memory=False):
        super().__init__()
        self.transform = transform
        self.data = None
        self.cache_in_memory = cache_in_memory
        self._cache = {} if cache_in_memory else None

    def __len__(self):
        return len(self.data)

    def load_data(self, path):
        """子类必须实现，返回原始数据"""
        raise NotImplementedError("Subclasses must implement load_item()")

    def __getitem__(self, index):
        # 使用缓存
        if self.cache_in_memory and index in self._cache:
            item = self._cache[index]
        else:
            item = self.data[index]
            if self.cache_in_memory:
                self._cache[index] = item

        # transform
        if self.transform:
            item = self.transform(item)
        return item

    def collate_fn(self, batch):
        """
        可选的 batch 处理函数（DataLoader 可用）
        默认直接使用 PyTorch 的 default_collate
        子类可以覆盖实现自定义 collate 逻辑
        """
        return torch.utils.data._utils.collate.default_collate(batch)


class IterableDatasetBase(IterableDataset):
    """
    基于 Hugging Face Dataset 的流式 Dataset 基类（IterableDataset）
    
    特点：
    - 支持从磁盘加载 Arrow/Parquet 格式数据（load_from_disk）
    - 可选 transform
    - 子类可覆盖 parse_item 定义如何处理单条数据
    """
    def __init__(self, dataset_path: str, split: str = None, transform=None):
        """
        Args:
            dataset_path (str): save_to_disk 保存的根目录或具体 split 目录
            split (str, optional): 如果 dataset_path 是 DatasetDict 保存的根目录，指定 'train' / 'validation' / 'test'
            transform (callable, optional): 对每条样本进行变换
        """
        super().__init__()
        self.transform = transform

        # 加载 Hugging Face Dataset
        if split is not None:
            # 从 DatasetDict 的某个 split 加载
            self.dataset = load_from_disk(dataset_path)[split]
        else:
            # 直接加载单个 Dataset
            self.dataset = load_from_disk(dataset_path)

        # 内部迭代器
        self._length = len(self.dataset)

    def parse_item(self, item):
        """
        子类可覆盖，定义如何解析原始 item
        默认返回原始 item，可加 transform
        """
        if self.transform:
            item = self.transform(item)
        return item

    def __iter__(self):
        for item in self.dataset:
            yield self.parse_item(item)

    def __len__(self):
        """
        IterableDataset 不要求实现 __len__，但是可以提供方便信息
        """
        return self._length

    def collate_fn(self, batch):
        """
        可选 batch 处理函数
        默认使用 PyTorch 的 default_collate
        """
        return torch.utils.data._utils.collate.default_collate(batch)