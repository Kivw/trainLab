import os
import pickle
import re
import json
from trainlab.builder import DATASETS
from .dataset_base import DatasetBase, Datum


@DATASETS.register_module()
class UCF101Dataset(DatasetBase):

    dataset_dir = "ucf101"

    def __init__(self, root, split='train', transform=None, cache_in_memory=False):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101_chinese.json")
        self.split_using_data = os.path.join(self.dataset_dir, "split_using_data") # 存放划分好的数据集
        if not os.path.exists(self.split_using_data):
            os.makedirs(self.split_using_data)
        super().__init__(transform, cache_in_memory)

        # 将数据及其路径从json文件中读取出来
        train, val, test = self.read_split(self.split_path, self.image_dir)

        # 预处理样本数据
        preprocessed = os.path.join(self.split_using_data, f"using_data.pkl")
            
        if os.path.exists(preprocessed):
            print(f"Loading preprocessed few-shot data from {preprocessed}")
            with open(preprocessed, "rb") as file:
                data = pickle.load(file)
                train, val = data["train"], data["val"]
        else:

            data = {"train": train, "val": val}
            print(f"Saving preprocessed few-shot data to {preprocessed}")
            with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        if split == 'train':
                self.data = train
        elif split == 'val':
                self.data = val
        elif split == 'test':
                self.data = test
        else:
                raise ValueError(f"Invalid split value: {split}")


        

    def read_split(self, split_path, path_prefix):
        def _convert(item_list):# 将相对路径转换为绝对路径
            out = []
            for impath, label, classname in item_list:
                impath = os.path.join(path_prefix,impath)
                item = (impath, label, classname)
                out.append(item)

            return out
        
        with open(split_path, "r") as f:
            data = json.load(f)
        train = _convert(data["train"])
        val = _convert(data["val"])
        test = _convert(data["test"])
        return train, val, test



    # def read_data(self, cname2lab, text_file):
    #     text_file = os.path.join(self.dataset_dir, text_file)
    #     items = []

    #     with open(text_file, "r") as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             line = line.strip().split(" ")[0]  # trainlist: filename, label
    #             action, filename = line.split("/")
    #             label = cname2lab[action]

    #             elements = re.findall("[A-Z][^A-Z]*", action)
    #             renamed_action = "_".join(elements)

    #             filename = filename.replace(".avi", ".jpg")
    #             impath = os.path.join(self.image_dir, renamed_action, filename)

    #             item = Datum(impath=impath, label=label, classname=renamed_action)
    #             items.append(item)

    #     return items


if __name__ == "__main__":




    dataset = UCF101Dataset("/data/lj/task/CoOp/data")
    
    print("类别数:", len(dataset.data))
    # print("训练集样本数:", dataset.data[:5])
    data = iter(dataset)
    item = next(data)
    print(item)
    # print("测试集样本数:", len(dataset.test))
