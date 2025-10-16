# TrainLab
## Introduction
TrainLab框架简介
TrainLab 是一个使用注册器Registry管理的模块化多卡训练框架，特别支持hugging face格式的大模型的微调。当前版本处于开发初期，还有很多功能不够完善，但是目前已经成功跑通了BERT模型的多卡微调，欢迎大家尝试并提交代码。
trainlab框架参考OpenMMLab采用注册模式（Registry Pattern）组织代码，整个框架将深度学习训练过程抽象为三大模块：
- MODELS: 注册所有深度学习模型
- DATASETS：注册所有数据集
- TRAINERS：注册所有训练方法

注册机制示例（自定义数据集，其他模块方法相同，注意所有注册文件都需要在train.py中import注册机制才能成功运行）：
<pre>
```import os
from .dataset_base import DatasetBase
from trainlab.builder import DATASETS # DATASETS注册表

@DATASETS.register_module()  # 调用注册函数，将StaockOverflowDataset这个类注册到DATASETS注册表中
class StackOverflowDataset(DatasetBase):
    name = 'train-sample'

    def __init__(self, root, split='train', transform=None, cache_in_memory=False):
        super().__init__(transform, cache_in_memory)

        self.data_dir = os.path.join(root, self.name, split)
        # self.data = self.load_data(self.data_dir).select(range(100))
        # 在trainlab框架中，DatasetBase继承了torch.Dataset, 重写了__item__(self,index)，只需将数据整理好赋值给self.data即可
        self.data = self.load_data(self.data_dir)

    def load_data(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")

        dataset = load_from_disk(path)
        return dataset
    
    def collate_fn(self, batch):
        """ Instructs how the DataLoader should process the data into a batch"""
        
        text = [item['text'] for item in batch]
        tabular = torch.stack([torch.tensor(item['tabular']) for item in batch])
        labels = torch.stack([torch.tensor(item['label']) for item in batch])

        return {'text': text, 'tabular': tabular, 'label': labels}```
</pre>
对于训练代码，在trainlab框架下我们仅需继承BaseTrainer然后只需要重写以下两个函数即可：
- `def custom_setup(self, rank, world_size): `由于DDP下的多卡训练需要初始化在每个进程中初始化一个模型，如果要对模型进行修改（如增加lora层）即在这个函数中定义。
- `def run_one_epoch(self, epoch,totoal_epcoh, data_loader, rank, device, train=True): `训练每个epoc的逻辑。
对于支持的模型，trainlab支持自定义模型以及trainsfomers风格的所有主流模型。transformers风格的模型支持请参考文件。
trainlab目录结构如图所示：
[图片]

## start 
trainlab的基础使用流程
#创建conda环境
<pre>```conda create --name trainlab python=3.10
conda activate trainlab
pip install -r requirements.txt

# clone项目
git clone https://github.com/Kivw/trainLab.git 
cd trainlab

# 修改好配置文件然后执行，'bash scripts cuda_visible_device config_file' 
bash scripts/peft_chinese_clip.sh 1,2,3 /data/lj/task/trainlab/trainlab/config/pert_chinese_clip.yaml```
</pre>
注意：关于pytorch的安装最好是手动安装适合自己设备的版本，然后注释掉requirements.txt中的torch,在执行：<br>
'pip install -r requirements.txt'<br>
环境配置好后我们的思路是：<br>
1. 数据准备：我们当前bert例子中使用的数据集是：Predict Closed Questions on Stack Overflow<br>
2. 数据集预处理：'python -m preprocess.stackoverlow'<br>
3. (模型和dataset已经准备好)<br>
4. bash scripts/bert_ft.sh 0,1,2,3 path/to/config.yaml<br>

## 开发日志
- [2025-09-24] v1.0.0
  - Added: 添加readme.md
  - Fixed: 修复在多进程下Logger类错乱的问题。
- [2025-09-25] v1.0.0
  - ADDed: 添加lora功能
  - Fiexed: 修复多进程下Logger错乱问题。