"""
Author: Kanna
Date: 2025-09-23
Version: 0.1
License: MIT
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from trainlab.utils import Logger
from tqdm import tqdm

SCHEDULER_MAP = {
    "StepLR": lr_scheduler.StepLR,
    "MultiStepLR": lr_scheduler.MultiStepLR,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    "CyclicLR": lr_scheduler.CyclicLR,
    "OneCycleLR": lr_scheduler.OneCycleLR,
    "LambdaLR": lr_scheduler.LambdaLR,
    "PolynomialLR": lr_scheduler.PolynomialLR if hasattr(lr_scheduler, "PolynomialLR") else None,  # 新版才有
}

OPTIMIZER_MAP = {
            "Adam": optim.Adam,
            "SGD": optim.SGD,
            "AdamW": optim.AdamW,
        }

LOSS_FUNCTION_MAP = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "BCELoss": nn.BCELoss,
    "KLDivLoss": nn.KLDivLoss,
    "NLLLoss": nn.NLLLoss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "HuberLoss": nn.HuberLoss,
    # 可以根据需要继续添加
}


class BaseTrainer:
    def __init__(
        self, 
        model, 
        log_queue,
        loss_fn_class=None, 
        epochs=1, 
        optimizer_class=None, 
        optimizer_kwargs={'lr':1e-3},
        scheduler_class=None,
        scheduler_kwargs=None,
        save=False,
        output_dir = './',
        output_filename='weight'
    ):
        """
        model: nn.Module, 待训练模型
        loss_fn: 损失函数，如果不提供，默认使用 CrossEntropyLoss
        epochs: 总训练轮数
        optimizer_class: 优化器类，如 torch.optim.AdamW
        optimizer_kwargs: 优化器初始化参数字典
        scheduler_class: 学习率调度器类，如 torch.optim.lr_scheduler.StepLR
        scheduler_kwargs: 调度器初始化参数字典
        """

        self.model = model
        self.epochs = epochs
        self.log_queue = log_queue # log队列
        self.loss_fn = LOSS_FUNCTION_MAP[loss_fn_class]()

        self.optimizer_class = OPTIMIZER_MAP[optimizer_class]
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_class = SCHEDULER_MAP[scheduler_class]
        self.scheduler_kwargs = scheduler_kwargs or {}

        # 每个进程的 optimizer 和 scheduler 会在 setup 中实例化
        self.optimizer = None
        self.scheduler = None

        self.output_dir = output_dir
        self.output_filename = output_filename
        self.epoch_best_model = None
        self.eval_loss = 0.
        self.save = save
    


    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        # 设置随机种子，确保每个进程的初始化参数一致
        torch.manual_seed(42)        # CPU
        torch.cuda.manual_seed(42)   # GPU

        """ 初始化 DDP 和每进程优化器/调度器 """
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{rank}")
        self.logger = Logger.get_logger(self.log_queue,name=f'rank{rank}')
        # 调用自定义初始化钩子
        self.custom_setup(rank, world_size)

        self.model.to(device)
        self.model = DDP(self.model, device_ids=[rank])
        



        if self.optimizer_class:
            self.optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
        if self.scheduler_class and self.optimizer:
            self.scheduler = self.scheduler_class(self.optimizer, **self.scheduler_kwargs)

        
        
        return device
    
    def custom_setup(self, rank, world_size):
        """
        默认实现：什么都不做
        子类可以重写此函数，执行特定初始化操作
        """
        pass

    def reduce_value(self, value, average=True):
        """ 多卡同步指标 """
        world_size = dist.get_world_size()
        if world_size < 2:
            return value
        with torch.no_grad():
            dist.all_reduce(value)
            if average:
                value /= world_size
        return value

    def prepare_dataloader(self, dataset, rank, world_size, batch_size=8, collate_fn=None, shuffle=True):
        """ 多卡分布式 sampler """
        # distributedSample依据rank将dataset分成不重叠的world_size个部分，然后每个rank实例化自己的dataloader
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

    def train(self, rank, world_size, train_dataset, val_dataset=None, collate_fn=None, evaluate=False):
        device = self.setup(rank, world_size)
        train_loader = self.prepare_dataloader(train_dataset, rank, world_size, collate_fn=collate_fn)
        val_loader = self.prepare_dataloader(val_dataset, rank, world_size,batch_size=2, collate_fn=collate_fn) if val_dataset else None

        for epoch in range(self.epochs):
            train_loader.sampler.set_epoch(epoch)
            self.run_one_epoch(epoch, train_loader, rank, device, train=True)

            if evaluate and val_loader:
                self.run_one_epoch(epoch, val_loader, rank, device, train=False)

        dist.destroy_process_group()

    def run_one_epoch(self, epoch, data_loader, rank, device, train=True):
        """ 子类必须实现具体训练/验证逻辑 """
        raise NotImplementedError("子类必须实现 iteration()")
    
    def wrap_dataloader_with_tqdm(self, rank, data_loader, epoch, train_model=True):
        if rank == 0:
            mode = 'train' if train_model else 'eval'
            data_loader_iter = tqdm(
                data_loader,
                desc=f"EP ({mode}) {epoch}",
                total=len(data_loader),
                bar_format="{l_bar}{r_bar}"
            )
        else:
            data_loader_iter = data_loader
        
        return data_loader_iter
    
    def save_best_model(self, epoch, avg_loss_epoch, train=False):
        """
        保存当前最优模型权重（仅在 eval 模式且损失更好时保存）
        
        Args:
            epoch (int): 当前训练轮数
            avg_loss_epoch (float): 当前轮平均损失
            train (bool): 是否为训练阶段（True 表示训练阶段，不保存）
        """
        if not self.save or train:
            return

        # 只有当当前轮损失更优时才保存
        if avg_loss_epoch >= self.eval_loss:
            return

        # 创建输出文件夹
        os.makedirs(self.output_dir, exist_ok=True)

        # 新模型保存路径
        file_path = os.path.join(
            os.path.abspath(self.output_dir),
            f"{self.output_filename}_epoch_{epoch}.pt"
        )

        # 删除上一个最优模型
        if hasattr(self, 'epoch_best_model') and self.epoch_best_model is not None:
            file_path_prev = os.path.join(
                os.path.abspath(self.output_dir),
                f"{self.output_filename}_epoch_{self.epoch_best_model}.pt"
            )
            if os.path.exists(file_path_prev):
                os.remove(file_path_prev)

        # 保存当前模型和优化器状态
        torch.save({
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }, file_path)

        # 更新最优 loss 和 epoch
        self.eval_loss = avg_loss_epoch
        self.epoch_best_model = epoch
