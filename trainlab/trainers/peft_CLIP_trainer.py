"""
Author: Kanna
Date: 2025-10-09
Version: 1.0
License: MIT
"""
import os
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import  ChineseCLIPProcessor
from trainlab.builder import TRAINERS
from trainlab.base_trainer import BaseTrainer
from trainlab.utils.tools import cls_acc
from PIL import Image



@TRAINERS.register_module()
class PEFTCLIPTrainer(BaseTrainer):
    def __init__(self, 
                 model, 
                 model_name_or_path,
                 log_queue,
                 project_name,
                 loss_fn=None, 
                 epochs=1, 
                 optimizer_class=None, 
                 optimizer_kwargs=None, 
                 scheduler_class=None, 
                 scheduler_kwargs=None,
                 save = True,
                 output_dir = './',
                 output_filename='weight'):
        super(PEFTCLIPTrainer, self).__init__(model, log_queue, project_name,loss_fn, epochs, optimizer_class, optimizer_kwargs, scheduler_class, scheduler_kwargs,save,output_dir, output_filename)
        self.peft_text_config = LoraConfig(
            # task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=['query','value','q_proj','v_proj'] # 实践经验表明微调这两个模块效果最好
        )
        # self.peft_vision_config = LoraConfig(
        #     task_type=TaskType.FEATURE_EXTRACTION,
        #     inference_mode=False,
        #     r=8,
        #     lora_alpha=16,
        #     lora_dropout=0.1,
        #     target_modules=['q_proj','v_proj'] # 实践经验表明微调这两个模块效果最好
        # )

        self.processor = ChineseCLIPProcessor.from_pretrained(model_name_or_path)
        self.instruct = '这张图片描述的内容是'

    def custom_setup(self, rank, world_size):
        '''再多进程情况下，每个进程中的模型是一个单独的实例，那么对模型进行操作时要在子进程中进行，重载这个函数即可插入特定的初始化逻辑再子进程中'''
        # transformes风格的模型被包装在warpper里面，但是没有暴露model的属性，而是用origin_model承接
        # add LoRA layers to text model
        self.model.origin_model = get_peft_model(self.model.origin_model, self.peft_text_config)
        # add LoRA layers to vision model
        # self.model.origin_model.vision_model = get_peft_model(self.model.origin_model.vision_model, self.peft_vision_config)
        # count the number of trainable parameters
        if rank == 0:
            self.model.origin_model.print_trainable_parameters()
        

    def run_one_epoch(self, epoch,totoal_epcoh, data_loader, rank, device, train=True):

        self.model.train() if train else self.model.eval()
       
        # initialize variables
        loss_accumulated = 0
        acc_train = 0
        tot_samples = 0
     

        # 只在主进程rank=1中打印日志
        data_loader_iter = self.wrap_dataloader_with_tqdm(
            rank=rank,
            data_loader=data_loader,
            epoch=epoch,
            train_model=train
        )

        for idx, (impath, labels, classname) in enumerate(data_loader_iter):

            images = [Image.open(path).convert('RGB') for path in impath]
            texts = [self.instruct + c for c in classname]
            

            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
            output = self.model(**inputs, return_loss=True)
            loss = output.loss
            logits_per_image = output.logits_per_image
            labels = torch.arange(logits_per_image.shape[0]).to(device)
            acc_train += cls_acc(logits_per_image, labels) * labels.shape[0]
            loss_accumulated += loss.item()*labels.shape[0]
            tot_samples += labels.shape[0]

            if train:
                self.optimizer.zero_grad()
                loss.backward() # 多卡中，会自动同步多gpu梯度
                self.optimizer.step()
                self.scheduler.step()

            # cpu等待当前GPU上所有计算完成,每个gpu进程都调用这个函数，则cpu就需要等待所有gpu完成梯度同步
            torch.cuda.synchronize(device)

        if rank == 0:
            acc_train /= tot_samples
            loss_accumulated /= tot_samples
            current_lr = self.scheduler.get_last_lr()[0]

            # print metrics to console
            self.logger.info(
                f"epoch={epoch},\
                loss={round(loss_accumulated, 3)},\
                acc_train={round(acc_train, 3)},\
                lr={round(current_lr, 6)}"
                )    

        self.save_best_model(epoch, loss_accumulated, train=train) # 仅在 eval 模式下保存最优模型   


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

        # 增加项目目录
        dir_path = os.path.join(self.output_dir, self.project_name)
        # 创建输出文件夹
        os.makedirs(dir_path, exist_ok=True)

        # 新模型保存路径
        self.file_path = os.path.join(
            os.path.abspath(dir_path),
            f"{self.output_filename}_epoch_{epoch}.pt"
        )

        # 删除上一个最优模型
        if hasattr(self, 'epoch_best_model') and self.epoch_best_model is not None:
            file_path_prev = os.path.join(
                os.path.abspath(dir_path),
                f"{self.output_filename}_epoch_{self.epoch_best_model}.pt"
            )
            if os.path.exists(file_path_prev):
                os.remove(file_path_prev)

        # 使用 PEFT 方法保存模型，只保存增量参数
        self.model.module.origin_model.save_pretrained(self.file_path)

        
        