"""
Author: Kanna
Date: 2025-09-23
Version: 0.1
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainlab.base_trainer import BaseTrainer
from trainlab.builder import TRAINERS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer


@TRAINERS.register_module()
class BERTTrainer(BaseTrainer):
    def __init__(self, 
                 model, 
                 model_name_or_path,
                 log_queue,
                 loss_fn=None, 
                 epochs=1, 
                 optimizer_class=None, 
                 optimizer_kwargs=None, 
                 scheduler_class=None, 
                 scheduler_kwargs=None,
                 save = True,
                 output_dir = './',
                 output_filename='weight'):
        super(BERTTrainer, self).__init__(model, log_queue,loss_fn, epochs, optimizer_class, optimizer_kwargs, scheduler_class, scheduler_kwargs,save,output_dir, output_filename)
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    def run_one_epoch(self, epoch, total_epoch, data_loader, rank, device, train=True):
        self.model.train() if train else self.model.eval()
       
        # initialize variables
        loss_accumulated = 0
        correct_accumulated = 0
        samples_accumulated = 0
        preds_all = []
        labels_all = []

        # 只在主进程rank=1中打印日志
        data_loader_iter = self.wrap_dataloader_with_tqdm(
            rank=rank,
            data_loader=data_loader,
            epoch=epoch,
            train_model=train
        )

        for idx, batch in enumerate(data_loader_iter):
            
            # tokenize
            batch_text = self.tokenizer(
                batch['text'],
                padding='max_length', 
                max_length=512, 
                truncation=True,
                return_tensors='pt', 
            )

            batch_text = {k:v.to(device) for k,v in batch_text.items()}
            batch_text["input_labels"] = batch["label"].to(device)
            batch_text["tabular_vectors"] = batch["tabular"].to(device)


            logits = self.model(
                    input_ids=batch_text["input_ids"], 
                    token_type_ids=batch_text["token_type_ids"], 
                    attention_mask=batch_text["attention_mask"],
                )

            loss = self.loss_fn(logits, batch_text['input_labels']) # 交叉熵损失

            if train:
                self.optimizer.zero_grad()
                loss.backward() # 多卡中，会自动同步多gpu梯度
                self.optimizer.step()

            # cpu等待当前GPU上所有计算完成,每个gpu进程都调用这个函数，则cpu就需要等待所有gpu完成梯度同步
            torch.cuda.synchronize(device)

            # 计算正确的预测数量
            preds = F.softmax(logits,dim=-1).argmax(dim=-1)
            correct = preds.eq(batch_text["input_labels"]).sum().item()
            
            # accumulate batch metrics and outputs
            loss_accumulated += self.reduce_value(loss,average=True).item() # reduce_value是计算当前batch多gpu的平均损失
            correct_accumulated += correct
            samples_accumulated += len(batch_text["input_labels"])
            preds_all.append(preds.detach())
            labels_all.append(batch_text['input_labels'].detach())

        # concatenate all batch tensors into one tensor and move to cpu for compatibility with sklearn metrics
        preds_all = torch.cat(preds_all, dim=0).cpu() # cat在亦有维度上拼接
        labels_all = torch.cat(labels_all, dim=0).cpu()

        if rank == 0:
            # metrics
            accuracy = accuracy_score(labels_all, preds_all)
            precision = precision_score(labels_all, preds_all, average='macro')
            recall = recall_score(labels_all, preds_all, average='macro')
            f1 = f1_score(labels_all, preds_all, average='macro')
            avg_loss_epoch = loss_accumulated / len(data_loader)

            # print metrics to console
            self.logger.info(
                f"samples={samples_accumulated}, \
                correct={correct_accumulated}, \
                acc={round(accuracy, 4)}, \
                recall={round(recall, 4)}, \
                prec={round(precision,4)}, \
                f1={round(f1, 4)}, \
                loss={round(avg_loss_epoch, 4)}"
            )    

            # 保存最优权重
            self.save_best_model(epoch=epoch, avg_loss_epoch=avg_loss_epoch,train=train)
                




