import os
import torch
import torch.distributed as dist
import argparse
from yacs.config import CfgNode
from trainlab.config.default_config import cfg as default_cfg
from trainlab.builder import build_from_cfg, MODELS, TRAINERS, DATASETS
import torch.multiprocessing as mp 
from transformers import BertTokenizer
from trainlab.utils import Logger
# 注册机制必须再这里导入才能注册成功
from trainlab.datasets.stackoverflowdataset import StackOverflowDataset
from trainlab.model.bert import BertForSequenceClassification
from trainlab.trainers.bert_trainer import BERTTrainer
from trainlab.trainers.lora_trainner import LORATrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    return parser.parse_args()


def train_worker(rank, world_size, trainer,train_dataset,val_dataset, evaluate=False):
    """
    DDP 多进程训练入口函数
    Args:
        rank: 当前进程 GPU 编号，由 mp.spawn 自动传入
        world_size: 总 GPU 数
        trainer: Trainer 实例
        evaluate: 是否在每个 epoch 结束后做评估
    """
    torch.cuda.empty_cache()

    # 每个进程调用 trainer.train 并传入 rank
    trainer.train(rank=rank, world_size=world_size,train_dataset = train_dataset, val_dataset=val_dataset,collate_fn=train_dataset.collate_fn,evaluate=evaluate)


def main():
    # 多进程日志队列
    log_queue = mp.Queue()
    logger_instance = Logger(queue=log_queue,log_dir='./logs')
    main_logger = logger_instance.get_logger(log_queue,name='main_logger')
    args = parse_args()

    # 1️⃣ 加载默认配置
    cfg = default_cfg.clone()

    # 2️⃣ 如果提供 YAML 文件，则合并
    if args.cfg:
        cfg.merge_from_file(args.cfg)

    # 3️⃣ 合并命令行覆盖参数
    if args.opts:
        cfg.merge_from_list(args.opts)

    # 5️⃣ 构建 Dataset / Model / Trainer
    train_dataset = build_from_cfg(cfg.DATASET.TRAIN, DATASETS)
    val_dataset = build_from_cfg(cfg.DATASET.VAL, DATASETS)
    
    model = build_from_cfg(cfg.MODEL, MODELS)
    tokenizer =  BertTokenizer.from_pretrained(cfg.MODEL.PRETRAINED)
    trainer = build_from_cfg(cfg.TRAINER, TRAINERS, model=model,tokenizer=tokenizer,log_queue=log_queue)
    main_logger.info(cfg) # 打印日志
    world_size = torch.cuda.device_count()

    # spawn 多进程，每个进程执行 train_worker
    mp.spawn(
        train_worker,
        args=(world_size, trainer, train_dataset, val_dataset, True),  # 这里 rank 会自动传入
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()