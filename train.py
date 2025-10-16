import os
import torch
import torch.distributed as dist
import argparse
from yacs.config import CfgNode
from trainlab.config.default_config import cfg as default_cfg
from trainlab.builder import build_from_cfg, MODELS, TRAINERS, DATASETS
import torch.multiprocessing as mp 

from trainlab.utils import Logger
# 注册机制必须再这里导入才能注册成功
from trainlab.datasets.stackoverflowdataset import StackOverflowDataset
from trainlab.model.bert import BertForSequenceClassification
from trainlab.model.transformers_model_warpper import TransformersModelWrapper
from trainlab.trainers.bert_trainer import BERTTrainer
from trainlab.trainers.lora_trainner import LORATrainer
from trainlab.trainers.peft_CLIP_trainer import PEFTCLIPTrainer
from trainlab.datasets.ucf101_dataset import UCF101Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    return parser.parse_args()


def train_worker(rank, world_size, cfg, log_queue, evaluate=False):
    """
    DDP 多进程训练入口函数
    Args:
        rank: 当前进程 GPU 编号，由 mp.spawn 自动传入
        world_size: 总 GPU 数
        trainer: Trainer 实例
        evaluate: 是否在每个 epoch 结束后做评估
    """
    torch.cuda.empty_cache()
    # 5️⃣ 构建 Dataset / Model / Trainer
    train_dataset = build_from_cfg(cfg.DATASET.TRAIN, DATASETS)
    val_dataset = build_from_cfg(cfg.DATASET.VAL, DATASETS)
    
    model = build_from_cfg(cfg.MODEL, MODELS)

    trainer = build_from_cfg(
        cfg.TRAINER, 
        TRAINERS, 
        model=model,
        model_name_or_path=cfg.MODEL.PRETRAINED,
        log_queue=log_queue)
    
    # 每个进程调用 trainer.train 并传入 rank
    trainer.train(
        rank=rank, 
        world_size=world_size, 
        batch_size_train=cfg.TRAINER.BATCH_SIZE_TRAIN,
        batch_size_eval=cfg.TRAINER.BATCH_SIZE_EVAL,
        train_dataset = train_dataset, 
        val_dataset=val_dataset,
        collate_fn=train_dataset.collate_fn,
        evaluate=evaluate)


def main():
    args = parse_args()
    # 1️⃣ 加载默认配置
    cfg = default_cfg.clone()
    # 2️⃣ 如果提供 YAML 文件，则合并
    if args.cfg:
        cfg.merge_from_file(args.cfg)

    # 3️⃣ 合并命令行覆盖参数
    if args.opts:
        cfg.merge_from_list(args.opts)

    # 多进程日志队列
    ctx = mp.get_context('spawn') # 必须启用spawn上下文初始化queue，不然会报错
    log_queue = ctx.Queue()
    logger_instance = Logger(queue=log_queue,log_dir=os.path.join(os.path.abspath('./logs'),cfg.TRAINER.PROJECT_NAME))
    main_logger = logger_instance.get_logger(log_queue,name='main_logger')

    
    main_logger.info(cfg) # 打印日志
    world_size = torch.cuda.device_count()

    # spawn 多进程，每个进程执行 train_worker
    mp.spawn(
        train_worker,
        args=(world_size, cfg, log_queue, True),  # 这里 rank 会自动传入
        nprocs=world_size,
        join=True
    )


    logger_instance.close() # 关闭日志相关工具


if __name__ == "__main__":
    main()