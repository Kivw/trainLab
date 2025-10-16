from yacs.config import CfgNode as CN

_C = CN()

# *****************************************Model 默认配置**********************************
_C.MODEL = CN()
_C.MODEL.NAME = 'BertForSequenceClassification'
# # Bert 配置参数
# _C.MODEL.MAX_SEQ_LENGTH = 512
# _C.MODEL.VOCAB_SIZE = 30522
# _C.MODEL.N_LAYERS = 12
# _C.MODEL.N_HEADS = (12,) * _C.MODEL.N_LAYERS
# _C.MODEL.EMB_SIZE = 768
# _C.MODEL.INTERMEDIATE_SIZE = _C.MODEL.EMB_SIZE * 4
# _C.MODEL.DROPOUT = 0.1
# _C.MODEL.N_CLASSES = 2
# _C.MODEL.LAYER_NORM_EPS = 1e-12
# _C.MODEL.PAD_TOKEN_ID = 103
# _C.MODEL.RETURN_POOLER_OUTPUT = False
_C.MODEL.PRETRAINED = '/data/lj/task/BERT-LoRA-TensorRT/bert-base-uncased'
# **********************************Trainer 配置节点**************************************
_C.TRAINER = CN()
_C.TRAINER.NAME = 'BERTTrainer'
_C.TRAINER.PROJECT_NAME = 'betrlora'
# 基本训练参数
_C.TRAINER.EPOCHS = 30
_C.TRAINER.BATCH_SIZE_TRAIN = 8
_C.TRAINER.BATCH_SIZE_EVAL = 1
_C.TRAINER.SAVE = True
_C.TRAINER.OUTPUT_DIR = './'
_C.TRAINER.OUTPUT_FILENAME = 'weight'
# 优化器配置
_C.TRAINER.OPTIMIZER_CLASS = 'AdamW'  # 可以改成你的优化器类名
_C.TRAINER.OPTIMIZER_KWARGS = CN()
_C.TRAINER.OPTIMIZER_KWARGS.LR = 1e-3
_C.TRAINER.OPTIMIZER_KWARGS.WEIGHT_DECAY = 0.01
# 学习率调度器配置
_C.TRAINER.SCHEDULER_CLASS = 'CosineAnnealingLR'  # 或其他 scheduler
_C.TRAINER.SCHEDULER_KWARGS = CN()
_C.TRAINER.SCHEDULER_KWARGS.T_MAX = 120 * _C.TRAINER.EPOCHS  # 这里假设每个 epoch 有 120 个 step
_C.TRAINER.SCHEDULER_KWARGS.ETA_MIN = 1e-6
# 损失函数，可以是字符串或者直接传类名
_C.TRAINER.LOSS_FN = 'CrossEntropyLoss'

# **************************************Dataset 默认配置****************************
_C.DATASET = CN()
# 训练集
_C.DATASET.TRAIN = CN()
_C.DATASET.TRAIN.NAME = 'StackOverflowDataset'
_C.DATASET.TRAIN.ROOT = "/data/lj/task/trainlab/data"
_C.DATASET.TRAIN.SPLIT = 'train'
_C.DATASET.TRAIN.TRANSFORM = None
_C.DATASET.TRAIN.CACHE_IN_MEMORY = False
# 验证集
_C.DATASET.VAL = CN()
_C.DATASET.VAL.NAME = 'StackOverflowDataset'
_C.DATASET.VAL.ROOT = "/data/lj/task/trainlab/data"
_C.DATASET.VAL.SPLIT = 'val'
_C.DATASET.VAL.TRANSFORM = None
_C.DATASET.VAL.CACHE_IN_MEMORY = False
# 测试集
_C.DATASET.TEST = CN()
_C.DATASET.TEST.NAME = 'StackOverflowDataset'
_C.DATASET.TEST.ROOT = "/data/lj/task/trainlab/data"
_C.DATASET.TEST.SPLIT = 'test'
_C.DATASET.TEST.TRANSFORM = None
_C.DATASET.TEST.CACHE_IN_MEMORY = False



cfg = _C
