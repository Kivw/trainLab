"""
Author: Kanna
Date: 2025-09-23
Version: 0.1
License: MIT
"""

from .utils import Registry
from yacs.config import CfgNode as CN

MODELS = Registry('models')
TRAINERS = Registry('trainers')
DATASETS = Registry('datasets')

def cfgnode_to_dict(cfg):
    if isinstance(cfg, CN):
        return {k.lower(): cfgnode_to_dict(v) for k, v in cfg.items()}
    elif isinstance(cfg, dict):
        return {k.lower(): cfgnode_to_dict(v) for k, v in cfg.items()}
    else:
        return cfg


def build_from_cfg(cfg, registry, *args, **kwargs):
    """
    根据配置和注册表构建实例
    Args:
        cfg (dict / CfgNode): 配置，必须包含 NAME 字段
        registry (Registry): 对应的注册表，如 MODELS / TRAINERS / DATASETS
        *args, **kwargs: 额外参数，会覆盖 cfg 中的字段
    Returns:
        实例化对象
    """
    if isinstance(cfg, dict):
        cfg = cfg.copy()
    else:
        cfg = dict(cfg)

    if 'NAME' not in cfg:
        raise KeyError("cfg must contain 'NAME' key")

    name = cfg.pop('NAME')

    if not registry.exists(name):
        raise KeyError(f"{name} is not registered in {registry.name}")

    cls_or_fn = registry.get(name)
    # --- 在这里把嵌套的 CfgNode 转成普通 dict ---
    cfg = cfgnode_to_dict(cfg)
    print('DDD',cfg)
    if registry == MODELS:
        return cls_or_fn(*args,cfg,**kwargs)
    return cls_or_fn(*args, **cfg, **kwargs)
