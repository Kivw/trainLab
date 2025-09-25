"""
Author: Kanna
Date: 2025-09-23
Version: 0.1
License: MIT
"""
import os
import logging
from logging.handlers import QueueHandler, QueueListener
import torch.multiprocessing as mp
import time

class Logger:
    """
    进程安全 Logger：
    - 主进程创建 listener 写文件 + 控制台
    - 子进程只挂载 QueueHandler
    """
    def __init__(self, queue, log_dir, level=logging.INFO):
        self.level = level
        self.queue = queue

        # 自动生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f"log_{timestamp}.txt"
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, filename)

        # 文件 Handler
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))

        # 控制台 Handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            "[%(asctime)s | %(name)s | %(filename)s:%(lineno)d | %(levelname)s]: %(message)s"
        ))

        # QueueListener 绑定两个 handler
        self.listener = QueueListener(self.queue, fh, ch)
        self.listener.start()
    
    @staticmethod
    def get_logger(queue, level=logging.INFO, name=None):
        logger = logging.getLogger(name)
        if not getattr(logger, "_queue_handler_set", False):
            qh = QueueHandler(queue)
            qh.setFormatter(logging.Formatter(
                "[%(asctime)s | %(name)s | %(filename)s:%(lineno)d | %(levelname)s]: %(message)s"
            ))
            logger.addHandler(qh)
            logger.setLevel(level)
            logger._queue_handler_set = True

        # 阻止日志传播到 root logger，避免重复打印
        logger.propagate = False
        return logger

    def close(self):
        """关闭监听器"""
        if self.listener:
            self.listener.stop()
            self.listener = None




