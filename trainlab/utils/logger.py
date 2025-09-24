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
    - 子进程只需挂载队列 QueueHandler
    """
    def __init__(self, queue, log_dir, level=logging.INFO):
        """
        初始化主进程 Logger
        :param queue: torch.multiprocessing.Queue
        :param log_dir: 日志文件夹
        :param level: 日志等级
        """
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
            "[%(asctime)s | %(name)s | %(filename)s:%(lineno)d | "
            "%(process)d | %(thread)d | %(levelname)s]: %(message)s"
        ))

        # QueueListener 同时绑定两个 handler
        self.listener = QueueListener(self.queue, fh, ch)
        self.listener.start()

    @staticmethod
    def get_logger(queue, level=logging.INFO, name=None):
        """
        子进程调用：把 QueueHandler 绑定到 logger
        """
        logger = logging.getLogger(name)
        if not any(isinstance(h, QueueHandler) for h in logger.handlers):
            qh = QueueHandler(queue)
            qh.setFormatter(logging.Formatter(
                "[%(asctime)s | %(name)s | %(filename)s:%(lineno)d | "
                "%(process)d | %(thread)d | %(levelname)s]: %(message)s"
            ))
            logger.setLevel(level)
            logger.addHandler(qh)
        return logger

    def close(self):
        """关闭监听器"""
        if self.listener:
            self.listener.stop()
            self.listener = None




