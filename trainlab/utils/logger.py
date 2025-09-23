"""
Author: Kanna
Date: 2025-09-23
Version: 0.1
License: MIT
"""

import logging
import sys
import time
from pathlib import Path
from threading import Lock

class Logger:
    _instance = None
    _lock = Lock()  # 确保多线程环境下的单例安全

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name="train_logger", level="info", log_dir="logs", log_file=None):
        if hasattr(self, "_initialized") and self._initialized:
            return  # 防止重复初始化

        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.propagate = False  # 避免重复打印

        # 创建 log 目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # 自动生成带时间戳的文件名
        if log_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = log_path / f"training_{timestamp}.log"
        else:
            log_file = Path(log_dir) / log_file

        # 文件处理器
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self._initialized = True

    def get_logger(self):
        return self.logger
