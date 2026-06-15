"""标准日志工具。"""

import logging
import sys
from typing import Optional


__all__ = ["setup_logging", "get_logger"]


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(message)s",
) -> None:
    """配置根日志记录器，输出到控制台（可选文件）。"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """返回指定名称的日志记录器。"""
    return logging.getLogger(name)
