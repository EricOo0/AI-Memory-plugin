"""自定义异常定义"""


class MemoryError(Exception):
    """记忆系统基础异常"""
    pass


class EmbeddingError(MemoryError):
    """嵌入生成异常"""
    pass


class RetrievalError(MemoryError):
    """检索异常"""
    pass


class SyncError(MemoryError):
    """同步异常"""
    pass
