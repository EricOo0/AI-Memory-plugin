"""自定义异常定义"""


class MemoryError(Exception):
    """记忆系统基础异常"""
    pass


class MemoryNotInitializedError(MemoryError):
    """记忆系统未初始化异常"""
    def __init__(self):
        super().__init__(
            "记忆系统未初始化。请先调用 init() 或创建 MemoryManager 实例。\n"
            "示例：\n"
            "  from ai_memory import init\n"
            "  init()  # 使用默认配置\n"
            "  # 或者\n"
            "  init(memory_dir='./my-memory')  # 自定义目录"
        )


class InvalidConfigError(MemoryError):
    """无效配置异常"""
    def __init__(self, message: str):
        super().__init__(f"配置错误: {message}")


class EmbeddingError(MemoryError):
    """嵌入生成异常"""
    pass


class EmbeddingProviderError(EmbeddingError):
    """嵌入提供者异常"""
    def __init__(self, provider: str, message: str):
        super().__init__(
            f"嵌入提供者 '{provider}' 错误: {message}\n"
            "请检查模型是否正确安装和配置。"
        )


class RetrievalError(MemoryError):
    """检索异常"""
    pass


class VectorStoreError(MemoryError):
    """向量存储异常"""
    pass


class VectorStoreNotFoundError(VectorStoreError):
    """向量存储后端未找到异常"""
    def __init__(self, backend: str):
        super().__init__(
            f"向量存储后端 '{backend}' 未找到。\n"
            f"支持的后端: 'sqlite', 'chroma'\n"
            f"如果使用 ChromaDB，请确保已安装: pip install chromadb"
        )


class SyncError(MemoryError):
    """同步异常"""
    pass


class FileNotFoundError(MemoryError):
    """记忆文件未找到异常"""
    def __init__(self, path: str):
        super().__init__(f"记忆文件未找到: {path}")


class DatabaseError(MemoryError):
    """数据库操作异常"""
    pass
