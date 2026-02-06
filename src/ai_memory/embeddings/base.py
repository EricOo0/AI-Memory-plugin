"""嵌入提供者抽象接口"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """嵌入提供者抽象接口"""

    def __init__(self, model: str = None):
        self.model = model

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """生成单段文本的嵌入向量"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        pass

    @abstractmethod
    def dimensions(self) -> int:
        """获取嵌入向量维度"""
        pass
