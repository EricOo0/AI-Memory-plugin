"""向量存储抽象基类"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    id: str
    score: float
    metadata: dict


class VectorStore(ABC):
    """向量存储抽象接口"""

    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[dict]
    ) -> None:
        """添加向量"""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[dict] = None
    ) -> List[VectorSearchResult]:
        """向量搜索"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """删除向量"""
        pass

    @abstractmethod
    def get(self, ids: List[str]) -> List[dict]:
        """获取向量"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空所有数据"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭连接"""
        pass
