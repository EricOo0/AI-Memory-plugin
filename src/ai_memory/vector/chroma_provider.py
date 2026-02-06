"""ChromaDB 向量存储提供者"""

import logging
from typing import List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from ai_memory.vector.base import VectorStore, VectorSearchResult

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB 向量存储"""

    def __init__(self, collection_name: str = "memory_chunks", config: dict = None):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB 未安装，请运行: pip install chromadb")

        self.config = config or {}
        self.collection_name = collection_name
        self.client = self._create_client()
        self.collection = self._get_or_create_collection()

    def _create_client(self):
        """创建 ChromaDB 客户端"""
        if "chroma_host" in self.config and "chroma_port" in self.config:
            # 远程服务器
            return chromadb.Client(
                host=self.config["chroma_host"],
                port=self.config["chroma_port"]
            )
        elif "chroma_persist_dir" in self.config:
            # 本地持久化
            return chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(self.config["chroma_persist_dir"])
                )
            )
        else:
            # 内存模式
            return chromadb.Client(Settings(anonymized_telemetry=False))

    def _get_or_create_collection(self):
        """获取或创建集合"""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "AI Memory Chunks"}
            )

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[dict]
    ) -> None:
        """添加向量"""
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            raise

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[dict] = None
    ) -> List[VectorSearchResult]:
        """向量搜索（使用 HNSW 索引）"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )

            vector_results = []
            for i, id_ in enumerate(results["ids"][0]):
                vector_results.append(
                    VectorSearchResult(
                        id=id_,
                        score=1.0 - results["distances"][0][i],  # 距离转分数
                        metadata=results["metadatas"][0][i]
                    )
                )

            return vector_results
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            raise

    def delete(self, ids: List[str]) -> None:
        """删除向量"""
        try:
            self.collection.delete(ids=ids)
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            raise

    def get(self, ids: List[str]) -> List[dict]:
        """获取向量"""
        try:
            results = self.collection.get(ids=ids, include=["embeddings", "documents", "metadatas"])
            return [
                {
                    "id": id_,
                    "embedding": results["embeddings"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                }
                for i, id_ in enumerate(results["ids"])
            ]
        except Exception as e:
            logger.error(f"获取向量失败: {e}")
            raise

    def clear(self) -> None:
        """清空所有数据"""
        try:
            # 删除并重新创建集合
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "AI Memory Chunks"}
            )
        except Exception as e:
            logger.error(f"清空向量存储失败: {e}")
            raise

    def close(self) -> None:
        """关闭连接"""
        # ChromaDB 客户端无需显式关闭
        pass

    def count(self) -> int:
        """获取向量数量"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"获取向量数量失败: {e}")
            return 0
