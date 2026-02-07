"""混合检索器"""

import logging
from typing import List, Optional

from ai_memory.storage.database import Database
from ai_memory.vector.base import VectorStore
from ai_memory.vector.sqlite_provider import SQLiteVectorStore
from ai_memory.vector.chroma_provider import ChromaVectorStore
from ai_memory.retrieval.scorer import MultiDimensionScorer
from ai_memory.core.types import MemorySearchResult, MemorySource

logger = logging.getLogger(__name__)


class HybridSearcher:
    """混合检索器（支持多种向量存储后端）"""

    def __init__(
        self,
        database: Database,
        vector_store: Optional[VectorStore] = None,
        vector_store_config: dict = None,
        **kwargs
    ):
        self.db = database
        self.scorer = MultiDimensionScorer(**kwargs)

        # 初始化向量存储
        if vector_store:
            self.vector_store = vector_store
        elif vector_store_config:
            backend = vector_store_config.get("backend", "sqlite")
            if backend == "chroma":
                self.vector_store = ChromaVectorStore(config=vector_store_config)
            else:
                self.vector_store = SQLiteVectorStore(database)
        else:
            self.vector_store = SQLiteVectorStore(database)

        logger.info(f"使用向量存储: {type(self.vector_store).__name__}")

    def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        max_results: int = 6,
        min_score: float = 0.35
    ) -> List[MemorySearchResult]:
        """混合检索"""
        results = []

        # 文本搜索
        text_results = self.db.search_by_text(query, limit=max_results * 2)

        # 向量搜索
        vector_results = []
        if query_embedding:
            try:
                vector_search_results = self.vector_store.search(
                    query_embedding,
                    n_results=max_results * 2
                )
                # ChromaDB 返回的结果包含 documents，需要提取
                # 格式可能是 VectorSearchResult 或其他格式
                for r in vector_search_results:
                    path = None
                    start_line = None
                    end_line = None
                    vector_score = 0  # 默认值
                    text = ""

                    # 尝试获取 score
                    if hasattr(r, "score"):
                        vector_score = r.score

                    if hasattr(r, "metadata"):
                        path = r.metadata.get("path")
                        start_line = r.metadata.get("start_line")
                        end_line = r.metadata.get("end_line")
                    elif isinstance(r, dict):
                        path = r.get("path")
                        start_line = r.get("start_line")
                        end_line = r.get("end_line")
                        vector_score = r.get("score", 0)

                    # 获取文本
                    if path and start_line is not None and end_line is not None:
                        chunk_id = f"{path}:{start_line}-{end_line}"
                        # 从数据库获取文本
                        try:
                            chunk_data = self.db.get([chunk_id])
                            if chunk_data and len(chunk_data) > 0:
                                text = chunk_data[0].get("text", "")
                        except Exception:
                            text = ""

                    vector_results.append({
                        "path": path or "",
                        "start_line": start_line or 0,
                        "end_line": end_line or 0,
                        "vector_score": vector_score,
                        "text": text
                    })
            except Exception as e:
                logger.error(f"向量搜索失败: {e}")

        # 合并结果
        merged = self._merge_results(text_results, vector_results)

        # 评分和过滤
        for item in merged:
            score = self.scorer.score(
                vector_score=item.get("vector_score", 0),
                text_score=item.get("text_score", 0)
            )

            if score >= min_score:
                self.scorer.record_access(item["path"])

                results.append(MemorySearchResult(
                    path=item["path"],
                    start_line=item["start_line"],
                    end_line=item["end_line"],
                    score=score,
                    snippet=item["text"],
                    source=MemorySource.MEMORY,
                    citation=f"{item['path']}#L{item['start_line']}-L{item['end_line']}"
                ))

        # 排序并限制结果
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

    def _merge_results(
        self,
        text_results: List[dict],
        vector_results: List[dict]
    ) -> List[dict]:
        """合并文本和向量结果"""
        merged = {}
        all_results = list(zip(text_results, ["text"] * len(text_results))) + \
                     list(zip(vector_results, ["vector"] * len(vector_results)))

        for result, source in all_results:
            # 确保所有必需字段存在
            if "path" not in result:
                logger.debug(f"跳过无 path 的结果: {result}")
                continue

            key = f"{result['path']}#{result['start_line']}-{result['end_line']}"
            if key not in merged:
                # 向量结果可能没有 text 字段，设置为空字符串
                merged[key] = {
                    "path": result["path"],
                    "start_line": result["start_line"],
                    "end_line": result["end_line"],
                    "text": result.get("text", ""),  # 安全访问
                    "text_score": 0,
                    "vector_score": 0
                }
            # 安全访问 score（向量结果使用 vector_score 键）
            if source == "vector" and "vector_score" in result:
                merged[key]["vector_score"] = result["vector_score"]
            elif "score" in result:
                merged[key][f"{source}_score"] = result["score"]

        return list(merged.values())
