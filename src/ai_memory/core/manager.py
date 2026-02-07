"""MemoryManager - 记忆管理器核心入口"""

import logging
from typing import List, Optional
from pathlib import Path

from ai_memory.config.settings import MemoryConfig
from ai_memory.storage.database import Database
from ai_memory.storage.file_manager import FileManager
from ai_memory.embeddings.local import LocalEmbeddingProvider
from ai_memory.retrieval.hybrid_searcher import HybridSearcher
from ai_memory.sync.indexer import MemoryIndexer
from ai_memory.core.types import MemorySearchResult, MemoryStatus
from ai_memory.vector.sqlite_provider import SQLiteVectorStore
from ai_memory.vector.chroma_provider import ChromaVectorStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器 - 统一入口"""

    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()

        logger.info("初始化 MemoryManager")
        logger.info(f"  存储目录: {self.config.storage.dir}")
        logger.info(f"  向量后端: {self.config.storage.vector_store.backend}")
        logger.info(f"  嵌入模型: {self.config.embedding.model}")

        # 初始化各组件
        self.fm = FileManager(Path(self.config.storage.dir))
        self.db = Database(Path(self.config.storage.dir) / self.config.storage.db_name)
        self.db.create_tables()

        # 嵌入提供者
        if self.config.embedding.provider == "local":
            self.provider = LocalEmbeddingProvider(self.config.embedding.model)
        else:
            self.provider = LocalEmbeddingProvider(self.config.embedding.model)

        # 向量存储初始化
        vector_store_config = self.config.storage.vector_store.model_dump()
        if vector_store_config.get("backend") == "chroma":
            logger.info(f"  ChromaDB 配置: host={vector_store_config.get('chroma_host')}, port={vector_store_config.get('chroma_port', 18432)}")
            self.vector_store = ChromaVectorStore(config=vector_store_config)
        else:
            logger.info("  使用 SQLite 向量存储")
            self.vector_store = SQLiteVectorStore(self.db)

        # 索引器（传入向量存储）
        self.indexer = MemoryIndexer(
            self.db,
            self.fm,
            self.provider,
            vector_store=self.vector_store
        )

        # 检索器（传入已初始化的向量存储）
        self.searcher = HybridSearcher(
            self.db,
            vector_store=self.vector_store,
            vector_weight=self.config.retrieval.vector_weight,
            text_weight=self.config.retrieval.text_weight,
            time_weight=self.config.retrieval.time_weight,
            frequency_weight=self.config.retrieval.frequency_weight
        )

    def add_memory(
        self,
        content: str,
        tags: Optional[List[str]] = None
    ) -> str:
        """添加新记忆（向后兼容，映射到短期记忆）"""
        return self.add_daily_memory(content, tags)

    def add_long_term_memory(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> str:
        """添加长期记忆到 MEMORY.md

        Args:
            content: 记忆内容
            tags: 标签列表
            category: 分类章节（可选，默认根据标签自动推断）

        Returns:
            文件路径 "MEMORY.md"
        """
        logger.info(f"添加长期记忆: category={category}, tags={tags}, 内容长度={len(content)}")
        path = self.fm.add_to_long_term(content, tags, category)
        logger.info(f"  文件路径: {path}")
        self._sync_file(path)
        logger.info(f"  长期记忆已保存并同步")
        return path

    def add_daily_memory(
        self,
        content: str,
        tags: Optional[List[str]] = None
    ) -> str:
        """添加短期记忆到今日文件

        Args:
            content: 记忆内容
            tags: 标签列表

        Returns:
            文件路径 "memory/DD-MM-YYYY.md"
        """
        logger.info(f"添加短期记忆: tags={tags}, 内容长度={len(content)}")
        path = self.fm.add_memory(content, tags)
        logger.info(f"  文件路径: {path}")
        self._sync_file(path)
        logger.info(f"  短期记忆已保存并同步")
        return path

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[MemorySearchResult]:
        """搜索记忆"""
        logger.info(f"搜索记忆: query='{query}'")
        # 生成查询嵌入
        query_embedding = self.provider.embed(query)

        results = self.searcher.search(
            query=query,
            query_embedding=query_embedding,
            max_results=max_results or self.config.retrieval.max_results,
            min_score=min_score or self.config.retrieval.min_score
        )
        logger.info(f"  返回 {len(results)} 条结果")
        return results

    def get_memory(
        self,
        path: str,
        from_line: Optional[int] = None,
        lines: int = 20
    ) -> str:
        """获取特定记忆内容"""
        content = self.fm.read_file(path)
        if from_line:
            line_list = content.split("\n")
            start = max(0, from_line - 1)
            end = min(len(line_list), start + lines)
            return "\n".join(line_list[start:end])
        return content

    def sync(self) -> None:
        """同步所有文件"""
        self.indexer.sync()

    def status(self) -> MemoryStatus:
        """获取系统状态"""
        backend = self.config.storage.vector_store.backend
        return MemoryStatus(
            backend=backend,
            files=len(self.fm.get_memory_files()),
            chunks=0,  # TODO: 从数据库或向量存储获取
            embedding_model=self.provider.model
        )

    def _sync_file(self, relative_path: str) -> None:
        """同步单个文件"""
        full_path = self.fm._resolve_path(relative_path)
        self.indexer._sync_file(full_path)
