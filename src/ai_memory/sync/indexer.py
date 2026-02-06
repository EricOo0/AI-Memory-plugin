"""记忆索引器"""

from typing import List, Tuple, Optional
from pathlib import Path

from ai_memory.storage.database import Database
from ai_memory.storage.file_manager import FileManager
from ai_memory.embeddings.base import EmbeddingProvider
from ai_memory.vector.base import VectorStore


class MemoryIndexer:
    """记忆索引器"""

    def __init__(
        self,
        database: Database,
        file_manager: FileManager,
        embedding_provider: EmbeddingProvider,
        vector_store: Optional[VectorStore] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 80
    ):
        self.db = database
        self.fm = file_manager
        self.provider = embedding_provider
        self.vector_store = vector_store  # 新增：向量存储
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def sync(self) -> None:
        """同步文件到数据库"""
        files = self.fm.get_memory_files()
        for file_path in files:
            self._sync_file(file_path)

    def _sync_file(self, file_path: Path) -> None:
        """同步单个文件"""
        relative_path = file_path.relative_to(self.fm.base_dir).as_posix()

        # 获取当前文件状态
        current_entry = self.fm.get_file_entry(relative_path)
        existing = self.db.get_file(relative_path)

        # 检查是否需要更新
        if existing and existing["hash"] == current_entry.hash:
            return

        # 删除旧的块
        if existing:
            self.db.delete_file(relative_path)

        # 插入新文件记录
        self.db.insert_file(current_entry)

        # 分块并索引
        chunks = self._chunk_text(current_entry.content)
        self._index_chunks(relative_path, chunks)

    def _chunk_text(self, text: str) -> List[Tuple[int, int, str]]:
        """将文本分块"""
        lines = text.split("\n")
        chunks = []
        start_line = 0

        while start_line < len(lines):
            # 计算结束行
            end_line = min(start_line + self.chunk_size, len(lines))
            chunk_text = "\n".join(lines[start_line:end_line])

            chunks.append((start_line + 1, end_line, chunk_text))

            # 下一块的起始位置（有重叠）
            start_line = max(end_line - self.chunk_overlap, start_line + 1)

            if start_line >= len(lines):
                break

        return chunks

    def _index_chunks(
        self,
        path: str,
        chunks: List[Tuple[int, int, str]]
    ) -> None:
        """索引文本块"""
        texts = [chunk[2] for chunk in chunks]
        embeddings = self.provider.embed_batch(texts)

        ids = []
        documents = []
        metadatas = []

        for (start, end, text), embedding in zip(chunks, embeddings):
            chunk_id = f"{path}:{start}-{end}"
            ids.append(chunk_id)
            documents.append(text)
            metadatas.append({
                "path": path,
                "start_line": start,
                "end_line": end,
                "model": self.provider.model
            })

            # 始终插入到 SQLite（元数据和全文搜索）
            self.db.insert_chunk(
                id=chunk_id,
                path=path,
                start_line=start,
                end_line=end,
                text=text,
                embedding=embedding,
                model=self.provider.model
            )

        # 如果配置了向量存储，也添加到向量存储
        if self.vector_store:
            self.vector_store.add(ids, embeddings, documents, metadatas)
