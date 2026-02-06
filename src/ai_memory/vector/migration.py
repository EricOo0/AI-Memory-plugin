"""向量存储迁移工具"""

import logging
from typing import Optional, Callable
from pathlib import Path

from ai_memory.config.settings import MemoryConfig
from ai_memory.storage.database import Database
from ai_memory.vector.sqlite_provider import SQLiteVectorStore
from ai_memory.vector.chroma_provider import ChromaVectorStore

logger = logging.getLogger(__name__)


class VectorStoreMigrator:
    """向量存储迁移工具"""

    def __init__(
        self,
        source_config: MemoryConfig,
        target_config: MemoryConfig
    ):
        self.source_config = source_config
        self.target_config = target_config

    def migrate(
        self,
        batch_size: int = 1000,
        progress_callback: Optional[Callable[[dict], None]] = None
    ) -> dict:
        """执行迁移"""
        # 初始化源和目标
        source_db = Database(
            Path(self.source_config.storage.dir) / self.source_config.storage.db_name
        )
        source_db.create_tables()

        target_store = ChromaVectorStore(
            config={
                "chroma_persist_dir": str(
                    Path(self.target_config.storage.vector_store.chroma_persist_dir)
                )
            }
        )

        # 获取所有 chunks
        cursor = source_db.connect().cursor()
        cursor.execute("SELECT * FROM chunks")
        all_chunks = cursor.fetchall()

        total = len(all_chunks)
        migrated = 0
        failed = 0

        logger.info(f"开始迁移 {total} 个向量...")

        # 批量迁移
        for i in range(0, total, batch_size):
            batch = all_chunks[i:i + batch_size]

            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for chunk in batch:
                try:
                    ids.append(chunk["id"])
                    embeddings.append(eval(chunk["embedding"]))  # SQLite 存的是 JSON 字符串
                    documents.append(chunk["text"])
                    metadatas.append({
                        "path": chunk["path"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "model": chunk["model"]
                    })
                    migrated += 1
                except Exception as e:
                    logger.warning(f"迁移 chunk {chunk['id']} 失败: {e}")
                    failed += 1

            # 添加到 ChromaDB
            if ids:
                target_store.add(ids, embeddings, documents, metadatas)

            # 进度回调
            if progress_callback:
                progress_callback({
                    "migrated": migrated,
                    "total": total,
                    "failed": failed,
                    "progress": migrated / total
                })

        target_store.close()
        source_db.close()

        return {
            "total": total,
            "migrated": migrated,
            "failed": failed,
            "success": failed == 0
        }


def migrate_command(
    memory_dir: str,
    chroma_dir: str,
    batch_size: int = 1000
):
    """CLI 迁移命令"""
    import sys

    # 配置
    source_config = MemoryConfig(storage={"dir": memory_dir})
    target_config = MemoryConfig(
        storage={
            "dir": memory_dir,
            "vector_store": {"backend": "chroma", "chroma_persist_dir": chroma_dir}
        }
    )

    # 执行迁移
    migrator = VectorStoreMigrator(source_config, target_config)

    def on_progress(data):
        print(f"\r进度: {data['migrated']}/{data['total']} ({data['progress']*100:.1f}%)", end="")

    result = migrator.migrate(batch_size=batch_size, progress_callback=on_progress)
    print()

    # 输出结果
    print(f"迁移完成: {result['migrated']}/{result['total']} 成功")
    if result['failed'] > 0:
        print(f"警告: {result['failed']} 个向量迁移失败")
        sys.exit(1)
    else:
        print("成功!")
