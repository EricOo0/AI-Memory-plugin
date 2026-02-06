"""向量存储迁移测试"""

import pytest
from pathlib import Path
from ai_memory.vector.migration import VectorStoreMigrator
from ai_memory.config.settings import MemoryConfig
from ai_memory.storage.database import Database


def test_migration(tmp_path):
    """测试数据迁移"""
    # 创建源数据
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db_path = memory_dir / "memory.db"

    db = Database(db_path)
    db.create_tables()

    # 添加测试数据
    db.insert_chunk(
        id="MEMORY.md:1:3",
        path="MEMORY.md",
        start_line=1,
        end_line=3,
        text="Test content",
        embedding=[0.1, 0.2, 0.3],
        model="test"
    )

    # 迁移
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    source_config = MemoryConfig(storage={"dir": str(memory_dir)})
    target_config = MemoryConfig(
        storage={
            "dir": str(memory_dir),
            "vector_store": {"backend": "chroma", "chroma_persist_dir": str(chroma_dir)}
        }
    )

    migrator = VectorStoreMigrator(source_config, target_config)
    result = migrator.migrate(batch_size=100)

    assert result["success"]
    assert result["migrated"] > 0
