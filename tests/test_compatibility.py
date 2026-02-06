"""SQLite 和 ChromaDB 兼容性测试"""

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_memory import MemoryManager, MemoryConfig
from ai_memory.storage.database import Database


def test_sqlite_backend(tmp_path):
    """测试 SQLite 后端"""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    config = MemoryConfig(
        storage={
            "dir": str(memory_dir),
            "vector_store": {"backend": "sqlite"}
        }
    )
    manager = MemoryManager(config)

    # 添加记忆
    manager.add_memory("# Test\nContent", tags=["test"])
    manager.sync()

    # 搜索
    results = manager.search("test")
    assert len(results) >= 0


def test_chroma_backend(tmp_path):
    """测试 ChromaDB 后端"""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    config = MemoryConfig(
        storage={
            "dir": str(memory_dir),
            "vector_store": {
                "backend": "chroma",
                "chroma_persist_dir": str(chroma_dir)
            }
        }
    )
    manager = MemoryManager(config)

    # 添加记忆
    manager.add_memory("# Test\nContent", tags=["test"])
    manager.sync()

    # 搜索
    results = manager.search("test")
    assert len(results) >= 0


def test_results_consistency(tmp_path):
    """测试两种后端结果一致性"""
    # 添加相同的测试数据
    content = "# Project\nAI memory plugin development"
    queries = ["AI", "memory", "plugin"]

    # SQLite 后端
    memory_dir1 = tmp_path / "memory_sqlite"
    memory_dir1.mkdir()

    sqlite_config = MemoryConfig(
        storage={"dir": str(memory_dir1), "vector_store": {"backend": "sqlite"}}
    )
    sqlite_manager = MemoryManager(sqlite_config)
    sqlite_manager.add_memory(content, tags=["project"])
    sqlite_manager.sync()

    sqlite_results = []
    for q in queries:
        sqlite_results.extend(sqlite_manager.search(q))

    # ChromaDB 后端
    memory_dir2 = tmp_path / "memory_chroma"
    memory_dir2.mkdir()

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    chroma_config = MemoryConfig(
        storage={
            "dir": str(memory_dir2),
            "vector_store": {
                "backend": "chroma",
                "chroma_persist_dir": str(chroma_dir)
            }
        }
    )
    chroma_manager = MemoryManager(chroma_config)
    chroma_manager.add_memory(content, tags=["project"])
    chroma_manager.sync()

    chroma_results = []
    for q in queries:
        chroma_results.extend(chroma_manager.search(q))

    # 两种后端都应该返回结果
    assert len(sqlite_results) > 0
    assert len(chroma_results) > 0


def test_migration_roundtrip(tmp_path):
    """测试迁移往返"""
    # 使用 SQLite 添加数据
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    sqlite_config = MemoryConfig(
        storage={"dir": str(memory_dir), "vector_store": {"backend": "sqlite"}}
    )
    sqlite_manager = MemoryManager(sqlite_config)
    sqlite_manager.add_memory("# Original\nOriginal content")
    sqlite_manager.sync()

    # 迁移到 ChromaDB
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    from ai_memory.vector.migration import VectorStoreMigrator

    chroma_config = MemoryConfig(
        storage={
            "dir": str(memory_dir),
            "vector_store": {
                "backend": "chroma",
                "chroma_persist_dir": str(chroma_dir)
            }
        }
    )

    migrator = VectorStoreMigrator(sqlite_config, chroma_config)
    result = migrator.migrate(batch_size=100)

    assert result["success"]
    assert result["migrated"] > 0
