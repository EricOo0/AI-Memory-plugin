"""SQLite 向量存储测试"""

import pytest
from pathlib import Path
from ai_memory.storage.database import Database
from ai_memory.vector.sqlite_provider import SQLiteVectorStore


def test_sqlite_provider_init(tmp_path):
    """测试 SQLite 向量存储初始化"""
    db_path = tmp_path / "memory.db"
    db = Database(db_path)
    db.create_tables()

    store = SQLiteVectorStore(db)
    assert store is not None


def test_sqlite_provider_add_search(tmp_path):
    """测试 SQLite 向量存储添加和搜索"""
    db_path = tmp_path / "memory.db"
    db = Database(db_path)
    db.create_tables()

    store = SQLiteVectorStore(db)

    # 添加向量
    store.add(
        ids=["MEMORY.md:1:5"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Test content"],
        metadatas=[{"path": "MEMORY.md", "start_line": 1, "end_line": 5, "model": "test"}]
    )

    # 搜索
    results = store.search([0.1, 0.2, 0.3], n_results=1)
    assert len(results) == 1
    assert results[0].id == "MEMORY.md:1:5"
    assert results[0].metadata["path"] == "MEMORY.md"


def test_sqlite_provider_get(tmp_path):
    """测试获取向量"""
    db_path = tmp_path / "memory.db"
    db = Database(db_path)
    db.create_tables()

    store = SQLiteVectorStore(db)

    # 添加向量
    store.add(
        ids=["MEMORY.md:1:5"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Test content"],
        metadatas=[{"path": "MEMORY.md", "start_line": 1, "end_line": 5, "model": "test"}]
    )

    # 获取
    vectors = store.get(["MEMORY.md:1:5"])
    assert len(vectors) == 1
    assert vectors[0]["id"] == "MEMORY.md:1:5"
    assert vectors[0]["text"] == "Test content"


def test_sqlite_provider_clear(tmp_path):
    """测试清空向量存储"""
    db_path = tmp_path / "memory.db"
    db = Database(db_path)
    db.create_tables()

    store = SQLiteVectorStore(db)

    # 添加向量
    store.add(
        ids=["MEMORY.md:1:5"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Test content"],
        metadatas=[{"path": "MEMORY.md", "start_line": 1, "end_line": 5, "model": "test"}]
    )

    # 清空
    store.clear()

    # 验证
    results = store.search([0.1, 0.2, 0.3])
    assert len(results) == 0
