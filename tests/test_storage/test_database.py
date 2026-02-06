"""数据库层测试"""

import pytest
from datetime import datetime
from ai_memory.storage.database import Database
from ai_memory.core.types import MemoryEntry, MemorySource


def test_database_init(memory_dir):
    """测试数据库初始化"""
    db = Database(memory_dir / "memory.db")
    assert db is not None
    assert db.db_path == memory_dir / "memory.db"


def test_create_tables(memory_dir):
    """测试创建表"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    # 表创建成功，无异常
    db.close()


def test_connect(memory_dir):
    """测试数据库连接"""
    db = Database(memory_dir / "memory.db")
    conn = db.connect()
    assert conn is not None
    db.close()


def test_insert_file(memory_dir):
    """测试插入文件记录"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    entry = MemoryEntry(
        path="MEMORY.md",
        content="# Test\nContent",
        hash="abc123",
        size=100,
        modified_at=datetime.now(),
        source=MemorySource.MEMORY
    )
    db.insert_file(entry)
    retrieved = db.get_file("MEMORY.md")
    assert retrieved is not None
    assert retrieved["path"] == "MEMORY.md"
    db.close()


def test_get_file_not_found(memory_dir):
    """测试获取不存在的文件"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    retrieved = db.get_file("nonexistent.md")
    assert retrieved is None
    db.close()


def test_delete_file(memory_dir):
    """测试删除文件记录"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    entry = MemoryEntry(
        path="MEMORY.md",
        content="# Test\nContent",
        hash="abc123",
        size=100,
        modified_at=datetime.now(),
        source=MemorySource.MEMORY
    )
    db.insert_file(entry)
    db.delete_file("MEMORY.md")
    retrieved = db.get_file("MEMORY.md")
    assert retrieved is None
    db.close()


def test_insert_chunk(memory_dir):
    """测试插入文本块"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    db.insert_chunk(
        id="chunk1",
        path="MEMORY.md",
        start_line=1,
        end_line=5,
        text="Test content",
        embedding=[0.1, 0.2, 0.3],
        model="test-model"
    )
    chunks = db.search_by_vector([0.1, 0.2, 0.3], limit=1)
    assert len(chunks) == 1
    assert chunks[0]["id"] == "chunk1"
    db.close()


def test_search_by_vector(memory_dir):
    """测试向量搜索"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    db.insert_chunk(
        id="chunk1",
        path="MEMORY.md",
        start_line=1,
        end_line=5,
        text="Test content",
        embedding=[0.1, 0.2, 0.3],
        model="test-model"
    )
    db.insert_chunk(
        id="chunk2",
        path="MEMORY.md",
        start_line=6,
        end_line=10,
        text="Another content",
        embedding=[0.9, 0.8, 0.7],
        model="test-model"
    )
    results = db.search_by_vector([0.1, 0.2, 0.3], limit=10)
    assert len(results) > 0
    assert results[0]["id"] == "chunk1"
    db.close()


def test_search_by_text(memory_dir):
    """测试全文搜索"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    db.insert_chunk(
        id="chunk1",
        path="MEMORY.md",
        start_line=1,
        end_line=5,
        text="This is a test about memory",
        embedding=[0.1] * 3,
        model="test-model"
    )
    results = db.search_by_text("memory", limit=10)
    assert len(results) > 0
    assert "score" in results[0]
    db.close()


def test_cosine_similarity():
    """测试余弦相似度计算"""
    db = Database(__file__)  # 临时路径
    # 相同向量
    score1 = db._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert abs(score1 - 1.0) < 0.001
    # 正交向量
    score2 = db._cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert abs(score2 - 0.0) < 0.001
    # 相反向量
    score3 = db._cosine_similarity([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
    assert abs(score3 - (-1.0)) < 0.001
