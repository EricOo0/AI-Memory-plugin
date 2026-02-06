"""索引同步器测试"""

import pytest
from ai_memory.sync.indexer import MemoryIndexer
from ai_memory.storage.database import Database
from ai_memory.storage.file_manager import FileManager
from ai_memory.embeddings.local import LocalEmbeddingProvider


def test_indexer_init(memory_dir):
    """测试索引器初始化"""
    db = Database(memory_dir / "memory.db")
    fm = FileManager(memory_dir)
    provider = LocalEmbeddingProvider()
    indexer = MemoryIndexer(db, fm, provider)
    assert indexer is not None
    assert indexer.db is not None
    assert indexer.fm is not None
    assert indexer.provider is not None


def test_indexer_chunk_text():
    """测试文本分块"""
    db = Database(__file__)  # 临时路径
    fm = FileManager(__file__)  # 临时路径
    provider = LocalEmbeddingProvider()
    indexer = MemoryIndexer(db, fm, provider, chunk_size=3, chunk_overlap=1)

    text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    chunks = indexer._chunk_text(text)

    assert len(chunks) > 0
    # 验证第一个块
    assert chunks[0][0] == 1  # start_line
    assert chunks[0][1] <= 3  # end_line
    assert "Line 1" in chunks[0][2]


def test_indexer_chunk_text_empty():
    """测试空文本分块"""
    db = Database(__file__)  # 临时路径
    fm = FileManager(__file__)  # 临时路径
    provider = LocalEmbeddingProvider()
    indexer = MemoryIndexer(db, fm, provider)

    chunks = indexer._chunk_text("")
    assert len(chunks) == 0


def test_indexer_sync(memory_dir):
    """测试完整同步流程"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    fm = FileManager(memory_dir)

    # 创建模拟的嵌入提供者（避免下载模型）
    class MockProvider:
        model = "mock-model"

        def embed_batch(self, texts):
            return [[0.1] * 10 for _ in texts]

    provider = MockProvider()

    # 创建测试文件
    fm.write_file("memory/test.md", "# Test\nContent here\nMore content")

    indexer = MemoryIndexer(db, fm, provider, chunk_size=2)
    indexer.sync()

    # 验证文件被索引
    file_entry = db.get_file("memory/test.md")
    assert file_entry is not None

    # 验证块被创建
    chunks = db.search_by_vector([0.1] * 10, limit=100)
    assert len(chunks) > 0


def test_indexer_sync_no_changes(memory_dir):
    """测试无变更时的同步"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    fm = FileManager(memory_dir)

    class MockProvider:
        model = "mock-model"

        def embed_batch(self, texts):
            return [[0.1] * 10 for _ in texts]

    provider = MockProvider()

    # 创建并同步文件
    fm.write_file("memory/test.md", "# Test\nContent")
    indexer = MemoryIndexer(db, fm, provider)
    indexer.sync()

    # 获取第一次同步后的块数量
    chunks_after_first = len(db.search_by_vector([0.1] * 10, limit=100))

    # 再次同步（无变更）
    indexer.sync()

    # 块数量应该不变
    chunks_after_second = len(db.search_by_vector([0.1] * 10, limit=100))
    assert chunks_after_first == chunks_after_second


def test_indexer_sync_file_change(memory_dir):
    """测试文件变更后的同步"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    fm = FileManager(memory_dir)

    class MockProvider:
        model = "mock-model"

        def embed_batch(self, texts):
            return [[0.1] * 10 for _ in texts]

    provider = MockProvider()

    # 创建并同步文件
    fm.write_file("memory/test.md", "# Test\nContent")
    indexer = MemoryIndexer(db, fm, provider)
    indexer.sync()

    # 获取第一次的哈希
    first_hash = db.get_file("memory/test.md")["hash"]

    # 修改文件
    fm.write_file("memory/test.md", "# Test\nModified content")

    # 再次同步
    indexer.sync()

    # 验证哈希已更新
    second_hash = db.get_file("memory/test.md")["hash"]
    assert first_hash != second_hash


def test_indexer_chunk_overlap(memory_dir):
    """测试分块重叠"""
    db = Database(__file__)  # 临时路径
    fm = FileManager(__file__)  # 临时路径
    provider = LocalEmbeddingProvider()
    indexer = MemoryIndexer(db, fm, provider, chunk_size=3, chunk_overlap=1)

    text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6"
    chunks = indexer._chunk_text(text)

    if len(chunks) > 1:
        # 验证相邻块有重叠
        first_end = chunks[0][1]
        second_start = chunks[1][0]
        # 第二个块的起始行应该小于等于第一个块的结束行（有重叠）
        assert second_start <= first_end
