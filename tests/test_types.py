"""类型定义测试"""

import pytest
from datetime import datetime
from ai_memory.core.types import MemorySearchResult, MemorySource, MemoryEntry, MemoryStatus
from ai_memory.core.exceptions import MemoryError, EmbeddingError, RetrievalError, SyncError


def test_memory_search_result():
    """测试记忆搜索结果类型"""
    result = MemorySearchResult(
        path="MEMORY.md",
        start_line=1,
        end_line=5,
        score=0.85,
        snippet="Test content",
        source=MemorySource.MEMORY,
        citation="MEMORY.md#L1-L5"
    )
    assert result.score == 0.85
    assert result.citation == "MEMORY.md#L1-L5"
    assert result.path == "MEMORY.md"


def test_memory_source_enum():
    """测试记忆来源枚举"""
    assert MemorySource.MEMORY == "memory"
    assert MemorySource.DAILY == "daily"
    assert MemorySource.SESSION == "session"


def test_memory_entry():
    """测试记忆条目类型"""
    entry = MemoryEntry(
        path="memory/01-01-2025.md",
        content="# Daily log\nContent here",
        hash="abc123",
        size=100,
        modified_at=datetime.now(),
        source=MemorySource.DAILY
    )
    assert entry.path == "memory/01-01-2025.md"
    assert entry.hash == "abc123"
    assert entry.source == MemorySource.DAILY


def test_memory_status():
    """测试记忆系统状态类型"""
    status = MemoryStatus(
        backend="sqlite",
        files=10,
        chunks=100,
        embedding_model="test-model"
    )
    assert status.backend == "sqlite"
    assert status.files == 10
    assert status.chunks == 100
    assert status.embedding_model == "test-model"


def test_memory_error():
    """测试记忆基础异常"""
    with pytest.raises(MemoryError):
        raise MemoryError("Test error")


def test_embedding_error():
    """测试嵌入异常"""
    with pytest.raises(EmbeddingError):
        raise EmbeddingError("Embedding failed")


def test_retrieval_error():
    """测试检索异常"""
    with pytest.raises(RetrievalError):
        raise RetrievalError("Retrieval failed")


def test_sync_error():
    """测试同步异常"""
    with pytest.raises(SyncError):
        raise SyncError("Sync failed")


def test_error_inheritance():
    """测试异常继承关系"""
    with pytest.raises(MemoryError):
        raise EmbeddingError("Test")
    with pytest.raises(MemoryError):
        raise RetrievalError("Test")
    with pytest.raises(MemoryError):
        raise SyncError("Test")
