"""ChromaDB 向量存储测试"""

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_chroma_provider_init(tmp_path):
    """测试 ChromaDB 提供者初始化"""
    config = {"chroma_persist_dir": str(tmp_path / "chroma")}
    from ai_memory.vector.chroma_provider import ChromaVectorStore

    store = ChromaVectorStore(config=config)
    assert store is not None


def test_chroma_provider_add_search(tmp_path):
    """测试 ChromaDB 添加和搜索"""
    config = {"chroma_persist_dir": str(tmp_path / "chroma")}
    from ai_memory.vector.chroma_provider import ChromaVectorStore

    store = ChromaVectorStore(config=config)

    # 添加向量
    store.add(
        ids=["chunk1", "chunk2"],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        documents=["Test content 1", "Test content 2"],
        metadatas=[
            {"path": "MEMORY.md", "start_line": 1, "end_line": 5},
            {"path": "MEMORY.md", "start_line": 6, "end_line": 10}
        ]
    )

    # 搜索
    results = store.search([0.1, 0.2, 0.3], n_results=1)
    assert len(results) == 1
    assert results[0].id == "chunk1"
    assert results[0].metadata["path"] == "MEMORY.md"


def test_chroma_provider_delete(tmp_path):
    """测试删除向量"""
    config = {"chroma_persist_dir": str(tmp_path / "chroma")}
    from ai_memory.vector.chroma_provider import ChromaVectorStore

    store = ChromaVectorStore(config=config)

    # 添加向量
    store.add(
        ids=["chunk1"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Test"],
        metadatas=[{"path": "test.md"}]
    )

    # 删除
    store.delete(["chunk1"])

    # 验证
    count = store.count()
    assert count == 0
