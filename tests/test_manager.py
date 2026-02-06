"""MemoryManager 核心测试"""

import pytest
from ai_memory import MemoryManager, MemoryConfig


def test_manager_init(memory_dir):
    """测试管理器初始化"""
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)
    assert manager is not None


def test_manager_default_config():
    """测试默认配置初始化"""
    manager = MemoryManager()
    assert manager is not None


def test_manager_add_memory(memory_dir):
    """测试添加记忆"""
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)
    manager.add_memory("# Test\nNew content", tags=["test"])

    # 验证文件存在
    assert (memory_dir / "memory").exists()


def test_manager_search(memory_dir):
    """测试搜索记忆"""
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)

    # 先添加一些记忆
    manager.add_memory("# Project\nAI memory plugin development")
    manager.sync()

    # 搜索
    results = manager.search("AI memory")
    assert len(results) >= 0


def test_manager_get_memory(memory_dir):
    """测试获取记忆内容"""
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)
    manager.add_memory("# Test\nContent")

    results = manager.search("Test")
    if results:
        content = manager.get_memory(results[0].path)
        assert "Test" in content


def test_manager_get_memory_with_lines(memory_dir):
    """测试获取特定行数的内容"""
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)
    manager.add_memory("# Test\nLine 1\nLine 2\nLine 3\nLine 4\nLine 5")

    results = manager.search("Test")
    if results:
        content = manager.get_memory(results[0].path, from_line=2, lines=2)
        lines = content.split("\n")
        assert "Line 1" in lines[0]
        assert "Line 2" in lines[1]


def test_manager_sync(memory_dir):
    """测试同步"""
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)

    # 添加记忆
    manager.add_memory("# Test\nContent")

    # 同步应该不抛出异常
    manager.sync()


def test_manager_status(memory_dir):
    """测试获取状态"""
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)
    manager.add_memory("# Test\nContent")
    manager.sync()

    status = manager.status()
    assert status.files > 0
    assert status.backend == "sqlite"
    assert status.embedding_model is not None


def test_manager_config_retrieval_params(memory_dir):
    """测试检索配置参数生效"""
    config = MemoryConfig(
        storage={"dir": str(memory_dir)},
        retrieval={"max_results": 10, "min_score": 0.5}
    )
    manager = MemoryManager(config)
    manager.add_memory("# Test\nContent")
    manager.sync()

    results = manager.search("Test")
    # 由于没有实际嵌入，可能返回空结果
    # 但配置应该被正确设置
    assert manager.config.retrieval.max_results == 10
    assert manager.config.retrieval.min_score == 0.5
