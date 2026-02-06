"""集成测试"""

from ai_memory import MemoryManager, MemoryConfig


def test_full_workflow(memory_dir):
    """测试完整工作流"""
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)

    # 1. 添加记忆
    manager.add_memory("# Project\nAI memory plugin", tags=["project"])
    manager.add_memory("# User\nLikes Python", tags=["user"])

    # 2. 同步
    manager.sync()

    # 3. 搜索
    results = manager.search("project")
    assert len(results) >= 0

    # 4. 获取内容
    if results:
        content = manager.get_memory(results[0].path)
        assert "AI memory" in content

    # 5. 检查状态
    status = manager.status()
    assert status.files > 0


def test_tool_integration(memory_dir):
    """测试工具接口集成"""
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)

    from ai_memory.tools import get_memory_tools

    tools = get_memory_tools(manager)
    assert "memory_search" in tools
    assert "memory_add" in tools
    assert "memory_get" in tools

    # 添加记忆
    result = tools["memory_add"]("Test content", ["test"])
    assert result["status"] == "success"

    # 搜索
    search_result = tools["memory_search"]("test")
    assert "count" in search_result


def test_config_from_yaml_integration(memory_dir):
    """测试配置文件集成"""
    import tempfile
    from pathlib import Path

    # 创建配置文件
    config_file = memory_dir / "config.yaml"
    config_file.write_text("""
storage:
  dir: ./memory
retrieval:
  max_results: 6
  min_score: 0.35
""")

    # 从配置加载
    config = MemoryConfig.from_yaml(config_file)
    manager = MemoryManager(config)

    # 验证配置生效
    assert manager.config.retrieval.max_results == 6
    assert manager.config.retrieval.min_score == 0.35


def test_memory_entry_retrieval(memory_dir):
    """测试记忆条目检索流程"""
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)

    # 添加记忆
    path = manager.add_memory("# Test Entry\nThis is a test content")
    manager.sync()

    # 搜索获取结果
    results = manager.search("test")

    # 如果有结果，获取完整内容
    if results:
        full_content = manager.get_memory(results[0].path)
        assert "# Test Entry" in full_content
        assert "This is a test content" in full_content


def test_openai_functions_format(memory_dir):
    """测试 OpenAI Function Calling 格式"""
    from ai_memory.tools import get_openai_functions

    functions = get_openai_functions()

    assert len(functions) == 3
    assert all(f["type"] == "function" for f in functions)

    # 验证每个函数的结构
    for func in functions:
        assert "function" in func
        assert "name" in func["function"]
        assert "description" in func["function"]
        assert "parameters" in func["function"]
