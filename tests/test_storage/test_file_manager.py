"""文件管理器测试"""

import pytest
from datetime import datetime
from ai_memory.storage.file_manager import FileManager
from ai_memory.core.types import MemorySource


def test_get_memory_files(memory_dir):
    """测试获取记忆文件列表"""
    fm = FileManager(memory_dir)
    files = fm.get_memory_files()
    assert len(files) >= 1  # MEMORY.md 至少存在


def test_read_file(memory_dir):
    """测试读取文件内容"""
    fm = FileManager(memory_dir)
    content = fm.read_file("MEMORY.md")
    assert "# Long-term memory" in content


def test_write_file(memory_dir):
    """测试写入文件内容"""
    fm = FileManager(memory_dir)
    fm.write_file("test.md", "# Test\nNew content")
    content = fm.read_file("test.md")
    assert "# Test" in content
    assert "New content" in content


def test_get_file_hash(memory_dir):
    """测试获取文件哈希"""
    fm = FileManager(memory_dir)
    hash1 = fm.get_file_hash("MEMORY.md")
    hash2 = fm.get_file_hash("MEMORY.md")
    assert hash1 == hash2
    assert len(hash1) > 0


def test_add_memory(memory_dir):
    """测试添加新记忆"""
    fm = FileManager(memory_dir)
    path = fm.add_memory("# New entry\nNew content", tags=["test"])
    assert "memory/" in path
    assert path.endswith(".md")
    # 验证文件存在
    assert fm._resolve_path(path).exists()
    content = fm.read_file(path)
    assert "# New entry" in content


def test_add_memory_with_date(memory_dir):
    """测试添加带日期的记忆"""
    fm = FileManager(memory_dir)
    test_date = datetime(2025, 2, 5)
    path = fm.add_memory("# Test\nContent", tags=["test"], target_date=test_date)
    assert "memory/05-02-2025.md" in path


def test_get_file_entry(memory_dir):
    """测试获取文件元数据"""
    fm = FileManager(memory_dir)
    entry = fm.get_file_entry("MEMORY.md")
    assert entry.path == "MEMORY.md"
    assert entry.hash == fm.get_file_hash("MEMORY.md")
    assert entry.source == MemorySource.MEMORY
    assert entry.size > 0
    assert isinstance(entry.modified_at, datetime)


def test_get_file_entry_daily(memory_dir):
    """测试获取日期文件元数据"""
    fm = FileManager(memory_dir)
    fm.add_memory("# Daily entry\nContent")
    files = fm.get_memory_files()
    daily_files = [f for f in files if "memory/" in str(f)]
    if daily_files:
        relative_path = daily_files[0].relative_to(memory_dir).as_posix()
        entry = fm.get_file_entry(relative_path)
        assert entry.source == MemorySource.DAILY


def test_resolve_path_memory(memory_dir):
    """测试解析 MEMORY.md 路径"""
    fm = FileManager(memory_dir)
    path = fm._resolve_path("MEMORY.md")
    assert path == memory_dir / "MEMORY.md"


def test_resolve_path_daily(memory_dir):
    """测试解析日期文件路径"""
    fm = FileManager(memory_dir)
    path = fm._resolve_path("memory/05-02-2025.md")
    assert path == memory_dir / "memory" / "05-02-2025.md"
