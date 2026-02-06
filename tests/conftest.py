"""pytest 配置和共享 fixtures"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """创建临时目录用于测试"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def memory_dir(temp_dir):
    """创建记忆目录结构"""
    memory_dir = temp_dir / "memory"
    memory_dir.mkdir()
    (temp_dir / "MEMORY.md").write_text("# Long-term memory\n")
    return temp_dir
