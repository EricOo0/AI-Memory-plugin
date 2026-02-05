# AI-Memory Plugin Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个基于文件系统的可插拔 AI 记忆插件，支持混合检索、多维度评分，作为 Python 库供各类 Agent 框架使用。

**Architecture:** 分层架构设计，包括存储层（文件系统 + SQLite）、检索层（混合检索器）、嵌入提供者（插件化）、Agent 接口层（System Prompt + Tool 接口 + 框架适配器）。

**Tech Stack:** Python 3.10+, SQLite (FTS5), sentence-transformers (本地嵌入), FastAPI (可选 HTTP 接口), Pydantic (数据验证), pytest (测试)

---

## 项目结构

```
ai_memory/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py          # 配置管理
├── core/
│   ├── __init__.py
│   ├── manager.py           # MemoryManager 统一入口
│   ├── types.py             # 类型定义
│   └── exceptions.py        # 自定义异常
├── storage/
│   ├── __init__.py
│   ├── file_manager.py      # 文件管理器
│   └── database.py          # SQLite 封装
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_searcher.py   # 混合检索器
│   └── scorer.py            # 多维度评分器
├── embeddings/
│   ├── __init__.py
│   ├── base.py              # 嵌入提供者抽象
│   ├── local.py             # 本地模型实现
│   └── openai.py            # OpenAI 实现
├── sync/
│   ├── __init__.py
│   ├── watcher.py           # 文件监控器
│   └── indexer.py           # 增量索引
└── tools/
    ├── memory_tools.py      # Tool 函数定义
    └── system_prompt.py     # System Prompt 模板

adapters/
├── __init__.py
├── langchain.py             # LangChain 适配器
└── base.py                  # 基础适配器接口

tests/
├── __init__.py
├── conftest.py              # pytest 配置
├── test_manager.py
├── test_storage/
├── test_retrieval/
├── test_embeddings/
└── test_sync/

docs/
├── MEMORY_INSTRUCTIONS.md  # Agent 使用指南
├── examples/
└── API.md                   # API 文档
```

---

## Task 1: 项目初始化

**Files:**
- Create: `pyproject.toml`
- Create: `src/ai_memory/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: 创建 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-memory"
version = "0.1.0"
description = "可插拔的 AI 记忆系统插件"
authors = [
    {name = "Your Name", email = "your@email.com"}
]
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "sentence-transformers>=2.2.0",
    "numpy>=1.24.0",
    "watchfiles>=0.20.0",
]
optional-dependencies = [
    "openai>=1.0.0",
    "langchain>=0.0.200",
]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 2: 创建 src/ai_memory/__init__.py**

```python
"""AI Memory Plugin - 可插拔的 AI 记忆系统

提供基于文件系统的记忆存储和混合检索能力，支持：
- 时间日记式记忆组织 (MEMORY.md + memory/DD-MM-YYYY.md)
- 混合检索（向量 + 关键词）
- 多维度评分（相似度、时间、频率）
- 可插拔的嵌入模型
"""

__version__ = "0.1.0"

from ai_memory.core.manager import MemoryManager
from ai_memory.config.settings import MemoryConfig

__all__ = ["MemoryManager", "MemoryConfig"]
```

**Step 3: 创建 tests/conftest.py**

```python
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
```

**Step 4: 运行基础测试确保项目结构正确**

运行: `pytest tests/conftest.py -v`
Expected: PASS (收集测试，无测试运行)

**Step 5: 提交**

```bash
git add pyproject.toml src/ai_memory/__init__.py tests/conftest.py
git commit -m "feat: 初始化项目结构"
```

---

## Task 2: 配置管理

**Files:**
- Create: `src/ai_memory/config/__init__.py`
- Create: `src/ai_memory/config/settings.py`
- Test: `tests/test_config.py`

**Step 1: 编写配置测试**

```python
# tests/test_config.py
import pytest
from ai_memory.config.settings import MemoryConfig

def test_default_config():
    config = MemoryConfig()
    assert config.storage.dir.name == "memory"
    assert config.retrieval.max_results == 6
    assert config.retrieval.min_score == 0.35

def test_config_from_dict():
    data = {
        "storage": {"dir": "/custom/path"},
        "retrieval": {"max_results": 10}
    }
    config = MemoryConfig(**data)
    assert str(config.storage.dir) == "/custom/path"
    assert config.retrieval.max_results == 10

def test_config_from_yaml(temp_dir):
    yaml_file = temp_dir / "config.yaml"
    yaml_file.write_text("""
storage:
  dir: ./memory
retrieval:
  max_results: 6
  min_score: 0.35
""")
    config = MemoryConfig.from_yaml(yaml_file)
    assert config.retrieval.max_results == 6
```

**Step 2: 运行测试确认失败**

运行: `pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError" 或类似错误

**Step 3: 实现配置类**

```python
# src/ai_memory/config/__init__.py
from ai_memory.config.settings import MemoryConfig, EmbeddingConfig, RetrievalConfig

__all__ = ["MemoryConfig", "EmbeddingConfig", "RetrievalConfig"]
```

```python
# src/ai_memory/config/settings.py
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
import yaml


class StorageConfig(BaseModel):
    """存储配置"""
    dir: Path = Field(default=Path("memory"))
    db_name: str = Field(default="memory.db")


class EmbeddingConfig(BaseModel):
    """嵌入配置"""
    provider: str = Field(default="local")
    model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    dimensions: Optional[int] = None


class RetrievalConfig(BaseModel):
    """检索配置"""
    max_results: int = Field(default=6, ge=1, le=20)
    min_score: float = Field(default=0.35, ge=0, le=1)
    hybrid: bool = Field(default=True)
    vector_weight: float = Field(default=0.7, ge=0, le=1)
    text_weight: float = Field(default=0.3, ge=0, le=1)
    time_weight: float = Field(default=0.1, ge=0, le=1)
    frequency_weight: float = Field(default=0.1, ge=0, le=1)


class SyncConfig(BaseModel):
    """同步配置"""
    watch: bool = Field(default=True)
    debounce_ms: int = Field(default=1000, ge=100)


class MemoryConfig(BaseModel):
    """记忆系统总配置"""
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "MemoryConfig":
        """从 YAML 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """保存配置到 YAML 文件"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(mode="json"), f, allow_unicode=True)
```

**Step 4: 运行测试确认通过**

运行: `pytest tests/test_config.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/ai_memory/config/ tests/test_config.py
git commit -m "feat: 实现配置管理系统"
```

---

## Task 3: 类型定义

**Files:**
- Create: `src/ai_memory/core/__init__.py`
- Create: `src/ai_memory/core/types.py`
- Create: `src/ai_memory/core/exceptions.py`
- Test: `tests/test_types.py`

**Step 1: 编写类型测试**

```python
# tests/test_types.py
from ai_memory.core.types import MemorySearchResult, MemorySource, MemoryEntry

def test_memory_search_result():
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

def test_memory_entry():
    entry = MemoryEntry(
        path="memory/01-01-2025.md",
        content="# Daily log\nContent here",
        hash="abc123"
    )
    assert entry.path == "memory/01-01-2025.md"
```

**Step 2: 运行测试确认失败**

运行: `pytest tests/test_types.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: 实现类型定义**

```python
# src/ai_memory/core/__init__.py
from ai_memory.core.types import (
    MemorySearchResult,
    MemorySource,
    MemoryEntry,
    MemoryStatus
)
from ai_memory.core.exceptions import (
    MemoryError,
    EmbeddingError,
    RetrievalError
)

__all__ = [
    "MemorySearchResult",
    "MemorySource",
    "MemoryEntry",
    "MemoryStatus",
    "MemoryError",
    "EmbeddingError",
    "RetrievalError"
]
```

```python
# src/ai_memory/core/types.py
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class MemorySource(str, Enum):
    """记忆来源类型"""
    MEMORY = "memory"      # 长期记忆文件
    DAILY = "daily"        # 日期日记
    SESSION = "session"    # 会话记录


class MemorySearchResult(BaseModel):
    """记忆检索结果"""
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    source: MemorySource
    citation: Optional[str] = None


class MemoryEntry(BaseModel):
    """记忆条目"""
    path: str
    content: str
    hash: str
    size: int
    modified_at: datetime
    source: MemorySource


class MemoryStatus(BaseModel):
    """记忆系统状态"""
    backend: str = "sqlite"
    files: int = 0
    chunks: int = 0
    last_sync: Optional[datetime] = None
    embedding_model: Optional[str] = None
```

```python
# src/ai_memory/core/exceptions.py


class MemoryError(Exception):
    """记忆系统基础异常"""
    pass


class EmbeddingError(MemoryError):
    """嵌入生成异常"""
    pass


class RetrievalError(MemoryError):
    """检索异常"""
    pass


class SyncError(MemoryError):
    """同步异常"""
    pass
```

**Step 4: 运行测试确认通过**

运行: `pytest tests/test_types.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/ai_memory/core/ tests/test_types.py
git commit -m "feat: 定义核心类型和异常"
```

---

## Task 4: SQLite 数据库层

**Files:**
- Create: `src/ai_memory/storage/__init__.py`
- Create: `src/ai_memory/storage/database.py`
- Test: `tests/test_storage/test_database.py`

**Step 1: 编写数据库测试**

```python
# tests/test_storage/test_database.py
import pytest
from ai_memory.storage.database import Database
from ai_memory.core.types import MemoryEntry, MemorySource
from datetime import datetime

def test_database_init(memory_dir):
    db = Database(memory_dir / "memory.db")
    assert db is not None

def test_create_tables(memory_dir):
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    # 表创建成功，无异常

def test_insert_file(memory_dir):
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
    assert retrieved.path == "MEMORY.md"

def test_insert_chunk(memory_dir):
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
```

**Step 2: 运行测试确认失败**

运行: `pytest tests/test_storage/test_database.py -v`
Expected: FAIL

**Step 3: 实现数据库类**

```python
# src/ai_memory/storage/__init__.py
from ai_memory.storage.database import Database
from ai_memory.storage.file_manager import FileManager

__all__ = ["Database", "FileManager"]
```

```python
# src/ai_memory/storage/database.py
import sqlite3
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from ai_memory.core.types import MemoryEntry, MemorySource


class Database:
    """SQLite 数据库封装"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        """创建数据库连接"""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
            # 启用 FTS5
            self.conn.execute("PRAGMA journal_mode=WAL")
        return self.conn

    def close(self) -> None:
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_tables(self) -> None:
        """创建数据表"""
        conn = self.connect()
        cursor = conn.cursor()

        # 文件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                hash TEXT NOT NULL,
                size INTEGER NOT NULL,
                modified_at INTEGER NOT NULL
            )
        """)

        # 文本块表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                model TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
        """)

        # FTS5 全文搜索表
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks
            USING fts5(text, id UNINDEXED, path UNINDEXED, start_line UNINDEXED, end_line UNINDEXED)
        """)

        conn.commit()

    def insert_file(self, entry: MemoryEntry) -> None:
        """插入文件记录"""
        conn = self.connect()
        conn.execute("""
            INSERT OR REPLACE INTO files (path, source, hash, size, modified_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            entry.path,
            entry.source.value,
            entry.hash,
            entry.size,
            int(entry.modified_at.timestamp())
        ))
        conn.commit()

    def get_file(self, path: str) -> Optional[Dict]:
        """获取文件记录"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM files WHERE path = ?", (path,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete_file(self, path: str) -> None:
        """删除文件记录及相关块"""
        conn = self.connect()
        conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        conn.execute("DELETE FROM fts_chunks WHERE path = ?", (path,))
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
        conn.commit()

    def insert_chunk(
        self,
        id: str,
        path: str,
        start_line: int,
        end_line: int,
        text: str,
        embedding: List[float],
        model: str
    ) -> None:
        """插入文本块"""
        conn = self.connect()
        now = int(datetime.now().timestamp())

        conn.execute("""
            INSERT OR REPLACE INTO chunks
            (id, path, start_line, end_line, text, embedding, model, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (id, path, start_line, end_line, text, json.dumps(embedding), model, now))

        # 同步到 FTS
        conn.execute("""
            INSERT OR REPLACE INTO fts_chunks (rowid, text, id, path, start_line, end_line)
            SELECT rowid, text, id, path, start_line, end_line FROM chunks
            WHERE id = ?
        """, (id,))

        conn.commit()

    def search_by_vector(
        self,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[Dict]:
        """通过向量搜索（余弦相似度）"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("SELECT id, path, start_line, end_line, text, embedding FROM chunks")
        rows = cursor.fetchall()

        results = []
        for row in rows:
            embedding = json.loads(row["embedding"])
            score = self._cosine_similarity(query_embedding, embedding)
            if score > 0:
                results.append({
                    "id": row["id"],
                    "path": row["path"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "text": row["text"],
                    "score": score
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def search_by_text(self, query: str, limit: int = 10) -> List[Dict]:
        """通过全文搜索"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, path, start_line, end_line, text, rank
            FROM fts_chunks
            WHERE fts_chunks MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))

        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "path": row["path"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "text": row["text"],
                "score": 1.0 / (1.0 + row["rank"])  # BM25 转分数
            }
            for row in rows
        ]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
```

**Step 4: 运行测试确认通过**

运行: `pytest tests/test_storage/test_database.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/ai_memory/storage/ tests/test_storage/
git commit -m "feat: 实现 SQLite 数据库层"
```

---

## Task 5: 文件管理器

**Files:**
- Create: `src/ai_memory/storage/file_manager.py`
- Test: `tests/test_storage/test_file_manager.py`

**Step 1: 编写文件管理器测试**

```python
# tests/test_storage/test_file_manager.py
from ai_memory.storage.file_manager import FileManager
from pathlib import Path

def test_get_memory_files(memory_dir):
    fm = FileManager(memory_dir)
    files = fm.get_memory_files()
    assert len(files) >= 1  # MEMORY.md 至少存在

def test_read_file(memory_dir):
    fm = FileManager(memory_dir)
    content = fm.read_file("MEMORY.md")
    assert "# Long-term memory" in content

def test_get_file_hash(memory_dir):
    fm = FileManager(memory_dir)
    hash1 = fm.get_file_hash("MEMORY.md")
    hash2 = fm.get_file_hash("MEMORY.md")
    assert hash1 == hash2

def test_add_memory(memory_dir):
    fm = FileManager(memory_dir)
    fm.add_memory("# New entry\nNew content", tags=["test"])
    # 验证文件被创建或更新
```

**Step 2: 运行测试确认失败**

运行: `pytest tests/test_storage/test_file_manager.py -v`
Expected: FAIL

**Step 3: 实现文件管理器**

```python
# src/ai_memory/storage/file_manager.py
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from ai_memory.core.types import MemoryEntry, MemorySource


class FileManager:
    """文件系统管理器"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.memory_dir = base_dir / "memory"
        self.memory_dir.mkdir(exist_ok=True)

    def get_memory_files(self) -> List[Path]:
        """获取所有记忆文件"""
        files = []

        # 主记忆文件
        main_file = self.base_dir / "MEMORY.md"
        if main_file.exists():
            files.append(main_file)

        # 日期日记目录
        if self.memory_dir.exists():
            files.extend(self.memory_dir.glob("*.md"))

        return files

    def read_file(self, relative_path: str) -> str:
        """读取文件内容"""
        full_path = self._resolve_path(relative_path)
        return full_path.read_text(encoding="utf-8")

    def write_file(self, relative_path: str, content: str) -> None:
        """写入文件内容"""
        full_path = self._resolve_path(relative_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def get_file_hash(self, relative_path: str) -> str:
        """计算文件哈希"""
        content = self.read_file(relative_path)
        return hashlib.sha256(content.encode()).hexdigest()

    def get_file_entry(self, relative_path: str) -> MemoryEntry:
        """获取文件元数据"""
        full_path = self._resolve_path(relative_path)
        stat = full_path.stat()

        return MemoryEntry(
            path=relative_path,
            content=self.read_file(relative_path),
            hash=self.get_file_hash(relative_path),
            size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            source=MemorySource.MEMORY if relative_path == "MEMORY.md" else MemorySource.DAILY
        )

    def add_memory(
        self,
        content: str,
        tags: List[str] = None,
        target_date: datetime = None
    ) -> str:
        """添加新记忆"""
        if target_date is None:
            target_date = datetime.now()

        date_str = target_date.strftime("%d-%m-%Y")
        relative_path = f"memory/{date_str}.md"

        current_content = ""
        if self._resolve_path(relative_path).exists():
            current_content = self.read_file(relative_path)
            current_content += "\n\n"

        # 添加标签
        tag_line = ""
        if tags:
            tag_line = f"\n**Tags:** {' '.join(f'#{t}' for t in tags)}\n"

        new_content = current_content + content + tag_line
        self.write_file(relative_path, new_content)

        return relative_path

    def _resolve_path(self, relative_path: str) -> Path:
        """解析相对路径"""
        if relative_path.startswith("memory/"):
            return self.base_dir / relative_path
        return self.base_dir / relative_path
```

**Step 4: 运行测试确认通过**

运行: `pytest tests/test_storage/test_file_manager.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/ai_memory/storage/file_manager.py tests/test_storage/test_file_manager.py
git commit -m "feat: 实现文件管理器"
```

---

## Task 6: 嵌入提供者抽象与实现

**Files:**
- Create: `src/ai_memory/embeddings/__init__.py`
- Create: `src/ai_memory/embeddings/base.py`
- Create: `src/ai_memory/embeddings/local.py`
- Test: `tests/test_embeddings/test_base.py`
- Test: `tests/test_embeddings/test_local.py`

**Step 1: 编写嵌入提供者测试**

```python
# tests/test_embeddings/test_base.py
from abc import ABC, abstractmethod
import pytest

def test_abstract_interface():
    # 测试接口定义
    pass

# tests/test_embeddings/test_local.py
from ai_memory.embeddings.local import LocalEmbeddingProvider

def test_local_provider_init():
    provider = LocalEmbeddingProvider()
    assert provider.model == "sentence-transformers/all-MiniLM-L6-v2"

def test_local_provider_embed():
    provider = LocalEmbeddingProvider()
    embedding = provider.embed("Hello world")
    assert len(embedding) == 384  # MiniLM-L6-v2 维度
    assert all(isinstance(x, float) for x in embedding)

def test_local_provider_embed_batch():
    provider = LocalEmbeddingProvider()
    texts = ["Hello", "World"]
    embeddings = provider.embed_batch(texts)
    assert len(embeddings) == 2
    assert all(len(e) == 384 for e in embeddings)
```

**Step 2: 运行测试确认失败**

运行: `pytest tests/test_embeddings/ -v`
Expected: FAIL

**Step 3: 实现嵌入提供者**

```python
# src/ai_memory/embeddings/__init__.py
from ai_memory.embeddings.base import EmbeddingProvider
from ai_memory.embeddings.local import LocalEmbeddingProvider

__all__ = ["EmbeddingProvider", "LocalEmbeddingProvider"]
```

```python
# src/ai_memory/embeddings/base.py
from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """嵌入提供者抽象接口"""

    def __init__(self, model: str = None):
        self.model = model

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """生成单段文本的嵌入向量"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        pass

    @abstractmethod
    def dimensions(self) -> int:
        """获取嵌入向量维度"""
        pass
```

```python
# src/ai_memory/embeddings/local.py
from typing import List
from sentence_transformers import SentenceTransformer

from ai_memory.embeddings.base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """本地嵌入提供者（使用 sentence-transformers）"""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model: str = None):
        super().__init__(model or self.DEFAULT_MODEL)
        self._model = None

    @property
    def model_instance(self) -> SentenceTransformer:
        """延迟加载模型"""
        if self._model is None:
            self._model = SentenceTransformer(self.model)
        return self._model

    def embed(self, text: str) -> List[float]:
        """生成单段文本的嵌入向量"""
        return self.model_instance.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        embeddings = self.model_instance.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def dimensions(self) -> int:
        """获取嵌入向量维度"""
        return self.model_instance.get_sentence_embedding_dimension()
```

**Step 4: 运行测试确认通过**

运行: `pytest tests/test_embeddings/ -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/ai_memory/embeddings/ tests/test_embeddings/
git commit -m "feat: 实现嵌入提供者抽象和本地实现"
```

---

## Task 7: 混合检索器

**Files:**
- Create: `src/ai_memory/retrieval/__init__.py`
- Create: `src/ai_memory/retrieval/hybrid_searcher.py`
- Create: `src/ai_memory/retrieval/scorer.py`
- Test: `tests/test_retrieval/test_hybrid_searcher.py`

**Step 1: 编写检索器测试**

```python
# tests/test_retrieval/test_hybrid_searcher.py
from ai_memory.retrieval.hybrid_searcher import HybridSearcher
from ai_memory.storage.database import Database
from pathlib import Path

def test_hybrid_search_init(memory_dir):
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    searcher = HybridSearcher(db, vector_weight=0.7, text_weight=0.3)
    assert searcher is not None

def test_hybrid_search(memory_dir):
    db = Database(memory_dir / "memory.db")
    db.create_tables()

    # 插入测试数据
    db.insert_chunk(
        id="chunk1",
        path="MEMORY.md",
        start_line=1,
        end_line=3,
        text="This is a test about memory",
        embedding=[0.1] * 384,
        model="test"
    )

    db.insert_chunk(
        id="chunk2",
        path="MEMORY.md",
        start_line=4,
        end_line=6,
        text="Another chunk of information",
        embedding=[0.2] * 384,
        model="test"
    )

    searcher = HybridSearcher(db)
    results = searcher.search(
        query="test memory",
        query_embedding=[0.1] * 384,
        max_results=2
    )

    assert len(results) <= 2
    assert all("score" in r for r in results)
```

**Step 2: 运行测试确认失败**

运行: `pytest tests/test_retrieval/test_hybrid_searcher.py -v`
Expected: FAIL

**Step 3: 实现混合检索器**

```python
# src/ai_memory/retrieval/__init__.py
from ai_memory.retrieval.hybrid_searcher import HybridSearcher
from ai_memory.retrieval.scorer import MultiDimensionScorer

__all__ = ["HybridSearcher", "MultiDimensionScorer"]
```

```python
# src/ai_memory/retrieval/scorer.py
from datetime import datetime, timedelta
from typing import Dict, List
from ai_memory.core.types import MemorySearchResult, MemorySource


class MultiDimensionScorer:
    """多维度评分器"""

    def __init__(
        self,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        time_weight: float = 0.1,
        frequency_weight: float = 0.1
    ):
        self.vector_weight = vector_weight
        self.text_weight = text_weight
        self.time_weight = time_weight
        self.frequency_weight = frequency_weight
        self.access_counts: Dict[str, int] = {}

    def score(
        self,
        vector_score: float,
        text_score: float,
        created_at: datetime = None,
        path: str = None
    ) -> float:
        """计算综合分数"""
        # 基础分数（向量 + 文本）
        base_score = (
            self.vector_weight * vector_score +
            self.text_weight * text_score
        )

        # 时间权重（越近越高）
        time_score = self._time_score(created_at) if created_at else 1.0

        # 频率权重（访问越多越高）
        freq_score = self._frequency_score(path) if path else 1.0

        # 综合评分
        total = base_score * time_score * freq_score
        return min(max(total, 0.0), 1.0)

    def record_access(self, path: str) -> None:
        """记录访问"""
        self.access_counts[path] = self.access_counts.get(path, 0) + 1

    def _time_score(self, created_at: datetime) -> float:
        """时间分数"""
        age = (datetime.now() - created_at).days
        # 7天内有额外加分
        if age < 7:
            return 1.0 + (7 - age) * 0.05 * self.time_weight
        return 1.0

    def _frequency_score(self, path: str) -> float:
        """频率分数"""
        count = self.access_counts.get(path, 0)
        return 1.0 + min(count * 0.1, 0.5) * self.frequency_weight
```

```python
# src/ai_memory/retrieval/hybrid_searcher.py
from typing import List, Optional
from datetime import datetime

from ai_memory.storage.database import Database
from ai_memory.retrieval.scorer import MultiDimensionScorer
from ai_memory.core.types import MemorySearchResult


class HybridSearcher:
    """混合检索器"""

    def __init__(
        self,
        database: Database,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        time_weight: float = 0.1,
        frequency_weight: float = 0.1
    ):
        self.db = database
        self.scorer = MultiDimensionScorer(
            vector_weight=vector_weight,
            text_weight=text_weight,
            time_weight=time_weight,
            frequency_weight=frequency_weight
        )

    def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        max_results: int = 6,
        min_score: float = 0.35
    ) -> List[MemorySearchResult]:
        """混合检索"""
        results = []

        # 文本搜索
        text_results = self.db.search_by_text(query, limit=max_results * 2)

        # 向量搜索
        vector_results = []
        if query_embedding:
            vector_results = self.db.search_by_vector(
                query_embedding,
                limit=max_results * 2
            )

        # 合并结果
        merged = self._merge_results(text_results, vector_results)

        # 评分和过滤
        for item in merged:
            score = self.scorer.score(
                vector_score=item.get("vector_score", 0),
                text_score=item.get("text_score", 0)
            )

            if score >= min_score:
                self.scorer.record_access(item["path"])

                results.append(MemorySearchResult(
                    path=item["path"],
                    start_line=item["start_line"],
                    end_line=item["end_line"],
                    score=score,
                    snippet=item["text"],
                    source=MemorySource.MEMORY,
                    citation=f"{item['path']}#L{item['start_line']}-L{item['end_line']}"
                ))

        # 排序并限制结果
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

    def _merge_results(
        self,
        text_results: List[dict],
        vector_results: List[dict]
    ) -> List[dict]:
        """合并文本和向量结果"""
        merged = {}
        all_results = list(zip(text_results, ["text"] * len(text_results))) + \
                     list(zip(vector_results, ["vector"] * len(vector_results)))

        for result, source in all_results:
            key = f"{result['path']}#{result['start_line']}-{result['end_line']}"
            if key not in merged:
                merged[key] = {
                    "path": result["path"],
                    "start_line": result["start_line"],
                    "end_line": result["end_line"],
                    "text": result["text"],
                    "text_score": 0,
                    "vector_score": 0
                }
            merged[key][f"{source}_score"] = result["score"]

        return list(merged.values())
```

**Step 4: 运行测试确认通过**

运行: `pytest tests/test_retrieval/test_hybrid_searcher.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/ai_memory/retrieval/ tests/test_retrieval/
git commit -m "feat: 实现混合检索器和多维度评分"
```

---

## Task 8: 文件监控与索引同步

**Files:**
- Create: `src/ai_memory/sync/__init__.py`
- Create: `src/ai_memory/sync/watcher.py`
- Create: `src/ai_memory/sync/indexer.py`
- Test: `tests/test_sync/test_indexer.py`

**Step 1: 编写索引同步测试**

```python
# tests/test_sync/test_indexer.py
from ai_memory.sync.indexer import MemoryIndexer
from ai_memory.storage.database import Database
from ai_memory.storage.file_manager import FileManager
from ai_memory.embeddings.local import LocalEmbeddingProvider
from pathlib import Path

def test_indexer_init(memory_dir):
    db = Database(memory_dir / "memory.db")
    fm = FileManager(memory_dir)
    provider = LocalEmbeddingProvider()
    indexer = MemoryIndexer(db, fm, provider)
    assert indexer is not None

def test_indexer_sync(memory_dir):
    db = Database(memory_dir / "memory.db")
    fm = FileManager(memory_dir)
    provider = LocalEmbeddingProvider()

    # 创建测试文件
    fm.write_file("memory/test.md", "# Test\nContent here")

    indexer = MemoryIndexer(db, fm, provider)
    indexer.sync()

    # 验证文件被索引
    file_entry = db.get_file("memory/test.md")
    assert file_entry is not None

    # 验证块被创建
    chunks = db.search_by_vector([0.1] * 384, limit=100)
    assert len(chunks) > 0
```

**Step 2: 运行测试确认失败**

运行: `pytest tests/test_sync/test_indexer.py -v`
Expected: FAIL

**Step 3: 实现索引同步**

```python
# src/ai_memory/sync/__init__.py
from ai_memory.sync.indexer import MemoryIndexer

__all__ = ["MemoryIndexer"]
```

```python
# src/ai_memory/sync/indexer.py
import logging
from typing import List, Tuple
from pathlib import Path

from ai_memory.storage.database import Database
from ai_memory.storage.file_manager import FileManager
from ai_memory.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class MemoryIndexer:
    """记忆索引器"""

    def __init__(
        self,
        database: Database,
        file_manager: FileManager,
        embedding_provider: EmbeddingProvider,
        chunk_size: int = 400,
        chunk_overlap: int = 80
    ):
        self.db = database
        self.fm = file_manager
        self.provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def sync(self) -> None:
        """同步文件到数据库"""
        files = self.fm.get_memory_files()
        for file_path in files:
            self._sync_file(file_path)

    def _sync_file(self, file_path: Path) -> None:
        """同步单个文件"""
        relative_path = file_path.relative_to(self.fm.base_dir).as_posix()

        # 获取当前文件状态
        current_entry = self.fm.get_file_entry(relative_path)
        existing = self.db.get_file(relative_path)

        # 检查是否需要更新
        if existing and existing["hash"] == current_entry.hash:
            return

        # 删除旧的块
        if existing:
            self.db.delete_file(relative_path)

        # 插入新文件记录
        self.db.insert_file(current_entry)

        # 分块并索引
        chunks = self._chunk_text(current_entry.content)
        self._index_chunks(relative_path, chunks)

        logger.info(f"Indexed: {relative_path}")

    def _chunk_text(self, text: str) -> List[Tuple[int, int, str]]:
        """将文本分块"""
        lines = text.split("\n")
        chunks = []
        start_line = 0

        while start_line < len(lines):
            # 计算结束行
            end_line = min(start_line + self.chunk_size, len(lines))
            chunk_text = "\n".join(lines[start_line:end_line])

            chunks.append((start_line + 1, end_line, chunk_text))

            # 下一块的起始位置（有重叠）
            start_line = max(end_line - self.chunk_overlap, start_line + 1)

            if start_line >= len(lines):
                break

        return chunks

    def _index_chunks(
        self,
        path: str,
        chunks: List[Tuple[int, int, str]]
    ) -> None:
        """索引文本块"""
        texts = [chunk[2] for chunk in chunks]
        embeddings = self.provider.embed_batch(texts)

        for (start, end, text), embedding in zip(chunks, embeddings):
            chunk_id = f"{path}:{start}-{end}"
            self.db.insert_chunk(
                id=chunk_id,
                path=path,
                start_line=start,
                end_line=end,
                text=text,
                embedding=embedding,
                model=self.provider.model
            )
```

**Step 4: 运行测试确认通过**

运行: `pytest tests/test_sync/test_indexer.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/ai_memory/sync/ tests/test_sync/
git commit -m "feat: 实现文件索引同步"
```

---

## Task 9: MemoryManager 核心入口

**Files:**
- Modify: `src/ai_memory/core/manager.py`
- Test: `tests/test_manager.py`

**Step 1: 编写 MemoryManager 测试**

```python
# tests/test_manager.py
from ai_memory import MemoryManager, MemoryConfig
from pathlib import Path

def test_manager_init(memory_dir):
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)
    assert manager is not None

def test_manager_add_memory(memory_dir):
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)
    manager.add_memory("# Test\nNew content", tags=["test"])

    # 验证文件存在
    assert (memory_dir / "memory").exists()

def test_manager_search(memory_dir):
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)

    # 先添加一些记忆
    manager.add_memory("# Project\nAI memory plugin development")
    manager.sync()

    # 搜索
    results = manager.search("AI memory")
    assert len(results) >= 0

def test_manager_get_memory(memory_dir):
    config = MemoryConfig(storage={"dir": str(memory_dir)})
    manager = MemoryManager(config)
    manager.add_memory("# Test\nContent")

    results = manager.search("Test")
    if results:
        content = manager.get_memory(results[0].path)
        assert "Test" in content
```

**Step 2: 运行测试确认失败**

运行: `pytest tests/test_manager.py -v`
Expected: FAIL

**Step 3: 实现 MemoryManager**

```python
# src/ai_memory/core/manager.py
import logging
from typing import List, Optional
from pathlib import Path

from ai_memory.config.settings import MemoryConfig
from ai_memory.storage.database import Database
from ai_memory.storage.file_manager import FileManager
from ai_memory.embeddings.local import LocalEmbeddingProvider
from ai_memory.retrieval.hybrid_searcher import HybridSearcher
from ai_memory.sync.indexer import MemoryIndexer
from ai_memory.core.types import MemorySearchResult, MemoryStatus

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器 - 统一入口"""

    def __init__(self, config: MemoryConfig):
        self.config = config

        # 初始化各组件
        self.fm = FileManager(Path(config.storage.dir))
        self.db = Database(Path(config.storage.dir) / config.storage.db_name)
        self.db.create_tables()

        # 嵌入提供者
        if config.embedding.provider == "local":
            self.provider = LocalEmbeddingProvider(config.embedding.model)
        else:
            self.provider = LocalEmbeddingProvider(config.embedding.model)

        # 索引器
        self.indexer = MemoryIndexer(self.db, self.fm, self.provider)

        # 检索器
        self.searcher = HybridSearcher(
            self.db,
            vector_weight=config.retrieval.vector_weight,
            text_weight=config.retrieval.text_weight,
            time_weight=config.retrieval.time_weight,
            frequency_weight=config.retrieval.frequency_weight
        )

    def add_memory(
        self,
        content: str,
        tags: Optional[List[str]] = None
    ) -> str:
        """添加新记忆"""
        path = self.fm.add_memory(content, tags)
        self._sync_file(path)
        return path

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[MemorySearchResult]:
        """搜索记忆"""
        # 生成查询嵌入
        query_embedding = self.provider.embed(query)

        return self.searcher.search(
            query=query,
            query_embedding=query_embedding,
            max_results=max_results or self.config.retrieval.max_results,
            min_score=min_score or self.config.retrieval.min_score
        )

    def get_memory(
        self,
        path: str,
        from_line: Optional[int] = None,
        lines: int = 20
    ) -> str:
        """获取特定记忆内容"""
        content = self.fm.read_file(path)
        if from_line:
            line_list = content.split("\n")
            start = max(0, from_line - 1)
            end = min(len(line_list), start + lines)
            return "\n".join(line_list[start:end])
        return content

    def sync(self) -> None:
        """同步所有文件"""
        self.indexer.sync()

    def status(self) -> MemoryStatus:
        """获取系统状态"""
        return MemoryStatus(
            backend="sqlite",
            files=len(self.fm.get_memory_files()),
            chunks=0,  # TODO: 从数据库获取
            embedding_model=self.provider.model
        )

    def _sync_file(self, relative_path: str) -> None:
        """同步单个文件"""
        full_path = self.fm._resolve_path(relative_path)
        self.indexer._sync_file(full_path)
```

**Step 4: 更新 src/ai_memory/core/__init__.py**

```python
# src/ai_memory/core/__init__.py
from ai_memory.core.types import (
    MemorySearchResult,
    MemorySource,
    MemoryEntry,
    MemoryStatus
)
from ai_memory.core.exceptions import (
    MemoryError,
    EmbeddingError,
    RetrievalError
)
from ai_memory.core.manager import MemoryManager

__all__ = [
    "MemorySearchResult",
    "MemorySource",
    "MemoryEntry",
    "MemoryStatus",
    "MemoryError",
    "EmbeddingError",
    "RetrievalError",
    "MemoryManager",
]
```

**Step 5: 运行测试确认通过**

运行: `pytest tests/test_manager.py -v`
Expected: PASS

**Step 6: 提交**

```bash
git add src/ai_memory/core/manager.py tests/test_manager.py
git commit -m "feat: 实现 MemoryManager 核心入口"
```

---

## Task 10: Agent 工具接口

**Files:**
- Create: `src/ai_memory/tools/__init__.py`
- Create: `src/ai_memory/tools/memory_tools.py`
- Create: `src/ai_memory/tools/system_prompt.py`
- Create: `docs/MEMORY_INSTRUCTIONS.md`

**Step 1: 创建工具函数定义**

```python
# src/ai_memory/tools/__init__.py
from ai_memory.tools.memory_tools import (
    memory_search,
    memory_add,
    memory_get,
    get_memory_tools,
    get_openai_functions
)

__all__ = [
    "memory_search",
    "memory_add",
    "memory_get",
    "get_memory_tools",
    "get_openai_functions"
]
```

```python
# src/ai_memory/tools/memory_tools.py
from typing import Optional, List, Dict, Any
from ai_memory.core.manager import MemoryManager


class MemoryTools:
    """记忆工具类"""

    def __init__(self, manager: MemoryManager):
        self.manager = manager

    def search(
        self,
        query: str,
        max_results: int = 6,
        min_score: float = 0.35
    ) -> Dict[str, Any]:
        """搜索记忆"""
        results = self.manager.search(query, max_results, min_score)
        return {
            "query": query,
            "results": [
                {
                    "path": r.path,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "score": r.score,
                    "snippet": r.snippet,
                    "citation": r.citation
                }
                for r in results
            ],
            "count": len(results)
        }

    def add(
        self,
        content: str,
        tags: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """添加记忆"""
        path = self.manager.add_memory(content, tags)
        return {
            "status": "success",
            "path": path
        }

    def get(
        self,
        path: str,
        from_line: Optional[int] = None,
        lines: int = 20
    ) -> Dict[str, Any]:
        """获取记忆"""
        content = self.manager.get_memory(path, from_line, lines)
        return {
            "path": path,
            "content": content
        }


def get_memory_tools(manager: MemoryManager) -> Dict[str, callable]:
    """获取工具函数字典"""
    tools = MemoryTools(manager)
    return {
        "memory_search": tools.search,
        "memory_add": tools.add,
        "memory_get": tools.get
    }


def get_openai_functions() -> List[Dict[str, Any]]:
    """获取 OpenAI Function Calling 格式的工具定义"""
    return [
        {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "Search memories for relevant information about past work, decisions, user preferences, or project history",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for semantic search"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 6)",
                            "default": 6
                        },
                        "min_score": {
                            "type": "number",
                            "description": "Minimum relevance score (0-1, default: 0.35)",
                            "default": 0.35
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "memory_add",
                "description": "Add a new memory entry. Use for important decisions, user preferences, or information worth remembering",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to remember"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization"
                        }
                    },
                    "required": ["content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "memory_get",
                "description": "Retrieve specific lines from a memory file by path",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to memory directory (e.g., 'MEMORY.md' or 'memory/01-02-2025.md')"
                        },
                        "from_line": {
                            "type": "integer",
                            "description": "Starting line number (1-indexed)"
                        },
                        "lines": {
                            "type": "integer",
                            "description": "Number of lines to retrieve (default: 20)",
                            "default": 20
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    ]
```

```python
# src/ai_memory/tools/system_prompt.py
"""System Prompt 模板"""

MEMORY_SYSTEM_PROMPT = """
## Memory System Instructions

You have access to a memory system that stores and retrieves information across conversations.

### When to Search Memory

Before answering any question about:
- Past work, decisions, or actions
- User preferences, goals, or context
- Project history, timelines, or dates
- Previously discussed topics or concepts

ALWAYS run a memory search first.

### How to Use Memory

1. **Search**: Use `memory_search(query)` to find relevant memories
2. **Retrieve**: Use `memory_get(path, from, lines)` to read specific sections
3. **Record**: Use `memory_add(content, tags)` to save important information

### Memory Format

- Long-term memories are in `MEMORY.md`
- Daily memories are in `memory/DD-MM-YYYY.md`
- Citations follow format: `path#Lstart-Lend`

### Best Practices

- Be specific in search queries
- Use tags when adding memories
- Verify source citations before relying on content
- Say "I checked my memory" when search yields low-confidence results
"""


def get_system_prompt() -> str:
    """获取记忆系统 System Prompt"""
    return MEMORY_SYSTEM_PROMPT


def get_agent_instructions(tools_available: List[str]) -> str:
    """根据可用工具生成 Agent 指令"""
    if not tools_available:
        return ""

    instructions = """
## Memory System

You have access to a memory system with the following tools:
"""
    for tool in tools_available:
        if tool == "memory_search":
            instructions += "\n- `memory_search(query)`: Search memories for relevant information"
        elif tool == "memory_add":
            instructions += "\n- `memory_add(content, tags)`: Add new memory entries"
        elif tool == "memory_get":
            instructions += "\n- `memory_get(path, from, lines)`: Retrieve specific memory content"

    instructions += """

### When to Use Memory

Search memory before answering about:
- Past work, decisions, or actions
- User preferences or context
- Project history or dates
- Previously discussed topics

### Citation Format

When referencing memories, use: `path#Lstart-Lend`
"""
    return instructions
```

**Step 2: 创建 Agent 使用指南**

```markdown
# docs/MEMORY_INSTRUCTIONS.md

## Memory System Instructions

You have access to a memory system that stores and retrieves information across conversations.

### When to Search Memory

Before answering any question about:
- Past work, decisions, or actions
- User preferences, goals, or context
- Project history, timelines, or dates
- Previously discussed topics or concepts

ALWAYS run a memory search first.

### How to Use Memory

1. **Search**: Use `memory_search(query)` to find relevant memories
2. **Retrieve**: Use `memory_get(path, from, lines)` to read specific sections
3. **Record**: Use `memory_add(content, tags)` to save important information

### Memory Format

- Long-term memories are in `MEMORY.md`
- Daily memories are in `memory/DD-MM-YYYY.md`
- Citations follow format: `path#Lstart-Lend`

### Best Practices

- Be specific in search queries
- Use tags when adding memories
- Verify source citations before relying on content
- Say "I checked my memory" when search yields low-confidence results
```

**Step 3: 提交**

```bash
git add src/ai_memory/tools/ docs/MEMORY_INSTRUCTIONS.md
git commit -m "feat: 实现 Agent 工具接口"
```

---

## Task 11: 框架适配器

**Files:**
- Create: `adapters/__init__.py`
- Create: `adapters/langchain.py`
- Create: `adapters/base.py`
- Test: `tests/test_adapters/`

**Step 1: 创建适配器**

```python
# adapters/__init__.py
from adapters.langchain import LangChainMemoryTool
from adapters.base import BaseAdapter

__all__ = ["LangChainMemoryTool", "BaseAdapter"]
```

```python
# adapters/base.py
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """基础适配器接口"""

    @abstractmethod
    def get_tools(self):
        """获取工具列表"""
        pass

    @abstractmethod
    def get_instructions(self) -> str:
        """获取使用指令"""
        pass
```

```python
# adapters/langchain.py
from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ai_memory.core.manager import MemoryManager


class MemorySearchSchema(BaseModel):
    query: str = Field(description="Search query for semantic search")
    max_results: int = Field(default=6, description="Maximum number of results")
    min_score: float = Field(default=0.35, description="Minimum relevance score (0-1)")


class MemoryAddSchema(BaseModel):
    content: str = Field(description="Content to remember")
    tags: str = Field(default="", description="Comma-separated tags")


class MemoryGetSchema(BaseModel):
    path: str = Field(description="File path (e.g., 'MEMORY.md' or 'memory/01-02-2025.md')")
    from_line: int = Field(default=None, description="Starting line number")
    lines: int = Field(default=20, description="Number of lines to retrieve")


class LangChainMemoryTool(BaseTool):
    """LangChain 记忆工具"""

    name: str = "memory_search"
    description: str = "Search memories for relevant information about past work, decisions, user preferences, or project history"
    args_schema: type[BaseModel] = MemorySearchSchema

    def __init__(self, manager: MemoryManager):
        super().__init__()
        self.manager = manager

    def _run(self, query: str, max_results: int = 6, min_score: float = 0.35) -> str:
        results = self.manager.search(query, max_results, min_score)
        output = f"Found {len(results)} memories:\n\n"
        for r in results:
            output += f"- {r.citation} (score: {r.score:.2f})\n  {r.snippet[:200]}...\n\n"
        return output


def get_langchain_tools(manager: MemoryManager) -> list[BaseTool]:
    """获取 LangChain 工具列表"""
    return [
        LangChainMemoryTool(manager),
        StructuredTool.from_function(
            func=lambda content, tags: manager.add_memory(content, tags.split(",") if tags else None),
            name="memory_add",
            description="Add a new memory entry",
            args_schema=MemoryAddSchema
        ),
        StructuredTool.from_function(
            func=lambda path, from_line, lines: manager.get_memory(path, from_line, lines),
            name="memory_get",
            description="Retrieve specific lines from a memory file",
            args_schema=MemoryGetSchema
        )
    ]
```

**Step 2: 提交**

```bash
git add adapters/ tests/test_adapters/
git commit -m "feat: 实现 LangChain 框架适配器"
```

---

## Task 12: 文档和示例

**Files:**
- Create: `docs/README.md`
- Create: `docs/USAGE.md`
- Create: `examples/basic_usage.py`

**Step 1: 创建文档**

```markdown
# docs/README.md

# AI Memory Plugin

可插拔的 AI 记忆系统插件，为各类 Agent 提供基于文件系统的记忆存储和混合检索能力。

## 特性

- 基于文件系统的简单存储（MEMORY.md + memory/DD-MM-YYYY.md）
- 混合检索（向量语义 + 关键词全文搜索）
- 多维度评分（相似度、时间、访问频率）
- 可插拔的嵌入模型（本地、OpenAI、自定义）
- Agent 工具接口（OpenAI Function Calling）
- 多框架适配器（LangChain、AutoGPT）

## 安装

```bash
pip install ai-memory
```

## 快速开始

```python
from ai_memory import MemoryManager, MemoryConfig

# 初始化
config = MemoryConfig(storage={"dir": "./my_memory"})
manager = MemoryManager(config)

# 添加记忆
manager.add_memory("# Project\nAI memory plugin development", tags=["project"])

# 同步索引
manager.sync()

# 搜索
results = manager.search("AI memory")
for result in results:
    print(f"{result.citation}: {result.snippet}")
```

## 文档

- [使用指南](USAGE.md)
- [API 文档](API.md)
- [示例代码](examples/)
```

```markdown
# docs/USAGE.md

# 使用指南

## 基础用法

### 初始化

```python
from ai_memory import MemoryManager, MemoryConfig

# 使用默认配置
manager = MemoryManager()

# 自定义配置
config = MemoryConfig(
    storage={"dir": "./my_memory"},
    retrieval={"max_results": 10, "min_score": 0.4}
)
manager = MemoryManager(config)
```

### 添加记忆

```python
# 添加简单记忆
manager.add_memory("# User Preference\nPrefers dark mode")

# 带标签的记忆
manager.add_memory(
    "Project deadline: 2025-03-01",
    tags=["deadline", "important"]
)
```

### 搜索记忆

```python
results = manager.search("project deadline")

for result in results:
    print(f"[{result.citation}] {result.snippet}")
    print(f"Score: {result.score:.2f}\n")
```

### 获取具体内容

```python
# 获取文件内容
content = manager.get_memory("memory/05-02-2025.md")

# 获取特定行
snippet = manager.get_memory("MEMORY.md", from_line=10, lines=5)
```

## Agent 集成

### OpenAI Function Calling

```python
from ai_memory import MemoryManager
from ai_memory.tools import get_openai_functions

manager = MemoryManager()
tools = get_openai_functions()

# 传递给 OpenAI API
# response = client.chat.completions.create(
#     model="gpt-4",
#     messages=[...],
#     tools=tools
# )
```

### LangChain 集成

```python
from ai_memory import MemoryManager
from adapters.langchain import get_langchain_tools

manager = MemoryManager()
tools = get_langchain_tools(manager)

# 传递给 LangChain Agent
# agent = Agent(tools=tools, llm=llm)
```

## 配置选项

### 存储配置

```python
MemoryConfig(
    storage={
        "dir": "./memory",      # 记忆目录
        "db_name": "memory.db"  # 数据库文件名
    }
)
```

### 检索配置

```python
MemoryConfig(
    retrieval={
        "max_results": 6,          # 最大结果数
        "min_score": 0.35,         # 最小相关分数
        "hybrid": True,            # 启用混合检索
        "vector_weight": 0.7,       # 向量权重
        "text_weight": 0.3,        # 文本权重
        "time_weight": 0.1,        # 时间权重
        "frequency_weight": 0.1    # 频率权重
    }
)
```

### 嵌入配置

```python
MemoryConfig(
    embedding={
        "provider": "local",  # local, openai
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
)
```
```

```python
# examples/basic_usage.py
"""基础用法示例"""

from ai_memory import MemoryManager, MemoryConfig

def main():
    # 初始化记忆管理器
    config = MemoryConfig(storage={"dir": "./memory"})
    manager = MemoryManager(config)

    # 添加一些记忆
    print("Adding memories...")
    manager.add_memory("# 用户偏好\n喜欢使用 Python 进行开发", tags=["user", "preference"])
    manager.add_memory("# 项目决策\n选择 SQLite 作为数据库", tags=["project", "decision"])
    manager.add_memory("# 待办事项\n- 完成记忆系统\n- 编写文档", tags=["todo"])

    # 同步索引
    print("Syncing index...")
    manager.sync()

    # 搜索
    print("\nSearching for 'Python'...")
    results = manager.search("Python")
    for r in results:
        print(f"\n[{r.citation}] (score: {r.score:.2f})")
        print(f"{r.snippet[:100]}...")

    # 获取状态
    print("\n\nSystem Status:")
    status = manager.status()
    print(f"  Files: {status.files}")
    print(f"  Model: {status.embedding_model}")

if __name__ == "__main__":
    main()
```

**Step 2: 提交**

```bash
git add docs/ examples/
git commit -m "docs: 添加文档和示例代码"
```

---

## Task 13: 最终集成测试

**Files:**
- Test: `tests/test_integration.py`

**Step 1: 编写集成测试**

```python
# tests/test_integration.py
"""集成测试"""

import pytest
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
    assert len(results) >= 1

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
    tools["memory_add"]("Test content", ["test"])

    # 搜索
    result = tools["memory_search"]("test")
    assert "count" in result
```

**Step 2: 运行集成测试**

运行: `pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: 提交**

```bash
git add tests/test_integration.py
git commit -m "test: 添加集成测试"
```

---

## 总结

本实现计划涵盖了 AI Memory Plugin 的完整开发流程：

1. **项目初始化** - 基础结构和依赖配置
2. **配置管理** - 灵活的配置系统
3. **类型定义** - 核心数据类型和异常
4. **数据库层** - SQLite 封装和 FTS5 支持
5. **文件管理器** - 文件读写和哈希追踪
6. **嵌入提供者** - 可插拔的嵌入模型接口
7. **混合检索器** - 向量 + 文本混合搜索
8. **索引同步** - 自动分块和索引
9. **核心管理器** - 统一的 API 入口
10. **工具接口** - Agent 集成支持
11. **框架适配器** - LangChain 等框架支持
12. **文档示例** - 用户文档和代码示例
13. **集成测试** - 端到端验证

每个任务都遵循 TDD 原则，确保代码质量和可维护性。
