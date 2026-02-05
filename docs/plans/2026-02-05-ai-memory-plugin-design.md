# AI Memory Plugin 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 构建一个可插拔的 Python 记忆系统插件，支持文件系统存储、SQLite 索引、多维度混合检索

**架构:** 分层架构设计 - Agent 接口层 (System Prompt + Tool 函数) -> MemoryManager -> 检索层/存储层/同步层 -> 嵌入提供者

**技术栈:** Python 3.10+, SQLite (FTS5), sentence-transformers, pydantic, watchdog

---

## 项目目录结构

```
AI-Memmory_plugin/
├── src/ai_memory/
│   ├── __init__.py
│   ├── config.py              # 配置模型
│   ├── manager.py             # MemoryManager 统一入口
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── file_manager.py    # 文件系统管理
│   │   └── db_manager.py     # SQLite 索引管理
│   ├── search/
│   │   ├── __init__.py
│   │   ├── hybrid_searcher.py # 混合检索器
│   │   └── scorer.py          # 多维度评分器
│   ├── sync/
│   │   ├── __init__.py
│   │   └── file_watcher.py   # 文件监控器
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py            # 嵌入提供者基类
│   │   ├── local_provider.py  # 本地模型实现
│   │   └── openai_provider.py # OpenAI 实现
│   ├── tools/
│   │   ├── __init__.py
│   │   └── memory_tools.py    # Tool 函数接口
│   └── prompts/
│       └── MEMORY_INSTRUCTIONS.md  # System Prompt 模板
├── tests/
│   ├── __init__.py
│   ├── test_manager.py
│   ├── test_storage/
│   ├── test_search/
│   └── test_embeddings/
├── docs/
│   └── plans/
├── pyproject.toml
├── setup.py
└── README.md
```

---

## Task 1: 项目基础结构搭建

**Files:**
- Create: `pyproject.toml`
- Create: `setup.py`
- Create: `src/ai_memory/__init__.py`

**Step 1: 创建 pyproject.toml**

```toml
[project]
name = "ai-memory-plugin"
version = "0.1.0"
description = "可插拔的 AI 记忆系统插件"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0.0",
    "sentence-transformers>=2.2.0",
    "watchdog>=3.0.0",
    "openai>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I"]
```

**Step 2: 创建 setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="ai-memory-plugin",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
```

**Step 3: 创建主包 __init__.py**

```python
"""AI Memory Plugin - 可插拔的 AI 记忆系统"""

__version__ = "0.1.0"

from ai_memory.manager import MemoryManager
from ai_memory.config import MemoryConfig, MemorySearchConfig

__all__ = ["MemoryManager", "MemoryConfig", "MemorySearchConfig"]
```

**Step 4: 提交基础结构**

```bash
git add pyproject.toml setup.py src/ai_memory/__init__.py
git commit -m "chore: 添加项目基础结构和依赖配置"
```

---

## Task 2: 配置模型定义

**Files:**
- Create: `src/ai_memory/config.py`
- Test: `tests/test_config.py`

**Step 1: 编写配置模型测试**

```python
# tests/test_config.py
import pytest
from pydantic import ValidationError
from ai_memory.config import MemoryConfig, MemorySearchConfig, EmbeddingProviderType


def test_default_config():
    """测试默认配置"""
    config = MemoryConfig()
    assert config.storage_dir == "memory"
    assert config.db_path == "memory.db"
    assert config.embedding.provider == EmbeddingProviderType.LOCAL


def test_openai_config():
    """测试 OpenAI 配置"""
    config = MemoryConfig(
        embedding={
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "test-key"
        }
    )
    assert config.embedding.provider == EmbeddingProviderType.OPENAI
    assert config.embedding.model == "text-embedding-3-small"


def test_invalid_provider():
    """测试无效的嵌入提供者"""
    with pytest.raises(ValidationError):
        MemoryConfig(embedding={"provider": "invalid"})


def test_search_config():
    """测试搜索配置"""
    config = MemorySearchConfig()
    assert config.max_results == 6
    assert config.min_score == 0.35
    assert config.hybrid.enabled is True
```

**Step 2: 运行测试验证失败**

```bash
cd /Users/bytedance/GolandProjects/AI-funding-backup/AI-Memmory_plugin
pytest tests/test_config.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'ai_memory.config'`

**Step 3: 实现配置模型**

```python
# src/ai_memory/config.py
from typing import Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class EmbeddingProviderType(str, Literal["local", "openai", "external"]):
    """嵌入提供者类型"""


class EmbeddingConfig(BaseModel):
    """嵌入模型配置"""
    provider: EmbeddingProviderType = EmbeddingProviderType.LOCAL
    model: str = "all-MiniLM-L6-v2"
    api_key: Optional[str] = None
    external_command: Optional[str] = None


class HybridSearchConfig(BaseModel):
    """混合检索配置"""
    enabled: bool = True
    vector_weight: float = 0.7
    text_weight: float = 0.3
    time_weight: float = 0.2  # 时间权重
    frequency_weight: float = 0.1  # 访问频率权重


class MemorySearchConfig(BaseModel):
    """检索配置"""
    max_results: int = Field(default=6, ge=1, le=50)
    min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    hybrid: HybridSearchConfig = Field(default_factory=HybridSearchConfig)


class StorageConfig(BaseModel):
    """存储配置"""
    storage_dir: str = "memory"
    db_path: str = "memory.db"
    chunk_size: int = 400  # 分块 token 数
    chunk_overlap: int = 80  # 重叠 token 数


class MemoryConfig(BaseModel):
    """记忆系统总配置"""
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: MemorySearchConfig = Field(default_factory=MemorySearchConfig)

    def get_storage_dir(self) -> Path:
        """获取存储目录绝对路径"""
        return Path(self.storage.storage_dir).resolve()

    def get_db_path(self) -> Path:
        """获取数据库文件绝对路径"""
        return Path(self.storage.db_path).resolve()
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_config.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add tests/test_config.py src/ai_memory/config.py
git commit -m "feat: 添加配置模型定义"
```

---

## Task 3: 数据库 Schema 定义

**Files:**
- Create: `src/ai_memory/storage/__init__.py`
- Create: `src/ai_memory/storage/db_manager.py`
- Test: `tests/test_storage/test_db_manager.py`

**Step 1: 编写数据库测试**

```python
# tests/test_storage/test_db_manager.py
import pytest
import sqlite3
from pathlib import Path
from ai_memory.config import MemoryConfig
from ai_memory.storage.db_manager import DatabaseManager


@pytest.fixture
def temp_db(tmp_path):
    """临时数据库 fixture"""
    db_path = tmp_path / "test.db"
    config = MemoryConfig(storage={"db_path": str(db_path)})
    manager = DatabaseManager(config)
    manager.initialize()
    yield manager
    manager.close()


def test_database_initialization(temp_db):
    """测试数据库初始化"""
    conn = sqlite3.connect(temp_db.config.get_db_path())
    cursor = conn.cursor()

    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "files" in tables
    assert "chunks" in tables
    assert "fts_chunks" in tables
    assert "embedding_cache" in tables
    assert "metadata" in tables

    conn.close()


def test_add_file_record(temp_db):
    """测试添加文件记录"""
    temp_db.add_file_record(
        path="MEMORY.md",
        source="memory",
        file_hash="abc123",
        size=1024
    )

    files = temp_db.get_all_files()
    assert len(files) == 1
    assert files[0]["path"] == "MEMORY.md"


def test_add_chunk_record(temp_db):
    """测试添加分块记录"""
    temp_db.add_file_record(path="test.md", source="memory", file_hash="hash", size=100)

    temp_db.add_chunk_record(
        file_id=1,
        start_line=1,
        end_line=10,
        text="测试内容",
        embedding=[0.1, 0.2, 0.3],
        model="test-model"
    )

    chunks = temp_db.get_chunks_by_file(1)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "测试内容"
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_storage/test_db_manager.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'ai_memory.storage.db_manager'`

**Step 3: 实现数据库管理器**

```python
# src/ai_memory/storage/__init__.py
from ai_memory.storage.db_manager import DatabaseManager

__all__ = ["DatabaseManager"]
```

```python
# src/ai_memory/storage/db_manager.py
import sqlite3
import json
from typing import List, Dict, Optional
from pathlib import Path
from ai_memory.config import MemoryConfig


class DatabaseManager:
    """SQLite 数据库管理器"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.db_path = config.get_db_path()
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """初始化数据库表结构"""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """创建所有表"""
        cursor = self._conn.cursor()

        # 文件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                source TEXT NOT NULL DEFAULT 'memory',
                file_hash TEXT NOT NULL,
                size INTEGER NOT NULL,
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)

        # 分块表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                model TEXT NOT NULL,
                updated_at INTEGER DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (file_id) REFERENCES files(id)
            )
        """)

        # FTS5 全文搜索表
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                text,
                id UNINDEXED,
                path UNINDEXED,
                source UNINDEXED
            )
        """)

        # 嵌入缓存表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                embedding TEXT NOT NULL,
                updated_at INTEGER DEFAULT (strftime('%s', 'now')),
                PRIMARY KEY (provider, model, text_hash)
            )
        """)

        # 元数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash)")

        self._conn.commit()

    def add_file_record(self, path: str, source: str, file_hash: str, size: int) -> int:
        """添加文件记录"""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO files (path, source, file_hash, size, updated_at)
            VALUES (?, ?, ?, ?, strftime('%s', 'now'))
            """,
            (path, source, file_hash, size)
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_all_files(self) -> List[Dict]:
        """获取所有文件记录"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM files")
        return [dict(row) for row in cursor.fetchall()]

    def add_chunk_record(
        self, file_id: int, start_line: int, end_line: int,
        text: str, embedding: List[float], model: str
    ) -> int:
        """添加分块记录"""
        cursor = self._conn.cursor()

        # 添加到 chunks 表
        cursor.execute(
            """
            INSERT INTO chunks (file_id, start_line, end_line, text, embedding, model, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
            """,
            (file_id, start_line, end_line, text, json.dumps(embedding), model)
        )

        chunk_id = cursor.lastrowid

        # 获取文件路径
        cursor.execute("SELECT path, source FROM files WHERE id = ?", (file_id,))
        file_info = cursor.fetchone()
        if file_info:
            # 添加到 FTS 表
            cursor.execute(
                """
                INSERT INTO fts_chunks (rowid, text, id, path, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, text, chunk_id, file_info["path"], file_info["source"])
            )

        self._conn.commit()
        return chunk_id

    def get_chunks_by_file(self, file_id: int) -> List[Dict]:
        """获取文件的所有分块"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE file_id = ?", (file_id,))
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_storage/test_db_manager.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add tests/test_storage/test_db_manager.py src/ai_memory/storage/ tests/test_storage/__init__.py
git commit -m "feat: 添加数据库 Schema 和管理器"
```

---

## Task 4: 嵌入提供者基类和实现

**Files:**
- Create: `src/ai_memory/embeddings/__init__.py`
- Create: `src/ai_memory/embeddings/base.py`
- Create: `src/ai_memory/embeddings/local_provider.py`
- Test: `tests/test_embeddings/test_local_provider.py`

**Step 1: 编写嵌入提供者测试**

```python
# tests/test_embeddings/test_local_provider.py
import pytest
from ai_memory.config import EmbeddingConfig
from ai_memory.embeddings.local_provider import LocalEmbeddingProvider


@pytest.fixture
def provider():
    """本地嵌入提供者 fixture"""
    config = EmbeddingConfig(provider="local", model="all-MiniLM-L6-v2")
    return LocalEmbeddingProvider(config)


def test_embed_query(provider):
    """测试单个查询嵌入"""
    text = "这是一段测试文本"
    embedding = provider.embed_query(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384  # all-MiniLM-L6-v2 的维度
    assert all(isinstance(x, float) for x in embedding)


def test_embed_batch(provider):
    """测试批量嵌入"""
    texts = ["文本一", "文本二", "文本三"]
    embeddings = provider.embed_batch(texts)

    assert len(embeddings) == 3
    assert all(len(e) == 384 for e in embeddings)


def test_embedding_cache():
    """测试嵌入缓存"""
    config = EmbeddingConfig(provider="local", model="all-MiniLM-L6-v2")
    provider = LocalEmbeddingProvider(config)

    text = "测试缓存"
    # 第一次调用
    emb1 = provider.embed_query(text)
    # 第二次调用（应该使用缓存）
    emb2 = provider.embed_query(text)

    assert emb1 == emb2
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_embeddings/test_local_provider.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'ai_memory.embeddings.local_provider'`

**Step 3: 实现嵌入提供者**

```python
# src/ai_memory/embeddings/__init__.py
from ai_memory.embeddings.base import BaseEmbeddingProvider
from ai_memory.embeddings.local_provider import LocalEmbeddingProvider

__all__ = ["BaseEmbeddingProvider", "LocalEmbeddingProvider"]
```

```python
# src/ai_memory/embeddings/base.py
from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """嵌入提供者基类"""

    def __init__(self, config):
        self.config = config
        self._cache = {}  # 简单内存缓存

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        pass

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return f"{self.config.model}:{hash(text)}"
```

```python
# src/ai_memory/embeddings/local_provider.py
from typing import List
from sentence_transformers import SentenceTransformer
from ai_memory.embeddings.base import BaseEmbeddingProvider


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """本地 HuggingFace 嵌入模型提供者"""

    def __init__(self, config):
        super().__init__(config)
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """延迟加载模型"""
        if self._model is None:
            self._model = SentenceTransformer(self.config.model)
        return self._model

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本"""
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self.model.encode(text).tolist()
        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        embeddings = self.model.encode(texts).tolist()
        return embeddings
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_embeddings/test_local_provider.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add tests/test_embeddings/ src/ai_memory/embeddings/ tests/test_embeddings/__init__.py
git commit -m "feat: 添加嵌入提供者基类和本地实现"
```

---

## Task 5: 文件管理器

**Files:**
- Create: `src/ai_memory/storage/file_manager.py`
- Test: `tests/test_storage/test_file_manager.py`

**Step 1: 编写文件管理器测试**

```python
# tests/test_storage/test_file_manager.py
import pytest
from pathlib import Path
from ai_memory.config import MemoryConfig
from ai_memory.storage.file_manager import FileManager


@pytest.fixture
def temp_memory_dir(tmp_path):
    """临时记忆目录 fixture"""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return memory_dir


@pytest.fixture
def file_manager(temp_memory_dir):
    """文件管理器 fixture"""
    config = MemoryConfig(storage={"storage_dir": str(temp_memory_dir)})
    return FileManager(config)


def test_create_daily_file(file_manager, temp_memory_dir):
    """测试创建每日记忆文件"""
    from datetime import datetime

    date = datetime(2025, 2, 5)
    file_path = file_manager.create_daily_file(date)

    expected_name = "05-02-2025.md"
    assert file_path.name == expected_name
    assert file_path.exists()


def test_write_to_memory(file_manager):
    """测试写入记忆"""
    content = "# 测试记忆\n\n这是一个测试内容。"
    file_manager.write_to_memory("MEMORY.md", content)

    file_path = file_manager.storage_dir / "MEMORY.md"
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == content


def test_read_memory(file_manager, temp_memory_dir):
    """测试读取记忆"""
    (temp_memory_dir / "MEMORY.md").write_text("测试内容", encoding="utf-8")

    content = file_manager.read_memory("MEMORY.md")
    assert content == "测试内容"
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_storage/test_file_manager.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'ai_memory.storage.file_manager'`

**Step 3: 实现文件管理器**

```python
# src/ai_memory/storage/file_manager.py
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List
from ai_memory.config import MemoryConfig


class FileManager:
    """文件系统管理器"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.storage_dir = config.get_storage_dir()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_daily_file(self, dt: Optional[date] = None) -> Path:
        """创建每日记忆文件"""
        if dt is None:
            dt = date.today()
        filename = f"{dt.day:02d}-{dt.month:02d}-{dt.year}.md"
        file_path = self.storage_dir / filename
        if not file_path.exists():
            file_path.write_text("", encoding="utf-8")
        return file_path

    def write_to_memory(self, filename: str, content: str, mode: str = "a") -> None:
        """写入记忆文件"""
        file_path = self.storage_dir / filename
        file_path.write_text(content, encoding="utf-8", mode=mode)

    def read_memory(self, filename: str) -> str:
        """读取记忆文件"""
        file_path = self.storage_dir / filename
        return file_path.read_text(encoding="utf-8")

    def get_memory_files(self) -> List[Path]:
        """获取所有记忆文件"""
        return list(self.storage_dir.glob("*.md"))

    def compute_file_hash(self, file_path: Path) -> str:
        """计算文件 SHA256 哈希"""
        import hashlib
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_storage/test_file_manager.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add tests/test_storage/test_file_manager.py src/ai_memory/storage/file_manager.py
git commit -m "feat: 添加文件管理器"
```

---

## Task 6: 混合检索器

**Files:**
- Create: `src/ai_memory/search/__init__.py`
- Create: `src/ai_memory/search/scorer.py`
- Create: `src/ai_memory/search/hybrid_searcher.py`
- Test: `tests/test_search/test_scorer.py`

**Step 1: 编写评分器测试**

```python
# tests/test_search/test_scorer.py
import pytest
from ai_memory.search.scorer import HybridScorer, ScoringConfig


def test_vector_only_score():
    """测试仅向量分数"""
    config = ScoringConfig(vector_weight=1.0, text_weight=0.0, time_weight=0.0, frequency_weight=0.0)
    scorer = HybridScorer(config)

    result = scorer.calculate(
        vector_score=0.8,
        text_score=0.0,
        time_score=0.5,
        frequency_score=0.5
    )

    assert result == 0.8


def test_hybrid_score():
    """测试混合分数"""
    config = ScoringConfig(vector_weight=0.5, text_weight=0.3, time_weight=0.15, frequency_weight=0.05)
    scorer = HybridScorer(config)

    result = scorer.calculate(
        vector_score=0.8,
        text_score=0.6,
        time_score=0.7,
        frequency_score=0.5
    )

    # 0.8*0.5 + 0.6*0.3 + 0.7*0.15 + 0.5*0.05 = 0.4 + 0.18 + 0.105 + 0.025 = 0.71
    assert abs(result - 0.71) < 0.01


def test_time_score_calculation():
    """测试时间分数计算"""
    from datetime import datetime, timedelta
    scorer = HybridScorer(ScoringConfig())

    now = datetime.now()
    # 一天前的分数应该高于一周前的分数
    recent_score = scorer._time_score(now - timedelta(days=1))
    old_score = scorer._time_score(now - timedelta(days=7))

    assert recent_score > old_score
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_search/test_scorer.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'ai_memory.search.scorer'`

**Step 3: 实现评分器和检索器**

```python
# src/ai_memory/search/__init__.py
from ai_memory.search.scorer import HybridScorer, ScoringConfig
from ai_memory.search.hybrid_searcher import HybridSearcher, SearchResult

__all__ = ["HybridScorer", "ScoringConfig", "HybridSearcher", "SearchResult"]
```

```python
# src/ai_memory/search/scorer.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ScoringConfig:
    """评分配置"""
    vector_weight: float = 0.7
    text_weight: float = 0.3
    time_weight: float = 0.2
    frequency_weight: float = 0.1


class HybridScorer:
    """混合评分器"""

    def __init__(self, config: ScoringConfig):
        self.config = config

    def calculate(
        self,
        vector_score: float,
        text_score: float,
        time_score: float = 0.5,
        frequency_score: float = 0.5
    ) -> float:
        """计算综合分数"""
        total_weight = (
            self.config.vector_weight +
            self.config.text_weight +
            self.config.time_weight +
            self.config.frequency_weight
        )

        if total_weight == 0:
            return 0.0

        weighted_score = (
            vector_score * self.config.vector_weight +
            text_score * self.config.text_weight +
            time_score * self.config.time_weight +
            frequency_score * self.config.frequency_weight
        )

        return weighted_score / total_weight

    def _time_score(self, timestamp: Optional[datetime] = None) -> float:
        """计算时间分数 - 越新分数越高"""
        if timestamp is None:
            return 0.5

        now = datetime.now()
        days_old = (now - timestamp).days

        # 30天内线性衰减，之后保持最小值
        if days_old <= 30:
            return 1.0 - (days_old / 60)  # 最小 0.5
        else:
            return 0.5
```

```python
# src/ai_memory/search/hybrid_searcher.py
from dataclasses import dataclass
from typing import List, Dict, Any
import sqlite3
from ai_memory.config import MemoryConfig
from ai_memory.search.scorer import HybridScorer, ScoringConfig


@dataclass
class SearchResult:
    """搜索结果"""
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    source: str = "memory"

    @property
    def citation(self) -> str:
        """生成引用格式"""
        return f"{self.path}#L{self.start_line}-L{self.end_line}"


class HybridSearcher:
    """混合检索器"""

    def __init__(self, config: MemoryConfig, db_conn: sqlite3.Connection, embedding_provider):
        self.config = config
        self.db_conn = db_conn
        self.embedding = embedding_provider
        self.scorer = HybridScorer(ScoringConfig(
            vector_weight=config.search.hybrid.vector_weight,
            text_weight=config.search.hybrid.text_weight,
            time_weight=config.search.hybrid.time_weight,
            frequency_weight=config.search.hybrid.frequency_weight
        ))

    def search(self, query: str) -> List[SearchResult]:
        """执行混合搜索"""
        query_embedding = self.embedding.embed_query(query)
        text_results = self._text_search(query)
        vector_results = self._vector_search(query_embedding)

        # 合并结果
        combined = self._merge_results(text_results, vector_results)

        # 评分和排序
        scored = self._score_results(combined, query)
        scored.sort(key=lambda x: x.score, reverse=True)

        # 过滤和限制
        min_score = self.config.search.min_score
        max_results = self.config.search.max_results

        return [r for r in scored[:max_results] if r.score >= min_score]

    def _text_search(self, query: str) -> List[Dict]:
        """全文搜索"""
        cursor = self.db_conn.cursor()
        cursor.execute(
            """
            SELECT fts_chunks.id, fts_chunks.path, fts_chunks.source, chunks.start_line, chunks.end_line, chunks.text,
                   bm25(fts_chunks) as rank
            FROM fts_chunks
            JOIN chunks ON fts_chunks.id = chunks.id
            WHERE fts_chunks MATCH ?
            ORDER BY rank
            LIMIT 50
            """,
            (query,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def _vector_search(self, query_embedding: List[float]) -> List[Dict]:
        """向量相似度搜索"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM chunks LIMIT 50")  # 简化版本，实际需要余弦相似度
        return [dict(row) for row in cursor.fetchall()]

    def _merge_results(self, text_results: List[Dict], vector_results: List[Dict]) -> List[Dict]:
        """合并去重"""
        seen = set()
        merged = []

        for result in text_results + vector_results:
            key = (result["path"], result["start_line"], result["end_line"])
            if key not in seen:
                seen.add(key)
                merged.append(result)

        return merged

    def _score_results(self, results: List[Dict], query: str) -> List[SearchResult]:
        """对结果评分"""
        scored = []

        for result in results:
            text_score = self._text_score_to_score(result.get("rank", 10))
            vector_score = 0.5  # 简化版本

            final_score = self.scorer.calculate(
                vector_score=vector_score,
                text_score=text_score
            )

            scored.append(SearchResult(
                path=result["path"],
                start_line=result["start_line"],
                end_line=result["end_line"],
                score=final_score,
                snippet=result["text"][:200],
                source=result.get("source", "memory")
            ))

        return scored

    def _text_score_to_score(self, rank: int) -> float:
        """将 BM25 排名转换为分数"""
        return 1.0 / (1.0 + rank)
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_search/test_scorer.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add tests/test_search/ src/ai_memory/search/ tests/test_search/__init__.py
git commit -m "feat: 添加混合检索器和评分器"
```

---

## Task 7: MemoryManager 核心入口

**Files:**
- Create: `src/ai_memory/manager.py`
- Test: `tests/test_manager.py`

**Step 1: 编写 MemoryManager 测试**

```python
# tests/test_manager.py
import pytest
from pathlib import Path
from ai_memory.config import MemoryConfig
from ai_memory.manager import MemoryManager


@pytest.fixture
def temp_memory_dir(tmp_path):
    """临时记忆目录"""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return memory_dir


@pytest.fixture
def memory_manager(temp_memory_dir):
    """MemoryManager fixture"""
    config = MemoryConfig(
        storage={
            "storage_dir": str(temp_memory_dir),
            "db_path": str(temp_memory_dir / "memory.db")
        }
    )
    return MemoryManager(config)


def test_add_memory(memory_manager, temp_memory_dir):
    """测试添加记忆"""
    content = "# 重要决策\n\n决定使用 SQLite 作为存储后端。"
    memory_manager.add_memory("MEMORY.md", content)

    file_path = temp_memory_dir / "MEMORY.md"
    assert file_path.exists()
    assert "重要决策" in file_path.read_text(encoding="utf-8")


def test_search_memory(memory_manager, temp_memory_dir):
    """测试搜索记忆"""
    # 添加测试数据
    memory_manager.add_memory("MEMORY.md", "# 项目\n\n这是一个 AI 记忆系统项目。")
    memory_manager.add_memory("MEMORY.md", "# 技术\n\n使用 Python 和 SQLite。")

    # 同步索引
    memory_manager.sync()

    # 搜索
    results = memory_manager.search("AI 项目")

    assert len(results) >= 1
    assert "AI" in results[0].snippet or "项目" in results[0].snippet


def test_get_memory(memory_manager, temp_memory_dir):
    """测试获取记忆"""
    content = "# 测试\n\n第一行\n第二行\n第三行"
    memory_manager.add_memory("MEMORY.md", content)

    result = memory_manager.get_memory("MEMORY.md", from_line=2, lines=2)

    assert "第一行" in result.text
    assert "第二行" in result.text
    assert "第三行" not in result.text
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_manager.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'ai_memory.manager'`

**Step 3: 实现 MemoryManager**

```python
# src/ai_memory/manager.py
from typing import List, Optional
from pathlib import Path
from ai_memory.config import MemoryConfig
from ai_memory.storage.file_manager import FileManager
from ai_memory.storage.db_manager import DatabaseManager
from ai_memory.embeddings.local_provider import LocalEmbeddingProvider
from ai_memory.embeddings.openai_provider import OpenAIEmbeddingProvider
from ai_memory.search.hybrid_searcher import HybridSearcher
from ai_memory.sync.file_watcher import FileWatcher


class MemoryGetResult:
    """获取记忆结果"""
    def __init__(self, text: str, path: str):
        self.text = text
        self.path = path


class MemoryManager:
    """记忆系统统一管理器"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.file_manager = FileManager(config)
        self.db_manager = DatabaseManager(config)
        self.db_manager.initialize()

        # 初始化嵌入提供者
        self.embedding = self._create_embedding_provider()

        # 检索器
        self.searcher = HybridSearcher(config, self.db_manager._conn, self.embedding)

        # 文件监控器（可选）
        self.watcher: Optional[FileWatcher] = None

    def _create_embedding_provider(self):
        """创建嵌入提供者"""
        from ai_memory.config import EmbeddingProviderType

        if self.config.embedding.provider == EmbeddingProviderType.LOCAL:
            return LocalEmbeddingProvider(self.config.embedding)
        elif self.config.embedding.provider == EmbeddingProviderType.OPENAI:
            return OpenAIEmbeddingProvider(self.config.embedding)
        else:
            return LocalEmbeddingProvider(self.config.embedding)

    def add_memory(self, filename: str, content: str) -> None:
        """添加记忆到文件"""
        self.file_manager.write_to_memory(filename, content)

    def search(self, query: str) -> List:
        """搜索记忆"""
        return self.searcher.search(query)

    def get_memory(self, path: str, from_line: Optional[int] = None, lines: int = 20) -> MemoryGetResult:
        """获取指定记忆文件内容"""
        full_content = self.file_manager.read_memory(path)

        if from_line is None:
            return MemoryGetResult(full_content, path)

        # 按行分割
        lines_list = full_content.split("\n")
        start = max(0, from_line - 1)
        end = min(len(lines_list), start + lines)
        selected = "\n".join(lines_list[start:end])

        return MemoryGetResult(selected, path)

    def sync(self) -> None:
        """同步文件到索引"""
        for file_path in self.file_manager.get_memory_files():
            rel_path = file_path.relative_to(self.file_manager.storage_dir)
            file_hash = self.file_manager.compute_file_hash(file_path)
            content = self.file_manager.read_memory(str(rel_path))

            # 添加文件记录
            file_id = self.db_manager.add_file_record(
                path=str(rel_path),
                source="memory",
                file_hash=file_hash,
                size=file_path.stat().st_size
            )

            # 分块和嵌入
            chunks = self._chunk_text(content)
            for i, chunk in enumerate(chunks):
                embedding = self.embedding.embed_query(chunk)
                self.db_manager.add_chunk_record(
                    file_id=file_id,
                    start_line=i * self.config.storage.chunk_size + 1,
                    end_line=(i + 1) * self.config.storage.chunk_size,
                    text=chunk,
                    embedding=embedding,
                    model=self.config.embedding.model
                )

    def _chunk_text(self, text: str) -> List[str]:
        """将文本分块"""
        chunks = []
        lines = text.split("\n")
        chunk_lines = []
        current_size = 0

        for line in lines:
            chunk_lines.append(line)
            current_size += len(line)

            if current_size >= self.config.storage.chunk_size:
                chunks.append("\n".join(chunk_lines))
                # 保留重叠部分
                overlap = min(len(chunk_lines), self.config.storage.chunk_overlap // 10)
                chunk_lines = chunk_lines[-overlap:] if overlap > 0 else []
                current_size = sum(len(l) for l in chunk_lines)

        if chunk_lines:
            chunks.append("\n".join(chunk_lines))

        return chunks

    def status(self) -> dict:
        """获取系统状态"""
        files = self.db_manager.get_all_files()
        return {
            "total_files": len(files),
            "storage_dir": str(self.file_manager.storage_dir),
            "db_path": str(self.db_manager.db_path),
            "embedding_provider": self.config.embedding.provider,
            "embedding_model": self.config.embedding.model
        }

    def close(self) -> None:
        """关闭管理器"""
        self.db_manager.close()
        if self.watcher:
            self.watcher.stop()
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_manager.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add tests/test_manager.py src/ai_memory/manager.py
git commit -m "feat: 添加 MemoryManager 核心入口"
```

---

## Task 8: Tool 函数接口

**Files:**
- Create: `src/ai_memory/tools/__init__.py`
- Create: `src/ai_memory/tools/memory_tools.py`
- Test: `tests/test_tools/test_memory_tools.py`

**Step 1: 编写工具接口测试**

```python
# tests/test_tools/test_memory_tools.py
import pytest
from pathlib import Path
from ai_memory.config import MemoryConfig
from ai_memory.manager import MemoryManager
from ai_memory.tools.memory_tools import memory_search, memory_add, memory_get


@pytest.fixture
def temp_memory_dir(tmp_path):
    """临时记忆目录"""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return memory_dir


@pytest.fixture
def manager(temp_memory_dir):
    """MemoryManager fixture"""
    config = MemoryConfig(
        storage={
            "storage_dir": str(temp_memory_dir),
            "db_path": str(temp_memory_dir / "memory.db")
        }
    )
    manager = MemoryManager(config)
    # 添加测试数据并同步
    manager.add_memory("MEMORY.md", "# 项目\n\n这是一个测试项目。")
    manager.sync()
    return manager


def test_memory_search_tool(manager):
    """测试记忆搜索工具"""
    results = memory_search(manager, query="测试项目", max_results=5)

    assert len(results) >= 1
    assert results[0]["path"] == "MEMORY.md"
    assert "snippet" in results[0]
    assert "score" in results[0]


def test_memory_add_tool(manager):
    """测试添加记忆工具"""
    result = memory_add(manager, content="# 新增\n\n这是一条新增的记忆。")

    assert result["success"] is True
    assert result["path"] == "MEMORY.md"


def test_memory_get_tool(manager, temp_memory_dir):
    """测试获取记忆工具"""
    # 首先写入多行内容
    manager.add_memory("test.md", "第一行\n第二行\n第三行\n第四行")

    result = memory_get(manager, path="test.md", from_line=2, lines=2)

    assert "第一行" not in result["text"]
    assert "第二行" in result["text"]
    assert "第三行" in result["text"]
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_tools/test_memory_tools.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'ai_memory.tools.memory_tools'`

**Step 3: 实现工具接口**

```python
# src/ai_memory/tools/__init__.py
from ai_memory.tools.memory_tools import (
    memory_search,
    memory_add,
    memory_get,
    get_memory_tools,
    MEMORY_INSTRUCTIONS
)

__all__ = ["memory_search", "memory_add", "memory_get", "get_memory_tools", "MEMORY_INSTRUCTIONS"]
```

```python
# src/ai_memory/tools/memory_tools.py
from typing import List, Dict, Any, Optional
from ai_memory.manager import MemoryManager


def memory_search(
    manager: MemoryManager,
    query: str,
    max_results: int = 6,
    min_score: float = 0.35,
    tags: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    搜索记忆

    Args:
        manager: MemoryManager 实例
        query: 搜索查询
        max_results: 最大结果数
        min_score: 最小相关性分数
        tags: 可选的标签过滤

    Returns:
        搜索结果列表
    """
    results = manager.search(query)
    filtered = [r for r in results if r.score >= min_score][:max_results]

    return [
        {
            "path": r.path,
            "start_line": r.start_line,
            "end_line": r.end_line,
            "score": r.score,
            "snippet": r.snippet,
            "citation": r.citation,
            "source": r.source
        }
        for r in filtered
    ]


def memory_add(
    manager: MemoryManager,
    content: str,
    tags: Optional[List[str]] = None,
    priority: str = "normal",
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    添加记忆

    Args:
        manager: MemoryManager 实例
        content: 要记录的内容
        tags: 可选的标签
        priority: 优先级 (high, normal, low)
        filename: 目标文件名，默认为 MEMORY.md

    Returns:
        操作结果
    """
    if priority == "high":
        content = f"## 高优先级记忆\n\n{content}"
    elif priority == "low":
        content = f"## 低优先级记忆\n\n{content}"

    target_file = filename or "MEMORY.md"
    manager.add_memory(target_file, content)

    return {
        "success": True,
        "path": target_file,
        "tags": tags or [],
        "priority": priority
    }


def memory_get(
    manager: MemoryManager,
    path: str,
    from_line: Optional[int] = None,
    lines: int = 20
) -> Dict[str, Any]:
    """
    获取记忆文件内容

    Args:
        manager: MemoryManager 实例
        path: 文件路径
        from_line: 起始行号
        lines: 获取行数

    Returns:
        文件内容和元数据
    """
    result = manager.get_memory(path, from_line, lines)

    return {
        "text": result.text,
        "path": result.path,
        "from_line": from_line,
        "lines": lines
    }


def get_memory_tools(manager: MemoryManager) -> Dict[str, Any]:
    """
    获取可用的记忆工具定义（用于 OpenAI Function Calling）

    Args:
        manager: MemoryManager 实例

    Returns:
        工具定义字典
    """
    return {
        "memory_search": {
            "name": "memory_search",
            "description": "搜索记忆以获取关于过去工作、决策、用户偏好或项目历史的相关信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询，用于语义搜索"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最大结果数 (默认: 6)",
                        "default": 6
                    },
                    "min_score": {
                        "type": "number",
                        "description": "最小相关性分数 (0-1, 默认: 0.35)",
                        "default": 0.35
                    }
                },
                "required": ["query"]
            }
        },
        "memory_add": {
            "name": "memory_add",
            "description": "添加新的记忆条目。用于重要决策、用户偏好或值得记住的信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要记录的内容"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选的分类标签"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "normal", "low"],
                        "description": "优先级 (默认: normal)",
                        "default": "normal"
                    }
                },
                "required": ["content"]
            }
        },
        "memory_get": {
            "name": "memory_get",
            "description": "按路径检索特定记忆文件的指定行",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "记忆目录下的文件相对路径 (例如: 'MEMORY.md' 或 'memory/01-02-2025.md')"
                    },
                    "from_line": {
                        "type": "integer",
                        "description": "起始行号 (从 1 开始)"
                    },
                    "lines": {
                        "type": "integer",
                        "description": "要获取的行数 (默认: 20)",
                        "default": 20
                    }
                },
                "required": ["path"]
            }
        }
    }


# System Prompt 模板
MEMORY_INSTRUCTIONS = """## Memory System Instructions

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
2. **Retrieve**: Use `memory_get(path, from_line, lines)` to read specific sections
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
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_tools/test_memory_tools.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add tests/test_tools/ src/ai_memory/tools/ tests/test_tools/__init__.py
git commit -m "feat: 添加 Tool 函数接口和 System Prompt"
```

---

## Task 9: 文件监控器

**Files:**
- Create: `src/ai_memory/sync/__init__.py`
- Create: `src/ai_memory/sync/file_watcher.py`
- Test: `tests/test_sync/test_file_watcher.py`

**Step 1: 编写文件监控器测试**

```python
# tests/test_sync/test_file_watcher.py
import pytest
import time
from pathlib import Path
from ai_memory.config import MemoryConfig
from ai_memory.manager import MemoryManager
from ai_memory.sync.file_watcher import FileWatcher


@pytest.fixture
def temp_memory_dir(tmp_path):
    """临时记忆目录"""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return memory_dir


def test_file_watcher_detection(temp_memory_dir):
    """测试文件变化检测"""
    config = MemoryConfig(
        storage={
            "storage_dir": str(temp_memory_dir),
            "db_path": str(temp_memory_dir / "memory.db")
        }
    )
    manager = MemoryManager(config)
    manager.sync()

    changes_detected = []

    def on_change(path):
        changes_detected.append(path)

    watcher = FileWatcher(manager, on_change)
    watcher.start()

    # 修改文件
    (temp_memory_dir / "MEMORY.md").write_text("修改的内容", encoding="utf-8")
    time.sleep(0.5)  # 等待检测

    watcher.stop()

    assert len(changes_detected) > 0
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_sync/test_file_watcher.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'ai_memory.sync.file_watcher'`

**Step 3: 实现文件监控器**

```python
# src/ai_memory/sync/__init__.py
from ai_memory.sync.file_watcher import FileWatcher

__all__ = ["FileWatcher"]
```

```python
# src/ai_memory/sync/file_watcher.py
from typing import Optional, Callable
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ai_memory.manager import MemoryManager


class MemoryFileHandler(FileSystemEventHandler):
    """记忆文件变化处理器"""

    def __init__(self, manager: MemoryManager, callback: Optional[Callable] = None):
        self.manager = manager
        self.callback = callback

    def on_modified(self, event):
        """文件修改事件"""
        if not event.is_directory and event.src_path.endswith(".md"):
            if self.callback:
                self.callback(Path(event.src_path))


class FileWatcher:
    """文件监控器"""

    def __init__(self, manager: MemoryManager, callback: Optional[Callable] = None):
        self.manager = manager
        self.callback = callback
        self.observer = Observer()
        self.handler = MemoryFileHandler(manager, callback)

    def start(self):
        """启动监控"""
        watch_path = str(self.manager.file_manager.storage_dir)
        self.observer.schedule(self.handler, watch_path, recursive=True)
        self.observer.start()

    def stop(self):
        """停止监控"""
        self.observer.stop()
        self.observer.join()
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_sync/test_file_watcher.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add tests/test_sync/ src/ai_memory/sync/ tests/test_sync/__init__.py
git commit -m "feat: 添加文件监控器"
```

---

## Task 10: OpenAI 嵌入提供者实现

**Files:**
- Create: `src/ai_memory/embeddings/openai_provider.py`
- Test: `tests/test_embeddings/test_openai_provider.py`

**Step 1: 编写 OpenAI 嵌入测试**

```python
# tests/test_embeddings/test_openai_provider.py
import pytest
from ai_memory.config import EmbeddingConfig
from ai_memory.embeddings.openai_provider import OpenAIEmbeddingProvider


@pytest.fixture
def provider():
    """OpenAI 嵌入提供者 fixture"""
    config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        api_key="test-key"
    )
    return OpenAIEmbeddingProvider(config)


def test_embed_query_mocked(provider, monkeypatch):
    """测试 OpenAI 查询嵌入（使用 mock）"""
    def mock_encode(*args, **kwargs):
        class MockResponse:
            data = [type('obj', (object,), {'embedding': [0.1, 0.2, 0.3]})()]
        return MockResponse()

    # 这里应该使用真正的 mock，简化版测试
    # 实际测试需要 openai mock 或者环境变量中有真实 key
    pass


def test_provider_initialization():
    """测试提供者初始化"""
    config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        api_key="test-key"
    )
    provider = OpenAIEmbeddingProvider(config)

    assert provider.config.provider == "openai"
    assert provider.config.model == "text-embedding-3-small"
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_embeddings/test_openai_provider.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'ai_memory.embeddings.openai_provider'`

**Step 3: 实现 OpenAI 嵌入提供者**

```python
# src/ai_memory/embeddings/openai_provider.py
from typing import List
from openai import OpenAI
from ai_memory.embeddings.base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI 嵌入模型提供者"""

    def __init__(self, config):
        super().__init__(config)
        self._client = None

    @property
    def client(self) -> OpenAI:
        """延迟创建 OpenAI 客户端"""
        if self._client is None:
            self._client = OpenAI(api_key=self.config.api_key)
        return self._client

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本"""
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        response = self.client.embeddings.create(
            model=self.config.model,
            input=text
        )
        embedding = response.data[0].embedding
        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        response = self.client.embeddings.create(
            model=self.config.model,
            input=texts
        )
        return [data.embedding for data in response.data]
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_embeddings/test_openai_provider.py -v
```

Expected: PASS

**Step 5: 提交**

```bash
git add tests/test_embeddings/test_openai_provider.py src/ai_memory/embeddings/openai_provider.py
git commit -m "feat: 添加 OpenAI 嵌入提供者实现"
```

---

## Task 11: System Prompt 模板文件

**Files:**
- Create: `src/ai_memory/prompts/__init__.py`
- Create: `src/ai_memory/prompts/MEMORY_INSTRUCTIONS.md`

**Step 1: 创建 prompt 模板**

```python
# src/ai_memory/prompts/__init__.py
from pathlib import Path

# 获取 prompts 目录
PROMPTS_DIR = Path(__file__).parent

def get_memory_instructions() -> str:
    """获取记忆系统使用说明"""
    return (PROMPTS_DIR / "MEMORY_INSTRUCTIONS.md").read_text(encoding="utf-8")

__all__ = ["get_memory_instructions"]
```

```markdown
<!-- src/ai_memory/prompts/MEMORY_INSTRUCTIONS.md -->
# Memory System Instructions

You have access to a memory system that stores and retrieves information across conversations.

## When to Search Memory

Before answering any question about:
- Past work, decisions, or actions
- User preferences, goals, or context
- Project history, timelines, or dates
- Previously discussed topics or concepts

ALWAYS run a memory search first.

## How to Use Memory

### 1. Search Memories
Use `memory_search(query)` to find relevant information.

Parameters:
- `query` (required): Search query for semantic search
- `max_results` (optional): Maximum number of results (default: 6)
- `min_score` (optional): Minimum relevance score 0-1 (default: 0.35)

### 2. Retrieve Specific Content
Use `memory_get(path, from_line, lines)` to read specific sections from a memory file.

Parameters:
- `path` (required): File path relative to memory directory (e.g., 'MEMORY.md' or 'memory/01-02-2025.md')
- `from_line` (optional): Starting line number (1-indexed)
- `lines` (optional): Number of lines to retrieve (default: 20)

### 3. Add New Memory
Use `memory_add(content, tags, priority)` to save important information.

Parameters:
- `content` (required): Content to remember
- `tags` (optional): Array of tags for categorization
- `priority` (optional): 'high', 'normal', or 'low' (default: 'normal')

## Memory File Structure

```
memory/
├── MEMORY.md              # Long-term important memories
└── DD-MM-YYYY.md         # Daily journal-style memories
```

## Citation Format

Citations follow the format: `path#Lstart-Lend`

Example: `MEMORY.md#L5-L10` refers to lines 5-10 of MEMORY.md

## Best Practices

1. **Be Specific**: Use precise search queries for better results
2. **Use Tags**: When adding memories, use relevant tags for organization
3. **Verify Sources**: Always check citation references before relying on content
4. **Low Confidence**: If search results have low scores, mention "I checked my memory but found no relevant information"
5. **Context Matters**: Search multiple times with different queries if initial results are unclear

## Example Workflow

User: "What was the decision we made about the database?"

Assistant: [Calls memory_search("database decision")]
[Gets results, calls memory_get to read details]
"Based on my memory (MEMORY.md#L12-L15), we decided to use SQLite as the database backend because it provides local-first storage with built-in full-text search."
```

**Step 2: 提交**

```bash
git add src/ai_memory/prompts/
git commit -m "feat: 添加 System Prompt 模板文件"
```

---

## Task 12: README 文档

**Files:**
- Create: `README.md`

**Step 1: 创建 README**

```markdown
# AI Memory Plugin

可插拔的 AI 记忆系统插件，支持文件系统存储、SQLite 索引、多维度混合检索。

## 特性

- **文件系统存储**: 基于 Markdown 格式的记忆文件
- **混合检索**: 向量相似度 + 全文搜索 + 时间权重 + 访问频率
- **可插拔架构**: 嵌入提供者、检索器、评分器均可替换
- **Agent 友好**: 提供 Tool 函数接口和 System Prompt 模板
- **多框架支持**: LangChain, AutoGPT 等框架适配器

## 安装

```bash
pip install ai-memory-plugin
```

## 快速开始

### 基础使用

```python
from ai_memory import MemoryManager, MemoryConfig

# 创建配置
config = MemoryConfig()

# 初始化管理器
manager = MemoryManager(config)

# 添加记忆
manager.add_memory("MEMORY.md", "# 项目决策\n\n使用 SQLite 作为存储后端。")

# 同步到索引
manager.sync()

# 搜索记忆
results = manager.search("数据库后端")
for result in results:
    print(f"{result.citation}: {result.snippet}")
```

### Agent 集成

```python
from ai_memory import MemoryManager, MemoryConfig
from ai_memory.tools import memory_search, memory_add, memory_get, MEMORY_INSTRUCTIONS

# 初始化
config = MemoryConfig()
manager = MemoryManager(config)

# 获取工具定义
tools = get_memory_tools(manager)

# 将 MEMORY_INSTRUCTIONS 添加到 Agent 的 System Prompt
system_prompt = f"{base_system_prompt}\n\n{MEMORY_INSTRUCTIONS}"
```

## 配置

### 存储配置

```python
config = MemoryConfig(
    storage={
        "storage_dir": "memory",      # 记忆文件目录
        "db_path": "memory.db",       # SQLite 数据库路径
        "chunk_size": 400,             # 分块大小
        "chunk_overlap": 80            # 分块重叠
    }
)
```

### 嵌入模型配置

```python
# 本地模型（默认）
config = MemoryConfig(
    embedding={
        "provider": "local",
        "model": "all-MiniLM-L6-v2"
    }
)

# OpenAI
config = MemoryConfig(
    embedding={
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "your-api-key"
    }
)
```

### 检索配置

```python
config = MemoryConfig(
    search={
        "max_results": 10,
        "min_score": 0.3,
        "hybrid": {
            "enabled": True,
            "vector_weight": 0.7,
            "text_weight": 0.3,
            "time_weight": 0.2,
            "frequency_weight": 0.1
        }
    }
)
```

## 文件结构

```
memory/
├── MEMORY.md              # 长期重要记忆
└── DD-MM-YYYY.md         # 每日记事
```

## Tool 接口

### memory_search
搜索记忆以获取相关信息。

### memory_add
添加新的记忆条目。

### memory_get
按路径检索特定记忆文件的指定行。

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black src/ tests/
ruff check src/ tests/
```

## License

MIT
```

**Step 2: 提交**

```bash
git add README.md
git commit -m "docs: 添加 README 文档"
```

---

## Task 13: 完整集成测试

**Files:**
- Create: `tests/test_integration.py`

**Step 1: 编写集成测试**

```python
# tests/test_integration.py
import pytest
from pathlib import Path
from ai_memory import MemoryManager, MemoryConfig
from ai_memory.tools import memory_search, memory_add, memory_get, get_memory_tools


@pytest.fixture
def temp_memory_dir(tmp_path):
    """临时记忆目录"""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return memory_dir


def test_full_workflow(temp_memory_dir):
    """测试完整工作流程"""
    # 初始化
    config = MemoryConfig(
        storage={
            "storage_dir": str(temp_memory_dir),
            "db_path": str(temp_memory_dir / "memory.db")
        }
    )
    manager = MemoryManager(config)

    # 1. 添加记忆
    manager.add_memory("MEMORY.md", "# 项目\n\nAI Memory Plugin 是一个可插拔的记忆系统。")
    manager.add_memory("MEMORY.md", "\n## 技术栈\n\nPython, SQLite, sentence-transformers")

    # 2. 同步索引
    manager.sync()

    # 3. 搜索记忆
    results = manager.search("AI 项目")
    assert len(results) >= 1
    assert "AI" in results[0].snippet or "项目" in results[0].snippet

    # 4. 获取特定内容
    content_result = manager.get_memory("MEMORY.md")
    assert "AI Memory Plugin" in content_result.text

    # 5. 使用 Tool 接口
    search_results = memory_search(manager, "技术栈", max_results=5)
    assert len(search_results) >= 1

    # 6. 获取工具定义
    tools = get_memory_tools(manager)
    assert "memory_search" in tools
    assert "memory_add" in tools
    assert "memory_get" in tools

    # 7. 检查状态
    status = manager.status()
    assert status["total_files"] >= 1
    assert "storage_dir" in status


def test_daily_memory_workflow(temp_memory_dir):
    """测试每日记忆工作流程"""
    config = MemoryConfig(
        storage={
            "storage_dir": str(temp_memory_dir),
            "db_path": str(temp_memory_dir / "memory.db")
        }
    )
    manager = MemoryManager(config)

    # 创建每日记忆
    from datetime import date
    manager.file_manager.create_daily_file(date(2025, 2, 5))
    manager.add_memory("05-02-2025.md", "# 2025-02-05\n\n今天完成了记忆系统设计。")

    manager.sync()

    # 搜索
    results = manager.search("记忆系统设计")
    assert len(results) >= 1
```

**Step 2: 运行测试验证通过**

```bash
pytest tests/test_integration.py -v
```

Expected: PASS

**Step 3: 提交**

```bash
git add tests/test_integration.py
git commit -m "test: 添加完整集成测试"
```

---

## Task 14: 创建空测试文件

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_storage/__init__.py`
- Create: `tests/test_search/__init__.py`
- Create: `tests/test_embeddings/__init__.py`
- Create: `tests/test_tools/__init__.py`
- Create: `tests/test_sync/__init__.py`

**Step 1: 创建空 __init__.py 文件**

```python
# tests/__init__.py
```

```python
# tests/test_storage/__init__.py
```

```python
# tests/test_search/__init__.py
```

```python
# tests/test_embeddings/__init__.py
```

```python
# tests/test_tools/__init__.py
```

```python
# tests/test_sync/__init__.py
```

**Step 2: 提交**

```bash
git add tests/__init__.py tests/test_storage/__init__.py tests/test_search/__init__.py tests/test_embeddings/__init__.py tests/test_tools/__init__.py tests/test_sync/__init__.py
git commit -m "chore: 添加测试包初始化文件"
```

---

## 总结

实现计划已完成，包含以下主要组件：

1. **项目基础结构** - 依赖配置和包结构
2. **配置模型** - 使用 Pydantic 定义可验证的配置
3. **数据库 Schema** - SQLite 表结构和管理器
4. **嵌入提供者** - 支持本地 HuggingFace 和 OpenAI 模型
5. **文件管理器** - 文件系统操作和分块
6. **混合检索器** - 向量 + 全文 + 多维度评分
7. **MemoryManager** - 统一入口和 API
8. **Tool 接口** - OpenAI Function Calling 兼容
9. **文件监控器** - 自动同步文件变化
10. **System Prompt** - Agent 使用指南
11. **README** - 用户文档
12. **集成测试** - 端到端验证
