"""核心类型定义"""

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
