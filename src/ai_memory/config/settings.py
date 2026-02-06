"""配置管理模块"""

from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
import yaml


class VectorStoreConfig(BaseModel):
    """向量存储配置"""
    backend: str = Field(default="sqlite")  # sqlite, chroma
    chroma_persist_dir: Optional[Path] = Field(default=None)
    chroma_host: Optional[str] = Field(default=None)
    chroma_port: Optional[int] = Field(default=None)


class StorageConfig(BaseModel):
    """存储配置"""
    dir: Path = Field(default=Path("memory"))
    db_name: str = Field(default="memory.db")
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)


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
