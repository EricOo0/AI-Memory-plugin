"""配置管理测试"""

import pytest
from ai_memory.config.settings import (
    MemoryConfig, StorageConfig, EmbeddingConfig,
    RetrievalConfig, SyncConfig, VectorStoreConfig
)


def test_default_config():
    """测试默认配置"""
    config = MemoryConfig()
    assert config.storage.dir.name == "memory"
    assert config.retrieval.max_results == 6
    assert config.retrieval.min_score == 0.35


def test_config_from_dict():
    """测试从字典创建配置"""
    data = {
        "storage": {"dir": "/custom/path"},
        "retrieval": {"max_results": 10}
    }
    config = MemoryConfig(**data)
    assert str(config.storage.dir) == "/custom/path"
    assert config.retrieval.max_results == 10


def test_config_from_yaml(temp_dir):
    """测试从 YAML 文件加载配置"""
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


def test_config_to_yaml(temp_dir):
    """测试保存配置到 YAML 文件"""
    config = MemoryConfig()
    yaml_file = temp_dir / "config_output.yaml"
    config.to_yaml(yaml_file)
    content = yaml_file.read_text()
    assert "storage:" in content


def test_storage_config():
    """测试存储配置"""
    config = StorageConfig(dir="/custom", db_name="test.db")
    assert str(config.dir) == "/custom"
    assert config.db_name == "test.db"


def test_embedding_config():
    """测试嵌入配置"""
    config = EmbeddingConfig(provider="local", model="test-model")
    assert config.provider == "local"
    assert config.model == "test-model"


def test_retrieval_config():
    """测试检索配置"""
    config = RetrievalConfig(max_results=10, min_score=0.5)
    assert config.max_results == 10
    assert config.min_score == 0.5
    assert config.vector_weight == 0.7


def test_sync_config():
    """测试同步配置"""
    config = SyncConfig(watch=True, debounce_ms=2000)
    assert config.watch is True
    assert config.debounce_ms == 2000


def test_vector_store_config():
    """测试向量存储配置"""
    config = VectorStoreConfig(
        backend="chroma",
        chroma_persist_dir=Path("./chroma_data")
    )
    assert config.backend == "chroma"
    assert config.chroma_persist_dir == Path("./chroma_data")


def test_vector_store_config_defaults():
    """测试向量存储配置默认值"""
    config = VectorStoreConfig()
    assert config.backend == "sqlite"
    assert config.chroma_persist_dir is None


def test_chroma_config():
    """测试 ChromaDB 配置集成"""
    config = MemoryConfig(
        storage={
            "vector_store": {
                "backend": "chroma",
                "chroma_persist_dir": "./chroma_data"
            }
        }
    )
    assert config.storage.vector_store.backend == "chroma"
    assert config.storage.vector_store.chroma_persist_dir == Path("./chroma_data")


def test_chroma_config_with_host():
    """测试 ChromaDB 远程服务器配置"""
    config = MemoryConfig(
        storage={
            "vector_store": {
                "backend": "chroma",
                "chroma_host": "localhost",
                "chroma_port": 8000
            }
        }
    )
    assert config.storage.vector_store.backend == "chroma"
    assert config.storage.vector_store.chroma_host == "localhost"
    assert config.storage.vector_store.chroma_port == 8000
