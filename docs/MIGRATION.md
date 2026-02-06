# 从 SQLite 迁移到 ChromaDB

## 为什么要迁移？

| 特性 | SQLite | ChromaDB |
|------|--------|----------|
| 向量搜索 | 暴力计算 O(N) | HNSW 索引 O(log N) |
| 性能 (10K) | ~1s | ~10ms |
| 可扩展性 | 受限于单机 | 支持分布式 |
| 依赖 | 无 | ChromaDB |

## 迁移步骤

### 1. 安装 ChromaDB

```bash
pip install chromadb
```

### 2. 运行迁移脚本

```bash
python scripts/migrate_to_chroma.py \
    --memory-dir ./memory \
    --chroma-dir ./chroma_data \
    --batch-size 1000
```

参数说明：
- `--memory-dir`: 记忆目录
- `--chroma-dir`: ChromaDB 数据目录
- `--batch-size`: 每批迁移的向量数量（默认 1000）

### 3. 更新配置

```python
from ai_memory import MemoryConfig, MemoryManager

# 旧配置
config = MemoryConfig(
    storage={
        "dir": "./memory",
        "vector_store": {"backend": "sqlite"}  # 旧
    }
)

# 新配置
config = MemoryConfig(
    storage={
        "dir": "./memory",
        "vector_store": {  # 新
            "backend": "chroma",
            "chroma_persist_dir": "./chroma_data"
        }
    }
)

manager = MemoryManager(config)
```

### 4. 验证迁移

```python
# 检查迁移结果
from ai_memory.vector.chroma_provider import ChromaVectorStore

store = ChromaVectorStore(
    config={"chroma_persist_dir": "./chroma_data"}
)

count = store.count()
print(f"已迁移 {count} 个向量")
```

## 迁移期间的服务可用性

迁移过程**不影响**现有服务：

1. 可以在后台运行迁移脚本
2. SQLite 服务继续正常工作
3. 迁移完成后切换配置即可

## 回滚

如果需要回滚到 SQLite：

```python
config = MemoryConfig(
    storage={
        "dir": "./memory",
        "vector_store": {"backend": "sqlite"}
    }
)
```

ChromaDB 数据可以保留，以便随时切回。

## 故障排除

### 迁移失败

```bash
# 查看详细错误
python scripts/migrate_to_chroma.py --memory-dir ./memory --chroma-dir ./chroma_data 2>&1 | tee migrate.log
```

### ChromaDB 连接失败

检查 ChromaDB 目录权限：

```bash
ls -la ./chroma_data
chmod -R 755 ./chroma_data
```

### 内存不足

减小批量大小：

```bash
python scripts/migrate_to_chroma.py --batch-size 500
```

## 性能对比

测试环境：M2 MacBook Pro, 10,000 chunks

| 操作 | SQLite | ChromaDB | 提升 |
|------|--------|----------|------|
| 添加向量 | 50ms | 100ms | - |
| 搜索 | 1000ms | 10ms | 100x |
| 批量添加 (100) | 500ms | 200ms | 2.5x |

---

## 最佳实践

### 初期开发

使用 SQLite，无需额外依赖：

```python
config = MemoryConfig()  # 默认 SQLite
```

### 生产环境

使用 ChromaDB：

```python
config = MemoryConfig(
    storage={
        "vector_store": {
            "backend": "chroma",
            "chroma_persist_dir": "./data/chroma"
        }
    }
)
```

### 分布式部署

使用 ChromaDB 远程服务器：

```python
config = MemoryConfig(
    storage={
        "vector_store": {
            "backend": "chroma",
            "chroma_host": "vector-db.example.com",
            "chroma_port": 8000
        }
    }
)
```
