# ChromaDB 设置指南

## 本地安装

### 方式一：包安装（推荐）

```bash
pip install chromadb
```

### 方式二：源码安装

```bash
git clone https://github.com/chroma-core/chroma.git
cd chroma
pip install .
```

## Docker 部署

### 本地 Docker

```bash
docker run -p 8000:8000 chromadb/chroma:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_data:/chroma/chroma
```

```bash
docker-compose up -d
```

## 连接配置

### 本地持久化

```python
from ai_memory import MemoryConfig

config = MemoryConfig(
    storage={
        "vector_store": {
            "backend": "chroma",
            "chroma_persist_dir": "./data/chroma"
        }
    }
)
```

### 远程服务器

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

### 内存模式（测试用）

```python
config = MemoryConfig(
    storage={
        "vector_store": {
            "backend": "chroma"
            # 不设置 persist_dir 即为内存模式
        }
    }
)
```

## 性能调优

### HNSW 索引参数

```python
from ai_memory.vector.chroma_provider import ChromaVectorStore

# 创建自定义配置的 ChromaDB 客户端
import chromadb

client = chromadb.Client(Settings(
    anonymized_telemetry=False,
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_data"
))

# HNSW 参数
collection = client.create_collection(
    name="memory_chunks",
    metadata={
        "hnsw:space": "cosine",  # 距离度量：cosine, l2, ip
        "hnsw:M": 16,           # 连接数（越大越精确但越慢）
        "hnsw:ef_construction": 200  # 构建时搜索深度
    }
)
```

### 批量操作

```python
# 批量添加（推荐）
ids = [f"chunk_{i}" for i in range(1000)]
embeddings = [...]  # 1000 个向量
documents = [...]  # 1000 个文档
metadatas = [...]  # 1000 个元数据

vector_store.add(ids, embeddings, documents, metadatas)

# 避免：逐个添加
for i in range(1000):
    vector_store.add([id], [embedding], [document], [metadata])  # 慢
```

## 监控

### 查看向量数量

```python
from ai_memory.vector.chroma_provider import ChromaVectorStore

store = ChromaVectorStore(config={"chroma_persist_dir": "./chroma_data"})
print(f"总向量数: {store.count()}")
```

### 查看集合信息

```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_data")

collection = client.get_collection("memory_chunks")
print(f"向量数: {collection.count()}")
print(f"元数据: {collection.metadata}")
```

## 备份和恢复

### 备份

```bash
# ChromaDB 数据目录
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz ./chroma_data/
```

### 恢复

```bash
tar -xzf chroma_backup_20250205.tar.gz
```

## 故障排除

### 端口冲突

```bash
# 检查端口占用
lsof -i :8000

# 使用其他端口
docker run -p 8001:8000 chromadb/chroma:latest
```

### 磁盘空间不足

```python
# 清空集合
store.clear()

# 删除数据目录
rm -rf ./chroma_data/*
```

### 连接超时

```python
# 增加超时时间
import chromadb

client = chromadb.HttpClient(
    host="vector-db.example.com",
    port=8000,
    timeout=60  # 秒
)
```

## 生产环境建议

1. **使用持久化存储**：避免数据丢失
2. **定期备份**：备份 ChromaDB 数据目录
3. **监控磁盘使用**：向量数据会持续增长
4. **使用远程服务器**：支持水平扩展
5. **配置资源限制**：Docker 中设置 CPU/内存限制

```yaml
# docker-compose.yml
services:
  chromadb:
    image: chromadb/chroma:latest
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    volumes:
      - ./chroma_data:/chroma/chroma
```
