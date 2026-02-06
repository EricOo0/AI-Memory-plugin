# AI Memory Plugin

可插拔的 AI 记忆系统插件，为各类 Agent 提供基于文件系统的记忆存储和混合检索能力。

## 特性

- 基于文件系统的简单存储（MEMORY.md + memory/DD-MM-YYYY.md）
- 混合检索（向量语义 + 关键词全文搜索）
- 多维度评分（相似度、时间、访问频率）
- 可插拔的向量存储后端（SQLite / ChromaDB）
- 可插拔的嵌入模型（本地、OpenAI、自定义）
- Agent 工具接口（OpenAI Function Calling）
- 多框架适配器（LangChain）

## 安装

```bash
pip install ai-memory[chromadb]  # 包含 ChromaDB
# 或
pip install ai-memory             # 仅 SQLite
```

可选依赖：

```bash
pip install ai-memory[openai]    # OpenAI 嵌入
pip install ai-memory[langchain]   # LangChain 集成
```

## 快速开始

```python
from ai_memory import MemoryManager, MemoryConfig

# 使用默认 SQLite 后端
config = MemoryConfig()
manager = MemoryManager(config)

# 或使用 ChromaDB 后端
config = MemoryConfig(
    storage={
        "vector_store": {
            "backend": "chroma",
            "chroma_persist_dir": "./chroma_data"
        }
    }
)
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

## 向量存储后端选择

| 后端 | 性能 | 适用场景 | 依赖 |
|------|------|----------|------|
| **SQLite** | ⭐⭐ | < 10K chunks | 无额外依赖 |
| **ChromaDB** | ⭐⭐⭐⭐ | 10K - 1M chunks | ChromaDB |

### 何时使用 ChromaDB

- 数据量超过 10,000 个 chunks
- 需要更快的检索速度
- 需要水平扩展能力

## 从 SQLite 迁移到 ChromaDB

```bash
python scripts/migrate_to_chroma.py \
    --memory-dir ./memory \
    --chroma-dir ./chroma_data \
    --batch-size 1000
```

详细迁移指南请参考 [MIGRATION.md](MIGRATION.md)。

## 文档

- [使用指南](USAGE.md)
- [ChromaDB 设置指南](CHROMADB_SETUP.md)
- [迁移指南](MIGRATION.md)
- [Agent 指令](MEMORY_INSTRUCTIONS.md)
- [示例代码](../examples/)
