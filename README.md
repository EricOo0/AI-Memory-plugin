# AI Memory Plugin

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/yourusername/ai-memory)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**可插拔的 AI 记忆系统插件** - 为各类 Agent 提供基于文件系统的记忆存储和混合检索能力。

## ✨ 特性

- 🗂️ **时间日记式记忆组织** - `MEMORY.md` + `memory/DD-MM-YYYY.md`
- 🔍 **混合检索** - 向量语义搜索 + 关键词全文搜索 + 多维度评分
- 🔌 **可插拔向量存储** - SQLite（轻量级）/ ChromaDB（高性能）
- 🎯 **多框架支持** - LangChain、OpenAI Function Calling
- 🛠️ **易于集成** - 纯函数式 API，三行代码开始使用
- 🖥️ **CLI 支持** - 适用于 Claude Code 等 CLI Agent

## 🚀 快速开始

### 安装

```bash
# 基础安装（SQLite 向量存储）
pip install -e .

# 包含 ChromaDB（推荐用于大规模数据）
pip install -e .[chroma]

# 完整安装（包含所有可选依赖）
pip install -e .[all]
```

### 5 分钟上手

```python
from ai_memory import init, memory_search, memory_add, get_system_prompt

# 1. 初始化（自动创建 .ai-memory/ 目录）
init()

# 2. 添加记忆
memory_add("用户喜欢深色主题", tags=["preference"])
memory_add("项目使用 FastAPI 框架", tags=["tech-stack"])

# 3. 搜索记忆
results = memory_search("用户的偏好是什么？")
print(f"找到 {results['count']} 条记忆")

# 4. 获取 System Prompt（用于 Agent）
prompt = get_system_prompt()
```

运行完整示例：
```bash
python examples/quickstart.py
```

## 📚 使用场景

### 1. 代码类 Agent - 纯函数式 API

```python
from ai_memory import init, memory_search, memory_add

init()  # 初始化全局单例

# 在你的 Agent 中调用
memory_add("重要决策：使用 Redis 作为缓存层", tags=["decision"])
results = memory_search("缓存相关的决策")
```

### 2. LangChain Agent

```python
from ai_memory import init, get_langchain_tools
from langchain.agents import create_react_agent

init()
tools = get_langchain_tools()  # 返回 LangChain 工具列表
agent = create_react_agent(llm, tools, prompt)
```

### 3. OpenAI Function Calling

```python
from ai_memory import init, get_openai_tools, execute_tool_calls
from openai import OpenAI

init()
tools, functions = get_openai_tools()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tools=tools
)

# 执行工具调用
if response.choices[0].message.tool_calls:
    results = execute_tool_calls(
        response.choices[0].message.tool_calls,
        functions
    )
```

### 4. CLI Agent（Skill 方式）

```bash
# 适用于 Claude Code 等支持 Skill 的 CLI Agent
/memory search "项目的技术栈"
/memory add "用户偏好使用 TypeScript" --tags preference
/memory get MEMORY.md --from 1 --lines 20
```

## 🏗️ 架构设计

```
┌─────────────────────────────────────────┐
│  接口层                                 │
│  - 纯函数式 API                         │
│  - LangChain 工具                       │
│  - OpenAI 工具                          │
│  - CLI / Skill                          │
├─────────────────────────────────────────┤
│  管理层                                 │
│  - MemoryManager（统一入口）           │
├─────────────────────────────────────────┤
│  业务层                                 │
│  - HybridSearcher（混合检索）          │
│  - MultiDimensionScorer（多维度评分）  │
│  - MemoryIndexer（同步索引）           │
├─────────────────────────────────────────┤
│  存储层                                 │
│  - FileManager（文件管理）             │
│  - Database（SQLite）                   │
│  - VectorStore（可插拔）               │
├─────────────────────────────────────────┤
│  基础层                                 │
│  - EmbeddingProvider（嵌入模型）       │
└─────────────────────────────────────────┘
```

## 🔧 配置选项

### 使用便捷配置函数

```python
from ai_memory import init_with_sqlite, init_with_chroma

# SQLite（轻量级，< 10K 记忆）
init_with_sqlite()

# ChromaDB 本地持久化（10K - 1M 记忆）
init_with_chroma()

# ChromaDB 远程服务器
init_with_chroma(chroma_host="localhost", chroma_port=8000)
```

### 自定义配置

```python
from ai_memory import MemoryManager, MemoryConfig

config = MemoryConfig(
    storage={
        "dir": "./my-memory",
        "vector_store": {
            "backend": "chroma",  # 或 "sqlite"
            "chroma_persist_dir": "./chroma_data"
        }
    },
    retrieval={
        "max_results": 10,
        "min_score": 0.25,
        "vector_weight": 0.7,  # 向量相似度权重
        "text_weight": 0.3     # 关键词匹配权重
    }
)

manager = MemoryManager(config)
```

## 📖 API 参考

### 核心函数

| 函数 | 说明 | 返回值 |
|------|------|--------|
| `init(memory_dir=None)` | 初始化记忆系统（全局单例） | `MemoryManager` |
| `memory_search(query, max_results=6, min_score=0.35)` | 搜索记忆 | `SearchResponse` |
| `memory_add(content, tags=None)` | 添加记忆 | `AddResponse` |
| `memory_get(path, from_line=None, lines=20)` | 获取记忆内容 | `GetResponse` |

### 便捷配置

| 函数 | 说明 |
|------|------|
| `init_with_sqlite(memory_dir=None)` | 快速初始化 SQLite 配置 |
| `init_with_chroma(memory_dir=None, chroma_host=None, chroma_port=None)` | 快速初始化 ChromaDB 配置 |

### System Prompt

| 函数 | 说明 |
|------|------|
| `get_system_prompt()` | 获取记忆系统 System Prompt |
| `get_agent_instructions(tools_available)` | 获取 Agent 使用指令 |

### 框架集成

| 函数 | 说明 |
|------|------|
| `get_langchain_tools(manager=None)` | 获取 LangChain 工具列表 |
| `get_openai_tools(manager=None)` | 获取 OpenAI 工具定义和函数映射 |
| `execute_tool_calls(tool_calls, functions)` | 执行 OpenAI 工具调用 |

## 🎯 混合检索策略

AI Memory Plugin 使用多维度混合检索，确保找到最相关的记忆：

1. **向量语义搜索**（权重 0.7）- 理解语义相似度
2. **关键词全文搜索**（权重 0.3）- 精确匹配关键词
3. **时间因子**（权重 0.1）- 越新的记忆越重要
4. **访问频率**（权重 0.1）- 常访问的记忆更相关

**评分公式**：
```
最终分数 = (向量分数 × 0.7 + 文本分数 × 0.3) × 时间因子 × 频率因子
```

## 📁 文件组织

```
.ai-memory/
├── MEMORY.md              # 长期记忆
├── memory/
│   ├── 07-02-2026.md     # 每日记忆
│   └── 06-02-2026.md
├── memory.db              # SQLite 数据库
└── chroma_data/          # ChromaDB 数据（可选）
```

## 🔌 向量存储对比

| 特性 | SQLite | ChromaDB |
|------|--------|----------|
| **适用规模** | < 10K 记忆 | 10K - 1M 记忆 |
| **额外依赖** | 无 | 需要安装 chromadb |
| **检索速度** | 较慢（纯 Python） | 快速（HNSW 索引） |
| **存储效率** | 一般 | 优秀 |
| **部署方式** | 仅本地 | 本地 / 远程服务器 |

**建议**：
- 开发/小项目：使用 SQLite
- 生产/大规模：使用 ChromaDB

## 🛠️ CLI 使用

### 直接使用

```bash
# 搜索
python -m ai_memory.cli search "查询内容" --max-results 6

# 添加
python -m ai_memory.cli add "记忆内容" --tags tag1,tag2

# 获取
python -m ai_memory.cli get MEMORY.md --from 1 --lines 20

# JSON 输出（适用于脚本）
python -m ai_memory.cli search "查询" --json
```

### 通过 Skill（Claude Code）

```bash
/memory search "查询内容"
/memory add "记忆内容" --tags tag1,tag2
/memory get MEMORY.md --from 1 --lines 20
```

## 📊 示例项目

查看 `examples/` 目录获取完整示例：

| 文件 | 说明 |
|------|------|
| `quickstart.py` | 5 分钟快速开始 |
| `agent_integration.py` | 五种集成方式完整演示 |
| `langchain_example.py` | LangChain 集成示例 |
| `openai_example.py` | OpenAI 集成示例 |
| `basic_usage.py` | 基础 API 用法 |

运行示例：
```bash
python examples/quickstart.py
python examples/agent_integration.py
```

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行端到端测试
python test_e2e.py
```

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 许可证

MIT License - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🔗 相关链接

- [文档](docs/)
- [示例代码](examples/)
- [问题反馈](https://github.com/yourusername/ai-memory/issues)
- [更新日志](CHANGELOG.md)

## ❓ 常见问题

### Q: 记忆数据存储在哪里？

默认存储在项目根目录的 `.ai-memory/` 目录。可以通过 `init(memory_dir="路径")` 自定义。

### Q: 如何迁移到 ChromaDB？

```python
from ai_memory.vector.migration import migrate_to_chroma

migrate_to_chroma(
    db_path=".ai-memory/memory.db",
    chroma_path="./chroma_data"
)
```

### Q: 支持哪些嵌入模型？

当前支持：
- 本地：`sentence-transformers/all-MiniLM-L6-v2`（默认）
- 可扩展：实现 `EmbeddingProvider` 接口支持其他模型

### Q: 如何清空所有记忆？

```bash
rm -rf .ai-memory/
```

或在代码中：
```python
manager = MemoryManager(config)
# 清空向量存储
manager.vector_store.clear()
```

---

**Made with ❤️ for AI Agents**
