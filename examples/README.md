# AI Memory Plugin 示例

这个目录包含了 AI Memory Plugin 的各种使用示例。

## 快速开始

### 1. 基础示例 - 5分钟快速上手

运行快速开始示例，了解基本功能：

```bash
python examples/quickstart.py
```

这个示例展示：
- 如何初始化记忆系统
- 如何添加记忆
- 如何搜索记忆
- 如何获取特定记忆内容
- 如何获取 System Prompt

### 2. Agent 集成示例

运行集成示例，了解如何集成到你的 Agent：

```bash
python examples/agent_integration.py
```

这个示例展示五种集成方式：
1. **纯函数式 API**（最简单，推荐）
2. **LangChain 集成**
3. **OpenAI Function Calling 集成**
4. **自定义 Agent**（直接使用 MemoryManager）
5. **便捷配置函数**

## 框架集成示例

### LangChain 集成

```bash
python examples/langchain_example.py
```

展示如何将记忆系统集成到 LangChain Agent 中。

### OpenAI 集成

```bash
python examples/openai_example.py
```

展示如何使用 OpenAI Function Calling 调用记忆系统。

## 基础用法示例

```bash
python examples/basic_usage.py
```

展示核心 API 的基本用法。

## 选择合适的集成方式

| 场景 | 推荐方式 | 示例文件 |
|------|---------|---------|
| 简单项目，只需基本功能 | 纯函数式 API | `quickstart.py` |
| LangChain 项目 | LangChain 工具 | `langchain_example.py` |
| OpenAI 项目 | OpenAI Function Calling | `openai_example.py` |
| 自定义 Agent | MemoryManager | `agent_integration.py` |
| CLI 使用 | Skill | 见 `Skills/memory/` |

## 安装依赖

基础功能：
```bash
pip install -e .
```

LangChain 集成：
```bash
pip install -e .[langchain]
```

OpenAI 集成：
```bash
pip install -e .[openai]
```

完整功能：
```bash
pip install -e .[all]
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `quickstart.py` | 5分钟快速开始，最佳入门点 |
| `agent_integration.py` | 完整的集成方式演示 |
| `basic_usage.py` | 基础 API 用法 |
| `langchain_example.py` | LangChain 集成示例 |
| `openai_example.py` | OpenAI 集成示例 |

## 下一步

1. 运行 `quickstart.py` 快速了解功能
2. 根据你的项目类型选择合适的集成方式
3. 查看 `../docs/` 目录了解更多配置选项
4. 阅读 API 文档了解高级功能

## 常见问题

### Q: 如何选择向量存储后端？

- **SQLite**（默认）：适合小规模（< 10K 记忆），无额外依赖
- **ChromaDB**：适合大规模（10K - 1M 记忆），需要安装 `chromadb`

### Q: 如何配置 ChromaDB？

```python
from ai_memory import init_with_chroma

# 本地持久化
init_with_chroma()

# 远程服务器
init_with_chroma(chroma_host="localhost", chroma_port=8000)
```

### Q: 记忆存储在哪里？

默认存储在项目根目录的 `.ai-memory/` 目录：
- `MEMORY.md`：长期记忆
- `memory/DD-MM-YYYY.md`：每日记忆
- `memory.db`：SQLite 数据库
- `chroma_data/`：ChromaDB 数据（如果使用）

### Q: 如何在 CLI 中使用？

```bash
# 通过 Skill（适用于 Claude Code 等 CLI Agent）
/memory search "查询内容"
/memory add "记忆内容" --tags tag1,tag2

# 或直接使用 CLI
python -m ai_memory.cli search "查询内容"
python -m ai_memory.cli add "记忆内容" --tags tag1,tag2
```

## 获取帮助

- 查看文档: `../docs/`
- 提交问题: GitHub Issues
- 查看测试: `../tests/`
