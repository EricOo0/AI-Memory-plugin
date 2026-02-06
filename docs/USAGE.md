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
