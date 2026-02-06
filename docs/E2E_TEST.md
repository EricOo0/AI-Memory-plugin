# 端到端测试指南

本文档介绍如何使用 `test_e2e.py` 进行端到端测试，验证 AI Memory Plugin 与 LangChain 的集成。

## 准备工作

### 1. 安装依赖

```bash
# 安装 AI Memory Plugin 及 LangChain 集成
pip install ai-memory[langchain]

# 或使用开发版本
cd /path/to/ai-memory-impl
pip install -e ".[langchain]"
```

### 2. 准备 API Key

设置 OpenAI API Key（两种方式）：

**方式一：环境变量**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**方式二：命令行参数**
```bash
python test_e2e.py --api-key your-api-key-here
```

## 运行测试

### 基础测试（SQLite 后端）

```bash
python test_e2e.py --model gpt-3.5-turbo --api-key YOUR_API_KEY
```

**测试内容：**
1. 添加 3 条记忆（项目信息、用户偏好、待办事项）
2. 同步到向量存储
3. 通过 LangChain Agent 搜索记忆
4. 直接搜索验证
5. 检查系统状态

### ChromaDB 后端测试

```bash
python test_e2e.py \
    --model gpt-3.5-turbo \
    --api-key YOUR_API_KEY \
    --use-chroma
```

### 交互模式

```bash
python test_e2e.py \
    --model gpt-4 \
    --api-key YOUR_API_KEY \
    --interactive
```

在交互模式中，你可以：
- 输入查询语句
- 查看 Agent 如何使用记忆工具
- 输入 `quit` 退出

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|----------|------|
| `--model` | gpt-3.5-turbo | OpenAI 模型名称 |
| `--api-key` | （空） | OpenAI API Key |
| `--use-chroma` | False | 使用 ChromaDB 后端 |
| `--chroma-dir` | ./test_chroma_data | ChromaDB 数据目录 |
| `--memory-dir` | ./test_memory | 记忆目录 |
| `--interactive` | False | 进入交互模式 |

## 测试示例

### 示例 1：使用 gpt-4 + SQLite

```bash
export OPENAI_API_KEY="sk-..."

python test_e2e.py --model gpt-4
```

### 示例 2：使用 gpt-3.5-turbo + ChromaDB

```bash
python test_e2e.py \
    --model gpt-3.5-turbo \
    --use-chroma \
    --chroma-dir ./my_chroma_data
```

### 示例 3：交互式对话

```bash
python test_e2e.py --interactive

# 然后输入查询：
你: 项目的特点是什么？
Agent: [搜索记忆并回答]

你: 用户有什么偏好？
Agent: [搜索记忆并回答]

你: quit
```

## 预期输出

```
============================================================
端到端测试开始
============================================================

[测试 1] 添加记忆...
  ✓ 添加了 3 条记忆

[测试 2] 同步到向量存储...
  ✓ 同步完成

[测试 3] 通过 Agent 搜索记忆...

  查询 1: 项目的特点是什么？
  ✓ Agent 回应: 根据记忆，AI 记忆插件是一个使用 Python 开发的项目...

  查询 2: 用户有什么偏好？
  ✓ Agent 回应: 用户喜欢使用深色主题...

  查询 3: 有什么待办事项？
  ✓ Agent 回应: 根据记忆，待办事项包括...

[测试 4] 直接搜索验证...
  查询 '向量存储': 找到 1 条结果
  查询 '偏好': 找到 1 条结果
  查询 '待办': 找到 1 条结果

[测试 5] 检查系统状态...
  ✓ 后端: sqlite
  ✓ 文件数: 3
  ✓ 嵌入模型: sentence-transformers/all-MiniLM-L6-v2

============================================================
端到端测试完成！
============================================================
```

## 故障排除

### 错误：LangChain 未安装

```
LangChain 未安装: No module named 'langchain'
```

**解决方案：**
```bash
pip install langchain langchain-openai
```

### 错误：API Key 缺失

```
错误: 请提供 API Key
```

**解决方案：**
```bash
# 设置环境变量
export OPENAI_API_KEY="your-key-here"

# 或使用参数
python test_e2e.py --api-key your-key-here
```

### 错误：ChromaDB 连接失败

```
添加向量失败: ...
```

**解决方案：**
```bash
# 清除旧的 ChromaDB 数据
rm -rf ./test_chroma_data

# 重新运行测试
python test_e2e.py --use-chroma
```

### 错误：导入错误

```
ImportError: cannot import name 'BaseTool' from 'langchain.tools'
```

**解决方案：** 更新到最新版本的 LangChain
```bash
pip install --upgrade langchain langchain-core
```

## 清理测试数据

测试完成后，可以删除生成的测试数据：

```bash
# 删除测试记忆
rm -rf ./test_memory

# 删除 ChromaDB 测试数据
rm -rf ./test_chroma_data
```
