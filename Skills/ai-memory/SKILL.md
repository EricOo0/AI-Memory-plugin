---
name: ai-memory
description: 记忆系统 - 搜索、添加长期/短期记忆、获取记忆。用于跨对话存储和检索信息。
argument-hint: [search|add-long-term|add-daily|get] [参数]
allowed-tools: Bash
disable-model-invocation: true
---

# 记忆系统

你可以访问记忆系统，跨对话存储和检索信息。记忆分为**长期记忆**（MEMORY.md）和**短期记忆**（每日文件）。

## 安装方法

### 方法 1：一键安装（推荐）

在 `Skills/ai-memory/` 目录运行：

```bash
cd Skills/ai-memory
bash install.sh
```

这会自动：
1. 检查并安装 ai-memory 包（从 PyPI）
2. 检测 Claude Code 安装目录
3. 复制 skill 文件到正确位置
4. 提示重启 Claude Code

### 方法 2：手动安装

1. 安装 ai-memory 包（从 PyPI）：
   ```bash
   pip install ai-memory
   ```

2. 复制 `Skills/ai-memory/SKILL.md` 到 Claude Code 技能目录：
   - **macOS**: `~/.claude/skills/ai-memory/`
   - **Windows**: `%APPDATA%\.claude\skills\ai-memory\`
   - **Linux**: `~/.claude/skills/ai-memory/`

3. 重启 Claude Code

### 开发模式安装

如果你正在开发此项目，可以使用开发模式安装：

```bash
# 从本地源码安装（开发模式）
cd /path/to/AI-Memmory_plugin
pip install -e .
```

这会在修改代码后立即生效，无需重新安装。

## 使用方法

### 搜索记忆
搜索过去的工作、决策、用户偏好或项目历史：

```bash
/ai-memory search "项目技术栈是什么？"
```

### 添加长期记忆
保存需要长期保留的结构化信息到 MEMORY.md：

```bash
# 用户偏好
/ai-memory add-long-term "用户喜欢使用深色主题的界面" --tags preference,ui

# 项目信息
/ai-memory add-long-term "后端使用 Python 3.8+ 和 FastAPI 框架" --tags project,tech-stack

# 重要决策
/ai-memory add-long-term "决定使用 ChromaDB 作为向量存储方案" --tags decision,architecture --category "重要决策"
```

### 添加短期记忆
保存临时性、可能过期的信息到今日文件：

```bash
# 对话上下文
/ai-memory add-daily "今天讨论了认证模块的实现方案" --tags discussion,auth

# 调试信息
/ai-memory add-daily "遇到 ChromaDB 连接超时错误，可能是网络问题" --tags debug,error

# 每日活动
/ai-memory add-daily "完成了用户注册功能的单元测试" --tags progress,testing
```

### 获取记忆
读取特定记忆文件的内容：

```bash
# 获取长期记忆
/ai-memory get MEMORY.md

# 获取今日记忆
/ai-memory get memory/07-02-2026.md --from 1 --lines 20
```

## 长期 vs 短期记忆

### 📌 长期记忆 (MEMORY.md)
**用于保存需要长期保留的结构化信息：**
- 用户偏好和设置（界面主题、工作习惯等）
- 项目核心信息（技术栈、架构设计等）
- 重要决策和里程碑（架构选型、重大变更等）
- 工作流程和规范（开发流程、代码规范等）
- 联系人信息（团队成员、协作者等）

**命令**: `add-long-term`

### 📝 短期记忆 (memory/DD-MM-YYYY.md)
**用于保存临时性、可能过期的信息：**
- 对话上下文和进度（讨论内容、当前任务等）
- 调试和排查信息（错误日志、临时发现等）
- 每日活动记录（今天做了什么、遇到的问题等）
- 不确定是否需要长期保留的信息

**命令**: `add-daily`

## 何时使用记忆

在回答以下问题前，先搜索记忆：
- 过去的工作、决策或行动
- 用户偏好、目标或上下文
- 项目历史、时间线或日期
- 之前讨论的主题或概念

**决策规则**：
- 用户偏好/设置 → `add-long-term`
- 项目核心信息 → `add-long-term`
- 重要决策 → `add-long-term`
- 临时/可能过期 → `add-daily`
- 不确定 → `add-daily`（安全默认）

## 记忆格式

### MEMORY.md（长期记忆）
按 5 个章节组织：用户偏好、项目信息、重要决策、工作流程、联系人信息

### memory/DD-MM-YYYY.md（短期记忆）
按日期分文件存储

### 引用格式
`path#Lstart-Lend`（如：`MEMORY.md#L10-L15`）

## 执行脚本

所有操作通过 `ai-memory` CLI 执行（需要先安装包）：

```bash
# 方式 1：从 PyPI 安装（推荐）
pip install ai-memory

# 方式 2：从本地源码安装（开发模式）
cd /path/to/AI-Memmory_plugin
pip install -e .

# 使用 CLI
ai-memory search "查询内容"
ai-memory add-long-term "记忆内容" --tags preference
ai-memory add-daily "临时记忆" --tags debug
ai-memory get MEMORY.md --from 1 --lines 20
```

**注意**：CLI 通过已安装的 ai-memory 包提供，不依赖项目根目录。
