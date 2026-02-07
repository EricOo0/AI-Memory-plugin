"""AI Memory Plugin - 可插拔的 AI 记忆系统

提供基于文件系统的记忆存储和混合检索能力，支持：
- 时间日记式记忆组织 (MEMORY.md + memory/DD-MM-YYYY.md)
- 混合检索（向量 + 关键词）
- 多维度评分（相似度、时间、频率）
- 可插拔的嵌入模型
"""

__version__ = "0.2.0"

# 核心管理器
from ai_memory.core.manager import MemoryManager
from ai_memory.config.settings import MemoryConfig

# 工具函数（推荐用户使用方式）
from ai_memory.tools.functions import (
    init,
    memory_search,
    memory_add,
    memory_add_long_term,
    memory_add_daily,
    memory_get,
    init_with_chroma,
    init_with_sqlite,
)
from ai_memory.tools import get_langchain_tools, get_openai_tools, execute_tool_calls
from ai_memory.tools.system_prompt import get_system_prompt, get_agent_instructions

__all__ = [
    # 核心类
    "MemoryManager",
    "MemoryConfig",
    # 工具函数
    "init",
    "memory_search",
    "memory_add",
    "memory_add_long_term",
    "memory_add_daily",
    "memory_get",
    # 便捷配置
    "init_with_chroma",
    "init_with_sqlite",
    # System Prompt
    "get_system_prompt",
    "get_agent_instructions",
    # 框架封装
    "get_langchain_tools",
    "get_openai_tools",
    "execute_tool_calls",
]
