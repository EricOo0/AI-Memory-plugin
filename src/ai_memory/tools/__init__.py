"""工具模块导出"""

# 核心工具函数
from ai_memory.tools.functions import init, memory_search, memory_add, memory_get

# 框架封装（可选导入）
try:
    from ai_memory.tools.langchain import get_langchain_tools
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    get_langchain_tools = None

try:
    from ai_memory.tools.openai import get_openai_tools, execute_tool_calls
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    get_openai_tools = execute_tool_calls = None

# 保留旧接口用于向后兼容
from ai_memory.tools.memory_tools import (
    MemoryTools,
    get_memory_tools,
    get_openai_functions
)
from ai_memory.tools.system_prompt import (
    get_system_prompt,
    get_agent_instructions
)

__all__ = [
    # 核心工具函数
    "init",
    "memory_search",
    "memory_add",
    "memory_get",
    # 框架封装
    "get_langchain_tools",
    "get_openai_tools",
    "execute_tool_calls",
    "LANGCHAIN_AVAILABLE",
    "OPENAI_AVAILABLE",
    # 向后兼容
    "MemoryTools",
    "get_memory_tools",
    "get_openai_functions",
    "get_system_prompt",
    "get_agent_instructions",
]
