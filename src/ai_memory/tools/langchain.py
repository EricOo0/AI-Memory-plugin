"""LangChain 工具封装"""

from typing import List
from ai_memory.tools.functions import init

try:
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    StructuredTool = None
    BaseModel = None
    Field = None


class SearchInput(BaseModel):
    """搜索参数"""
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=6, description="最大结果数")
    min_score: float = Field(default=0.35, description="最小相关度")


class AddInput(BaseModel):
    """添加记忆参数（向后兼容）"""
    content: str = Field(description="记忆内容")
    tags: str = Field(default="", description="标签，逗号分隔")


class AddLongTermInput(BaseModel):
    """添加长期记忆参数"""
    content: str = Field(description="记忆内容")
    tags: str = Field(default="", description="标签，逗号分隔")
    category: str = Field(default="", description="分类章节（可选）：用户偏好、项目信息、重要决策、工作流程、联系人信息")


class AddDailyInput(BaseModel):
    """添加短期记忆参数"""
    content: str = Field(description="记忆内容")
    tags: str = Field(default="", description="标签，逗号分隔")


class GetInput(BaseModel):
    """获取记忆参数"""
    path: str = Field(description="文件路径")
    from_line: int = Field(default=None, description="起始行号")
    lines: int = Field(default=20, description="行数")


def get_langchain_tools(manager=None) -> List["StructuredTool"]:
    """获取 LangChain 工具列表

    Args:
        manager: MemoryManager 实例，如不提供则使用全局单例

    Returns:
        LangChain StructuredTool 列表（包含搜索、长期记忆、短期记忆、获取 4 个工具）
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain 未安装，运行: pip install langchain langchain-core pydantic")

    # 确保管理器已初始化
    if manager is None:
        init()

    from ai_memory.tools.functions import (
        memory_search,
        memory_add_long_term,
        memory_add_daily,
        memory_get
    )

    return [
        StructuredTool.from_function(
            func=lambda **kwargs: memory_search(**kwargs),
            name="memory_search",
            description="搜索存储的记忆，用于查找过去的工作、决策、用户偏好或项目历史",
            args_schema=SearchInput
        ),
        StructuredTool.from_function(
            func=lambda content, tags="", category="": memory_add_long_term(
                content,
                tags.split(",") if tags else None,
                category if category else None
            ),
            name="memory_add_long_term",
            description=(
                "添加长期记忆到 MEMORY.md。用于保存需要长期保留的结构化信息：\n"
                "- 用户偏好和设置（界面主题、工作习惯等）\n"
                "- 项目核心信息（技术栈、架构设计等）\n"
                "- 重要决策和里程碑（架构选型、重大变更等）\n"
                "- 工作流程和规范（开发流程、代码规范等）\n"
                "- 联系人信息（团队成员、协作者等）"
            ),
            args_schema=AddLongTermInput
        ),
        StructuredTool.from_function(
            func=lambda content, tags="": memory_add_daily(
                content,
                tags.split(",") if tags else None
            ),
            name="memory_add_daily",
            description=(
                "添加短期记忆到今日文件。用于保存临时性、可能过期的信息：\n"
                "- 对话上下文和进度（讨论内容、当前任务等）\n"
                "- 调试和排查信息（错误日志、临时发现等）\n"
                "- 每日活动记录（今天做了什么、遇到的问题等）\n"
                "- 不确定是否需要长期保留的信息"
            ),
            args_schema=AddDailyInput
        ),
        StructuredTool.from_function(
            func=memory_get,
            name="memory_get",
            description="获取特定记忆文件的内容",
            args_schema=GetInput
        )
    ]
