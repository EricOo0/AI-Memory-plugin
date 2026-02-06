"""LangChain 框架适配器"""

from typing import List

try:
    from langchain.tools import BaseTool
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # 尝试旧版本导入
    try:
        from langchain.tools import BaseTool, StructuredTool
        from pydantic import BaseModel, Field
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        BaseTool = None
        StructuredTool = None
        BaseModel = None
        Field = None

from ai_memory.core.manager import MemoryManager


class MemorySearchSchema(BaseModel):
    """记忆搜索参数"""
    query: str = Field(description="Search query for semantic search")
    max_results: int = Field(default=6, description="Maximum number of results")
    min_score: float = Field(default=0.35, description="Minimum relevance score (0-1)")


class MemoryAddSchema(BaseModel):
    """添加记忆参数"""
    content: str = Field(description="Content to remember")
    tags: str = Field(default="", description="Comma-separated tags")


class MemoryGetSchema(BaseModel):
    """获取记忆参数"""
    path: str = Field(description="File path (e.g., 'MEMORY.md' or 'memory/01-02-2025.md')")
    from_line: int = Field(default=None, description="Starting line number")
    lines: int = Field(default=20, description="Number of lines to retrieve")


class LangChainMemoryTool(BaseTool):
    """LangChain 记忆工具"""

    name: str = "memory_search"
    description: str = "Search memories for relevant information about past work, decisions, user preferences, or project history"
    args_schema: type[BaseModel] = MemorySearchSchema

    def __init__(self, manager: MemoryManager):
        super().__init__()
        self.manager = manager

    def _run(self, query: str, max_results: int = 6, min_score: float = 0.35) -> str:
        results = self.manager.search(query, max_results, min_score)
        output = f"Found {len(results)} memories:\n\n"
        for r in results:
            output += f"- {r.citation} (score: {r.score:.2f})\n  {r.snippet[:200]}...\n\n"
        return output


def get_langchain_tools(manager: MemoryManager) -> List[BaseTool]:
    """获取 LangChain 工具列表"""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is not installed. Install it with: pip install ai-memory[langchain]"
        )

    return [
        LangChainMemoryTool(manager),
        StructuredTool.from_function(
            func=lambda content, tags: manager.add_memory(content, tags.split(",") if tags else None),
            name="memory_add",
            description="Add a new memory entry",
            args_schema=MemoryAddSchema
        ),
        StructuredTool.from_function(
            func=lambda path, from_line, lines: manager.get_memory(path, from_line, lines),
            name="memory_get",
            description="Retrieve specific lines from a memory file",
            args_schema=MemoryGetSchema
        )
    ]
