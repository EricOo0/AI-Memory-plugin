"""记忆工具函数定义"""

from typing import Optional, List, Dict, Any

from ai_memory.core.manager import MemoryManager


class MemoryTools:
    """记忆工具类"""

    def __init__(self, manager: MemoryManager):
        self.manager = manager

    def search(
        self,
        query: str,
        max_results: int = 6,
        min_score: float = 0.2
    ) -> Dict[str, Any]:
        """搜索记忆"""
        results = self.manager.search(query, max_results, min_score)
        return {
            "query": query,
            "results": [
                {
                    "path": r.path,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "score": r.score,
                    "snippet": r.snippet,
                    "citation": r.citation
                }
                for r in results
            ],
            "count": len(results)
        }

    def add(
        self,
        content: str,
        tags: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """添加记忆"""
        path = self.manager.add_memory(content, tags)
        return {
            "status": "success",
            "path": path
        }

    def get(
        self,
        path: str,
        from_line: Optional[int] = None,
        lines: int = 20
    ) -> Dict[str, Any]:
        """获取记忆"""
        content = self.manager.get_memory(path, from_line, lines)
        return {
            "path": path,
            "content": content
        }


def get_memory_tools(manager: MemoryManager) -> Dict[str, callable]:
    """获取工具函数字典"""
    tools = MemoryTools(manager)
    return {
        "memory_search": tools.search,
        "memory_add": tools.add,
        "memory_get": tools.get
    }


def get_openai_functions() -> List[Dict[str, Any]]:
    """获取 OpenAI Function Calling 格式的工具定义"""
    return [
        {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "Search memories for relevant information about past work, decisions, user preferences, or project history",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for semantic search"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 6)",
                            "default": 6
                        },
                        "min_score": {
                            "type": "number",
                            "description": "Minimum relevance score (0-1, default: 0.35)",
                            "default": 0.35
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "memory_add",
                "description": "Add a new memory entry. Use for important decisions, user preferences, or information worth remembering",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to remember"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization"
                        }
                    },
                    "required": ["content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "memory_get",
                "description": "Retrieve specific lines from a memory file by path",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to memory directory (e.g., 'MEMORY.md' or 'memory/01-02-2025.md')"
                        },
                        "from_line": {
                            "type": "integer",
                            "description": "Starting line number (1-indexed)"
                        },
                        "lines": {
                            "type": "integer",
                            "description": "Number of lines to retrieve (default: 20)",
                            "default": 20
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    ]
