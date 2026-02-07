"""OpenAI Function Calling 工具封装"""

from typing import List, Dict, Any, Tuple
from ai_memory.tools.functions import init


def get_openai_tools(manager=None) -> Tuple[List[Dict[str, Any]], Dict[str, callable]]:
    """获取 OpenAI 工具定义和函数映射

    Args:
        manager: MemoryManager 实例，如不提供则使用全局单例

    Returns:
        (tools_list, functions_dict) 元组
        - tools_list: OpenAI 工具定义列表（包含搜索、长期记忆、短期记忆、获取 4 个工具）
        - functions_dict: 函数名到函数的映射
    """
    # 确保管理器已初始化
    if manager is None:
        init()

    from ai_memory.tools.functions import (
        memory_search,
        memory_add_long_term,
        memory_add_daily,
        memory_get
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "搜索存储的记忆，用于查找过去的工作、决策、用户偏好或项目历史。在回答关于过去事件、用户偏好、项目信息的问题时，应该首先使用此工具搜索相关记忆。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "最大结果数（默认：6）",
                            "default": 6
                        },
                        "min_score": {
                            "type": "number",
                            "description": "最小相关度分数 0-1（默认：0.35）",
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
                "name": "memory_add_long_term",
                "description": (
                    "添加长期记忆到 MEMORY.md。用于保存需要长期保留的结构化信息。\n\n"
                    "适用场景：\n"
                    "1. 用户偏好和设置：界面主题、工作习惯、交互方式、语言偏好等\n"
                    "2. 项目核心信息：技术栈、架构设计、依赖关系、项目背景等\n"
                    "3. 重要决策和里程碑：架构选型、技术方案、重大变更的原因等\n"
                    "4. 工作流程和规范：开发流程、代码规范、发布流程等\n"
                    "5. 联系人信息：团队成员、重要联系人、协作者信息等\n\n"
                    "示例：\n"
                    "- '用户喜欢使用深色主题的界面' → tags=['preference', 'ui']\n"
                    "- '后端使用 Python 3.8+ 和 FastAPI 框架' → tags=['project', 'tech-stack']\n"
                    "- '决定使用 ChromaDB 作为向量存储' → tags=['decision', 'architecture']"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "记忆内容"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "可选的分类标签（如：preference, project, decision, workflow, contact）"
                        },
                        "category": {
                            "type": "string",
                            "description": "可选的分类章节（用户偏好、项目信息、重要决策、工作流程、联系人信息）。如不指定，将根据标签自动推断。"
                        }
                    },
                    "required": ["content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "memory_add_daily",
                "description": (
                    "添加短期记忆到今日文件。用于保存临时性、可能过期的信息。\n\n"
                    "适用场景：\n"
                    "1. 对话上下文和进度：讨论内容、当前任务、临时想法等\n"
                    "2. 调试和排查信息：错误日志、排查过程、临时发现等\n"
                    "3. 每日活动记录：今天做了什么、遇到的问题、待办事项等\n"
                    "4. 不确定是否需要长期保留的信息（默认选项）\n\n"
                    "示例：\n"
                    "- '今天讨论了认证模块的实现方案' → tags=['discussion', 'auth']\n"
                    "- '遇到 ChromaDB 连接超时错误' → tags=['debug', 'error']\n"
                    "- '完成了用户注册功能的单元测试' → tags=['progress', 'testing']"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "记忆内容"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "可选的分类标签"
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
                "description": "获取特定记忆文件的内容。用于读取已保存的记忆文件的完整内容或特定章节。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "文件路径（如：'MEMORY.md' 或 'memory/07-02-2026.md'）"
                        },
                        "from_line": {
                            "type": "integer",
                            "description": "起始行号（1-indexed，可选）"
                        },
                        "lines": {
                            "type": "integer",
                            "description": "获取行数（默认：20）",
                            "default": 20
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    ]

    functions = {
        "memory_search": memory_search,
        "memory_add_long_term": memory_add_long_term,
        "memory_add_daily": memory_add_daily,
        "memory_get": memory_get
    }

    return tools, functions


def execute_tool_calls(tool_calls, functions_dict) -> List[Any]:
    """执行 OpenAI 的工具调用

    Args:
        tool_calls: OpenAI 返回的 tool_calls 列表
        functions_dict: 函数名到函数的映射

    Returns:
        工具执行结果列表
    """
    import json
    results = []
    for call in tool_calls:
        func_name = call.function.name
        func_args = json.loads(call.function.arguments)
        func = functions_dict[func_name]
        result = func(**func_args)
        results.append(result)
    return results
