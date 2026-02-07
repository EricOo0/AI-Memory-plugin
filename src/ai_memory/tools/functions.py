"""纯工具函数实现 - 无状态，框架无关"""

from typing import List, Optional, TypedDict

from ai_memory.core.manager import MemoryManager

# 全局单例管理器
_manager = None
DEFAULT_DIR = ".ai-memory"


# ============ 类型定义 ============

class SearchResultItem(TypedDict):
    """搜索结果项"""
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    citation: str


class SearchResponse(TypedDict):
    """搜索响应"""
    query: str
    results: List[SearchResultItem]
    count: int


class AddResponse(TypedDict):
    """添加记忆响应"""
    status: str
    path: str


class AddLongTermResponse(TypedDict):
    """添加长期记忆响应"""
    status: str
    path: str
    type: str  # "long_term"


class AddDailyResponse(TypedDict):
    """添加短期记忆响应"""
    status: str
    path: str
    type: str  # "daily"


class GetResponse(TypedDict):
    """获取记忆响应"""
    path: str
    content: str


def init(config=None, memory_dir=None):
    """初始化记忆系统（全局单例）

    Args:
        config: MemoryConfig 配置对象
        memory_dir: 记忆存储目录，默认项目根目录 .ai-memory/

    Returns:
        MemoryManager 实例
    """
    global _manager
    if _manager is None:
        if config is None:
            from ai_memory.config.settings import MemoryConfig
            config = MemoryConfig(storage={"dir": memory_dir or DEFAULT_DIR})
        _manager = MemoryManager(config)
    return _manager


def memory_search(query: str, max_results: int = 6, min_score: float = 0.35) -> SearchResponse:
    """搜索记忆

    Args:
        query: 搜索关键词
        max_results: 最大结果数，默认 6
        min_score: 最小相关度分数，默认 0.35

    Returns:
        SearchResponse: {"query": str, "results": list, "count": int}
    """
    manager = _manager or init()
    results = manager.search(query, max_results, min_score)
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


def memory_add(content: str, tags: Optional[List[str]] = None) -> AddResponse:
    """添加记忆（向后兼容，映射到短期记忆）

    Args:
        content: 记忆内容
        tags: 标签列表

    Returns:
        AddResponse: {"status": "success", "path": str}
    """
    manager = _manager or init()
    path = manager.add_memory(content, tags)
    return {"status": "success", "path": path}


def memory_add_long_term(
    content: str,
    tags: Optional[List[str]] = None,
    category: Optional[str] = None
) -> AddLongTermResponse:
    """添加长期记忆到 MEMORY.md

    用于保存需要长期保留的结构化信息：
    - 用户偏好和设置（界面主题、工作习惯、交互方式等）
    - 项目核心信息（技术栈、架构、依赖关系等）
    - 重要决策和里程碑（架构选型、重大变更等）
    - 工作流程和规范（开发流程、代码规范等）
    - 联系人信息（团队成员、重要联系人等）

    Args:
        content: 记忆内容
        tags: 标签列表（可选），用于自动推断分类
        category: 分类章节（可选），可选值："用户偏好"、"项目信息"、"重要决策"、"工作流程"、"联系人信息"

    Returns:
        AddLongTermResponse: {"status": "success", "path": "MEMORY.md", "type": "long_term"}

    Examples:
        >>> # 用户偏好
        >>> memory_add_long_term("用户喜欢使用深色主题的界面", tags=["preference", "ui"])

        >>> # 项目信息
        >>> memory_add_long_term("后端使用 Python 3.8+ 和 FastAPI 框架", tags=["project", "tech-stack"])

        >>> # 重要决策
        >>> memory_add_long_term("决定使用 ChromaDB 作为向量存储方案", tags=["decision", "architecture"])
    """
    manager = _manager or init()
    path = manager.add_long_term_memory(content, tags, category)
    return {"status": "success", "path": path, "type": "long_term"}


def memory_add_daily(
    content: str,
    tags: Optional[List[str]] = None
) -> AddDailyResponse:
    """添加短期记忆到今日文件

    用于保存临时性、可能过期的信息：
    - 对话上下文和进度（讨论内容、当前任务等）
    - 调试和排查信息（错误日志、临时发现等）
    - 每日活动记录（今天做了什么、遇到的问题等）
    - 不确定是否需要长期保留的信息

    Args:
        content: 记忆内容
        tags: 标签列表（可选）

    Returns:
        AddDailyResponse: {"status": "success", "path": "memory/DD-MM-YYYY.md", "type": "daily"}

    Examples:
        >>> # 对话上下文
        >>> memory_add_daily("今天讨论了认证模块的实现方案", tags=["discussion", "auth"])

        >>> # 调试信息
        >>> memory_add_daily("遇到 ChromaDB 连接超时错误，可能是网络问题", tags=["debug", "error"])

        >>> # 每日记录
        >>> memory_add_daily("完成了用户注册功能的单元测试", tags=["progress", "testing"])
    """
    manager = _manager or init()
    path = manager.add_daily_memory(content, tags)
    return {"status": "success", "path": path, "type": "daily"}


def memory_get(path: str, from_line: Optional[int] = None, lines: int = 20) -> GetResponse:
    """获取记忆

    Args:
        path: 文件路径
        from_line: 起始行号
        lines: 行数

    Returns:
        GetResponse: {"path": str, "content": str}
    """
    manager = _manager or init()
    content = manager.get_memory(path, from_line, lines)
    return {"path": path, "content": content}


# ============ 便捷配置函数 ============

def init_with_chroma(
    memory_dir: Optional[str] = None,
    chroma_host: Optional[str] = None,
    chroma_port: Optional[int] = None,
    chroma_persist_dir: Optional[str] = None
) -> MemoryManager:
    """快速初始化 ChromaDB 配置

    Args:
        memory_dir: 记忆存储目录，默认 .ai-memory/
        chroma_host: ChromaDB 服务器地址（None 则使用本地持久化）
        chroma_port: ChromaDB 端口
        chroma_persist_dir: 本地持久化目录，默认 ./chroma_data（仅在 chroma_host 为 None 时有效）

    Returns:
        MemoryManager 实例

    Examples:
        >>> # 本地持久化 ChromaDB
        >>> init_with_chroma()

        >>> # 远程 ChromaDB 服务器
        >>> init_with_chroma(chroma_host="localhost", chroma_port=8000)
    """
    from ai_memory.config.settings import MemoryConfig

    vector_store_config = {
        "backend": "chroma",
    }

    if chroma_host:
        # 远程服务器模式
        vector_store_config["chroma_host"] = chroma_host
        if chroma_port:
            vector_store_config["chroma_port"] = chroma_port
    else:
        # 本地持久化模式
        vector_store_config["chroma_persist_dir"] = chroma_persist_dir or "./chroma_data"

    config = MemoryConfig(
        storage={
            "dir": memory_dir or DEFAULT_DIR,
            "vector_store": vector_store_config
        }
    )
    return init(config)


def init_with_sqlite(memory_dir: Optional[str] = None) -> MemoryManager:
    """快速初始化 SQLite 向量存储配置（轻量级，适用于小规模数据）

    Args:
        memory_dir: 记忆存储目录，默认 .ai-memory/

    Returns:
        MemoryManager 实例

    Examples:
        >>> # 使用 SQLite 向量存储
        >>> init_with_sqlite()
    """
    from ai_memory.config.settings import MemoryConfig

    config = MemoryConfig(
        storage={
            "dir": memory_dir or DEFAULT_DIR,
            "vector_store": {"backend": "sqlite"}
        }
    )
    return init(config)
