"""AI Memory Plugin - 可插拔的 AI 记忆系统

提供基于文件系统的记忆存储和混合检索能力，支持：
- 时间日记式记忆组织 (MEMORY.md + memory/DD-MM-YYYY.md)
- 混合检索（向量 + 关键词）
- 多维度评分（相似度、时间、频率）
- 可插拔的嵌入模型
"""

__version__ = "0.1.0"

from ai_memory.core.manager import MemoryManager
from ai_memory.config.settings import MemoryConfig

__all__ = ["MemoryManager", "MemoryConfig"]
