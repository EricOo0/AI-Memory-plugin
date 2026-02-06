"""嵌入模块初始化"""

from ai_memory.embeddings.base import EmbeddingProvider
from ai_memory.embeddings.local import LocalEmbeddingProvider

__all__ = ["EmbeddingProvider", "LocalEmbeddingProvider"]
