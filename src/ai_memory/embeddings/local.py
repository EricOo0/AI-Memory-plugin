"""本地嵌入提供者（使用 sentence-transformers）"""

from typing import List

from ai_memory.embeddings.base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """本地嵌入提供者（使用 sentence-transformers）"""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model: str = None):
        super().__init__(model or self.DEFAULT_MODEL)
        self._model = None

    @property
    def model_instance(self):
        """延迟加载模型"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model)
        return self._model

    def embed(self, text: str) -> List[float]:
        """生成单段文本的嵌入向量"""
        return self.model_instance.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        embeddings = self.model_instance.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def dimensions(self) -> int:
        """获取嵌入向量维度"""
        return self.model_instance.get_sentence_embedding_dimension()
