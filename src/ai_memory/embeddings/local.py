"""本地嵌入提供者（使用 sentence-transformers）"""

from typing import List
from pathlib import Path
import logging
import shutil

from ai_memory.embeddings.base import EmbeddingProvider

# 配置日志
logger = logging.getLogger(__name__)

# 模型缓存目录
CACHE_DIR = Path.home() / ".cache" / "ai-memory" / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 支持的模型配置
SUPPORTED_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "name": "MiniLM-L6-v2",
        "dimensions": 384,
        "description": "快速、轻量，适用于实时应用",
        "cache_key": "minilm-l6-v2"
    },
    "BAAI/bge-large-zh-v1.5": {
        "name": "bge-large-zh",
        "dimensions": 1024,
        "description": "大规模中文 embedding，质量更高",
        "cache_key": "bge-large-zh"
    },
}


class DownloadProgress:
    """下载进度跟踪器"""
    def __init__(self, model_name: str, cache_key: str):
        self.model_name = model_name
        self.cache_key = cache_key
        self.downloaded = 0
        self.total = 0
        self.last_percent = -1

    def __call__(self, block_num, block_size, total_size):
        """HuggingFace Hub 下载回调"""
        self.downloaded += block_size
        self.total = total_size

        if total_size > 0:
            percent = int(100 * self.downloaded / total_size)
            # 只在进度变化时打印，避免刷屏
            if percent != self.last_percent and percent % 5 == 0:
                self.last_percent = percent
                mb_downloaded = self.downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                logger.info(f"下载 {self.model_name}: {percent}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)")

    def mark_complete(self):
        """标记下载完成"""
        logger.info(f"✓ {self.model_name} 下载完成")
        # 创建完成标记文件
        complete_file = CACHE_DIR / f"{self.cache_key}.complete"
        complete_file.touch()


class LocalEmbeddingProvider(EmbeddingProvider):
    """本地嵌入提供者（使用 sentence-transformers）"""

    DEFAULT_MODEL = "BAAI/bge-large-zh-v1.5"  # 默认使用 BGE 中文模型
    CHINESE_MODEL = "BAAI/bge-large-zh-v1.5"  # 中文场景推荐

    def __init__(self, model: str = None, use_cache: bool = True):
        super().__init__(model or self.DEFAULT_MODEL)
        self._model = None
        self.use_cache = use_cache

    @property
    def cache_dir(self) -> Path:
        """获取当前模型的缓存目录"""
        model_info = SUPPORTED_MODELS.get(self.model, {})
        cache_key = model_info.get("cache_key", "default")
        return CACHE_DIR / cache_key

    def _is_cached(self) -> bool:
        """检查模型是否已缓存"""
        if not self.use_cache:
            return False

        cache_dir = self.cache_dir
        complete_file = cache_dir / f"{SUPPORTED_MODELS.get(self.model, {}).get('cache_key', 'default')}.complete"

        if complete_file.exists() and cache_dir.exists():
            # 检查目录是否非空
            model_files = list(cache_dir.glob("*"))
            return len(model_files) > 0
        return False

    @property
    def model_instance(self):
        """延迟加载模型，带进度显示和缓存"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            cache_dir = self.cache_dir

            if self._is_cached():
                logger.info(f"使用缓存: {self.model} → {cache_dir}")
                cache_dir.mkdir(parents=True, exist_ok=True)

                # BGE 模型需要特殊处理
                if self.model == self.CHINESE_MODEL:
                    self._model = SentenceTransformer(
                        str(cache_dir),  # 从缓存目录加载
                        trust_remote_code=True
                    )
                else:
                    self._model = SentenceTransformer(str(cache_dir))
            else:
                # 首次下载，带进度显示
                logger.info(f"首次加载模型: {self.model}")

                model_info = SUPPORTED_MODELS.get(self.model, {})
                progress = DownloadProgress(model_info.get("name", self.model), model_info.get("cache_key", "default"))

                cache_dir.mkdir(parents=True, exist_ok=True)

                # BGE 模型需要特殊处理
                if self.model == self.CHINESE_MODEL:
                    self._model = SentenceTransformer(
                        self.model,
                        cache_folder=str(CACHE_DIR),  # 使用统一缓存目录
                        trust_remote_code=True
                    )
                else:
                    self._model = SentenceTransformer(
                        self.model,
                        cache_folder=str(CACHE_DIR),
                    )

                # 暂时忽略下载进度（sentence-transformers 不直接支持进度回调）
                # HuggingFace Hub 使用自己的进度显示

                progress.mark_complete()

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

    def clear_cache(self) -> None:
        """清除当前模型的缓存"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info(f"已清除缓存: {self.cache_dir}")

    @classmethod
    def clear_all_caches(cls) -> None:
        """清除所有模型缓存"""
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            logger.info(f"已清除所有模型缓存: {CACHE_DIR}")
