"""多维度评分器"""

from datetime import datetime
from typing import Dict


class MultiDimensionScorer:
    """多维度评分器"""

    def __init__(
        self,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        time_weight: float = 0.1,
        frequency_weight: float = 0.1
    ):
        self.vector_weight = vector_weight
        self.text_weight = text_weight
        self.time_weight = time_weight
        self.frequency_weight = frequency_weight
        self.access_counts: Dict[str, int] = {}

    def score(
        self,
        vector_score: float,
        text_score: float,
        created_at: datetime = None,
        path: str = None
    ) -> float:
        """计算综合分数"""
        # 基础分数（向量 + 文本）
        base_score = (
            self.vector_weight * vector_score +
            self.text_weight * text_score
        )

        # 时间权重（越近越高）
        time_score = self._time_score(created_at) if created_at else 1.0

        # 频率权重（访问越多越高）
        freq_score = self._frequency_score(path) if path else 1.0

        # 综合评分
        total = base_score * time_score * freq_score
        return min(max(total, 0.0), 1.0)

    def record_access(self, path: str) -> None:
        """记录访问"""
        self.access_counts[path] = self.access_counts.get(path, 0) + 1

    def _time_score(self, created_at: datetime) -> float:
        """时间分数"""
        age = (datetime.now() - created_at).days
        # 7天内有额外加分
        if age < 7:
            return 1.0 + (7 - age) * 0.05 * self.time_weight
        return 1.0

    def _frequency_score(self, path: str) -> float:
        """频率分数"""
        count = self.access_counts.get(path, 0)
        return 1.0 + min(count * 0.1, 0.5) * self.frequency_weight
