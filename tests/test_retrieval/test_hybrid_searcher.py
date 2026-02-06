"""混合检索器测试"""

import pytest
from datetime import datetime
from ai_memory.retrieval.hybrid_searcher import HybridSearcher
from ai_memory.retrieval.scorer import MultiDimensionScorer
from ai_memory.storage.database import Database


def test_hybrid_searcher_init(memory_dir):
    """测试混合检索器初始化"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()
    searcher = HybridSearcher(db, vector_weight=0.7, text_weight=0.3)
    assert searcher is not None
    assert searcher.db is not None


def test_scorer_init():
    """测试多维度评分器初始化"""
    scorer = MultiDimensionScorer(
        vector_weight=0.7,
        text_weight=0.3,
        time_weight=0.1,
        frequency_weight=0.1
    )
    assert scorer.vector_weight == 0.7
    assert scorer.text_weight == 0.3
    assert scorer.time_weight == 0.1
    assert scorer.frequency_weight == 0.1


def test_scorer_score_basic():
    """测试基础评分"""
    scorer = MultiDimensionScorer()
    score = scorer.score(
        vector_score=0.8,
        text_score=0.6
    )
    # 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
    assert abs(score - 0.74) < 0.01


def test_scorer_score_with_time():
    """测试带时间的评分"""
    scorer = MultiDimensionScorer(time_weight=0.1)
    score = scorer.score(
        vector_score=0.5,
        text_score=0.5,
        created_at=datetime.now()
    )
    # 基础分数 0.5，近期应该有加分
    assert score >= 0.5


def test_scorer_score_clamped():
    """测试分数被限制在 0-1 之间"""
    scorer = MultiDimensionScorer()
    score1 = scorer.score(vector_score=0.0, text_score=0.0)
    score2 = scorer.score(vector_score=1.0, text_score=1.0)
    assert 0 <= score1 <= 1
    assert 0 <= score2 <= 1


def test_scorer_record_access():
    """测试记录访问"""
    scorer = MultiDimensionScorer()
    path = "test.md"
    scorer.record_access(path)
    scorer.record_access(path)
    assert scorer.access_counts.get(path) == 2


def test_scorer_frequency_score():
    """测试频率评分"""
    scorer = MultiDimensionScorer(frequency_weight=0.1)
    path = "test.md"
    # 多次访问应该提高分数
    score1 = scorer._frequency_score(path)
    scorer.record_access(path)
    scorer.record_access(path)
    scorer.record_access(path)
    score2 = scorer._frequency_score(path)
    assert score2 > score1


def test_scorer_time_score():
    """测试时间评分"""
    scorer = MultiDimensionScorer(time_weight=0.1)
    now = datetime.now()
    recent = now
    old = datetime(2024, 1, 1)
    score_recent = scorer._time_score(recent)
    score_old = scorer._time_score(old)
    assert score_recent >= score_old


def test_hybrid_search(memory_dir):
    """测试混合搜索"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()

    # 插入测试数据
    db.insert_chunk(
        id="chunk1",
        path="MEMORY.md",
        start_line=1,
        end_line=3,
        text="This is a test about memory",
        embedding=[0.1] * 384,
        model="test"
    )

    db.insert_chunk(
        id="chunk2",
        path="MEMORY.md",
        start_line=4,
        end_line=6,
        text="Another chunk of information",
        embedding=[0.2] * 384,
        model="test"
    )

    searcher = HybridSearcher(db)
    results = searcher.search(
        query="test memory",
        query_embedding=[0.1] * 384,
        max_results=2
    )

    assert len(results) <= 2
    assert all("score" in r.__dict__ for r in results)


def test_hybrid_search_min_score_filter(memory_dir):
    """测试最小分数过滤"""
    db = Database(memory_dir / "memory.db")
    db.create_tables()

    db.insert_chunk(
        id="chunk1",
        path="MEMORY.md",
        start_line=1,
        end_line=3,
        text="Test",
        embedding=[0.0] * 384,  # 低相似度
        model="test"
    )

    searcher = HybridSearcher(db)
    results = searcher.search(
        query="test",
        query_embedding=[0.9] * 384,
        min_score=0.5
    )

    # 低相似度的结果应该被过滤掉
    assert all(r.score >= 0.5 for r in results)


def test_merge_results():
    """测试结果合并"""
    db = Database(__file__)  # 临时路径
    searcher = HybridSearcher(db)

    text_results = [
        {"id": "chunk1", "path": "test.md", "start_line": 1, "end_line": 3, "text": "content", "score": 0.5}
    ]

    vector_results = [
        {"id": "chunk1", "path": "test.md", "start_line": 1, "end_line": 3, "text": "content", "score": 0.8},
        {"id": "chunk2", "path": "test.md", "start_line": 4, "end_line": 6, "text": "other", "score": 0.6}
    ]

    merged = searcher._merge_results(text_results, vector_results)
    assert len(merged) == 2  # chunk1 和 chunk2
    # chunk1 应该有 text_score 和 vector_score
    chunk1 = next((m for m in merged if m["id"] == "chunk1"), None)
    assert chunk1 is not None
    assert chunk1["text_score"] == 0.5
    assert chunk1["vector_score"] == 0.8
