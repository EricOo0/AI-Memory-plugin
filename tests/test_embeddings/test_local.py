"""本地嵌入提供者测试"""

import pytest
from ai_memory.embeddings.local import LocalEmbeddingProvider
from ai_memory.embeddings.base import EmbeddingProvider


def test_local_provider_init():
    """测试本地提供者初始化"""
    provider = LocalEmbeddingProvider()
    assert provider.model == "sentence-transformers/all-MiniLM-L6-v2"


def test_local_provider_init_with_model():
    """测试使用自定义模型初始化"""
    provider = LocalEmbeddingProvider(model="test-model")
    assert provider.model == "test-model"


def test_abstract_interface():
    """测试抽象接口"""
    # EmbeddingProvider 是抽象类，不能直接实例化
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_provider_dimensions():
    """测试获取嵌入维度"""
    provider = LocalEmbeddingProvider()
    # 由于需要下载模型，这里只测试方法存在
    assert hasattr(provider, 'dimensions')
    assert callable(provider.dimensions)


def test_provider_has_embed_methods():
    """测试嵌入方法存在"""
    provider = LocalEmbeddingProvider()
    assert hasattr(provider, 'embed')
    assert hasattr(provider, 'embed_batch')
    assert callable(provider.embed)
    assert callable(provider.embed_batch)


def test_local_provider_is_abstract_subclass():
    """测试本地提供者是抽象接口的子类"""
    provider = LocalEmbeddingProvider()
    assert isinstance(provider, EmbeddingProvider)


# 注意：实际的嵌入测试需要下载模型，暂时跳过
# 如果需要运行完整测试，可以取消以下注释

# @pytest.mark.skip(reason="需要下载模型，跳过")
# def test_local_provider_embed():
#     """测试单文本嵌入"""
#     provider = LocalEmbeddingProvider()
#     embedding = provider.embed("Hello world")
#     assert len(embedding) == 384  # MiniLM-L6-v2 维度
#     assert all(isinstance(x, float) for x in embedding)


# @pytest.mark.skip(reason="需要下载模型，跳过")
# def test_local_provider_embed_batch():
#     """测试批量嵌入"""
#     provider = LocalEmbeddingProvider()
#     texts = ["Hello", "World"]
#     embeddings = provider.embed_batch(texts)
#     assert len(embeddings) == 2
#     assert all(len(e) == 384 for e in embeddings)
