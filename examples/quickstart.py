#!/usr/bin/env python3
"""AI Memory Plugin 快速开始示例

这个示例展示了如何在 5 分钟内开始使用 AI Memory Plugin。
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_memory import (
    init,
    memory_search,
    memory_add,
    memory_get,
    get_system_prompt,
)


def main():
    print("=" * 60)
    print("AI Memory Plugin 快速开始示例")
    print("=" * 60)
    print()

    # ============ 步骤 1: 初始化记忆系统 ============
    print("步骤 1: 初始化记忆系统")
    print("-" * 60)

    # 自动创建 .ai-memory/ 目录
    init()
    print("✓ 记忆系统已初始化（使用默认配置）")
    print("  存储目录: .ai-memory/")
    print()

    # ============ 步骤 2: 添加记忆 ============
    print("步骤 2: 添加记忆")
    print("-" * 60)

    # 添加一些示例记忆
    memories = [
        ("用户喜欢深色主题和简洁的界面设计", ["preference", "ui"]),
        ("项目使用 FastAPI 框架和 PostgreSQL 数据库", ["tech-stack"]),
        ("上次讨论决定使用 Redis 作为缓存层", ["decision", "architecture"]),
        ("用户的工作时间是周一到周五 9:00-18:00", ["preference", "schedule"]),
    ]

    for content, tags in memories:
        result = memory_add(content, tags)
        print(f"✓ 已添加: {content[:50]}...")
        print(f"  位置: {result['path']}")

    print()

    # ============ 步骤 3: 搜索记忆 ============
    print("步骤 3: 搜索记忆")
    print("-" * 60)

    queries = [
        "用户的偏好是什么？",
        "技术栈包括哪些？",
        "关于缓存的决策",
    ]

    for query in queries:
        print(f"\n查询: {query}")
        result = memory_search(query, max_results=3, min_score=0.3)

        if result["count"] > 0:
            print(f"找到 {result['count']} 条相关记忆:")
            for i, r in enumerate(result["results"], 1):
                print(f"  {i}. [相关度: {r['score']:.2f}] {r['citation']}")
                print(f"     {r['snippet'][:80]}...")
        else:
            print("  未找到相关记忆")

    print()

    # ============ 步骤 4: 获取特定记忆 ============
    print("步骤 4: 获取特定记忆内容")
    print("-" * 60)

    # 获取 MEMORY.md 的前 10 行
    result = memory_get("MEMORY.md", from_line=1, lines=10)
    print(f"文件: {result['path']}")
    print("内容预览:")
    print(result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"])
    print()

    # ============ 步骤 5: 获取 System Prompt ============
    print("步骤 5: 获取 System Prompt（用于 Agent）")
    print("-" * 60)

    prompt = get_system_prompt()
    print("System Prompt 示例:")
    print(prompt[:300] + "...\n")

    # ============ 总结 ============
    print("=" * 60)
    print("✓ 快速开始完成！")
    print()
    print("接下来你可以:")
    print("  1. 查看 examples/langchain_example.py 学习 LangChain 集成")
    print("  2. 查看 examples/openai_example.py 学习 OpenAI 集成")
    print("  3. 查看 docs/ 目录了解更多功能")
    print()
    print("更多信息: https://github.com/your-repo/ai-memory")
    print("=" * 60)


if __name__ == "__main__":
    main()
