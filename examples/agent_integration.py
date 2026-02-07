#!/usr/bin/env python3
"""演示如何将 AI Memory 集成到自定义 Agent 中

这个示例展示了三种集成方式：
1. 纯函数式 API（最简单）
2. LangChain 集成
3. OpenAI Function Calling 集成
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============ 方式 1: 纯函数式 API ============
def example_pure_functions():
    """最简单的集成方式 - 纯函数式 API"""
    print("\n" + "=" * 60)
    print("方式 1: 纯函数式 API（推荐用于简单场景）")
    print("=" * 60)

    from ai_memory import init, memory_search, memory_add, get_system_prompt

    # 初始化（全局单例，只需调用一次）
    init()

    # 添加记忆
    memory_add("用户的名字是 Alice，职业是软件工程师", tags=["user-info"])
    print("✓ 已添加记忆")

    # 搜索记忆
    results = memory_search("用户的职业是什么？")
    print(f"✓ 搜索到 {results['count']} 条记忆")
    if results["count"] > 0:
        print(f"  相关度最高: {results['results'][0]['snippet']}")

    # 获取 System Prompt（可添加到 Agent 的系统提示中）
    prompt = get_system_prompt()
    print(f"✓ System Prompt 长度: {len(prompt)} 字符")

    print("\n使用示例:")
    print("```python")
    print("from ai_memory import init, memory_search, memory_add")
    print()
    print("init()  # 初始化")
    print('memory_add("重要信息", tags=["tag1"])')
    print('results = memory_search("查询关键词")')
    print("```")


# ============ 方式 2: LangChain 集成 ============
def example_langchain():
    """LangChain 框架集成"""
    print("\n" + "=" * 60)
    print("方式 2: LangChain 集成")
    print("=" * 60)

    try:
        from ai_memory import init, get_langchain_tools

        # 初始化
        init()

        # 获取 LangChain 工具
        tools = get_langchain_tools()
        print(f"✓ 获取到 {len(tools)} 个 LangChain 工具:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        print("\n使用示例:")
        print("```python")
        print("from ai_memory import init, get_langchain_tools")
        print("from langchain.agents import create_react_agent, AgentExecutor")
        print("from langchain_openai import ChatOpenAI")
        print()
        print("# 初始化")
        print("init()")
        print("tools = get_langchain_tools()")
        print()
        print("# 创建 Agent")
        print('llm = ChatOpenAI(model="gpt-4")')
        print("agent = create_react_agent(llm, tools, prompt)")
        print("executor = AgentExecutor(agent=agent, tools=tools)")
        print()
        print("# 运行")
        print('executor.invoke({"input": "记住：用户喜欢深色主题"})')
        print("```")

    except ImportError:
        print("⚠️  LangChain 未安装")
        print("   安装: pip install langchain")


# ============ 方式 3: OpenAI Function Calling ============
def example_openai():
    """OpenAI Function Calling 集成"""
    print("\n" + "=" * 60)
    print("方式 3: OpenAI Function Calling 集成")
    print("=" * 60)

    try:
        from ai_memory import init, get_openai_tools

        # 初始化
        init()

        # 获取 OpenAI 工具定义
        tools, functions = get_openai_tools()
        print(f"✓ 获取到 {len(tools)} 个 OpenAI 工具:")
        for tool in tools:
            print(f"  - {tool['function']['name']}: {tool['function']['description']}")

        print("\n使用示例:")
        print("```python")
        print("from ai_memory import init, get_openai_tools, execute_tool_calls")
        print("from openai import OpenAI")
        print()
        print("# 初始化")
        print("init()")
        print("tools, functions = get_openai_tools()")
        print("client = OpenAI()")
        print()
        print("# 第一次调用（获取工具调用）")
        print("response = client.chat.completions.create(")
        print('    model="gpt-4",')
        print("    messages=[")
        print('        {"role": "system", "content": get_system_prompt()},')
        print('        {"role": "user", "content": "记住：项目使用 FastAPI"}')
        print("    ],")
        print("    tools=tools")
        print(")")
        print()
        print("# 执行工具调用")
        print("if response.choices[0].message.tool_calls:")
        print("    results = execute_tool_calls(")
        print("        response.choices[0].message.tool_calls,")
        print("        functions")
        print("    )")
        print("```")

    except ImportError:
        print("⚠️  OpenAI SDK 未安装")
        print("   安装: pip install openai")


# ============ 方式 4: 自定义 Agent ============
def example_custom_agent():
    """自定义 Agent 集成（直接使用 MemoryManager）"""
    print("\n" + "=" * 60)
    print("方式 4: 自定义 Agent（高级用法）")
    print("=" * 60)

    from ai_memory import MemoryManager, MemoryConfig

    # 自定义配置
    config = MemoryConfig(
        storage={
            "dir": "./custom-memory",
            "vector_store": {"backend": "sqlite"}
        },
        retrieval={
            "max_results": 10,
            "min_score": 0.25
        }
    )

    # 创建管理器实例
    manager = MemoryManager(config)
    print("✓ 创建了自定义配置的 MemoryManager")

    # 使用管理器 API
    manager.add_memory("自定义记忆内容", tags=["custom"])
    results = manager.search("自定义", max_results=5)
    print(f"✓ 搜索到 {len(results)} 条结果")

    print("\n使用示例:")
    print("```python")
    print("from ai_memory import MemoryManager, MemoryConfig")
    print()
    print("# 创建自定义配置")
    print("config = MemoryConfig(")
    print('    storage={"dir": "./my-memory"}')
    print(")")
    print()
    print("# 创建管理器")
    print("manager = MemoryManager(config)")
    print()
    print("# 使用管理器 API")
    print('manager.add_memory("内容", tags=["tag"])')
    print('results = manager.search("查询")')
    print("```")


# ============ 方式 5: 使用便捷配置函数 ============
def example_convenience_configs():
    """使用便捷配置函数"""
    print("\n" + "=" * 60)
    print("方式 5: 便捷配置函数")
    print("=" * 60)

    from ai_memory import init_with_chroma, init_with_sqlite

    print("✓ 提供了以下便捷函数:")
    print("  - init_with_sqlite(): SQLite 向量存储（适合小规模）")
    print("  - init_with_chroma(): ChromaDB 向量存储（适合大规模）")

    print("\n使用示例:")
    print("```python")
    print("from ai_memory import init_with_chroma, init_with_sqlite")
    print()
    print("# SQLite（轻量级，< 10K 记忆）")
    print("init_with_sqlite()")
    print()
    print("# ChromaDB 本地持久化（10K - 1M 记忆）")
    print("init_with_chroma()")
    print()
    print("# ChromaDB 远程服务器")
    print('init_with_chroma(chroma_host="localhost", chroma_port=8000)')
    print("```")


# ============ 主函数 ============
def main():
    print("\n" + "=" * 60)
    print("AI Memory Plugin - Agent 集成示例")
    print("=" * 60)

    # 运行所有示例
    example_pure_functions()
    example_langchain()
    example_openai()
    example_custom_agent()
    example_convenience_configs()

    print("\n" + "=" * 60)
    print("✓ 所有示例演示完成！")
    print()
    print("选择最适合你的集成方式:")
    print("  - 简单项目: 使用纯函数式 API")
    print("  - LangChain 项目: 使用 get_langchain_tools()")
    print("  - OpenAI 项目: 使用 get_openai_tools()")
    print("  - 自定义需求: 直接使用 MemoryManager")
    print("=" * 60)


if __name__ == "__main__":
    main()
