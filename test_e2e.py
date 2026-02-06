#!/usr/bin/env python3
"""
端到端测试：使用 LangChain + AI Memory Plugin

使用方法:
    python test_e2e.py --model gpt-4 --api-key YOUR_API_KEY

或者使用 ChromaDB:
    python test_e2e.py --model gpt-4 --api-key YOUR_API_KEY --use-chroma

使用自定义 API 端点:
    python test_e2e.py --api-url https://api.example.com/v1 --api-key YOUR_API_KEY
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain_openai import ChatOpenAI
    from langchain.tools import StructuredTool
    from langchain_community.chat_models import ChatOpenAI as CustomChatOpenAI
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain 未安装: {e}")
    print("请运行: pip install ai-memory[langchain] langchain langchain-openai")
    sys.exit(1)

from ai_memory import MemoryManager, MemoryConfig
from adapters.langchain import get_langchain_tools


class TestConfig(BaseModel):
    """测试配置"""
    model: str = Field(default="gpt-3.5-turbo", description="模型名称")
    api_key: str = Field(default="", description="API Key (或设置 OPENAI_API_KEY 环境变量)")
    api_url: str = Field(default="", description="自定义 API URL（可选，留空使用默认）")
    api_base: str = Field(default="", description="API Base URL（可选，留空使用默认）")
    use_chroma: bool = Field(default=False, description="使用 ChromaDB 后端")
    chroma_dir: str = Field(default="./test_chroma_data", description="ChromaDB 数据目录")
    memory_dir: str = Field(default="./test_memory", description="记忆目录")


def setup_memory(use_chroma: bool = False, chroma_dir: str = "./test_chroma_data") -> MemoryManager:
    """设置记忆管理器"""
    from ai_memory.vector.chroma_provider import ChromaVectorStore

    if use_chroma:
        print(f"使用 ChromaDB 后端，数据目录: {chroma_dir}")
        config = MemoryConfig(
            storage={
                "dir": "./test_memory",
                "vector_store": {
                    "backend": "chroma",
                    "chroma_persist_dir": chroma_dir
                }
            }
        )
    else:
        print("使用 SQLite 后端")
        config = MemoryConfig()

    return MemoryManager(config)


def run_tests(manager: MemoryManager, agent, test_chroma: bool = False):
    """运行端到端测试"""
    print("\n" + "=" * 60)
    print("端到端测试开始")
    print("=" * 60)

    # 测试 1: 添加记忆
    print("\n[测试 1] 添加记忆...")
    manager.add_memory(
        "# 项目信息\n这是一个 AI 记忆插件项目，使用 Python 开发，"
        "支持向量存储（SQLite/ChromaDB）和混合检索。",
        tags=["project", "ai-memory"]
    )
    manager.add_memory(
        "# 用户偏好\n用户喜欢使用深色主题，"
        "偏好简洁的代码风格，关注性能优化。",
        tags=["preference"]
    )
    manager.add_memory(
        "# 待办事项\n- 完成 ChromaDB 集成\n- 编写测试\n- 更新文档",
        tags=["todo"]
    )
    print("  ✓ 添加了 3 条记忆")

    # 同步到向量存储
    print("\n[测试 2] 同步到向量存储...")
    manager.sync()
    print("  ✓ 同步完成")

    # 测试 3: 通过 Agent 搜索记忆
    print("\n[测试 3] 通过 Agent 搜索记忆...")
    queries = [
        "项目的特点是什么？",
        "用户有什么偏好？",
        "有什么待办事项？"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n  查询 {i}: {query}")
        try:
            result = agent.invoke({"input": query})
            print(f"  ✓ Agent 回应: {result['output'][:200]}...")
        except Exception as e:
            print(f"  ✗ 查询失败: {e}")

    # 测试 4: 直接搜索验证
    print("\n[测试 4] 直接搜索验证...")
    test_queries = ["向量存储", "偏好", "待办"]
    for query in test_queries:
        results = manager.search(query, max_results=2)
        print(f"  查询 '{query}': 找到 {len(results)} 条结果")

    # 测试 5: 状态检查
    print("\n[测试 5] 检查系统状态...")
    status = manager.status()
    print(f"  ✓ 后端: {status.backend}")
    print(f"  ✓ 文件数: {status.files}")
    print(f"  ✓ 嵌入模型: {status.embedding_model}")

    if test_chroma:
        print("\n[附加测试] 验证 ChromaDB...")
        from ai_memory.vector.chroma_provider import ChromaVectorStore
        store = ChromaVectorStore(
            config={"chroma_persist_dir": "./test_chroma_data"}
        )
        count = store.count()
        print(f"  ✓ ChromaDB 向量数量: {count}")

    print("\n" + "=" * 60)
    print("端到端测试完成！")
    print("=" * 60)


def interactive_mode(manager: MemoryManager, agent):
    """交互模式"""
    print("\n进入交互模式（输入 'quit' 退出）...")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("退出交互模式")
                break

            print(f"\n正在查询: {user_input}")
            result = agent.invoke({"input": user_input})
            print(f"\nAgent: {result['output']}")

        except KeyboardInterrupt:
            print("\n退出交互模式")
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="AI Memory Plugin 端到端测试")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                      help="模型名称（如: gpt-4, claude-3-opus）")
    parser.add_argument("--api-key", type=str, default="",
                      help="API Key (或设置 OPENAI_API_KEY 环境变量)")
    parser.add_argument("--api-url", type=str, default="",
                      help="自定义 API URL（如: https://api.example.com/v1）")
    parser.add_argument("--api-base", type=str, default="",
                      help="API Base URL（如: https://api.example.com）")
    parser.add_argument("--use-chroma", action="store_true",
                      help="使用 ChromaDB 后端")
    parser.add_argument("--chroma-dir", type=str, default="./test_chroma_data",
                      help="ChromaDB 数据目录")
    parser.add_argument("--memory-dir", type=str, default="./test_memory",
                      help="记忆目录")
    parser.add_argument("--interactive", action="store_true",
                      help="进入交互模式")

    args = parser.parse_args()

    # 获取 API Key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("错误: 请提供 API Key")
        print("方式 1: --api-key YOUR_API_KEY")
        print("方式 2: 设置环境变量 OPENAI_API_KEY")
        sys.exit(1)

    # 设置记忆管理器
    manager = setup_memory(
        use_chroma=args.use_chroma,
        chroma_dir=args.chroma_dir
    )

    # 创建 LLM 实例（支持自定义 API）
    print("\n初始化 LLM...")
    print(f"  模型: {args.model}")

    # 检查是否使用自定义 API
    if args.api_url or args.api_base:
        print(f"  自定义 API: {args.api_url or args.api_base}")
        llm = CustomChatOpenAI(
            model=args.model,
            api_key=api_key,
            temperature=0.7,
            base_url=args.api_url if args.api_url else args.api_base
        )
    else:
        llm = ChatOpenAI(
            model=args.model,
            api_key=api_key,
            temperature=0.7
        )

    tools = get_langchain_tools(manager)
    print(f"  工具: {[tool.name for tool in tools]}")

    agent = create_openai_functions_agent(
        llm=llm,
        tools=tools,
        verbose=True
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # 运行测试或交互模式
    if args.interactive:
        interactive_mode(manager, agent_executor)
    else:
        run_tests(manager, agent_executor, test_chroma=args.use_chroma)


if __name__ == "__main__":
    main()
