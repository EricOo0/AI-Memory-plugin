#!/usr/bin/env python3
"""
端到端测试：模拟代码 Agent 如何使用 AI Memory Plugin

测试模式:
    1. 模拟 Agent 模式（无需 API key）- 模拟代码 agent 的工具调用流程
       python test_e2e.py

    2. 真实 LLM 模式（需 API key）- 用 OpenAI function calling 测试真实 agent 循环
       python test_e2e.py --live --api-key YOUR_KEY
       python test_e2e.py --live --api-url https://api.example.com/v1 --api-key YOUR_KEY

    3. 交互模式
       python test_e2e.py --interactive --api-key YOUR_KEY

选项:
    --use-chroma        使用 ChromaDB 后端
    --memory-dir DIR    指定记忆目录（默认 .ai-memory）
    --model MODEL       指定模型（默认 gpt-4）
    --verbose           详细输出
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_memory import MemoryManager, MemoryConfig
from ai_memory.tools.functions import memory_search, memory_add, memory_get
from ai_memory.tools.system_prompt import get_system_prompt

# 配置日志
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================
# 测试基础设施
# ============================================================

# 测试计数器
_test_passed = 0
_test_failed = 0


def assert_test(condition: bool, name: str, detail: str = ""):
    """断言测试结果"""
    global _test_passed, _test_failed
    if condition:
        _test_passed += 1
        print(f"  ✓ {name}")
    else:
        _test_failed += 1
        msg = f"  ✗ {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def print_header(title: str):
    """打印测试段落标题"""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def print_summary():
    """打印测试总结"""
    total = _test_passed + _test_failed
    print(f"\n{'=' * 60}")
    print(f"  测试结果: {_test_passed}/{total} 通过", end="")
    if _test_failed > 0:
        print(f"  ({_test_failed} 失败)")
    else:
        print("  (全部通过)")
    print(f"{'=' * 60}")
    return _test_failed == 0


# ============================================================
# 第一部分：模拟 Agent 测试（无需 API key）
# ============================================================

def setup_clean_manager(
    memory_dir: str = "./test_memory",
    use_chroma: bool = False,
    chroma_dir: str = "./test_chroma_data",
    clean: bool = True,
) -> MemoryManager:
    """创建干净的 MemoryManager 实例

    Args:
        memory_dir: 记忆存储目录
        use_chroma: 是否使用 ChromaDB
        chroma_dir: ChromaDB 数据目录
        clean: 是否清理已有数据
    """
    # 重置全局单例
    import ai_memory.tools.functions as fn
    fn._manager = None

    if clean:
        for d in [memory_dir, chroma_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)

    if use_chroma:
        config = MemoryConfig(
            storage={
                "dir": memory_dir,
                "vector_store": {
                    "backend": "chroma",
                    "chroma_persist_dir": chroma_dir,
                },
            }
        )
    else:
        config = MemoryConfig(storage={"dir": memory_dir})

    manager = MemoryManager(config)

    # 同步设置到全局单例，以便 memory_search/add/get 可用
    fn._manager = manager
    return manager


def test_scenario_1_session_start(manager: MemoryManager):
    """场景 1：代码 Agent 启动 — 检查历史记忆

    模拟: Agent 在新会话开始时，先搜索是否有之前的项目上下文。
    这是代码 agent（如 Claude Code、Cursor）启动时的典型行为。
    """
    print_header("场景 1：Agent 启动 — 检查历史记忆")

    # Agent 首先检查系统状态
    status = manager.status()
    print(f"  系统状态: 后端={status.backend}, 文件数={status.files}")
    assert_test(status.backend in ("sqlite", "chroma"), "系统状态可查询")

    # Agent 尝试搜索历史记忆（空库应该不报错）
    result = memory_search("项目架构", max_results=3, min_score=0.1)
    assert_test(isinstance(result, dict), "空库搜索返回字典")
    assert_test(result["count"] == 0, "空库搜索结果为 0 条")

    # Agent 搜索用户偏好
    result = memory_search("用户偏好设置", min_score=0.1)
    assert_test(result["count"] == 0, "空库无用户偏好")
    print("  → Agent 确认这是首次会话，无历史记忆")


def test_scenario_2_save_code_decisions(manager: MemoryManager):
    """场景 2：Agent 保存代码决策

    模拟: Agent 在帮助用户做技术选型后，把关键决策保存到记忆。
    """
    print_header("场景 2：Agent 保存代码决策")

    # 决策 1: 技术栈选型
    r1 = memory_add(
        "# 技术栈决策\n"
        "项目选用 FastAPI 作为后端框架，理由：\n"
        "- 原生 async/await 支持\n"
        "- 自动生成 OpenAPI 文档\n"
        "- 类型安全（Pydantic 集成）\n"
        "用户明确要求高性能和类型安全。",
        tags=["decision", "backend", "fastapi"],
    )
    assert_test(r1["status"] == "success", "保存技术栈决策")
    assert_test("path" in r1, "返回文件路径")
    print(f"  → 保存到: {r1['path']}")

    # 决策 2: 数据库选型
    r2 = memory_add(
        "# 数据库选型\n"
        "选择 PostgreSQL + SQLAlchemy async，理由：\n"
        "- 需要复杂查询和事务支持\n"
        "- asyncpg 驱动性能优秀\n"
        "- 团队已有 PostgreSQL 运维经验",
        tags=["decision", "database", "postgresql"],
    )
    assert_test(r2["status"] == "success", "保存数据库决策")

    # 决策 3: 代码规范
    r3 = memory_add(
        "# 代码规范\n"
        "- 使用 ruff 进行代码检查和格式化\n"
        "- 使用 mypy 进行类型检查\n"
        "- 测试框架: pytest + pytest-asyncio\n"
        "- 最小测试覆盖率: 80%",
        tags=["decision", "code-style", "testing"],
    )
    assert_test(r3["status"] == "success", "保存代码规范")

    # 用户偏好
    r4 = memory_add(
        "# 用户偏好\n"
        "- 偏好函数式编程风格，避免过度使用类继承\n"
        "- 喜欢简洁的错误处理，不要过度防御性编程\n"
        "- 注释语言：中文\n"
        "- 编辑器: VS Code + Vim 模式",
        tags=["preference", "user"],
    )
    assert_test(r4["status"] == "success", "保存用户偏好")

    # 确保所有记忆完整索引
    manager.sync()

    print(f"  → 共保存 4 条记忆（已同步索引）")


def test_scenario_3_search_context(manager: MemoryManager):
    """场景 3：Agent 在编码过程中搜索相关上下文

    模拟: 用户问 "帮我搭建 API 框架"，Agent 先搜索记忆获取之前的决策。
    """
    print_header("场景 3：Agent 搜索编码上下文")

    # Agent 搜索后端框架相关记忆
    result = memory_search("FastAPI 后端框架 API", max_results=5, min_score=0.1)
    assert_test(result["count"] > 0, "搜索到后端框架记忆")
    if result["count"] > 0:
        top = result["results"][0]
        assert_test(top["score"] > 0, "搜索结果有相关性分数")
        assert_test(len(top["snippet"]) > 0, "搜索结果包含摘要")
        print(f"  → 最佳匹配: score={top['score']:.2f}, 来源={top['citation']}")

    # Agent 搜索数据库相关记忆
    result = memory_search("PostgreSQL 数据库", max_results=3, min_score=0.1)
    assert_test(result["count"] > 0, "搜索到数据库记忆")
    if result["count"] > 0:
        # 检查是否包含 PostgreSQL 相关内容
        snippets = " ".join(r["snippet"] for r in result["results"])
        has_pg = "PostgreSQL" in snippets or "postgresql" in snippets.lower()
        assert_test(has_pg, "搜索结果包含 PostgreSQL 决策")

    # Agent 搜索用户偏好
    result = memory_search("用户偏好编码风格", min_score=0.1)
    assert_test(result["count"] > 0, "搜索到用户偏好")

    # Agent 搜索不存在的内容
    result = memory_search("Kubernetes 部署配置", max_results=3, min_score=0.5)
    print(f"  → 无关搜索结果: {result['count']} 条（期望低匹配）")


def test_scenario_4_get_memory_detail(manager: MemoryManager):
    """场景 4：Agent 获取记忆详情

    模拟: Agent 搜索到相关记忆后，通过 memory_get 获取完整内容。
    """
    print_header("场景 4：Agent 获取记忆详情")

    # 先搜索
    result = memory_search("技术栈", max_results=1, min_score=0.1)
    assert_test(result["count"] > 0, "搜索到记忆")

    if result["count"] > 0:
        path = result["results"][0]["path"]
        # 获取完整内容
        detail = memory_get(path)
        assert_test("content" in detail, "获取到记忆内容")
        assert_test(len(detail["content"]) > 0, "记忆内容非空")
        print(f"  → 文件: {detail['path']}, 长度: {len(detail['content'])} 字符")

        # 获取指定行范围
        detail_partial = memory_get(path, from_line=1, lines=3)
        assert_test(len(detail_partial["content"]) > 0, "获取部分内容成功")
        line_count = len(detail_partial["content"].strip().split("\n"))
        assert_test(line_count <= 3, f"行数限制生效（返回 {line_count} 行）")

    # 获取今日记忆文件（记忆默认写入 memory/DD-MM-YYYY.md）
    from datetime import datetime
    today = datetime.now().strftime("%d-%m-%Y")
    daily_path = f"memory/{today}.md"
    daily_memory = memory_get(daily_path)
    assert_test("content" in daily_memory, f"获取 {daily_path} 成功")
    assert_test(len(daily_memory["content"]) > 0, "今日记忆文件非空")
    print(f"  → {daily_path} 长度: {len(daily_memory['content'])} 字符")


def test_scenario_5_multi_turn_accumulation(manager: MemoryManager):
    """场景 5：多轮对话中 Agent 持续积累记忆

    模拟: Agent 在多轮对话中不断保存新的上下文，并能检索到之前保存的内容。
    """
    print_header("场景 5：多轮对话记忆积累")

    # 第 1 轮：用户描述需求，Agent 保存
    memory_add(
        "# 需求：用户认证模块\n"
        "- 支持 JWT + Refresh Token\n"
        "- OAuth2 集成（Google, GitHub）\n"
        "- 基于角色的权限控制（RBAC）",
        tags=["requirement", "auth"],
    )
    print("  轮次 1: 保存用户认证需求")

    # 第 2 轮：用户确认实现细节，Agent 继续保存
    memory_add(
        "# 认证实现细节\n"
        "- JWT 过期时间: access=15min, refresh=7d\n"
        "- 密码哈希: bcrypt, rounds=12\n"
        "- Rate limit: 登录接口 5次/分钟",
        tags=["implementation", "auth"],
    )
    print("  轮次 2: 保存认证实现细节")

    # 第 3 轮：Agent 回忆之前的需求来写代码
    result = memory_search("JWT 认证 Token 权限", min_score=0.1)
    assert_test(result["count"] >= 1, f"搜索到多轮积累的记忆（{result['count']} 条）")

    # 验证能搜到相关内容
    all_snippets = " ".join(r["snippet"] for r in result["results"])
    assert_test(
        "RBAC" in all_snippets or "JWT" in all_snippets or "bcrypt" in all_snippets,
        "搜索到认证相关内容"
    )

    # 第 4 轮：Agent 保存实现成果
    memory_add(
        "# 完成：认证模块实现\n"
        "- 文件: src/auth/jwt.py, src/auth/oauth.py, src/auth/rbac.py\n"
        "- 测试: tests/test_auth.py (覆盖率 92%)\n"
        "- 状态: 已合并到 main 分支",
        tags=["progress", "auth", "done"],
    )
    print("  轮次 3-4: Agent 保存实现成果")

    # 验证完整的记忆链
    result = memory_search("认证模块", max_results=10, min_score=0.1)
    assert_test(result["count"] >= 1, f"认证模块记忆链可检索（{result['count']} 条）")
    print(f"  → 多轮积累: 共检索到 {result['count']} 条相关记忆")


def test_scenario_6_cross_session(
    memory_dir: str,
    use_chroma: bool,
    chroma_dir: str,
):
    """场景 6：跨会话记忆恢复

    模拟: Agent 关闭后重新启动，应该能检索到之前会话保存的记忆。
    这是代码 agent 最核心的价值 — 跨会话持久记忆。
    """
    print_header("场景 6：跨会话记忆恢复")

    # 创建新的 Manager 实例（模拟新会话），但不清理数据
    manager2 = setup_clean_manager(
        memory_dir=memory_dir,
        use_chroma=use_chroma,
        chroma_dir=chroma_dir,
        clean=False,  # 不清理，模拟重启
    )

    # 新会话中搜索之前保存的记忆（SQLite 数据已持久化，无需重新 sync）
    result = memory_search("FastAPI 后端框架", max_results=3, min_score=0.1)
    assert_test(result["count"] > 0, "新会话能搜索到历史记忆")

    result = memory_search("用户偏好编码风格", min_score=0.1)
    assert_test(result["count"] > 0, "新会话能搜索到用户偏好")

    result = memory_search("认证模块", min_score=0.1)
    assert_test(result["count"] > 0, "新会话能搜索到项目进度")

    # 检查文件状态
    status = manager2.status()
    assert_test(status.files > 0, f"新会话文件数 > 0（实际: {status.files}）")
    print(f"  → 新会话恢复: {status.files} 个记忆文件, 后端={status.backend}")


def test_scenario_7_openai_tool_format():
    """场景 7：验证 OpenAI Function Calling 工具格式

    模拟: 代码 agent 通过 OpenAI API 的 function calling 使用记忆工具。
    这里验证工具定义格式和函数映射是否正确。
    """
    print_header("场景 7：OpenAI Function Calling 工具格式")

    from ai_memory.tools.openai import get_openai_tools, execute_tool_calls

    tools, functions = get_openai_tools()

    # 验证工具定义
    assert_test(len(tools) >= 4, f"至少导出 4 个工具（实际: {len(tools)}）")

    tool_names = {t["function"]["name"] for t in tools}
    assert_test("memory_search" in tool_names, "包含 memory_search 工具")
    assert_test("memory_add_long_term" in tool_names, "包含 memory_add_long_term 工具")
    assert_test("memory_add_daily" in tool_names, "包含 memory_add_daily 工具")
    assert_test("memory_get" in tool_names, "包含 memory_get 工具")

    # 验证工具定义符合 OpenAI 格式
    for tool in tools:
        assert_test(tool["type"] == "function", f"{tool['function']['name']} type=function")
        func_def = tool["function"]
        assert_test("description" in func_def, f"{func_def['name']} 有描述")
        assert_test("parameters" in func_def, f"{func_def['name']} 有参数定义")
        params = func_def["parameters"]
        assert_test(params["type"] == "object", f"{func_def['name']} 参数类型为 object")
        assert_test("required" in params, f"{func_def['name']} 有 required 字段")

    # 验证函数映射
    assert_test(len(functions) >= 4, "函数映射包含至少 4 个函数")
    for name in ["memory_search", "memory_add_long_term", "memory_add_daily", "memory_get"]:
        assert_test(callable(functions[name]), f"{name} 是可调用的")

    # 模拟 OpenAI tool_call 执行
    class MockToolCall:
        """模拟 OpenAI 的 tool_call 对象"""
        def __init__(self, name: str, arguments: str):
            self.id = f"call_mock_{name}"
            self.function = type("Function", (), {"name": name, "arguments": arguments})()

    mock_calls = [
        MockToolCall("memory_search", json.dumps({"query": "FastAPI"})),
        MockToolCall("memory_add_daily", json.dumps({
            "content": "# 测试记忆\n通过 OpenAI tool call 添加",
            "tags": ["test"],
        })),
    ]

    results = execute_tool_calls(mock_calls, functions)
    assert_test(len(results) == 2, "执行了 2 个工具调用")
    assert_test(results[0]["count"] >= 0, "memory_search 返回结果")
    assert_test(results[1]["status"] == "success", "memory_add_daily 执行成功")
    print("  → OpenAI Function Calling 格式验证通过")


def test_scenario_8_agent_workflow_simulation():
    """场景 8：完整 Agent 工作流模拟

    模拟代码 agent 处理一个用户请求的完整工作流：
    1. 收到用户请求
    2. 搜索相关记忆
    3. 根据记忆+请求生成回应
    4. 保存新的上下文
    """
    print_header("场景 8：完整 Agent 工作流模拟")

    # ── 用户请求: "帮我给认证模块加上邮箱验证功能" ──

    # Step 1: Agent 搜索相关记忆
    print("  Step 1: 搜索相关记忆...")
    search_result = memory_search("认证模块", max_results=5, min_score=0.1)
    print(f"    找到 {search_result['count']} 条相关记忆")

    # Step 2: Agent 获取详细内容
    context_snippets = []
    for r in search_result["results"][:3]:
        detail = memory_get(r["path"])
        context_snippets.append(detail["content"])
    print(f"  Step 2: 获取了 {len(context_snippets)} 条记忆的详细内容")

    # Step 3: Agent 利用记忆上下文"生成回应"
    # （在真实场景中这里会调用 LLM，这里模拟检查 Agent 是否获取到了足够的上下文）
    combined_context = "\n".join(context_snippets)
    has_auth_context = "JWT" in combined_context or "认证" in combined_context
    assert_test(has_auth_context, "Agent 获取到了认证模块的上下文")

    # Step 4: Agent 保存新的决策
    save_result = memory_add(
        "# 邮箱验证功能设计\n"
        "- 注册时发送验证邮件（使用 SendGrid API）\n"
        "- 验证链接有效期 24 小时\n"
        "- 已验证邮箱才能使用 OAuth 绑定\n"
        "- 文件: src/auth/email_verify.py",
        tags=["implementation", "auth", "email"],
    )
    assert_test(save_result["status"] == "success", "Agent 保存了新的设计决策")

    # Step 5: 验证新记忆已可检索
    verify = memory_search("邮箱验证 SendGrid", max_results=1, min_score=0.1)
    assert_test(verify["count"] > 0, "新保存的记忆立即可检索")

    print("  → 完整工作流: 搜索 → 获取详情 → 利用上下文 → 保存决策 ✓")


def run_simulated_tests(
    memory_dir: str = "./test_memory",
    use_chroma: bool = False,
    chroma_dir: str = "./test_chroma_data",
):
    """运行模拟 Agent 测试（无需 API key）"""
    print("\n" + "=" * 60)
    print("  模拟代码 Agent 端到端测试")
    print("  （无需 API key，直接测试工具函数层）")
    print("=" * 60)

    manager = setup_clean_manager(memory_dir, use_chroma, chroma_dir, clean=True)

    backend = "ChromaDB" if use_chroma else "SQLite"
    print(f"\n  向量后端: {backend}")
    print(f"  记忆目录: {memory_dir}")

    # 按顺序执行场景
    test_scenario_1_session_start(manager)
    test_scenario_2_save_code_decisions(manager)
    test_scenario_3_search_context(manager)
    test_scenario_4_get_memory_detail(manager)
    test_scenario_5_multi_turn_accumulation(manager)
    test_scenario_6_cross_session(memory_dir, use_chroma, chroma_dir)
    test_scenario_7_openai_tool_format()
    test_scenario_8_agent_workflow_simulation()


# ============================================================
# 第二部分：真实 LLM Agent 测试（需 API key）
# ============================================================

def run_live_agent_test(
    model: str,
    api_key: str,
    api_base: str = "",
    memory_dir: str = "./test_memory",
    use_chroma: bool = False,
    chroma_dir: str = "./test_chroma_data",
):
    """用真实 LLM 测试 Agent 的记忆工具使用

    通过 OpenAI function calling 验证 LLM 能正确调用记忆工具。
    """
    print_header("真实 LLM Agent 测试 (OpenAI Function Calling)")

    try:
        from openai import OpenAI
    except ImportError:
        print("  ✗ 需要安装 openai: pip install openai")
        return

    from ai_memory.tools.openai import get_openai_tools, execute_tool_calls

    # 初始化
    manager = setup_clean_manager(memory_dir, use_chroma, chroma_dir, clean=True)

    # 预填充一些记忆
    manager.add_memory(
        "# 项目信息\n"
        "这是一个电商平台后端，使用 FastAPI + PostgreSQL。\n"
        "部署在 AWS ECS 上，使用 GitHub Actions 做 CI/CD。",
        tags=["project", "architecture"],
    )
    manager.add_memory(
        "# 用户偏好\n"
        "- 使用 Python type hints\n"
        "- 偏好 async/await\n"
        "- 注释用中文",
        tags=["preference"],
    )

    # 获取工具
    tools, tool_functions = get_openai_tools()

    # 创建 OpenAI 客户端
    client_kwargs = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base
    client = OpenAI(**client_kwargs)

    # 测试 API 是否支持标准 function calling
    print("检测 API 兼容性...")
    try:
        test_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            tools=[tools[0]],  # 只传一个工具测试
            max_tokens=10,
        )
        msg = test_response.choices[0].message
        # 检查是否包含 DSML 标记（DeepSeek 特殊格式）
        if msg.content and ("DSML" in msg.content or "function_calls" in msg.content):
            print("⚠️  检测到 DeepSeek 特殊响应格式 (DSML)")
            print("   DeepSeek 可能不完全兼容标准 OpenAI Function Calling")
            print("   建议：使用兼容 OpenAI 的模型（如 GPT-4、Claude、Qwen 等）\n")
    except Exception as e:
        print(f"API 兼容性测试失败: {e}\n")

    system_prompt = get_system_prompt()

    # ── 测试 1: Agent 应该搜索记忆来回答问题 ──
    print("\n  [测试 1] Agent 搜索记忆回答问题")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "这个项目用的什么技术栈？"},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0.1,
            timeout=30.0,
        )
    except Exception as api_error:
        print(f"    [错误] API 调用失败: {api_error}")
        return

    choice = response.choices[0]
    if choice.message.tool_calls:
        tool_call_names = [tc.function.name for tc in choice.message.tool_calls]
        print(f"    Agent 调用了工具: {tool_call_names}")
        assert_test("memory_search" in tool_call_names, "Agent 主动搜索了记忆")

        # 执行工具调用
        results = execute_tool_calls(choice.message.tool_calls, tool_functions)

        # 构建第二轮对话
        messages.append(choice.message)
        for tc, result in zip(choice.message.tool_calls, results):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # 获取最终回复
        final = client.chat.completions.create(
            model=model, messages=messages, temperature=0.1, timeout=30.0
        )
        answer = final.choices[0].message.content
        print(f"    Agent 回复: {answer[:150]}...")
        has_tech = any(kw in answer for kw in ["FastAPI", "PostgreSQL", "AWS", "电商"])
        assert_test(has_tech, "Agent 回复包含项目技术信息")
    else:
        print(f"    Agent 直接回复（未搜索记忆）: {choice.message.content[:150]}...")
        assert_test(False, "Agent 主动搜索了记忆", "Agent 未调用工具")

    # ── 测试 2: Agent 应该保存新的重要信息 ──
    print("\n  [测试 2] Agent 保存新信息")
    messages2 = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "我刚决定引入 Redis 做缓存层，请记住这个决策。",
        },
    ]

    try:
        response2 = client.chat.completions.create(
            model=model,
            messages=messages2,
            tools=tools,
            temperature=0.1,
            timeout=30.0,
        )
    except Exception as api_error:
        print(f"    [错误] API 调用失败: {api_error}")
        return

    choice2 = response2.choices[0]
    if choice2.message.tool_calls:
        tool_call_names = [tc.function.name for tc in choice2.message.tool_calls]
        print(f"    Agent 调用了工具: {tool_call_names}")
        assert_test("memory_add_long_term" in tool_call_names or "memory_add_daily" in tool_call_names, "Agent 主动保存了记忆")

        results2 = execute_tool_calls(choice2.message.tool_calls, tool_functions)
        for r in results2:
            if isinstance(r, dict) and r.get("status") == "success":
                print(f"    保存到: {r['path']}")

        # 验证保存的内容可搜索
        verify = memory_search("Redis 缓存", max_results=1)
        assert_test(verify["count"] > 0, "Agent 保存的记忆可搜索到")
    else:
        print(f"    Agent 直接回复（未保存记忆）: {choice2.message.content[:150]}...")
        assert_test(False, "Agent 主动保存了记忆", "Agent 未调用 memory_add_long_term 或 memory_add_daily")


# ============================================================
# 第三部分：交互模式
# ============================================================

def interactive_mode(
    model: str,
    api_key: str,
    api_base: str = "",
    memory_dir: str = "./test_memory",
    use_chroma: bool = False,
    chroma_dir: str = "./test_chroma_data",
):
    """交互模式 — 与 Agent 实时对话，观察记忆使用行为"""
    try:
        from openai import OpenAI
    except ImportError:
        print("需要安装 openai: pip install openai")
        return

    from ai_memory.tools.openai import get_openai_tools, execute_tool_calls

    manager = setup_clean_manager(memory_dir, use_chroma, chroma_dir, clean=False)
    tools, tool_functions = get_openai_tools()

    client_kwargs = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base
    client = OpenAI(**client_kwargs)

    # 测试 API 是否支持标准 function calling
    print("检测 API 兼容性...")
    try:
        test_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            tools=[tools[0]],  # 只传一个工具测试
            max_tokens=10,
        )
        msg = test_response.choices[0].message
        # 检查是否包含 DSML 标记（DeepSeek 特殊格式）
        if msg.content and ("DSML" in msg.content or "function_calls" in msg.content):
            print("⚠️  检测到 DeepSeek 特殊响应格式 (DSML)")
            print("   DeepSeek 可能不完全兼容标准 OpenAI Function Calling")
            print("   建议使用兼容 OpenAI 的模型（如 GPT-4、Claude、Qwen 等）")
            print("   继续运行可能遇到工具调用解析问题\n")
    except Exception as e:
        print(f"API 兼容性测试失败: {e}\n")

    system_prompt = get_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    print("\n" + "=" * 60)
    print("  交互模式 — Agent 自主决策记忆操作")
    print("  输入 'quit' 退出, 'status' 查看系统状态")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("退出交互模式")
                break
            if user_input.lower() == "status":
                s = manager.status()
                print(f"  后端: {s.backend}, 文件: {s.files}, 模型: {s.embedding_model}")
                continue

            messages.append({"role": "user", "content": user_input})

            # Agent 循环（支持多轮工具调用）
            max_iterations = 5
            for iteration in range(max_iterations):
                print(f"  [调试] 第 {iteration+1} 轮对话，请求 API...")
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        temperature=0.3,
                        timeout=30.0,  # 添加 30 秒超时
                    )
                    print(f"  [调试] API 响应成功")
                except Exception as api_error:
                    print(f"  [错误] API 调用失败: {api_error}")
                    raise
                choice = response.choices[0]

                if not choice.message.tool_calls:
                    # Agent 返回最终回复
                    messages.append(choice.message)
                    print(f"\nAgent: {choice.message.content}")
                    break

                # 执行工具调用
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    print(f"  [工具调用] {tc.function.name}({tc.function.arguments})")
                    func = tool_functions[tc.function.name]
                    args = json.loads(tc.function.arguments)
                    result = func(**args)
                    result_str = json.dumps(result, ensure_ascii=False, indent=2)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })
                    # 简要显示结果
                    if tc.function.name == "memory_search":
                        print(f"    → 找到 {result.get('count', 0)} 条结果")
                    elif tc.function.name in ("memory_add", "memory_add_long_term", "memory_add_daily"):
                        print(f"    → 保存到 {result.get('path', '?')}")
                    elif tc.function.name == "memory_get":
                        content = result.get("content", "")
                        print(f"    → 获取 {len(content)} 字符")

        except KeyboardInterrupt:
            print("\n退出交互模式")
            break
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI Memory Plugin 端到端测试 — 模拟代码 Agent 使用场景",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 模拟测试（无需 API key）
  python test_e2e.py

  # 真实 LLM 测试
  python test_e2e.py --live --api-key sk-xxx

  # 使用自定义 API 端点
  python test_e2e.py --live --api-url https://api.example.com/v1 --api-key sk-xxx

  # 交互模式
  python test_e2e.py --interactive --api-key sk-xxx

  # 使用 ChromaDB 后端
  python test_e2e.py --use-chroma

  # 指定记忆目录
  python test_e2e.py --interactive --api-key sk-xxx --memory-dir .ai-memory
""",
    )

    # 模式选择
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--live", action="store_true",
        help="使用真实 LLM 测试（需要 API key）",
    )
    mode_group.add_argument(
        "--interactive", action="store_true",
        help="交互模式（需要 API key）",
    )

# LLM 配置
    parser.add_argument("--model", type=str, default="gpt-4",
                        help="模型名称（默认: gpt-4）")
    parser.add_argument("--api-key", type=str, default="",
                        help="API Key（或设置 OPENAI_API_KEY 环境变量）")
    parser.add_argument("--api-url", type=str, default="",
                        help="自定义 API URL")

    # 存储配置
    parser.add_argument("--use-chroma", action="store_true",
                        help="使用 ChromaDB 后端")
    parser.add_argument("--chroma-dir", type=str, default="./test_chroma_data",
                        help="ChromaDB 数据目录")
    parser.add_argument("--memory-dir", type=str, default=".ai-memory",
                        help="记忆存储目录（默认: .ai-memory）")

    # 其它
    parser.add_argument("--verbose", action="store_true",
                        help="详细日志输出")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 获取 API key（live/interactive 模式需要）
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")

    if args.live or args.interactive:
        if not api_key:
            print("错误: --live / --interactive 模式需要 API Key")
            print("  方式 1: --api-key YOUR_API_KEY")
            print("  方式 2: export OPENAI_API_KEY=YOUR_API_KEY")
            sys.exit(1)

    # 执行
    if args.interactive:
        interactive_mode(
            model=args.model,
            api_key=api_key,
            api_base=args.api_url,
            memory_dir=args.memory_dir,
            use_chroma=args.use_chroma,
            chroma_dir=args.chroma_dir,
        )
    else:
        # 始终运行模拟测试
        run_simulated_tests(
            memory_dir=args.memory_dir,
            use_chroma=args.use_chroma,
            chroma_dir=args.chroma_dir,
        )

        # 如果指定了 --live，再跑真实 LLM 测试
        if args.live:
            run_live_agent_test(
                model=args.model,
                api_key=api_key,
                api_base=args.api_url,
                memory_dir=args.memory_dir,
                use_chroma=args.use_chroma,
                chroma_dir=args.chroma_dir,
            )

        # 打印总结
        success = print_summary()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
