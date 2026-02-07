#!/usr/bin/env python3
"""独立 CLI 命令"""

import argparse
import json
import sys
from pathlib import Path

from ai_memory.tools.functions import (
    init,
    memory_search,
    memory_add,
    memory_add_long_term,
    memory_add_daily,
    memory_get
)

# 默认路径：项目根目录的 .ai-memory/
DEFAULT_DIR = Path.cwd() / ".ai-memory"


def get_manager():
    """获取管理器实例"""
    return init(memory_dir=str(DEFAULT_DIR))


def format_search_results(result: dict, json_output: bool = False) -> str:
    """格式化搜索结果输出

    Args:
        result: 搜索结果字典
        json_output: 是否输出 JSON 格式

    Returns:
        格式化后的字符串
    """
    if json_output:
        return json.dumps(result, ensure_ascii=False, indent=2)

    # 人类可读格式
    lines = []
    lines.append(f"查询: {result['query']}")
    lines.append(f"找到 {result['count']} 条结果")

    if result['count'] == 0:
        lines.append("\n没有找到相关记忆。")
        return "\n".join(lines)

    lines.append("")
    for i, r in enumerate(result['results'], 1):
        lines.append(f"{i}. [{r['score']:.2f}] {r['citation']}")
        # 截断过长的摘要
        snippet = r['snippet'].strip()
        if len(snippet) > 150:
            snippet = snippet[:150] + "..."
        lines.append(f"   {snippet}")
        lines.append("")

    return "\n".join(lines)


def format_add_result(result: dict, json_output: bool = False) -> str:
    """格式化添加结果输出"""
    if json_output:
        return json.dumps(result, ensure_ascii=False, indent=2)

    result_type = result.get("type", "")
    type_label = {
        "long_term": "长期记忆",
        "daily": "短期记忆"
    }.get(result_type, "记忆")

    return f"✓ {type_label}已添加到: {result['path']}"


def main():
    parser = argparse.ArgumentParser(
        prog="ai-memory",
        description="AI Memory Plugin 命令行工具"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出（适用于程序化调用）"
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # search
    search_parser = subparsers.add_parser("search", help="搜索记忆")
    search_parser.add_argument("query", help="搜索关键词")
    search_parser.add_argument("--max-results", type=int, default=6, help="最大结果数")
    search_parser.add_argument("--min-score", type=float, default=0.35, help="最小相关度")

    # add (向后兼容,映射到短期记忆)
    add_parser = subparsers.add_parser("add", help="添加记忆（映射到短期记忆）")
    add_parser.add_argument("content", help="记忆内容")
    add_parser.add_argument("--tags", help="标签，逗号分隔")

    # add-long-term
    add_long_parser = subparsers.add_parser("add-long-term", help="添加长期记忆到 MEMORY.md")
    add_long_parser.add_argument("content", help="记忆内容")
    add_long_parser.add_argument("--tags", help="标签，逗号分隔")
    add_long_parser.add_argument("--category", help="分类章节（可选）：用户偏好、项目信息、重要决策、工作流程、联系人信息")

    # add-daily
    add_daily_parser = subparsers.add_parser("add-daily", help="添加短期记忆到今日文件")
    add_daily_parser.add_argument("content", help="记忆内容")
    add_daily_parser.add_argument("--tags", help="标签，逗号分隔")

    # get
    get_parser = subparsers.add_parser("get", help="获取记忆")
    get_parser.add_argument("path", help="文件路径")
    get_parser.add_argument("--from", type=int, dest="from_line", help="起始行号")
    get_parser.add_argument("--lines", type=int, default=20, help="行数")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        # 初始化管理器
        manager = get_manager()

        if args.command == "search":
            result = memory_search(args.query, args.max_results, args.min_score)
            output = format_search_results(result, json_output=args.json)
            print(output)

        elif args.command == "add":
            tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
            result = memory_add(args.content, tags)
            output = format_add_result(result, json_output=args.json)
            print(output)

        elif args.command == "add-long-term":
            tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
            category = args.category if hasattr(args, "category") else None
            result = memory_add_long_term(args.content, tags, category)
            output = format_add_result(result, json_output=args.json)
            print(output)

        elif args.command == "add-daily":
            tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
            result = memory_add_daily(args.content, tags)
            output = format_add_result(result, json_output=args.json)
            print(output)

        elif args.command == "get":
            result = memory_get(args.path, args.from_line, args.lines)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(result["content"])

    except KeyboardInterrupt:
        print("\n操作已取消", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        error_msg = f"错误: {str(e)}"
        if args.json:
            error_obj = {"error": str(e), "type": type(e).__name__}
            print(json.dumps(error_obj, ensure_ascii=False), file=sys.stderr)
        else:
            print(error_msg, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
