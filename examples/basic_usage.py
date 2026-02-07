"""基础用法示例 - 使用工具函数 API"""

from ai_memory import init, memory_search, memory_add, memory_get

def main():
    # 初始化（使用默认目录 .ai-memory/）
    init()

    # 添加记忆
    print("添加记忆...")
    result = memory_add("# 用户偏好\n喜欢使用 Python 进行开发", tags=["user", "preference"])
    print(f"  保存到: {result['path']}")

    result = memory_add("# 项目决策\n选择 SQLite 作为数据库", tags=["project", "decision"])
    print(f"  保存到: {result['path']}")

    result = memory_add("# 待办事项\n- 完成记忆系统\n- 编写文档", tags=["todo"])
    print(f"  保存到: {result['path']}")

    # 搜索记忆
    print("\n搜索 'Python'...")
    result = memory_search("Python", max_results=3)
    for item in result["results"]:
        print(f"\n[{item['citation']}] (score: {item['score']:.2f})")
        print(f"{item['snippet'][:100]}...")

    # 获取完整记忆内容
    print("\n获取 MEMORY.md 前 10 行...")
    result = memory_get("MEMORY.md", from_line=1, lines=10)
    print(result["content"])

    print(f"\n\n系统状态:")
    print(f"  搜索结果数: {result['count']}")


if __name__ == "__main__":
    main()
