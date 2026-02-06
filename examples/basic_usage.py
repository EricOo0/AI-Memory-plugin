"""基础用法示例"""

from ai_memory import MemoryManager, MemoryConfig


def main():
    # 初始化记忆管理器
    config = MemoryConfig(storage={"dir": "./memory"})
    manager = MemoryManager(config)

    # 添加一些记忆
    print("Adding memories...")
    manager.add_memory("# 用户偏好\n喜欢使用 Python 进行开发", tags=["user", "preference"])
    manager.add_memory("# 项目决策\n选择 SQLite 作为数据库", tags=["project", "decision"])
    manager.add_memory("# 待办事项\n- 完成记忆系统\n- 编写文档", tags=["todo"])

    # 同步索引
    print("Syncing index...")
    manager.sync()

    # 搜索
    print("\nSearching for 'Python'...")
    results = manager.search("Python")
    for r in results:
        print(f"\n[{r.citation}] (score: {r.score:.2f})")
        print(f"{r.snippet[:100]}...")

    # 获取状态
    print("\n\nSystem Status:")
    status = manager.status()
    print(f"  Files: {status.files}")
    print(f"  Model: {status.embedding_model}")


if __name__ == "__main__":
    main()
