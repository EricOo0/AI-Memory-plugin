"""System Prompt 模板"""

MEMORY_SYSTEM_PROMPT = """
## Memory System Instructions

You have access to a memory system that stores and retrieves information across conversations.

### When to Search Memory

Before answering any question about:
- Past work, decisions, or actions
- User preferences, goals, or context
- Project history, timelines, or dates
- Previously discussed topics or concepts

ALWAYS run a memory search first.

### How to Use Memory

1. **Search**: Use `memory_search(query)` to find relevant memories
2. **Retrieve**: Use `memory_get(path, from, lines)` to read specific sections
3. **Record**: Use `memory_add(content, tags)` to save important information

### Memory Format

- Long-term memories are in `MEMORY.md`
- Daily memories are in `memory/DD-MM-YYYY.md`
- Citations follow format: `path#Lstart-Lend`

### Best Practices

- Be specific in search queries
- Use tags when adding memories
- Verify source citations before relying on content
- Say "I checked my memory" when search yields low-confidence results
"""


def get_system_prompt() -> str:
    """获取记忆系统 System Prompt"""
    return MEMORY_SYSTEM_PROMPT


def get_agent_instructions(tools_available: List[str]) -> str:
    """根据可用工具生成 Agent 指令"""
    if not tools_available:
        return ""

    instructions = """
## Memory System

You have access to a memory system with the following tools:
"""
    for tool in tools_available:
        if tool == "memory_search":
            instructions += "\n- `memory_search(query)`: Search memories for relevant information"
        elif tool == "memory_add":
            instructions += "\n- `memory_add(content, tags)`: Add new memory entries"
        elif tool == "memory_get":
            instructions += "\n- `memory_get(path, from, lines)`: Retrieve specific memory content"

    instructions += """

### When to Use Memory

Search memory before answering about:
- Past work, decisions, or actions
- User preferences or context
- Project history or dates
- Previously discussed topics

### Citation Format

When referencing memories, use: `path#Lstart-Lend`
"""
    return instructions
