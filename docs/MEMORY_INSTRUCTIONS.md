# Memory System Instructions

You have access to a memory system that stores and retrieves information across conversations.

## When to Search Memory

Before answering any question about:
- Past work, decisions, or actions
- User preferences, goals, or context
- Project history, timelines, or dates
- Previously discussed topics or concepts

**ALWAYS run a memory search first.**

## How to Use Memory

### 1. Search
Use `memory_search(query)` to find relevant memories about the topic.

### 2. Retrieve
Use `memory_get(path, from, lines)` to read specific sections from a memory file.

### 3. Record
Use `memory_add(content, tags)` to save important information for future reference.

## Memory Format

- **Long-term memories** are stored in `MEMORY.md`
- **Daily memories** are stored in `memory/DD-MM-YYYY.md`
- **Citations** follow the format: `path#Lstart-Lend`

## Best Practices

- Be specific in search queries to get better results
- Use tags when adding memories for better organization
- Always verify source citations before relying on retrieved content
- Say "I checked my memory" when search yields low-confidence results

## Example Usage

```python
# Search for previous work
memory_search("project decisions last month")

# Get detailed content
memory_get("MEMORY.md", from_line=10, lines=20)

# Save important information
memory_add("User prefers API v2 over v1", tags=["preference"])
```
