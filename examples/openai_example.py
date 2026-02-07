import json
from ai_memory import init
from ai_memory.tools import get_openai_tools, execute_tool_calls
from openai import OpenAI

# 初始化
init()

# 获取封装好的工具定义
tools, tool_functions = get_openai_tools()

client = OpenAI()

# 第一次请求
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "搜索用户偏好"}],
    tools=tools
)

# 处理工具调用
if response.choices[0].message.tool_calls:
    # 执行工具
    tool_results = execute_tool_calls(response.choices[0].message.tool_calls, tool_functions)

    # 构建第二轮对话
    messages = [
        {"role": "user", "content": "搜索用户偏好"},
        response.choices[0].message,
        *[{"role": "tool", "tool_call_id": r.id, "content": json.dumps(r)}
          for r in tool_results]
    ]

    # 获取最终回复
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    print(final_response.choices[0].message.content)
