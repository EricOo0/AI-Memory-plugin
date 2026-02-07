from ai_memory import init
from ai_memory.tools import get_langchain_tools
from langchain.agents import create_agent

# 初始化
init()

# 获取封装好的 LangChain 工具
tools = get_langchain_tools()

# 创建 Agent
agent = create_agent(llm, tools)

# 使用
result = agent.invoke({"messages": [{"role": "user", "content": "搜索用户偏好"}]})
