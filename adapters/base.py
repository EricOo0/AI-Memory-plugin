"""基础适配器接口"""

from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """基础适配器接口"""

    @abstractmethod
    def get_tools(self):
        """获取工具列表"""
        pass

    @abstractmethod
    def get_instructions(self) -> str:
        """获取使用指令"""
        pass
