"""文件系统管理器"""

import hashlib
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from ai_memory.core.types import MemoryEntry, MemorySource

logger = logging.getLogger(__name__)


class FileManager:
    """文件系统管理器"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.memory_dir = base_dir / "memory"
        # 确保父目录存在
        base_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(exist_ok=True)
        logger.debug(f"FileManager 初始化: base_dir={base_dir}")

    def get_memory_files(self) -> List[Path]:
        """获取所有记忆文件"""
        files = []

        # 主记忆文件
        main_file = self.base_dir / "MEMORY.md"
        if main_file.exists():
            files.append(main_file)

        # 日期日记目录
        if self.memory_dir.exists():
            files.extend(self.memory_dir.glob("*.md"))

        return files

    def read_file(self, relative_path: str) -> str:
        """读取文件内容"""
        full_path = self._resolve_path(relative_path)
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {relative_path}")
        return full_path.read_text(encoding="utf-8")

    def write_file(self, relative_path: str, content: str) -> None:
        """写入文件内容"""
        full_path = self._resolve_path(relative_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def get_file_hash(self, relative_path: str) -> str:
        """计算文件哈希"""
        content = self.read_file(relative_path)
        return hashlib.sha256(content.encode()).hexdigest()

    def get_file_entry(self, relative_path: str) -> MemoryEntry:
        """获取文件元数据"""
        logger.debug(f"get_file_entry: relative_path={relative_path}")
        full_path = self._resolve_path(relative_path)
        logger.debug(f"  resolved to: {full_path}")
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {relative_path}")

        stat = full_path.stat()

        # 根据路径判断来源
        if relative_path == "MEMORY.md":
            source = MemorySource.MEMORY
        else:
            source = MemorySource.DAILY

        return MemoryEntry(
            path=relative_path,
            content=self.read_file(relative_path),
            hash=self.get_file_hash(relative_path),
            size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            source=source
        )

    def add_memory(
        self,
        content: str,
        tags: List[str] = None,
        target_date: datetime = None
    ) -> str:
        """添加新记忆到每日文件"""
        if target_date is None:
            target_date = datetime.now()

        date_str = target_date.strftime("%d-%m-%Y")
        relative_path = f"memory/{date_str}.md"

        current_content = ""
        if self._resolve_path(relative_path).exists():
            current_content = self.read_file(relative_path)
            current_content += "\n\n"

        # 添加标签
        tag_line = ""
        if tags:
            tag_line = f"\n**Tags:** {' '.join(f'#{t}' for t in tags)}\n"

        new_content = current_content + content + tag_line
        self.write_file(relative_path, new_content)

        return relative_path

    def add_to_long_term(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """添加内容到长期记忆 MEMORY.md

        Args:
            content: 记忆内容
            tags: 标签列表
            category: 分类章节（"用户偏好"、"项目信息"、"重要决策"、"工作流程"、"联系人信息"）
            timestamp: 时间戳（默认当前时间）

        Returns:
            "MEMORY.md"
        """
        if timestamp is None:
            timestamp = datetime.now()

        relative_path = "MEMORY.md"
        full_path = self._resolve_path(relative_path)

        # 读取现有内容或创建初始结构
        if full_path.exists():
            existing_content = self.read_file(relative_path)
        else:
            existing_content = self._create_initial_memory_structure()

        # 推断分类（如未指定）
        if category is None:
            category = self._infer_category(tags or [])

        # 提取标题
        title = self._extract_title(content)

        # 构造新条目
        time_str = timestamp.strftime("%Y-%m-%d %H:%M")
        entry = f"\n### [{time_str}] {title}\n"

        if tags:
            entry += f"**Tags:** {' '.join(f'#{t}' for t in tags)}\n\n"
        else:
            entry += "\n"

        entry += f"{content}\n\n---\n"

        # 插入到对应章节
        new_content = self._insert_into_category(existing_content, category, entry)

        # 写回文件
        self.write_file(relative_path, new_content)

        logger.info(f"添加长期记忆到 MEMORY.md: category={category}, title={title}")

        return relative_path

    def _create_initial_memory_structure(self) -> str:
        """创建初始的 MEMORY.md 结构（5个章节）"""
        return """# 长期记忆

## 用户偏好
<!--用户的个人偏好、工作习惯、交互方式等长期设置-->

---

## 项目信息
<!--项目的核心信息、技术栈、架构决策等-->

---

## 重要决策
<!--架构决策、技术选型、重大变更等-->

---

## 工作流程
<!--团队协作流程、开发规范等-->

---

## 联系人信息
<!--重要的人员、联系方式、协作者信息-->

---
"""

    def _infer_category(self, tags: List[str]) -> str:
        """根据标签推断分类

        Args:
            tags: 标签列表

        Returns:
            分类章节名称
        """
        tag_set = set(tags)

        if tag_set & {"preference", "user", "settings", "习惯", "偏好"}:
            return "用户偏好"
        if tag_set & {"project", "architecture", "tech-stack", "技术栈", "架构"}:
            return "项目信息"
        if tag_set & {"decision", "milestone", "决策", "里程碑"}:
            return "重要决策"
        if tag_set & {"workflow", "process", "standard", "流程", "规范"}:
            return "工作流程"
        if tag_set & {"contact", "team", "联系人", "团队"}:
            return "联系人信息"

        # 默认分类
        return "项目信息"

    def _extract_title(self, content: str, max_length: int = 60) -> str:
        """从内容提取标题（第一行或摘要）

        Args:
            content: 内容文本
            max_length: 最大标题长度

        Returns:
            标题字符串
        """
        # 去除开头的 Markdown 标题符号
        lines = content.strip().split("\n")
        first_line = lines[0].lstrip("#").strip()

        # 截断过长标题
        if len(first_line) > max_length:
            return first_line[:max_length] + "..."

        return first_line if first_line else "无标题"

    def _insert_into_category(self, content: str, category: str, entry: str) -> str:
        """将条目插入到指定章节

        Args:
            content: 现有文件内容
            category: 章节名称
            entry: 新条目内容

        Returns:
            更新后的文件内容
        """
        # 查找章节位置（章节标题 + 可选注释）
        pattern = rf"(## {re.escape(category)}\s*\n(?:<!--.*?-->\s*\n)?)"
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            # 章节不存在，追加到末尾
            logger.warning(f"章节 '{category}' 不存在，追加到文件末尾")
            return content + f"\n## {category}\n\n{entry}"

        # 找到章节插入位置（跳过章节标题和注释）
        insert_pos = match.end()

        # 插入新条目
        new_content = content[:insert_pos] + "\n" + entry + content[insert_pos:]

        return new_content

    def _resolve_path(self, relative_path: str) -> Path:
        """解析相对路径"""
        if relative_path.startswith("memory/"):
            return self.base_dir / relative_path
        return self.base_dir / relative_path
