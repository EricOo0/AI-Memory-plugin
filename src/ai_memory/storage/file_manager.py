"""文件系统管理器"""

import hashlib
from pathlib import Path
from datetime import datetime
from typing import List

from ai_memory.core.types import MemoryEntry, MemorySource


class FileManager:
    """文件系统管理器"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.memory_dir = base_dir / "memory"
        # 确保父目录存在
        base_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(exist_ok=True)

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
        full_path = self._resolve_path(relative_path)
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
        """添加新记忆"""
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

    def _resolve_path(self, relative_path: str) -> Path:
        """解析相对路径"""
        if relative_path.startswith("memory/"):
            return self.base_dir / relative_path
        return self.base_dir / relative_path
