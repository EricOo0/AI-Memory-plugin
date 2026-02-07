"""SQLite 数据库封装"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from ai_memory.core.types import MemoryEntry, MemorySource


class Database:
    """SQLite 数据库封装"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        """创建数据库连接"""
        if self.conn is None:
            self.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False  # 允许跨线程访问（需注意并发安全）
            )
            self.conn.row_factory = sqlite3.Row
            # 启用 WAL 模式
            self.conn.execute("PRAGMA journal_mode=WAL")
        return self.conn

    def close(self) -> None:
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_tables(self) -> None:
        """创建数据表"""
        conn = self.connect()
        cursor = conn.cursor()

        # 文件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                hash TEXT NOT NULL,
                size INTEGER NOT NULL,
                modified_at INTEGER NOT NULL
            )
        """)

        # 文本块表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                model TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
        """)

        # FTS5 全文搜索表
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks
            USING fts5(text, id UNINDEXED, path UNINDEXED, start_line UNINDEXED, end_line UNINDEXED)
        """)

        conn.commit()

    def insert_file(self, entry: MemoryEntry) -> None:
        """插入文件记录"""
        conn = self.connect()
        conn.execute("""
            INSERT OR REPLACE INTO files (path, source, hash, size, modified_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            entry.path,
            entry.source.value,
            entry.hash,
            entry.size,
            int(entry.modified_at.timestamp())
        ))
        conn.commit()

    def get_file(self, path: str) -> Optional[Dict]:
        """获取文件记录"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM files WHERE path = ?", (path,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete_file(self, path: str) -> None:
        """删除文件记录及相关块"""
        conn = self.connect()
        conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        conn.execute("DELETE FROM fts_chunks WHERE path = ?", (path,))
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
        conn.commit()

    def insert_chunk(
        self,
        id: str,
        path: str,
        start_line: int,
        end_line: int,
        text: str,
        embedding: List[float],
        model: str
    ) -> None:
        """插入文本块"""
        conn = self.connect()
        now = int(datetime.now().timestamp())

        # 插入块记录
        conn.execute("""
            INSERT OR REPLACE INTO chunks
            (id, path, start_line, end_line, text, embedding, model, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (id, path, start_line, end_line, text, json.dumps(embedding), model, now))

        # 同步到 FTS 表
        conn.execute("""
            INSERT OR REPLACE INTO fts_chunks (rowid, text, id, path, start_line, end_line)
            SELECT rowid, text, id, path, start_line, end_line FROM chunks
            WHERE id = ?
        """, (id,))

        conn.commit()

    def search_by_vector(
        self,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[Dict]:
        """通过向量搜索（余弦相似度）"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("SELECT id, path, start_line, end_line, text, embedding FROM chunks")
        rows = cursor.fetchall()

        results = []
        for row in rows:
            embedding = json.loads(row["embedding"])
            score = self._cosine_similarity(query_embedding, embedding)
            if score > 0:
                results.append({
                    "id": row["id"],
                    "path": row["path"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "text": row["text"],
                    "score": score
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def search_by_text(self, query: str, limit: int = 10) -> List[Dict]:
        """通过全文搜索"""
        conn = self.connect()
        cursor = conn.cursor()

        # 转义 FTS5 查询中的特殊字符
        safe_query = self._escape_fts_query(query)

        cursor.execute("""
            SELECT id, path, start_line, end_line, text, rank
            FROM fts_chunks
            WHERE fts_chunks MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (safe_query, limit))

        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "path": row["path"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "text": row["text"],
                "score": 1.0 / (1.0 + abs(row["rank"]))  # BM25 rank 在 FTS5 中为负值
            }
            for row in rows
        ]

    def get(self, ids: List[str]) -> List[Dict]:
        """获取向量"""
        results = []
        for id_ in ids:
            # 解析 chunk_id: "path:start-end"
            parts = id_.split(":")
            if len(parts) >= 2:
                path = parts[0]
                # parts[1] 格式是 "start-end"
                line_parts = parts[1].split("-")
                if len(line_parts) >= 1:
                    start = int(line_parts[0])

            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, path, start_line, end_line, text, embedding, model
                FROM chunks
                WHERE path = ? AND start_line = ?
            """, (path, start))
            row = cursor.fetchone()
            if row:
                results.append({
                    "id": id_,
                    "path": path,
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "text": row["text"],
                    "embedding": json.loads(row["embedding"]),
                    "model": row["model"]
                })
        return results

    def _escape_fts_query(self, query: str) -> str:
        """转义 FTS5 查询特殊字符"""
        # FTS5 特殊字符: " ' * ( ) [ ] { }
        # 将查询用双引号包裹，并进行转义
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
