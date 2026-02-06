"""SQLite 向量存储提供者（兼容现有实现）"""

import json
from typing import List, Optional

from ai_memory.vector.base import VectorStore, VectorSearchResult
from ai_memory.storage.database import Database


class SQLiteVectorStore(VectorStore):
    """SQLite 向量存储（兼容现有实现）"""

    def __init__(self, database: Database):
        self.db = database

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[dict]
    ) -> None:
        """添加向量到 SQLite"""
        for id_, embedding, text, metadata in zip(ids, embeddings, documents, metadatas):
            self.db.insert_chunk(
                id=id_,
                path=metadata["path"],
                start_line=metadata["start_line"],
                end_line=metadata["end_line"],
                text=text,
                embedding=embedding,
                model=metadata.get("model", "unknown")
            )

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[dict] = None
    ) -> List[VectorSearchResult]:
        """向量搜索（暴力计算）"""
        results = self.db.search_by_vector(query_embedding, limit=n_results * 3)

        return [
            VectorSearchResult(
                id=r["id"],
                score=r["score"],
                metadata={"path": r["path"], "start_line": r["start_line"], "end_line": r["end_line"]}
            )
            for r in results[:n_results]
        ]

    def delete(self, ids: List[str]) -> None:
        """删除向量"""
        for id_ in ids:
            # 从 chunk ID 中提取路径，删除相关记录
            parts = id_.split(":")
            if len(parts) >= 1:
                path = parts[0]
                # 删除文件的所有块
                self.db.delete_file(path)

    def get(self, ids: List[str]) -> List[dict]:
        """获取向量"""
        # SQLite 实现中，向量与 chunk 一起存储
        results = []
        for id_ in ids:
            parts = id_.split(":")
            if len(parts) >= 3:
                path = parts[0]
                start = int(parts[1])
                cursor = self.db.connect().cursor()
                cursor.execute("""
                    SELECT text, embedding, model FROM chunks
                    WHERE path = ? AND start_line = ?
                """, (path, start))
                row = cursor.fetchone()
                if row:
                    results.append({
                        "id": id_,
                        "text": row["text"],
                        "embedding": json.loads(row["embedding"]),
                        "model": row["model"]
                    })
        return results

    def clear(self) -> None:
        """清空所有数据"""
        conn = self.db.connect()
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM fts_chunks")
        conn.commit()

    def close(self) -> None:
        """关闭连接"""
        self.db.close()
