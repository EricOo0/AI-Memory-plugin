#!/usr/bin/env python3
"""迁移向量存储到 ChromaDB"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_memory.vector.migration import migrate_command


def main():
    parser = argparse.ArgumentParser(description="迁移向量存储到 ChromaDB")
    parser.add_argument(
        "--memory-dir",
        type=str,
        default="./memory",
        help="记忆目录"
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default="./chroma_data",
        help="ChromaDB 数据目录"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="批量大小"
    )

    args = parser.parse_args()

    migrate_command(
        memory_dir=args.memory_dir,
        chroma_dir=args.chroma_dir,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
