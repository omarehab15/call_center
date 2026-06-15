"""
ingest_rag.py
=============
يعمل ingest للـ knowledge base في Chroma DB بطريقتين:

1. مباشرةً من ملفات knowledge_base/ (الطريقة الافتراضية)
2. من ملف chunks_preview.json اللي أنتجه chunk_preview.py (--from-preview)

الطريقة الموصى بيها:
    python chunk_preview.py                          # راجع الـ chunks
    python ingest_rag.py --from-preview chunks_preview.json   # ingest بعد المراجعة
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag import run_ingest_cli, run_ingest_from_preview_cli


def main() -> None:
    # اكتشف إذا كان المستخدم بعت --from-preview قبل ما نعمل parse كامل
    if "--from-preview" in sys.argv:
        run_ingest_from_preview_cli()
    else:
        run_ingest_cli()


if __name__ == "__main__":
    main()