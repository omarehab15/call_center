"""
chunk_preview.py
================
بيعمل chunking ذكي للملفات في knowledge_base/ ويحفظ النتيجة في chunks_preview.json
قبل ما تتعمل embedding — عشان تقدر تراجعها وتعدل فيها.

الـ chunking logic:
- إذا الملف فيه Q&A بصيغة (س:/ج:) → كل سؤال+جواب = chunk منفرد
- ممكن نجمع أسئلة كتير تحت نفس القسم في chunk واحد (grouped mode)
- أي ملف تاني → character chunking عادي مع split على فقرات

الاستخدام:
    python chunk_preview.py                        # كل الملفات في knowledge_base/
    python chunk_preview.py --source path/to/dir   # مجلد مختلف
    python chunk_preview.py --group-by-section     # جمع Q&A تحت نفس القسم
    python chunk_preview.py --output my_chunks.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".htm", ".json", ".csv"}

# ───────────────────────────────────────────────
# Core chunking strategies
# ───────────────────────────────────────────────

def detect_qa_format(text: str) -> bool:
    """هل الملف فيه Q&A بصيغة س:/ج: ؟"""
    qa_lines = sum(1 for line in text.splitlines() if re.match(r"^\*?\*?س:", line.strip()))
    return qa_lines >= 3


def chunk_qa_individual(text: str, source: str, section_hint: str = "") -> list[dict]:
    """
    كل سؤال + جوابه → chunk منفصل.
    بيتعامل مع الصيغتين:
      - **س: ...** على سطر + ج: على سطر تاني
      - س: ... و ج: على نفس السطر
    """
    # الملف فيه **س: ...** و**  trailing on same line — شيّل الـ ** بس خلي السطر كما هو
    # **س: question text**  → س: question text
    text = re.sub(r"\*\*\s*(س:)\s*", r"\1 ", text)
    text = re.sub(r"\*\*\s*$", "", text, flags=re.MULTILINE)   # trailing **
    # شيل trailing spaces من markdown (السطر بيخلص بـ "  ")
    text = re.sub(r"\s+$", "", text, flags=re.MULTILINE)

    chunks = []
    current_section = section_hint
    current_q: str | None = None
    current_a_lines: list[str] = []
    state = "idle"  # idle | in_question | in_answer

    def flush():
        nonlocal current_q, current_a_lines, state
        if current_q and current_a_lines:
            q_text = current_q.strip()
            a_text = " ".join(current_a_lines).strip()
            full = f"س: {q_text}\nج: {a_text}"
            chunks.append({
                "text": full,
                "source": source,
                "section": current_section,
                "type": "qa",
                "question": q_text,
                "answer": a_text,
                "chunk_index": len(chunks),
            })
        current_q = None
        current_a_lines = []
        state = "idle"

    for line in text.splitlines():
        stripped = line.strip()

        # section header
        if stripped.startswith("## "):
            flush()
            current_section = stripped.lstrip("# ").strip()
            continue

        if stripped.startswith("---") or stripped.startswith("# "):
            continue

        if not stripped:
            # سطر فاضي بعد الجواب = نهاية الـ chunk
            if state == "in_answer":
                flush()
            continue

        # Question line: "س: ..."
        q_match = re.match(r"^س:\s*(.+)", stripped)
        if q_match:
            flush()
            current_q = q_match.group(1).strip()
            state = "in_question"
            continue

        # Answer line: "ج: ..."
        a_match = re.match(r"^ج:\s*(.*)", stripped)
        if a_match:
            rest = a_match.group(1).strip()
            current_a_lines = [rest] if rest else []
            state = "in_answer"
            continue

        # continuation of answer (any non-empty line while in_answer)
        if state == "in_answer":
            current_a_lines.append(stripped)

    flush()
    return chunks


def chunk_qa_grouped_by_section(text: str, source: str) -> list[dict]:
    """
    كل قسم (## Header) → chunk واحد يحتوي على كل Q&A تحته.
    مفيد لو الأقسام صغيرة وعايز كل section تكون وحدة واحدة.
    """
    # normalize bold
    text = re.sub(r"\*\*(س:|ج:)\*\*", r"\1", text)

    chunks = []
    current_section = "عام"
    current_lines: list[str] = []

    def flush_section():
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append({
                "text": content,
                "source": source,
                "section": current_section,
                "type": "section",
                "chunk_index": len(chunks),
            })
        current_lines.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            flush_section()
            current_section = stripped.lstrip("# ").strip()
            continue
        if stripped.startswith("---") or stripped.startswith("# "):
            continue
        if stripped:
            current_lines.append(stripped)

    flush_section()
    return chunks


def chunk_plain_text(text: str, source: str, chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    """Character-based chunking للنصوص العادية مع split على فقرات."""
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(normalized) <= chunk_size:
        return [{"text": normalized, "source": source, "type": "plain", "chunk_index": 0}]

    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        min_split = start + overlap + 1
        split_search = min(end, min_split + 1)
        split_at = normalized.rfind("\n\n", split_search, end)
        if split_at <= min_split:
            split_at = normalized.rfind("\n", split_search, end)
        if split_at <= min_split:
            split_at = end

        chunk_text = normalized[start:split_at].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": source,
                "type": "plain",
                "chunk_index": len(chunks),
            })
        if split_at >= len(normalized):
            break
        next_start = max(0, split_at - overlap)
        if next_start <= start:
            next_start = split_at
        if next_start <= start:
            break
        start = next_start

    return chunks


# ───────────────────────────────────────────────
# Main ingestion
# ───────────────────────────────────────────────

def process_file(path: Path, source_dir: Path, *, group_by_section: bool = False) -> list[dict]:
    source = path.relative_to(source_dir).as_posix()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if detect_qa_format(text):
        if group_by_section:
            chunks = chunk_qa_grouped_by_section(text, source)
            print(f"  ✓ QA-grouped: {len(chunks)} chunks (by section)")
        else:
            chunks = chunk_qa_individual(text, source)
            print(f"  ✓ QA-individual: {len(chunks)} chunks (per Q&A pair)")
    else:
        chunks = chunk_plain_text(text, source)
        print(f"  ✓ Plain text: {len(chunks)} chunks")

    return chunks


def run(source_dir: Path, output_path: Path, *, group_by_section: bool = False) -> None:
    all_chunks: list[dict[str, Any]] = []

    files = sorted(p for p in source_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)

    if not files:
        print(f"❌ مافيش ملفات في {source_dir}")
        sys.exit(1)

    print(f"\n📂 المجلد: {source_dir}")
    print(f"📄 عدد الملفات: {len(files)}\n")

    for fpath in files:
        print(f"📝 {fpath.name}")
        chunks = process_file(fpath, source_dir, group_by_section=group_by_section)
        all_chunks.extend(chunks)

    # إحصائيات
    total_chars = sum(len(c["text"]) for c in all_chunks)
    avg_chars = total_chars // len(all_chunks) if all_chunks else 0

    summary = {
        "meta": {
            "source_dir": str(source_dir),
            "total_chunks": len(all_chunks),
            "total_files": len(files),
            "total_chars": total_chars,
            "avg_chunk_chars": avg_chars,
            "group_by_section": group_by_section,
        },
        "chunks": all_chunks,
    }

    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n{'─'*50}")
    print(f"✅ تم الحفظ في: {output_path}")
    print(f"   إجمالي الـ chunks : {len(all_chunks)}")
    print(f"   متوسط حجم الـ chunk: {avg_chars} حرف")
    print(f"   إجمالي الأحرف     : {total_chars:,}")
    print(f"\n💡 راجع الملف، عدّل فيه، وبعدين شغّل ingest_rag.py\n")


# ───────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preview RAG chunks before embedding.")
    parser.add_argument("--source", default="knowledge_base",
                        help="مجلد الـ knowledge base (default: knowledge_base/)")
    parser.add_argument("--output", default="chunks_preview.json",
                        help="ملف الـ output (default: chunks_preview.json)")
    parser.add_argument("--group-by-section", action="store_true",
                        help="جمّع Q&A تحت نفس القسم في chunk واحد بدل chunk لكل سؤال")
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"❌ المجلد مش موجود: {source_dir}")
        sys.exit(1)

    run(source_dir, Path(args.output), group_by_section=args.group_by_section)


if __name__ == "__main__":
    main()
