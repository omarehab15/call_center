# Knowledge Base

Add Markdown, text, HTML, JSON, or CSV files here, then index them into Chroma:

```bash
uv run python src/ingest_rag.py --reset
```

The agent retrieves from this local Chroma collection with `BAAI/bge-m3` and injects relevant context before each LLM response.
