# sqlite-vec + Chonkie Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace ChromaDB with sqlite-vec and replace custom `chunk_by_heading()` with Chonkie's RecursiveChunker for a fully local, telemetry-free vector store with markdown-aware chunking.

**Architecture:** Rewrite `store.py` with sqlite-vec (two tables: metadata + vec0 virtual table, same public API). Replace custom chunking in `indexer.py` with Chonkie `RecursiveChunker` using the built-in `markdown` recipe. Clean break migration - user re-indexes after upgrade.

**Tech Stack:** Python, sqlite-vec, sqlite3 stdlib, Chonkie RecursiveChunker

---

### Task 1: Rewrite store.py with sqlite-vec

**Files:**
- Modify: `src/obsidian_rag/store.py`
- Test: `tests/test_store.py` (already exists with 11 contract tests)

Contract tests already exist and pass against ChromaDB. This task rewrites the internals so the same tests pass against sqlite-vec.

**Step 1: Rewrite store.py**

Replace the entire contents of `src/obsidian_rag/store.py` with:

```python
"""SQLite-vec vector store wrapper."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import sqlite_vec

from .indexer import Chunk

# Default embedding dimension (nomic-embed-text = 768, OpenAI small = 1536)
# Detected automatically on first upsert.
DEFAULT_DIM = 768


def _serialize_f32(vec: Sequence[float]) -> bytes:
    """Serialize a list of floats to a compact bytes format for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


class VectorStore:
    """SQLite-vec backed vector store for Obsidian notes."""

    def __init__(self, data_path: str, collection_name: str = "obsidian_notes"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.db_path = self.data_path / f"{collection_name}.db"

        self.db = sqlite3.connect(str(self.db_path))
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)

        self._dim: Optional[int] = None
        self._ensure_metadata_table()
        self._try_load_vec_table()

    def _ensure_metadata_table(self) -> None:
        """Create the metadata table if it doesn't exist."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                heading TEXT,
                heading_level INTEGER,
                type TEXT,
                tags TEXT,
                content TEXT NOT NULL
            )
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_file_path
            ON chunks(file_path)
        """)
        self.db.commit()

    def _try_load_vec_table(self) -> None:
        """Try to detect the dimension from an existing vec table."""
        try:
            row = self.db.execute(
                "SELECT embedding FROM chunks_vec LIMIT 1"
            ).fetchone()
            if row is not None:
                self._dim = len(row[0]) // 4  # 4 bytes per float32
        except sqlite3.OperationalError:
            pass  # Table doesn't exist yet

    def _ensure_vec_table(self, dim: int) -> None:
        """Create the vector table with the given dimension."""
        if self._dim == dim:
            return
        self._dim = dim
        self.db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{dim}]
            )
        """)
        self.db.commit()

    def upsert(self, chunk: Chunk, embedding: List[float]) -> None:
        """Add or update a chunk."""
        self._ensure_vec_table(len(embedding))
        meta = self._prepare_metadata(chunk)

        self.db.execute("""
            INSERT OR REPLACE INTO chunks (id, file_path, heading, heading_level, type, tags, content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (chunk.id, meta["file_path"], meta["heading"], meta["heading_level"],
              meta["type"], meta.get("tags", ""), chunk.content))

        # sqlite-vec: delete then insert (no native upsert on virtual tables)
        self.db.execute("DELETE FROM chunks_vec WHERE id = ?", (chunk.id,))
        self.db.execute(
            "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
            (chunk.id, _serialize_f32(embedding))
        )
        self.db.commit()

    def upsert_batch(self, chunks: List[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Add or update multiple chunks."""
        if not chunks:
            return
        self._ensure_vec_table(len(embeddings[0]))

        for chunk, embedding in zip(chunks, embeddings):
            meta = self._prepare_metadata(chunk)
            self.db.execute("""
                INSERT OR REPLACE INTO chunks (id, file_path, heading, heading_level, type, tags, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (chunk.id, meta["file_path"], meta["heading"], meta["heading_level"],
                  meta["type"], meta.get("tags", ""), chunk.content))

            self.db.execute("DELETE FROM chunks_vec WHERE id = ?", (chunk.id,))
            self.db.execute(
                "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk.id, _serialize_f32(embedding))
            )

        self.db.commit()

    def delete_by_file(self, file_path: str) -> None:
        """Delete all chunks from a specific file."""
        # Get IDs to delete from vec table
        ids = [row[0] for row in
               self.db.execute("SELECT id FROM chunks WHERE file_path = ?", (file_path,)).fetchall()]
        if ids:
            placeholders = ",".join("?" * len(ids))
            self.db.execute(f"DELETE FROM chunks_vec WHERE id IN ({placeholders})", ids)
            self.db.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
            self.db.commit()

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar chunks."""
        if self._dim is None:
            return []

        query_bytes = _serialize_f32(query_embedding)

        if where:
            # Build WHERE clause for metadata filter
            conditions = []
            params: list = []
            for key, value in where.items():
                conditions.append(f"c.{key} = ?")
                params.append(value)

            where_clause = " AND ".join(conditions)

            # Use subquery to filter candidates, then KNN on filtered set
            # sqlite-vec requires k parameter in the MATCH query
            # Fetch more than needed to account for filtering
            fetch_limit = limit * 5

            rows = self.db.execute(f"""
                SELECT c.id, c.file_path, c.heading, c.heading_level, c.type, c.tags, c.content, v.distance
                FROM chunks_vec v
                JOIN chunks c ON c.id = v.id
                WHERE v.embedding MATCH ? AND k = ?
                  AND {where_clause}
                ORDER BY v.distance
                LIMIT ?
            """, [query_bytes, fetch_limit] + params + [limit]).fetchall()
        else:
            rows = self.db.execute("""
                SELECT c.id, c.file_path, c.heading, c.heading_level, c.type, c.tags, c.content, v.distance
                FROM chunks_vec v
                JOIN chunks c ON c.id = v.id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
            """, [query_bytes, limit]).fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "metadata": {
                    "file_path": row[1],
                    "heading": row[2] or "",
                    "heading_level": row[3],
                    "type": row[4] or "note",
                    "tags": row[5] or "",
                },
                "content": row[6],
                "distance": row[7],
            })
        return results

    def get_stats(self) -> dict:
        """Get collection statistics."""
        count = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "collection": self.collection_name,
            "count": count,
            "data_path": str(self.data_path),
        }

    def clear(self) -> None:
        """Clear all data."""
        self.db.execute("DELETE FROM chunks")
        if self._dim is not None:
            self.db.execute("DROP TABLE IF EXISTS chunks_vec")
            self._dim = None
        self.db.commit()

    def _prepare_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Prepare metadata for storage."""
        meta = {
            "file_path": chunk.file_path,
            "heading": chunk.heading or "",
            "heading_level": chunk.heading_level,
            "type": chunk.metadata.get("type", "note"),
        }
        if "tags" in chunk.metadata:
            tags = chunk.metadata["tags"]
            if isinstance(tags, list):
                meta["tags"] = ",".join(str(t) for t in tags)
            else:
                meta["tags"] = str(tags)
        return meta
```

**Step 2: Run contract tests**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest tests/test_store.py -v`
Expected: All 11 tests PASS

**Step 3: Run ALL tests**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest -v`
Expected: All tests pass (store tests + CLI tests + indexer tests)

**Step 4: Commit**

```bash
cd /Users/ernestkoe/Projects/obsidian-notes-rag
git add src/obsidian_rag/store.py
git commit -m "feat: replace ChromaDB with sqlite-vec vector store

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Replace custom chunker with Chonkie RecursiveChunker

**Files:**
- Modify: `src/obsidian_rag/indexer.py`
- Modify: `tests/test_indexer.py`

**Step 1: Update tests for new chunking behavior**

The existing `TestChunkByHeading` tests reference `chunk_by_heading()` which is being removed. Replace them with tests for the new `chunk_markdown()` function that wraps Chonkie.

Replace the contents of `tests/test_indexer.py` with:

```python
"""Tests for the indexer module."""

import pytest

from obsidian_rag.indexer import chunk_markdown, parse_frontmatter


class TestParseFrontmatter:
    def test_no_frontmatter(self):
        content = "# Hello\n\nThis is content."
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert body == content

    def test_with_frontmatter(self):
        content = """---
title: Test Note
tags:
  - test
  - example
---

# Hello

This is content."""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter["title"] == "Test Note"
        assert frontmatter["tags"] == ["test", "example"]
        assert body.startswith("# Hello")

    def test_invalid_yaml(self):
        content = """---
invalid: yaml: content
---

Content here."""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}


class TestChunkMarkdown:
    def test_single_chunk_no_headings(self):
        """Short content without headings becomes one chunk."""
        content = "This is a simple note without headings."
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) == 1
        assert "simple note" in chunks[0].content

    def test_splits_on_headings(self):
        """Content with headings gets split at heading boundaries."""
        content = """## First Section

Content for first section with enough text to stand alone.

## Second Section

Content for second section with enough text to stand alone."""
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) >= 2

    def test_preserves_file_path(self):
        content = "## Test\n\nContent here that is long enough."
        chunks = chunk_markdown(content, "notes/test.md")
        assert all(c.file_path == "notes/test.md" for c in chunks)

    def test_extracts_frontmatter(self):
        """Frontmatter is parsed and stored in metadata, not in chunk content."""
        content = """---
tags:
  - project
---

## Section

Actual content here."""
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) >= 1
        # Frontmatter should not appear in chunk content
        assert "---" not in chunks[0].content or "tags" not in chunks[0].content

    def test_assigns_type_metadata(self):
        """Chunks get type metadata based on file path."""
        daily = chunk_markdown("Some content here.", "Daily Notes/2026-01-01.md")
        note = chunk_markdown("Some content here.", "Projects/foo.md")
        assert daily[0].metadata["type"] == "daily"
        assert note[0].metadata["type"] == "note"

    def test_empty_content_returns_empty(self):
        """Empty or whitespace-only content returns no chunks."""
        assert chunk_markdown("", "test.md") == []
        assert chunk_markdown("   \n\n  ", "test.md") == []
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest tests/test_indexer.py -v`
Expected: FAIL - `chunk_markdown` not found

**Step 3: Rewrite chunking in indexer.py**

Replace the chunking functions in `src/obsidian_rag/indexer.py`. Remove: `chunk_by_heading()`, `split_oversized_chunk()`, `_get_recursive_chunker()`, and the `_recursive_chunker` global. Add `chunk_markdown()`.

The key changes:
- Remove the `_recursive_chunker` lazy global (lines 52-65)
- Remove `split_oversized_chunk()` (lines 68-106)
- Remove `chunk_by_heading()` (lines 109-186)
- Add `chunk_markdown()` which uses Chonkie `RecursiveChunker` with the `markdown` recipe

New function to add (replacing lines 52-186):

```python
def chunk_markdown(content: str, file_path: str) -> List[Chunk]:
    """Split markdown content into chunks using Chonkie RecursiveChunker.

    Args:
        content: Raw markdown content (may include frontmatter)
        file_path: Relative path to the source file

    Returns:
        List of Chunk objects
    """
    frontmatter, body = parse_frontmatter(content)

    if not body.strip():
        return []

    chunker = RecursiveChunker(
        chunk_size=MAX_CHUNK_TOKENS,
        rules=RecursiveRules.from_recipe("markdown"),
        min_characters_per_chunk=50,
    )

    chonkie_chunks = chunker.chunk(body)

    # Determine note type from path
    note_type = "daily" if file_path.startswith("Daily Notes/") else "note"

    chunks = []
    for i, cc in enumerate(chonkie_chunks):
        text = cc.text.strip()
        if not text:
            continue

        # Try to extract heading from chunk start
        heading = None
        heading_level = 0
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                if 1 <= level <= 6:
                    heading = line.lstrip("# ").strip()
                    heading_level = level
                break

        chunk_id = _generate_chunk_id(file_path, heading, text, i)
        meta = {**frontmatter, "type": note_type, "file_path": file_path}

        chunks.append(Chunk(
            id=chunk_id,
            content=text,
            file_path=file_path,
            heading=heading,
            heading_level=heading_level,
            metadata=meta,
        ))

    return chunks
```

Also update the import at the top of the file - add `RecursiveRules`:

```python
from chonkie import RecursiveChunker, RecursiveRules
from chonkie import Chunk as ChonkieChunk  # keep if still used, otherwise remove
```

And update `VaultIndexer.index_file()` to call `chunk_markdown` instead of `chunk_by_heading`:

```python
    def index_file(self, file_path: Path) -> List[Tuple[Chunk, List[float]]]:
        """Index a single file, returning chunks with embeddings."""
        content = file_path.read_text(encoding="utf-8")
        rel_path = str(file_path.relative_to(self.vault_path))

        chunks = chunk_markdown(content, rel_path)

        results = []
        for chunk in chunks:
            embedding = self.embedder.embed(chunk.content)
            results.append((chunk, embedding))

        return results
```

Note: The type assignment (`daily`/`note`) now happens inside `chunk_markdown()` instead of `index_file()`.

**Step 4: Run indexer tests**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest tests/test_indexer.py -v`
Expected: All PASS

**Step 5: Run ALL tests**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest -v`
Expected: All tests pass

**Step 6: Commit**

```bash
cd /Users/ernestkoe/Projects/obsidian-notes-rag
git add src/obsidian_rag/indexer.py tests/test_indexer.py
git commit -m "feat: replace custom chunker with Chonkie RecursiveChunker

Uses the built-in markdown recipe which splits by heading levels,
then paragraphs, then lines, then sentences.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Update dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml dependencies**

In `pyproject.toml`, replace:
```toml
    "chromadb>=0.6.0,<1.0.0",
```
With:
```toml
    "sqlite-vec>=0.1.6",
```

Keep `chonkie>=1.0.0` (already present).

**Step 2: Sync dependencies**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv sync`
Expected: chromadb and its transitive deps removed, sqlite-vec added

**Step 3: Run all tests**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest -v`
Expected: All tests pass (no more chromadb imports anywhere)

**Step 4: Commit**

```bash
cd /Users/ernestkoe/Projects/obsidian-notes-rag
git add pyproject.toml uv.lock
git commit -m "build: replace chromadb dependency with sqlite-vec

Removes chromadb and its heavy transitive dependencies (fastapi, grpc,
opentelemetry, posthog telemetry). Adds sqlite-vec (~200KB, no deps).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Smoke test

**No files to modify - verification only.**

**Step 1: Reinstall as standalone tool**

Run: `uv tool install --force /Users/ernestkoe/Projects/obsidian-notes-rag`

**Step 2: Re-index vault**

Run: `obsidian-rag index --clear`
Expected: Indexes all vault files, reports chunk count

**Step 3: Test all CLI commands**

```bash
obsidian-rag stats
obsidian-rag search "project planning" -n 3
obsidian-rag similar "Main/Proof App Platform.md" -n 3
obsidian-rag context "Main/Proof App Platform.md" -n 3
```

Expected: All commands return results. Search results should be relevant.

**Step 4: Test path-filter**

Run: `obsidian-rag index --path-filter "Daily Notes/" --clear`
Expected: Only indexes Daily Notes files
