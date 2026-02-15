"""Tests for VectorStore - defines the contract for the sqlite-vec backend."""

import tempfile
from pathlib import Path

import pytest

from obsidian_rag.indexer import Chunk
from obsidian_rag.store import VectorStore


# Use 4-dimensional vectors for simplicity in tests.
# Real embeddings are 768 or 1536 dims, but the logic is the same.
DIM = 4


def make_chunk(id: str, content: str, file_path: str, heading: str = "",
               heading_level: int = 0, type: str = "note", tags: str = "") -> Chunk:
    """Helper to create a Chunk with metadata."""
    meta = {"type": type}
    if tags:
        meta["tags"] = tags
    return Chunk(
        id=id,
        content=content,
        file_path=file_path,
        heading=heading,
        heading_level=heading_level,
        metadata=meta,
    )


@pytest.fixture
def store(tmp_path):
    """Create a VectorStore in a temp directory."""
    return VectorStore(data_path=str(tmp_path))


class TestUpsert:
    def test_upsert_and_search(self, store):
        """Insert a chunk, search for it, get it back."""
        chunk = make_chunk("c1", "Hello world", "notes/hello.md")
        embedding = [1.0, 0.0, 0.0, 0.0]
        store.upsert(chunk, embedding)

        results = store.search([1.0, 0.0, 0.0, 0.0], limit=1)
        assert len(results) == 1
        assert results[0]["content"] == "Hello world"
        assert results[0]["metadata"]["file_path"] == "notes/hello.md"

    def test_upsert_batch(self, store):
        """Batch insert multiple chunks."""
        chunks = [
            make_chunk("c1", "First note", "a.md"),
            make_chunk("c2", "Second note", "b.md"),
            make_chunk("c3", "Third note", "c.md"),
        ]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        store.upsert_batch(chunks, embeddings)

        stats = store.get_stats()
        assert stats["count"] == 3

    def test_upsert_overwrites(self, store):
        """Upserting the same ID updates content."""
        chunk_v1 = make_chunk("c1", "Version 1", "note.md")
        chunk_v2 = make_chunk("c1", "Version 2", "note.md")
        embedding = [1.0, 0.0, 0.0, 0.0]

        store.upsert(chunk_v1, embedding)
        store.upsert(chunk_v2, embedding)

        results = store.search([1.0, 0.0, 0.0, 0.0], limit=1)
        assert results[0]["content"] == "Version 2"
        assert store.get_stats()["count"] == 1


class TestSearch:
    def test_search_returns_nearest(self, store):
        """Search returns results ordered by similarity."""
        chunks = [
            make_chunk("c1", "Very relevant", "a.md"),
            make_chunk("c2", "Somewhat relevant", "b.md"),
            make_chunk("c3", "Not relevant", "c.md"),
        ]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],   # closest to query
            [0.7, 0.7, 0.0, 0.0],   # medium distance
            [0.0, 0.0, 0.0, 1.0],   # far from query
        ]
        store.upsert_batch(chunks, embeddings)

        results = store.search([1.0, 0.0, 0.0, 0.0], limit=3)
        assert len(results) == 3
        assert results[0]["content"] == "Very relevant"
        # Distance should increase
        assert results[0]["distance"] <= results[1]["distance"]
        assert results[1]["distance"] <= results[2]["distance"]

    def test_search_limit(self, store):
        """Respects the limit parameter."""
        chunks = [make_chunk(f"c{i}", f"Note {i}", f"{i}.md") for i in range(10)]
        embeddings = [[float(i == j) for j in range(DIM)] for i in range(10)]
        # Only first 4 have unique directions, rest will overlap - that's fine
        store.upsert_batch(chunks, embeddings)

        results = store.search([1.0, 0.0, 0.0, 0.0], limit=3)
        assert len(results) == 3

    def test_search_with_where_filter(self, store):
        """Filtering by metadata field."""
        chunks = [
            make_chunk("c1", "Daily entry", "Daily Notes/2026-01-01.md", type="daily"),
            make_chunk("c2", "Project note", "Projects/foo.md", type="note"),
        ]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
        ]
        store.upsert_batch(chunks, embeddings)

        # Without filter - both returned
        all_results = store.search([1.0, 0.0, 0.0, 0.0], limit=10)
        assert len(all_results) == 2

        # With filter - only daily
        daily_results = store.search([1.0, 0.0, 0.0, 0.0], limit=10, where={"type": "daily"})
        assert len(daily_results) == 1
        assert daily_results[0]["metadata"]["type"] == "daily"

    def test_search_with_file_path_filter(self, store):
        """Filtering by file_path (used by similar/context commands)."""
        chunks = [
            make_chunk("c1", "Chunk from target", "target.md"),
            make_chunk("c2", "Chunk from other", "other.md"),
        ]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
        ]
        store.upsert_batch(chunks, embeddings)

        results = store.search([1.0, 0.0, 0.0, 0.0], limit=10, where={"file_path": "target.md"})
        assert len(results) == 1
        assert results[0]["metadata"]["file_path"] == "target.md"

    def test_empty_search(self, store):
        """Searching an empty store returns empty list."""
        results = store.search([1.0, 0.0, 0.0, 0.0], limit=5)
        assert results == []


class TestDelete:
    def test_delete_by_file(self, store):
        """Deletes all chunks for a given file_path."""
        chunks = [
            make_chunk("c1", "Keep this", "keep.md"),
            make_chunk("c2", "Delete this", "delete.md"),
            make_chunk("c3", "Also delete", "delete.md"),
        ]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        store.upsert_batch(chunks, embeddings)
        assert store.get_stats()["count"] == 3

        store.delete_by_file("delete.md")
        assert store.get_stats()["count"] == 1

        results = store.search([1.0, 0.0, 0.0, 0.0], limit=10)
        assert len(results) == 1
        assert results[0]["metadata"]["file_path"] == "keep.md"


class TestClear:
    def test_clear_removes_all(self, store):
        """Clear empties the store completely."""
        chunks = [make_chunk(f"c{i}", f"Note {i}", f"{i}.md") for i in range(5)]
        embeddings = [[1.0, 0.0, 0.0, 0.0]] * 5
        store.upsert_batch(chunks, embeddings)
        assert store.get_stats()["count"] == 5

        store.clear()
        assert store.get_stats()["count"] == 0


class TestStats:
    def test_get_stats(self, store):
        """Returns correct statistics."""
        stats = store.get_stats()
        assert stats["count"] == 0
        assert "data_path" in stats
        assert "collection" in stats
