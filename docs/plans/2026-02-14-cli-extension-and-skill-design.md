# CLI Extension + Claude Code Skill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the obsidian-notes-rag CLI to cover the full MCP tool surface, add a shorter `obsidian-rag` alias, and create a Claude Code skill documenting CLI usage.

**Architecture:** Add `similar` and `context` commands to `cli.py` following the same pattern as the existing `search` command (init embedder, init store, query, display). Add `--path-filter` to `index`. Register alias in `pyproject.toml`. Create standalone skill file.

**Tech Stack:** Python, Click, ChromaDB, OpenAI/Ollama embeddings

---

### Task 1: Add `obsidian-rag` alias

**Files:**
- Modify: `pyproject.toml:34-35`

**Step 1: Add alias entry point**

In `pyproject.toml`, change:

```toml
[project.scripts]
obsidian-notes-rag = "obsidian_rag.cli:main"
```

To:

```toml
[project.scripts]
obsidian-notes-rag = "obsidian_rag.cli:main"
obsidian-rag = "obsidian_rag.cli:main"
```

**Step 2: Reinstall to register alias**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv sync`

**Step 3: Verify alias works**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run obsidian-rag --help`
Expected: Same output as `obsidian-notes-rag --help`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add obsidian-rag as shorter CLI alias"
```

---

### Task 2: Add `--path-filter` to `index` command

**Files:**
- Modify: `src/obsidian_rag/cli.py:319-396` (index command)
- Test: `tests/test_cli.py` (new file)

**Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
"""Tests for CLI commands."""

from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from obsidian_rag.cli import main


class TestIndexCommand:
    def test_index_with_path_filter(self):
        """Verify --path-filter option is accepted and passed through."""
        runner = CliRunner()
        with patch("obsidian_rag.cli.create_embedder") as mock_embedder, \
             patch("obsidian_rag.cli.VectorStore") as mock_store, \
             patch("obsidian_rag.cli.VaultIndexer") as mock_indexer:
            mock_embedder.return_value = MagicMock()
            mock_embedder.return_value.close = MagicMock()
            mock_store.return_value = MagicMock()
            mock_store.return_value.get_stats.return_value = {"count": 0}
            mock_indexer.return_value.iter_markdown_files.return_value = []

            result = runner.invoke(main, ["index", "--path-filter", "Daily Notes/"])
            assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest tests/test_cli.py::TestIndexCommand::test_index_with_path_filter -v`
Expected: FAIL - no such option `--path-filter`

**Step 3: Add `--path-filter` option to index command**

In `src/obsidian_rag/cli.py`, modify the `index` command. Add the option and filtering logic:

```python
@main.command()
@click.option("--clear", is_flag=True, help="Clear existing index before indexing")
@click.option("--path-filter", default=None, help="Only index files under this path prefix (e.g. 'Daily Notes/')")
@click.pass_context
def index(ctx, clear, path_filter):
```

Then after `files = list(indexer.iter_markdown_files())`, add:

```python
    if path_filter:
        files = [f for f in files if str(f.relative_to(indexer.vault_path)).startswith(path_filter)]
```

Note: `indexer.vault_path` is set in `VaultIndexer.__init__` as `self.vault_path = Path(vault_path)`.

**Step 4: Run test to verify it passes**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest tests/test_cli.py::TestIndexCommand::test_index_with_path_filter -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/obsidian_rag/cli.py tests/test_cli.py
git commit -m "feat: add --path-filter option to index command"
```

---

### Task 3: Add `similar` command

**Files:**
- Modify: `src/obsidian_rag/cli.py` (add command after `search`)
- Test: `tests/test_cli.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
class TestSimilarCommand:
    def test_similar_shows_results(self):
        """Verify similar command accepts note-path and displays results."""
        runner = CliRunner()
        with patch("obsidian_rag.cli.create_embedder") as mock_embedder, \
             patch("obsidian_rag.cli.VectorStore") as mock_store:
            embedder_instance = MagicMock()
            embedder_instance.embed.return_value = [0.1] * 1536
            embedder_instance.close = MagicMock()
            mock_embedder.return_value = embedder_instance

            store_instance = MagicMock()
            # First search: find chunks of the source note
            # Second search: find similar notes
            store_instance.search.side_effect = [
                [{"content": "Note content", "metadata": {"file_path": "test.md", "heading": ""}, "distance": 0.0}],
                [{"content": "Similar note", "metadata": {"file_path": "other.md", "heading": "Section"}, "distance": 0.2}],
            ]
            mock_store.return_value = store_instance

            result = runner.invoke(main, ["similar", "test.md"])
            assert result.exit_code == 0
            assert "other.md" in result.output
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest tests/test_cli.py::TestSimilarCommand::test_similar_shows_results -v`
Expected: FAIL - no such command `similar`

**Step 3: Implement `similar` command**

Add to `src/obsidian_rag/cli.py` after the `search` command (after line 471). Follow the same pattern as `search` for embedder/store initialization. The logic mirrors `server.py:get_similar`:

```python
@main.command()
@click.argument("note_path")
@click.option("--limit", "-n", default=5, help="Number of similar notes")
@click.pass_context
def similar(ctx, note_path, limit):
    """Find notes similar to a given note."""
    data_path = ctx.obj["data"]
    provider = ctx.obj["provider"]
    ollama_url = ctx.obj["ollama_url"]
    lmstudio_url = ctx.obj["lmstudio_url"]
    config = ctx.obj["config"]

    model = ctx.obj["model"]
    if model is None:
        if provider == "openai":
            model = config.openai_model
        elif provider == "ollama":
            model = config.ollama_model
        elif provider == "lmstudio":
            model = config.lmstudio_model

    if provider == "ollama":
        base_url = ollama_url
    elif provider == "lmstudio":
        base_url = lmstudio_url
    else:
        base_url = None

    embedder = create_embedder(provider=provider, model=model, base_url=base_url)
    store = VectorStore(data_path=data_path)

    # Get chunks from the source note
    click.echo(f"Finding notes similar to: {note_path}\n")
    results = store.search(
        query_embedding=embedder.embed(note_path),
        limit=50,
        where={"file_path": note_path}
    )

    if not results:
        click.echo(f"Note not found in index: {note_path}")
        embedder.close()
        return

    # Combine content and search for similar
    note_content = "\n\n".join(r["content"] for r in results)
    note_embedding = embedder.embed(note_content[:8000])
    all_results = store.search(note_embedding, limit=limit + 10)

    similar_notes = [r for r in all_results if r["metadata"]["file_path"] != note_path][:limit]

    if not similar_notes:
        click.echo("No similar notes found.")
        embedder.close()
        return

    for i, result in enumerate(similar_notes, 1):
        meta = result["metadata"]
        similarity = 1 - result["distance"]
        click.echo(f"{'─' * 60}")
        click.echo(f"[{i}] {meta['file_path']}")
        if meta.get("heading"):
            click.echo(f"    Section: {meta['heading']}")
        click.echo(f"    Similarity: {similarity:.2%}")
        content = result["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        click.echo(f"\n    {content}\n")

    embedder.close()
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest tests/test_cli.py::TestSimilarCommand -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/obsidian_rag/cli.py tests/test_cli.py
git commit -m "feat: add similar command to CLI"
```

---

### Task 4: Add `context` command

**Files:**
- Modify: `src/obsidian_rag/cli.py` (add command after `similar`)
- Test: `tests/test_cli.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
class TestContextCommand:
    def test_context_shows_note_and_similar(self):
        """Verify context command shows note content and similar notes."""
        runner = CliRunner()
        with patch("obsidian_rag.cli.create_embedder") as mock_embedder, \
             patch("obsidian_rag.cli.VectorStore") as mock_store:
            embedder_instance = MagicMock()
            embedder_instance.embed.return_value = [0.1] * 1536
            embedder_instance.close = MagicMock()
            mock_embedder.return_value = embedder_instance

            store_instance = MagicMock()
            store_instance.search.side_effect = [
                # First call: get note chunks
                [{"content": "Note content here", "metadata": {"file_path": "test.md", "heading": ""}, "distance": 0.0}],
                # Second call: get similar (reused for similar lookup inside context)
                [{"content": "Note content here", "metadata": {"file_path": "test.md", "heading": ""}, "distance": 0.0}],
                # Third call: similar notes search
                [{"content": "Related note", "metadata": {"file_path": "related.md", "heading": "Intro"}, "distance": 0.3}],
            ]
            mock_store.return_value = store_instance

            result = runner.invoke(main, ["context", "test.md"])
            assert result.exit_code == 0
            assert "Note content here" in result.output
            assert "related.md" in result.output
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest tests/test_cli.py::TestContextCommand -v`
Expected: FAIL - no such command `context`

**Step 3: Implement `context` command**

Add to `src/obsidian_rag/cli.py` after the `similar` command. This combines getting the note content with finding similar notes:

```python
@main.command()
@click.argument("note_path")
@click.option("--limit", "-n", default=5, help="Number of similar notes to include")
@click.pass_context
def context(ctx, note_path, limit):
    """Get a note and its related context."""
    data_path = ctx.obj["data"]
    provider = ctx.obj["provider"]
    ollama_url = ctx.obj["ollama_url"]
    lmstudio_url = ctx.obj["lmstudio_url"]
    config = ctx.obj["config"]

    model = ctx.obj["model"]
    if model is None:
        if provider == "openai":
            model = config.openai_model
        elif provider == "ollama":
            model = config.ollama_model
        elif provider == "lmstudio":
            model = config.lmstudio_model

    if provider == "ollama":
        base_url = ollama_url
    elif provider == "lmstudio":
        base_url = lmstudio_url
    else:
        base_url = None

    embedder = create_embedder(provider=provider, model=model, base_url=base_url)
    store = VectorStore(data_path=data_path)

    # Get the note's chunks
    click.echo(f"Getting context for: {note_path}\n")
    results = store.search(
        query_embedding=embedder.embed(note_path),
        limit=50,
        where={"file_path": note_path}
    )

    if not results:
        click.echo(f"Note not found in index: {note_path}")
        embedder.close()
        return

    # Display note content
    note_content = "\n\n".join(r["content"] for r in results)
    click.echo(f"{'═' * 60}")
    click.echo(f"Note: {note_path}")
    click.echo(f"{'═' * 60}")
    click.echo(note_content)
    click.echo()

    # Find similar notes
    note_embedding = embedder.embed(note_content[:8000])
    all_results = store.search(note_embedding, limit=limit + 10)
    similar_notes = [r for r in all_results if r["metadata"]["file_path"] != note_path][:limit]

    if similar_notes:
        click.echo(f"{'─' * 60}")
        click.echo(f"Related Notes ({len(similar_notes)})")
        click.echo(f"{'─' * 60}")
        for i, result in enumerate(similar_notes, 1):
            meta = result["metadata"]
            similarity = 1 - result["distance"]
            click.echo(f"  [{i}] {meta['file_path']}")
            if meta.get("heading"):
                click.echo(f"      Section: {meta['heading']}")
            click.echo(f"      Similarity: {similarity:.2%}")
    else:
        click.echo("No related notes found.")

    embedder.close()
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest tests/test_cli.py::TestContextCommand -v`
Expected: PASS

**Step 5: Run all tests**

Run: `cd /Users/ernestkoe/Projects/obsidian-notes-rag && uv run pytest -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/obsidian_rag/cli.py tests/test_cli.py
git commit -m "feat: add context command to CLI"
```

---

### Task 5: Create Claude Code skill

**Files:**
- Create: `~/.claude/skills/obsidian-rag.md`

**Step 1: Create the skill file**

Create `~/.claude/skills/obsidian-rag.md`:

```markdown
name: obsidian-rag
description: Semantic search for Obsidian notes via CLI. Use when doing concept-based searches, finding similar notes, or getting note context. For exact text/filename searches, use obsidian-cli instead.

# Obsidian RAG CLI

Semantic search over Obsidian vault using embeddings + ChromaDB.

## When to Use

| Need | Tool |
|------|------|
| Semantic/concept search | This CLI or obsidian-notes-rag MCP |
| Find similar notes | This CLI (`similar`) |
| Get note + related context | This CLI (`context`) |
| Exact text search | obsidian-cli (`obsidian search`) |
| Read/write files | obsidian-cli |

## Commands

### Search

```bash
# Semantic search
obsidian-rag search "project architecture decisions" -n 10

# Filter by note type
obsidian-rag search "standup notes" --type daily
obsidian-rag search "design patterns" --type note
```

### Similar

```bash
# Find notes similar to a given note
obsidian-rag similar "Projects/Platform Hub.md"
obsidian-rag similar "Daily Notes/2026-02-14.md" -n 10
```

### Context

```bash
# Get note content + related notes
obsidian-rag context "Projects/Platform Hub.md"
obsidian-rag context "Projects/Platform Hub.md" -n 3
```

### Index

```bash
# Full reindex
obsidian-rag index

# Clear and rebuild
obsidian-rag index --clear

# Index only a subfolder
obsidian-rag index --path-filter "Daily Notes/"
```

### Stats

```bash
obsidian-rag stats
```

## Notes

- Paths are relative to vault root (e.g., `Daily Notes/2026-02-14.md`)
- Default limit is 5 for similar/context, 5 for search
- `obsidian-notes-rag` also works (longer alias)
```

**Step 2: Verify skill loads**

Run: `claude /skills` (manual verification - check that `obsidian-rag` appears in the skill list)

**Step 3: Commit skill to dotfiles if applicable, otherwise done**

No git commit needed - skill lives in `~/.claude/skills/` which is user config.

---

### Task 6: Manual smoke test

**No files to modify - verification only.**

**Step 1: Test alias**

Run: `obsidian-rag --help`
Expected: Shows all commands including `similar`, `context`

**Step 2: Test search**

Run: `obsidian-rag search "project planning" -n 3`
Expected: Returns semantic search results

**Step 3: Test similar**

Run: `obsidian-rag similar "Projects/Platform Hub.md"`
Expected: Returns similar notes or "Note not found"

**Step 4: Test context**

Run: `obsidian-rag context "Projects/Platform Hub.md"`
Expected: Shows note content + related notes

**Step 5: Test path-filter**

Run: `obsidian-rag index --path-filter "Daily Notes/" --clear`
Expected: Only indexes files under Daily Notes/
