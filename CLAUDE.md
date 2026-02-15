# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

obsidian-notes-rag is an MCP (Model Context Protocol) server that provides semantic search over Obsidian notes. It uses OpenAI embeddings by default (or Ollama/LM Studio for local processing) with sqlite-vec for vector storage.

**PyPI:** https://pypi.org/project/obsidian-notes-rag/

## Commands

```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest -v

# Type checking
uv run pyright

# Interactive setup wizard
uv run obsidian-rag setup

# Index vault (manual refresh)
uv run obsidian-rag index

# Run the MCP server (stdio transport)
uv run obsidian-rag serve

# Watch vault for changes
uv run obsidian-rag watch

# Search from CLI
uv run obsidian-rag search "query"
obsidian-rag similar "Path/To/Note.md"
obsidian-rag context "Path/To/Note.md"
```

## Architecture

### Data Flow

```
Obsidian Vault → VaultIndexer → Embedder (OpenAI/Ollama/LMStudio) → VectorStore (sqlite-vec)
                                                                          ↓
MCP Client ← FastMCP Server ← search_notes/get_similar/etc.
```

### Key Components (src/obsidian_rag/)

- **config.py**: `Config` dataclass, `load_config()`/`save_config()` for TOML config file, cross-platform paths via `platformdirs`
- **indexer.py**: `VaultIndexer` scans markdown files, `chunk_markdown()` uses Chonkie RecursiveChunker with markdown-aware rules, `OpenAIEmbedder`/`OllamaEmbedder`/`LMStudioEmbedder` generate embeddings, `create_embedder()` factory selects provider
- **store.py**: `VectorStore` wraps sqlite-vec with KNN vector search, two tables (chunks metadata + chunks_vec virtual table), handles upsert/delete by file path
- **server.py**: FastMCP server exposing 5 tools: `search_notes`, `get_similar`, `get_note_context`, `get_stats`, `reindex`
- **watcher.py**: `VaultWatcher` uses watchdog with debouncing (default 2s) to incrementally re-index on file changes
- **cli.py**: Click-based CLI with `setup` wizard, `--provider` option, commands for indexing, searching, similar, context, watching, and service management

### Chunking Strategy

Files are chunked using Chonkie's RecursiveChunker with markdown-aware splitting rules. Maximum chunk size is 1500 tokens with a minimum of 50 characters per chunk. The chunker splits by heading levels > paragraphs > lines > sentences > words.

### Metadata

Each chunk stores: `file_path`, `heading`, `heading_level`, `type` ("daily" if path starts with "Daily Notes/", else "note"), and tags from YAML frontmatter.

## Configuration

Config file location (created by `setup` command):
- macOS/Linux: `~/.config/obsidian-notes-rag/config.toml`
- Windows: `%APPDATA%/obsidian-notes-rag/config.toml`

Environment variables (override config file):
- `OPENAI_API_KEY` - OpenAI API key (required for default provider)
- `OBSIDIAN_RAG_PROVIDER` - `openai` (default) or `ollama`
- `OBSIDIAN_RAG_VAULT` - Path to Obsidian vault
- `OBSIDIAN_RAG_DATA` - sqlite-vec storage path
- `OBSIDIAN_RAG_OLLAMA_URL` - Ollama API (default: `http://localhost:11434`)
- `OBSIDIAN_RAG_MODEL` - Override embedding model

## Testing

Tests are in `tests/`:
- `test_store.py` - VectorStore contract tests (sqlite-vec backend)
- `test_indexer.py` - Frontmatter parsing and chunk_markdown tests
- `test_cli.py` - CLI command tests

```bash
# Run all tests
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_store.py -v
```
