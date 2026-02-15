[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/obsidian-notes-rag)](https://pypi.org/project/obsidian-notes-rag/)

# obsidian-notes-rag

MCP server and CLI for semantic search over your Obsidian vault. Generates embeddings with OpenAI, Ollama, or LM Studio. Stores vectors locally in sqlite-vec (~200KB, no telemetry, no network calls).

## What it does

Search your notes by meaning, not just keywords:

```bash
obsidian-rag search "project architecture decisions" -n 5
obsidian-rag similar "Projects/Platform Hub.md"
obsidian-rag context "Daily Notes/2026-02-14.md"
```

As an MCP server, it gives Claude the same capabilities — Claude can search your notes, find related content, and pull context during conversations.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (for running and installing)
- One of: `OPENAI_API_KEY`, [Ollama](https://ollama.ai/), or [LM Studio](https://lmstudio.ai/) for embeddings

## Setup

### 1. Run the setup wizard

```bash
uvx obsidian-notes-rag setup
```

This creates a config at `~/.config/obsidian-notes-rag/config.toml` with your vault path, embedding provider, and API key.

### 2. Build the index

```bash
uvx obsidian-notes-rag index
```

Parses your markdown files, chunks them by heading structure (using [Chonkie](https://github.com/chonkie-ai/chonkie) RecursiveChunker), generates embeddings, and stores everything in a local SQLite database.

### 3. Connect to Claude

**Claude Code (CLI):**

```bash
claude mcp add -s user obsidian-notes-rag -- uvx obsidian-notes-rag serve
```

**Claude Desktop (JSON config):**

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "obsidian-notes-rag": {
      "command": "uvx",
      "args": ["obsidian-notes-rag", "serve"]
    }
  }
}
```

### 4. Install the CLI (optional)

If you want `obsidian-rag` available as a standalone command:

```bash
uv tool install obsidian-notes-rag
```

This installs both `obsidian-rag` and `obsidian-notes-rag` to `~/.local/bin/`.

## CLI Reference

```bash
# Search
obsidian-rag search "query"                  # semantic search
obsidian-rag search "standup" --type daily   # filter by note type
obsidian-rag search "design" -n 10           # more results

# Explore
obsidian-rag similar "Path/To/Note.md"       # find related notes
obsidian-rag context "Path/To/Note.md"       # show note + related context

# Index
obsidian-rag index                            # re-index vault
obsidian-rag index --clear                    # rebuild from scratch
obsidian-rag index --path-filter "Daily Notes/"  # index subset

# Info
obsidian-rag stats                            # show index size

# Services
obsidian-rag serve                            # start MCP server
obsidian-rag watch                            # watch for changes, auto-reindex
obsidian-rag install-service                  # macOS launchd auto-start
obsidian-rag uninstall-service                # remove service
obsidian-rag service-status                   # check service status
```

## MCP Tools

Once connected, Claude has access to:

| Tool | What it does |
|------|--------------|
| `search_notes` | Find notes matching a query |
| `get_similar` | Find notes similar to a given note |
| `get_note_context` | Get a note with related context |
| `get_stats` | Show index statistics |
| `reindex` | Rebuild the index |

## Keeping the Index Fresh

**Manual:** `obsidian-rag index`

**Auto-reindex on file changes:** `obsidian-rag watch` (run in a terminal or background)

**macOS background service:** `obsidian-rag install-service` (starts on login, appears in System Settings > Login Items)

## Using Ollama (local, no API key)

```bash
ollama pull nomic-embed-text
obsidian-rag --provider ollama index
```

## Using LM Studio (local, no API key)

Load an embedding model in LM Studio, then:

```bash
obsidian-rag --provider lmstudio index
```

## Configuration

The setup wizard writes to `~/.config/obsidian-notes-rag/config.toml`. You can also override with environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `OBSIDIAN_RAG_PROVIDER` | `openai` (default), `ollama`, or `lmstudio` |
| `OBSIDIAN_RAG_VAULT` | Path to Obsidian vault |
| `OBSIDIAN_RAG_DATA` | Index storage path (default: platform-specific) |
| `OBSIDIAN_RAG_OLLAMA_URL` | Ollama URL (default: `http://localhost:11434`) |
| `OBSIDIAN_RAG_LMSTUDIO_URL` | LM Studio URL (default: `http://localhost:1234`) |
| `OBSIDIAN_RAG_MODEL` | Override embedding model |

## How it works

1. Parses markdown files, strips YAML frontmatter
2. Chunks content using Chonkie's RecursiveChunker (splits by headings > paragraphs > lines > sentences, max 1500 tokens per chunk)
3. Generates embeddings via your chosen provider
4. Stores metadata in SQLite, vectors in sqlite-vec (KNN search via vec0 virtual tables)
5. MCP server and CLI both query the same local database

## Upgrading

If you installed the CLI with `uv tool install`, upgrade with:

```bash
uv tool upgrade obsidian-notes-rag
```

If you use `uvx` to run commands or the MCP server, it automatically uses the latest version.

### Upgrading to v1.0.0

v1.0.0 replaces ChromaDB with sqlite-vec. After upgrading, rebuild your index:

```bash
obsidian-rag index --clear
```

The old ChromaDB data at `~/.local/share/obsidian-notes-rag/` (or your configured path) can be deleted.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

## Support

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/ernestkoe)

## License

MIT
