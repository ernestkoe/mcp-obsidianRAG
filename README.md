# Obsidian Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Local vector store for Obsidian notes with semantic search via MCP server.

## Features

- Semantic search across your Obsidian vault
- ChromaDB-backed vector storage
- Local embeddings via Ollama (nomic-embed-text)
- MCP server for AI assistant integration
- File watcher daemon for auto-indexing

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai/) with `nomic-embed-text` model
- [uv](https://github.com/astral-sh/uv) (recommended)

## Installation

```bash
# Clone the repository
git clone https://github.com/ernestkoe/obsidian-memory.git
cd obsidian-memory

# Install with uv
uv sync

# Pull the embedding model
ollama pull nomic-embed-text
```

## Usage

### Index your vault

```bash
uv run obsidian-memory index
```

### Search notes

```bash
uv run obsidian-memory search "your query"
```

### Watch for changes (daemon)

```bash
uv run obsidian-memory watch
```

### Install as macOS service

```bash
uv run obsidian-memory install-service
```

### View statistics

```bash
uv run obsidian-memory stats
```

## MCP Server

Add to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "obsidian-memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/obsidian-memory", "obsidian-memory-mcp"]
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `search_notes` | Semantic search with optional type filter |
| `get_similar` | Find notes similar to a given note |
| `get_note_context` | Get note content plus related context |
| `get_stats` | Collection statistics |
| `reindex` | Re-index vault (with optional clear and path filter) |

## Configuration

Default configuration is in `pyproject.toml`:

```toml
[tool.obsidian-memory]
vault_path = "/path/to/your/vault"
data_path = "./data"
ollama_url = "http://localhost:11434"
embedding_model = "nomic-embed-text"
exclude = ["attachments/**", ".obsidian/**", ".trash/**"]
```

## License

MIT
