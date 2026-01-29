# Obsidian Memory - Vector Store for AI-Assisted Notes

## Overview

Build a local vector store system for Obsidian notes that enables semantic search and AI-assisted note management.

## Architecture

```
┌─────────────────────┐
│   Obsidian Vault    │  /Users/ernestkoe/Documents/Obsidian/ProofKit/
└──────────┬──────────┘
           │ file watcher (watchdog)
           ▼
┌─────────────────────┐
│   Indexer Service   │  Background daemon
│   • Parse markdown  │
│   • Chunk by heading│
│   • Generate embeds │  via Ollama/OpenAI/LMStudio/CoreML
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ChromaDB          │  Local persistent store
│   ~/Projects/obsidian-memory/data/
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   MCP Server        │  stdio transport
│   • search_notes    │
│   • get_similar     │
│   • get_note_context│
└─────────────────────┘
```

## Project Structure

```
~/Projects/obsidian-memory/
├── pyproject.toml          # Project config, dependencies
├── README.md               # Documentation
├── src/
│   └── obsidian_memory/
│       ├── __init__.py
│       ├── indexer.py      # Markdown parsing, chunking, embedding
│       ├── store.py        # ChromaDB wrapper
│       ├── watcher.py      # File system watcher
│       ├── server.py       # MCP server
│       └── cli.py          # Command-line interface
├── data/                   # ChromaDB persistent storage
└── tests/
    └── test_indexer.py
```

## Components

### 1. Indexer (`indexer.py`)
- Parse markdown files
- Chunk by heading (preserve context)
- Generate embeddings via Ollama
- Store metadata: file path, heading, modified date, tags

### 2. Store (`store.py`)
- ChromaDB wrapper with collections:
  - `daily_notes` - Daily journal entries
  - `notes` - All other notes
- CRUD operations: add, update, delete, query
- Metadata filtering (by date, path, tags)

### 3. Watcher (`watcher.py`)
- Monitor Obsidian vault for changes
- Trigger incremental reindex on file changes
- Handle creates, updates, deletes, renames

### 4. MCP Server (`server.py`)
Tools exposed:
- `search_notes(query, limit, collection)` - Semantic search
- `get_similar(note_path, limit)` - Find related notes
- `get_note_context(note_path)` - Get note + related context
- `reindex(path?)` - Force reindex (optional path filter)
- `get_stats()` - Collection stats

### 5. CLI (`cli.py`)
Commands:
- `obsidian-memory index` - Full reindex
- `obsidian-memory search "query"` - Test search
- `obsidian-memory serve` - Start MCP server
- `obsidian-memory watch` - Start watcher daemon

## Configuration

```toml
# config.toml or pyproject.toml [tool.obsidian-memory]
vault_path = "/Users/ernestkoe/Documents/Obsidian/ProofKit"
data_path = "./data"
ollama_url = "http://localhost:11434"
embedding_model = "nomic-embed-text"

[collections]
daily_notes = "Daily Notes/**/*.md"
notes = "**/*.md"
exclude = ["attachments/**", ".obsidian/**"]
```

## Dependencies

- `chromadb` - Vector store
- `watchdog` - File system monitoring
- `mcp` - Model Context Protocol SDK
- `httpx` - Ollama/LMStudio API calls
- `openai` - OpenAI embeddings
- `pyyaml` - Frontmatter parsing
- `click` - CLI framework

**Optional:**
- `coremltools` - CoreML model conversion (for `[coreml]` extra)

## Implementation Phases

### Phase 1: Core Indexer (MVP)
1. Set up project structure with pyproject.toml
2. Implement markdown parser with heading-based chunking
3. Implement Ollama embedding client
4. Implement ChromaDB store wrapper
5. Create CLI with `index` and `search` commands
6. Test with Daily Notes folder

### Phase 2: MCP Server
1. Implement MCP server with search tools
2. Add to Claude Code MCP config
3. Test semantic search from Claude

### Phase 3: File Watcher
1. Implement watchdog-based file monitoring
2. Incremental updates (not full reindex)
3. Daemon mode for background operation

### Phase 4: Polish
1. Handle edge cases (empty files, binary files, etc.)
2. Add stats/health endpoints
3. Optimize chunking strategy based on real usage

### Phase 5: CoreML Embedding Provider (Future)
Add native macOS embedding support to eliminate external dependencies.

**Goals:**
- No need for Ollama or OpenAI API
- Runs on Neural Engine (power efficient on Apple Silicon)
- Fully offline operation

**Approach:**
1. Convert `all-MiniLM-L6-v2` (22M params) to CoreML format using `coremltools`
2. Add `coreml` as a provider option alongside openai/ollama/lmstudio
3. Create `CoreMLEmbedder` class following existing embedder pattern
4. Bundle or download the converted model on first use

**Tradeoffs:**
| Aspect | CoreML | Ollama/nomic-embed-text |
|--------|--------|-------------------------|
| Quality | Good (MiniLM) | Better (nomic) |
| Dependencies | None | Ollama running |
| Power usage | Low (Neural Engine) | Higher |
| Model size | ~90MB | ~270MB |

**Alternative models:**
- `bge-small-en-v1.5` - similar size, slightly better quality
- Apple's built-in `NLEmbedding` - zero dependencies but older/lower quality

**Implementation notes:**
- Add `coremltools` as optional dependency: `pip install obsidian-notes-rag[coreml]`
- Model stored in `~/Library/Application Support/obsidian-notes-rag/models/`
- First-run downloads or converts the model

## Design Decisions

1. **Chunking**: By heading (## and ###) - preserves semantic sections
2. **Scope**: Full vault - all markdown files
3. **Collections**: Single collection with metadata filtering (type, path, date)
