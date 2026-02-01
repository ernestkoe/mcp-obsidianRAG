# CHANGELOG


## v0.4.3 (2026-02-01)

### Bug Fixes

- Prevent infinite retry loop for deleted/temp files
  ([`8d01202`](https://github.com/ernestkoe/obsidian-notes-rag/commit/8d01202b36103b92feeba51f7d4710c47580dd4a))

- Check file exists before indexing to avoid retry loops for deleted files - Filter Obsidian
  temp/recovery files (.!NNNNN!filename.md pattern) - Remove duplicate StreamHandler to prevent
  unbounded stderr log growth

The retry mechanism had a flaw where _index_file() caught exceptions and called retry_queue.add()
  with attempt=0, preventing the attempt counter from ever incrementing. This caused deleted files
  to retry forever, filling watcher.err to 47GB.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Documentation

- Add Phase 5 CoreML embedding provider to roadmap
  ([`4574ffb`](https://github.com/ernestkoe/obsidian-notes-rag/commit/4574ffb3502ab22dca80e30b63a5a9a778deb266))

Document future plan to add native macOS embedding support using CoreML. This would eliminate
  Ollama/OpenAI dependency, run on Neural Engine, and enable fully offline operation. Includes
  tradeoffs analysis and implementation notes.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.4.2 (2026-01-29)

### Bug Fixes

- Prevent crash loop from filling disk with logs
  ([`0388bc8`](https://github.com/ernestkoe/obsidian-notes-rag/commit/0388bc8c1ac8e9e4cb07643b1e9dd38cc498af28))

Add ThrottleInterval (30s) to launchd plist to prevent rapid restarts from generating massive log
  files. Move logs from /tmp to proper macOS location ~/Library/Logs/obsidian-notes-rag/ and add
  rotating file handler (10MB max, 3 backups) to cap total log size at ~40MB.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.4.1 (2026-01-22)

### Bug Fixes

- Handle ChromaDB version incompatibility gracefully
  ([`bd03c4e`](https://github.com/ernestkoe/obsidian-notes-rag/commit/bd03c4e37a5b75eaf61d04123cb6953ead2f738b))

- Tighten chromadb constraint from >=0.4.0 to >=0.6.0 - Add ChromaDBMigrationError with clear
  instructions when users have databases created with older ChromaDB versions - Error message tells
  users to run `obsidian-notes-rag index --clear`

Fixes KeyError: '_type' caused by schema changes between ChromaDB 0.4.x and 0.6.x where
  config_json_str format changed.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Chores

- Gitignore .atlas directory
  ([`9fd567a`](https://github.com/ernestkoe/obsidian-notes-rag/commit/9fd567ac2eb38187715d5dfa4c1e3c2ff67fdc2a))

Remove .atlas/project.yaml from tracking - this is auto-generated metadata from the atlas project
  manager, not project source code.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.4.0 (2026-01-22)

### Bug Fixes

- Correct semantic-release config and workflow
  ([`9ca110f`](https://github.com/ernestkoe/obsidian-notes-rag/commit/9ca110f4867a2d43b88c5c4680cf29ef413b4e1b))

- Use official python-semantic-release GitHub Action - Configure branch settings properly - Disable
  built-in PyPI upload (using trusted publishing instead)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Remove build_command from semantic-release config
  ([`eac6a77`](https://github.com/ernestkoe/obsidian-notes-rag/commit/eac6a7785872a15eeaa88fb96953bfa880510dd0))

The semantic-release action runs in its own container without uv. Let the workflow handle building
  separately after version bump.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Resolve macOS path case issues and pin ChromaDB version
  ([`e5900c5`](https://github.com/ernestkoe/obsidian-notes-rag/commit/e5900c59fdc0df52b4dabbfc81b082d6a8043b25))

- Add resolve_path_case() to normalize vault paths on macOS where the filesystem is case-insensitive
  but case-preserving. Watchdog requires exact case to detect file changes properly. - Pin ChromaDB
  to <1.0.0 to avoid Rust bindings crashes (issue #5937)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Resolve pyright type error for chromadb Include parameter
  ([`b80b0d9`](https://github.com/ernestkoe/obsidian-notes-rag/commit/b80b0d9a8eab3d2d3bab8ec080e6a1205be602a8))

Cast the include parameter to chromadb's Include type to satisfy pyright's strict type checking.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Resolve pyright type errors
  ([`ebc7a4d`](https://github.com/ernestkoe/obsidian-notes-rag/commit/ebc7a4d4651270cd312607af1e117e7e00dcae11))

- Add task_type parameter to OpenAIEmbedder.embed() for interface consistency - Import ChonkieChunk
  and use chunker.chunk() method for proper typing - Add assertion for vault_path in setup wizard
  service installation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Continuous Integration

- Suppress pyright version warning by forcing latest
  ([`0af1c36`](https://github.com/ernestkoe/obsidian-notes-rag/commit/0af1c36817abe5e44d81ca2b050944e69e7eecf2))

Set PYRIGHT_PYTHON_FORCE_VERSION=latest to avoid version mismatch warnings in CI output.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Documentation

- Update README with LM Studio support and service improvements
  ([`c971fa3`](https://github.com/ernestkoe/obsidian-notes-rag/commit/c971fa360f4a19395994d7e9b1c134efb5073456))

- Add LM Studio as a supported embedding provider - Document the new descriptive service name in
  System Settings - Update configuration table with LM Studio URL option - Clarify that data path is
  platform-specific

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Features

- Show 'obsidian-notes-rag' in Activity Monitor instead of python3
  ([`b9c1502`](https://github.com/ernestkoe/obsidian-notes-rag/commit/b9c15023d5b4479797c9f1fa85ee9c73f67e74c4))

Use setproctitle to set a descriptive process name that appears in macOS Activity Monitor and ps
  output, making the background service easier to identify.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Show descriptive name in macOS System Settings
  ([`590d86c`](https://github.com/ernestkoe/obsidian-notes-rag/commit/590d86ccedb69fea66f4c0f2a3e1b181bb6820d9))

Use a wrapper script named 'obsidian-notes-rag-watcher' instead of calling python directly. This
  makes the service appear with a descriptive name in System Settings > Login Items & Extensions >
  App Background Activity, rather than showing "python3".

The wrapper script is installed to ~/.local/bin/ and the launchd plist now references it instead of
  the Python interpreter.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Switch from release-please to python-semantic-release
  ([`b3e3da6`](https://github.com/ernestkoe/obsidian-notes-rag/commit/b3e3da69bf8f152b185f1d2318d89ebc43a9d861))

Simpler auto-deploy: push a feat/fix commit → auto-release to PyPI. No intermediate PR required.

- Remove release-please config files - Add semantic_release config to pyproject.toml - Update
  release workflow for semantic-release

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.3.0 (2026-01-21)

### Bug Fixes

- Update launchd service name to obsidian-notes-rag
  ([`1ba651a`](https://github.com/ernestkoe/obsidian-notes-rag/commit/1ba651a54b8d0de8116051c1320fd3f06cd2f4c3))

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Chores

- Add TODO for improving macOS login item appearance
  ([`e99467b`](https://github.com/ernestkoe/obsidian-notes-rag/commit/e99467b13b04de8de9fbf34fc17091a874d7f96a))

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **main**: Release 0.3.0
  ([`d62b21f`](https://github.com/ernestkoe/obsidian-notes-rag/commit/d62b21f35477bd205df181417daff3ec431ece24))

### Continuous Integration

- Add manual publish option to release workflow
  ([`1c31b12`](https://github.com/ernestkoe/obsidian-notes-rag/commit/1c31b12d36ac9629f18da822e92a6cf32e8adb5b))

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Rename workflow to match PyPI trusted publisher
  ([`36dbe60`](https://github.com/ernestkoe/obsidian-notes-rag/commit/36dbe6097f7b12e656cec2cbd1d9f5d4ce558daf))

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Features

- Add LM Studio support with auto model detection
  ([`745f5aa`](https://github.com/ernestkoe/obsidian-notes-rag/commit/745f5aab0bce0650d9dad5c681d4debae1c9cd87))

LM Studio support: - Add LMStudioEmbedder class with OpenAI-compatible API - Add
  is_lmstudio_running() and get_lmstudio_models() helpers - Add lmstudio_url and lmstudio_model
  config fields - Add OBSIDIAN_RAG_LMSTUDIO_URL environment variable support

Auto model detection: - Add is_ollama_running() and get_ollama_models() helpers - CLI setup wizard
  auto-detects available embedding models - Server detection warns but allows manual input if
  detection fails - Shows helpful pull commands for Ollama models

Task-aware prefixes: - Add _get_prefix() method to OllamaEmbedder and LMStudioEmbedder - Apply
  'search_document:' prefix for indexing - Apply 'search_query:' prefix for searching - Improves
  retrieval for nomic-embed-text and qwen models

- **indexer**: Add Chonkie for smart chunking of oversized content
  ([`819d3dd`](https://github.com/ernestkoe/obsidian-notes-rag/commit/819d3ddc618d1d5aecae27fedba8695695475c5e))

- Add chonkie dependency for recursive text chunking - Set MAX_CHUNK_TOKENS=1500 (nomic-embed-text
  limit is 2048) - Use gpt2 tokenizer for accurate token counting - Split oversized chunks at
  semantic boundaries - Preserve heading metadata with split_index for sub-chunks

Previously, files without headings or with large sections would fail with Ollama 500 errors. Now all
  1126 files index successfully (1839 chunks vs 1666 before).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **setup**: Offer watcher service installation during setup
  ([`a3c96ff`](https://github.com/ernestkoe/obsidian-notes-rag/commit/a3c96ff48517881fbd59642d027ea04d1ef61abc))

- Add step 7 to setup wizard: install watcher as background service - macOS: prompts to install
  launchd service with configured values - Linux/Windows: shows guidance to run `watch` manually -
  Update README to document platform differences

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- **watcher**: Add health checks, retry queue, and notifications
  ([`6587d04`](https://github.com/ernestkoe/obsidian-notes-rag/commit/6587d04fea5124ecd2b0faa38360afcb142e1528))

- Add startup health check for Ollama (waits up to 5 min if not running) - Add periodic health check
  every 60 seconds - Add RetryQueue for failed files (retries up to 3 times) - Add macOS
  notifications when embedding fails persistently - Log provider type on startup for debugging

This ensures the watcher doesn't fail silently when Ollama is unavailable and provides user feedback
  via notifications.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.2.2 (2026-01-03)

### Chores

- **main**: Release 0.2.2
  ([`852a340`](https://github.com/ernestkoe/obsidian-notes-rag/commit/852a340c5bd555997a275ba1f9dd088a650968f8))

### Continuous Integration

- Combine release-please with PyPI publish
  ([`6ba322b`](https://github.com/ernestkoe/obsidian-notes-rag/commit/6ba322bb260ccf684017fd17011b98f18bba324e))

- release-please.yml now publishes to PyPI automatically when release is created - release.yml
  simplified to manual-only fallback for republishing

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Remove redundant manual release workflow
  ([`76799be`](https://github.com/ernestkoe/obsidian-notes-rag/commit/76799bebba1018509f6e2e378a865fedc287c401))

Auto-release in release-please.yml handles everything.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Documentation

- Add Buy Me A Coffee badge
  ([`9a40405`](https://github.com/ernestkoe/obsidian-notes-rag/commit/9a40405b8f47e35c968767768fb0b19640ca34a8))

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Update CLAUDE.md with correct package name and PyPI link
  ([`d710ed4`](https://github.com/ernestkoe/obsidian-notes-rag/commit/d710ed48c86789b4eb1ed7490436e49c9e4169ea))

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.2.1 (2026-01-03)

### Bug Fixes

- Update package branding to obsidian-notes-rag
  ([`7cad58b`](https://github.com/ernestkoe/obsidian-notes-rag/commit/7cad58b4027b28a70635ae1d6d3329e26f42e07a))

Update remaining references in docstrings and documentation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Chores

- **main**: Release 0.2.1
  ([`57fd57d`](https://github.com/ernestkoe/obsidian-notes-rag/commit/57fd57df71b6923873a90b8e6337b9b82b45e592))

### Refactoring

- Rename repo to obsidian-notes-rag
  ([`cf1e90b`](https://github.com/ernestkoe/obsidian-notes-rag/commit/cf1e90b651d8c1c5519b578eeca101590234cd39))

Update all references to match new repository name: - pyproject.toml URLs - config.py APP_NAME
  (affects config directory path) - README, CONTRIBUTING, CLAUDE.md documentation

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.2.0 (2026-01-03)

### Bug Fixes

- Suppress httpx logs during indexing progress bar
  ([`75e16a4`](https://github.com/ernestkoe/obsidian-notes-rag/commit/75e16a459742716f1e7466ea7cb416e7f06ad281))

HTTP request logs were interfering with click progress bar display.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Update setup completion message with correct package name
  ([`41ba859`](https://github.com/ernestkoe/obsidian-notes-rag/commit/41ba85906c20da50d67c69872c8dcf841ecd1020))

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Use pypi environment for trusted publishing
  ([`227fe16`](https://github.com/ernestkoe/obsidian-notes-rag/commit/227fe16629df9d6dd277ded144e8e07a87c34631))

Match working workflow pattern from mcp-obsidian-ek.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Chores

- **main**: Release 0.2.0
  ([`392aea4`](https://github.com/ernestkoe/obsidian-notes-rag/commit/392aea4523878e5c363f4c8b982b52a0a973edd6))

### Documentation

- Add Claude Desktop JSON config example
  ([`6b17b45`](https://github.com/ernestkoe/obsidian-notes-rag/commit/6b17b45aed6810d114803f66872927c7e90e3743))

Add both CLI and JSON config options for installing the MCP server.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Update CLAUDE.md CLI commands to use new package name
  ([`48d17ab`](https://github.com/ernestkoe/obsidian-notes-rag/commit/48d17ab1e937f879d7da1b768d81c25181fcfc97))

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Update CLI commands for new package name
  ([`71c708f`](https://github.com/ernestkoe/obsidian-notes-rag/commit/71c708f88f7dde15ea0b432d9c95c6f0919d18ee))

Update uvx and CLI examples to use obsidian-notes-rag.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Features

- Rename package to obsidian-notes-rag
  ([`ab78ec8`](https://github.com/ernestkoe/obsidian-notes-rag/commit/ab78ec80e3cdb59d9713e33e6a263a0446bce311))

Original name was too similar to existing PyPI project.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.1.2 (2026-01-03)

### Bug Fixes

- Disable attestations for token-based auth
  ([`8d70f98`](https://github.com/ernestkoe/obsidian-notes-rag/commit/8d70f989f3e50c7d9f6a5f045d0cc767646c0686))

Attestations require OIDC which fails for new projects.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Use API token for PyPI publish
  ([`51a2c09`](https://github.com/ernestkoe/obsidian-notes-rag/commit/51a2c0958e226b128a2a34c9ec0a8e2c75102a93))

Trusted publishing requires the project to already exist on PyPI. Use API token for initial publish
  to create the project.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Chores

- **main**: Release 0.1.2
  ([`f181c07`](https://github.com/ernestkoe/obsidian-notes-rag/commit/f181c0706ec6e91b7a6dedd276eea91081875ea9))


## v0.1.1 (2026-01-03)

### Bug Fixes

- Add manual trigger to release workflow
  ([`130dfd0`](https://github.com/ernestkoe/obsidian-notes-rag/commit/130dfd0cfddce817863998042ee783791b8204a9))

Allows publishing to PyPI for existing tags via workflow_dispatch. Useful when a release was created
  before the workflow existed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Chores

- **main**: Release 0.1.1
  ([`05c4e39`](https://github.com/ernestkoe/obsidian-notes-rag/commit/05c4e3917df64e92edfa8d515365c030bbd900ef))

### Continuous Integration

- Add PyPI release workflow
  ([`8ae9f77`](https://github.com/ernestkoe/obsidian-notes-rag/commit/8ae9f771194e53cec89b808f2695093d4055d849))

Add GitHub Actions workflow to publish to PyPI when a release is published. Uses trusted publishing
  (OIDC) with a "release" environment for secure, tokenless authentication.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.1.0 (2026-01-03)

### Chores

- **main**: Release 0.1.0
  ([`1132c90`](https://github.com/ernestkoe/obsidian-notes-rag/commit/1132c90e8ad1dfbcf429d04ac9c23e5e2e75b681))

### Continuous Integration

- Add manual trigger to release-please workflow
  ([`0548823`](https://github.com/ernestkoe/obsidian-notes-rag/commit/0548823ed9664a58609a16e7592b28636c2bda05))

### Documentation

- Add TODOs for Linux/Windows service support
  ([`569a830`](https://github.com/ernestkoe/obsidian-notes-rag/commit/569a830b5272c7fdd0c8ba826063b765706ecdcf))

Add TODO comments for implementing service management on Linux (systemd) and Windows (Task
  Scheduler). Update error messages to indicate that cross-platform support is planned.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Features

- Add file watcher daemon with launchd integration
  ([`23d2429`](https://github.com/ernestkoe/obsidian-notes-rag/commit/23d2429c61d82e98610c0180f7044b807ad9cd60))

- Add watcher.py module with debounced file system event handling - Add CLI commands: watch,
  install-service, uninstall-service, service-status - Add GitHub Actions workflows for CI and
  release-please - Add release-please configuration for automated releases - Add LICENSE (MIT) and
  enhance README with documentation - Add basic tests for indexer module - Add pyright configuration
  for type checking

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Add OpenAI embeddings and setup wizard
  ([`e1e363e`](https://github.com/ernestkoe/obsidian-notes-rag/commit/e1e363ee26a70d82635d7c8bf2bec38afaf28383))

- Add OpenAI as default embedding provider (text-embedding-3-small) - Keep Ollama as alternative for
  local/offline processing - Add interactive setup wizard (obsidian-rag setup) that configures: -
  Embedding provider selection - API key configuration - Vault path with validation - Data directory
  with cross-platform defaults - Add config.py for persistent configuration in TOML format - Config
  stored in platform-appropriate locations: - macOS/Linux: ~/.config/obsidian-rag/config.toml -
  Windows: %APPDATA%/obsidian-rag/config.toml - All commands now load config file with env var
  overrides - Add CLAUDE.md for Claude Code guidance - Add CONTRIBUTING.md for development setup

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Consolidate CLI entry points for uvx workflow
  ([`95eb546`](https://github.com/ernestkoe/obsidian-notes-rag/commit/95eb546b0a7d6d89d30da8d0dd4f7c4c64600657))

- Add `serve` command to start MCP server - Replace `obsidian-rag-mcp` entry point with
  `mcp-obsidianrag` - Update README with uvx quick start instructions - Package name entry point
  enables `uvx mcp-obsidianrag` workflow

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Refactoring

- Rename package to obsidian-rag
  ([`669d7e7`](https://github.com/ernestkoe/obsidian-notes-rag/commit/669d7e797edd64e03f2ea46b0e3a894c4763fb1b))

MCP is a feature, not the identity. The package is about semantic search/RAG for Obsidian notes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Rename project to mcp-obsidianRAG
  ([`16a83d9`](https://github.com/ernestkoe/obsidian-notes-rag/commit/16a83d9ebe142f25231a33824b7c27237f61da73))

BREAKING CHANGE: Project renamed from obsidian-memory to mcp-obsidianRAG

- Rename package from obsidian-memory to mcp-obsidianRAG - Rename module from obsidian_memory to
  obsidian_rag - Update CLI commands: obsidian-rag, obsidian-rag-mcp - Update environment variables:
  OBSIDIAN_RAG_* - Update launchd service name: com.obsidian-rag.watcher - Update all documentation
  and configuration

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

- Rename repo to obsidian-rag
  ([`96d64e4`](https://github.com/ernestkoe/obsidian-notes-rag/commit/96d64e4cc03ccb26a1d942ae3970620845b51dc0))

Update all references from mcp-obsidianRAG to obsidian-rag: - GitHub repo URLs in pyproject.toml -
  README and CONTRIBUTING clone instructions - CLAUDE.md project description - release-please config
  - plist template paths - Module docstrings

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
