"""Markdown parsing, chunking, and embedding generation."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Tuple

import httpx
import yaml
from chonkie import RecursiveChunker
from chonkie.types.recursive import RecursiveLevel, RecursiveRules

# Maximum tokens per chunk
# nomic-embed-text context length is 2048 tokens
# Using 1500 to leave headroom for tokenizer differences
MAX_CHUNK_TOKENS = 1500


@dataclass
class Chunk:
    """A chunk of text from a markdown file."""

    id: str
    content: str
    file_path: str
    heading: Optional[str]
    heading_level: int
    metadata: Dict


def parse_frontmatter(content: str) -> Tuple[Dict, str]:
    """Extract YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, parts[2].strip()


_chunker: Optional[RecursiveChunker] = None


def _get_chunker() -> RecursiveChunker:
    """Get or create the shared RecursiveChunker instance."""
    global _chunker
    if _chunker is None:
        # Markdown-aware splitting rules: headings > paragraphs > lines > sentences
        markdown_rules = RecursiveRules(levels=[
            RecursiveLevel(delimiters="\n# ", include_delim="next"),       # h1
            RecursiveLevel(delimiters="\n## ", include_delim="next"),      # h2
            RecursiveLevel(delimiters="\n### ", include_delim="next"),     # h3
            RecursiveLevel(delimiters="\n#### ", include_delim="next"),    # h4
            RecursiveLevel(delimiters="\n\n"),                             # paragraphs
            RecursiveLevel(delimiters="\n"),                               # lines
            RecursiveLevel(delimiters=[". ", "! ", "? "]),                 # sentences
            RecursiveLevel(whitespace=True),                              # words
        ])
        _chunker = RecursiveChunker(
            chunk_size=MAX_CHUNK_TOKENS,
            rules=markdown_rules,
            min_characters_per_chunk=50,
        )
    return _chunker


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

    chunker = _get_chunker()

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
                if 1 <= level <= 6 and len(line) > level and line[level] == " ":
                    heading = line[level:].strip()
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


def _generate_chunk_id(file_path: str, heading: Optional[str], content: str, chunk_index: int = 0) -> str:
    """Generate a stable ID for a chunk."""
    # Include file path, heading, content hash, and index for uniqueness
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    key = f"{file_path}:{heading or 'root'}:{content_hash}:{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class OpenAIEmbedder:
    """Generate embeddings using OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        self.model = model

    def embed(self, text: str, task_type: str = "search_document") -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            task_type: Ignored for OpenAI (included for interface consistency)
        """
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def close(self):
        """Close the client (no-op for OpenAI)."""
        pass


class OllamaEmbedder:
    """Generate embeddings using Ollama (local)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text"
    ):
        self.base_url = base_url
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    def _get_prefix(self, task_type: str) -> str:
        """Get the prefix for the current model and task.
        
        Some embedding models perform better with task-specific prefixes.
        See: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#usage
        """
        model = self.model.lower()
        if "nomic" in model:
            if task_type == "search_document":
                return "search_document: "
            elif task_type == "search_query":
                return "search_query: "
        elif "qwen" in model:
            if task_type == "search_query":
                return "Query: "
        return ""

    def embed(self, text: str, task_type: str = "search_document") -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            task_type: "search_document" or "search_query"
        """
        prefix = self._get_prefix(task_type)
        response = self.client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": f"{prefix}{text}"}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: List[str], task_type: str = "search_document") -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            task_type: "search_document" or "search_query"
        """
        # Ollama doesn't support batch, so we do sequential
        return [self.embed(text, task_type) for text in texts]

    def close(self):
        """Close the HTTP client."""
        self.client.close()


class LMStudioEmbedder:
    """Generate embeddings using LM Studio (local, OpenAI-compatible API)."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str = "text-embedding-nomic-embed-text-v1.5"
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    def _get_prefix(self, task_type: str) -> str:
        """Get the prefix for the current model and task."""
        model = self.model.lower()
        if "nomic" in model:
            if task_type == "search_document":
                return "search_document: "
            elif task_type == "search_query":
                return "search_query: "
        elif "qwen" in model:
            if task_type == "search_query":
                return "Query: "
        return ""

    def embed(self, text: str, task_type: str = "search_document") -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            task_type: "search_document" or "search_query"
        """
        prefix = self._get_prefix(task_type)
        response = self.client.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": f"{prefix}{text}"}
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def embed_batch(self, texts: List[str], task_type: str = "search_document") -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            task_type: "search_document" or "search_query"
        """
        prefix = self._get_prefix(task_type)
        prefixed_texts = [f"{prefix}{t}" for t in texts]
        
        response = self.client.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": prefixed_texts}
        )
        response.raise_for_status()
        data = response.json()["data"]
        # Sort by index to ensure correct order
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    def close(self):
        """Close the HTTP client."""
        self.client.close()


def is_lmstudio_running(base_url: str = "http://localhost:1234") -> bool:
    """Check if LM Studio server is running.
    
    Args:
        base_url: LM Studio API URL to check
        
    Returns:
        True if LM Studio is accessible, False otherwise
    """
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{base_url.rstrip('/')}/v1/models")
            return response.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        return False


def is_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running.
    
    Args:
        base_url: Ollama API URL to check
        
    Returns:
        True if Ollama is accessible, False otherwise
    """
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
            return response.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        return False


def get_lmstudio_models(base_url: str = "http://localhost:1234") -> List[str]:
    """Get list of available models from LM Studio, filtered for embedding models.
    
    Args:
        base_url: LM Studio API URL
        
    Returns:
        List of embedding model identifiers
    """
    embedding_keywords = ['embed', 'bge', 'minilm', 'e5', 'gte', 'instructor']
    
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url.rstrip('/')}/v1/models")
            if response.status_code != 200:
                return []
            
            data = response.json()
            models = []
            
            for model in data.get("data", []):
                model_id = model.get("id", "")
                # Check if model name contains embedding-related keywords
                model_lower = model_id.lower()
                if any(keyword in model_lower for keyword in embedding_keywords):
                    models.append(model_id)
            
            return sorted(models)
    except (httpx.RequestError, httpx.TimeoutException, ValueError):
        return []


def get_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Get list of available embedding models from Ollama.
    
    Args:
        base_url: Ollama API URL
        
    Returns:
        List of embedding model names
    """
    embedding_keywords = ['embed', 'bge', 'minilm', 'e5', 'gte', 'instructor', 'nomic']
    
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
            if response.status_code != 200:
                return []
            
            data = response.json()
            models = []
            
            for model in data.get("models", []):
                model_name = model.get("name", "")
                # Check if model name contains embedding-related keywords
                model_lower = model_name.lower()
                if any(keyword in model_lower for keyword in embedding_keywords):
                    models.append(model_name)
            
            return sorted(models)
    except (httpx.RequestError, httpx.TimeoutException, ValueError):
        return []


# Type alias for embedder
Embedder = OpenAIEmbedder | OllamaEmbedder | LMStudioEmbedder


def create_embedder(
    provider: str = "openai",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Embedder:
    """Create an embedder instance for the specified provider.

    Args:
        provider: "openai", "ollama", or "lmstudio"
        model: Model name (defaults to provider's default)
        base_url: Base URL for Ollama/LM Studio (ignored for OpenAI)

    Returns:
        An embedder instance
    """
    if provider == "openai":
        kwargs = {}
        if model:
            kwargs["model"] = model
        return OpenAIEmbedder(**kwargs)
    elif provider == "ollama":
        kwargs = {}
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        return OllamaEmbedder(**kwargs)
    elif provider == "lmstudio":
        kwargs = {}
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        return LMStudioEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'ollama', or 'lmstudio'.")


class VaultIndexer:
    """Index an Obsidian vault."""

    def __init__(
        self,
        vault_path,
        embedder: Embedder,
        exclude_patterns: Optional[List[str]] = None
    ):
        self.vault_path = Path(vault_path)
        self.embedder = embedder
        self.exclude_patterns = exclude_patterns or [
            "attachments/**",
            ".obsidian/**",
            ".trash/**"
        ]

    def iter_markdown_files(self) -> Iterator[Path]:
        """Iterate over all markdown files in the vault."""
        for md_file in self.vault_path.rglob("*.md"):
            rel_path = md_file.relative_to(self.vault_path)

            # Check exclusions
            skip = False
            for pattern in self.exclude_patterns:
                if rel_path.match(pattern):
                    skip = True
                    break

            if not skip:
                yield md_file

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

    def index_all(self) -> Iterator[Tuple[Chunk, List[float]]]:
        """Index all files in the vault."""
        for file_path in self.iter_markdown_files():
            try:
                for chunk, embedding in self.index_file(file_path):
                    yield chunk, embedding
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
