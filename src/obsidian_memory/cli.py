"""Command-line interface for obsidian-memory."""

import sys
from pathlib import Path

import click

from .indexer import OllamaEmbedder, VaultIndexer
from .store import VectorStore

# Default configuration
DEFAULT_VAULT = "/Users/ernestkoe/Documents/Brave Robot"
DEFAULT_DATA = "/Users/ernestkoe/Projects/obsidian-memory/data"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"


@click.group()
@click.option("--vault", default=DEFAULT_VAULT, help="Path to Obsidian vault")
@click.option("--data", default=str(DEFAULT_DATA), help="Path to vector store data")
@click.option("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama API URL")
@click.option("--model", default=DEFAULT_MODEL, help="Embedding model name")
@click.pass_context
def main(ctx, vault, data, ollama_url, model):
    """Obsidian Memory - Vector store for your notes."""
    ctx.ensure_object(dict)
    ctx.obj["vault"] = vault
    ctx.obj["data"] = data
    ctx.obj["ollama_url"] = ollama_url
    ctx.obj["model"] = model


@main.command()
@click.option("--clear", is_flag=True, help="Clear existing index before indexing")
@click.pass_context
def index(ctx, clear):
    """Index all markdown files in the vault."""
    vault_path = ctx.obj["vault"]
    data_path = ctx.obj["data"]
    ollama_url = ctx.obj["ollama_url"]
    model = ctx.obj["model"]

    click.echo(f"Indexing vault: {vault_path}")
    click.echo(f"Data path: {data_path}")

    # Initialize components
    embedder = OllamaEmbedder(base_url=ollama_url, model=model)
    store = VectorStore(data_path=data_path)
    indexer = VaultIndexer(vault_path=vault_path, embedder=embedder)

    if clear:
        click.echo("Clearing existing index...")
        store.clear()

    # Count files first
    files = list(indexer.iter_markdown_files())
    click.echo(f"Found {len(files)} markdown files")

    # Index with progress
    chunk_count = 0
    batch_chunks = []
    batch_embeddings = []
    batch_size = 50

    with click.progressbar(files, label="Indexing") as bar:
        for file_path in bar:
            try:
                for chunk, embedding in indexer.index_file(file_path):
                    batch_chunks.append(chunk)
                    batch_embeddings.append(embedding)
                    chunk_count += 1

                    # Batch insert
                    if len(batch_chunks) >= batch_size:
                        store.upsert_batch(batch_chunks, batch_embeddings)
                        batch_chunks = []
                        batch_embeddings = []

            except Exception as e:
                click.echo(f"\nError indexing {file_path}: {e}", err=True)

    # Insert remaining
    if batch_chunks:
        store.upsert_batch(batch_chunks, batch_embeddings)

    embedder.close()

    click.echo(f"\nIndexed {chunk_count} chunks from {len(files)} files")
    click.echo(f"Total documents in store: {store.get_stats()['count']}")


@main.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.option("--type", "note_type", default=None, help="Filter by type (daily, note)")
@click.pass_context
def search(ctx, query, limit, note_type):
    """Search notes semantically."""
    data_path = ctx.obj["data"]
    ollama_url = ctx.obj["ollama_url"]
    model = ctx.obj["model"]

    # Initialize components
    embedder = OllamaEmbedder(base_url=ollama_url, model=model)
    store = VectorStore(data_path=data_path)

    # Generate query embedding
    click.echo(f"Searching for: {query}\n")
    query_embedding = embedder.embed(query)

    # Build filter
    where = None
    if note_type:
        where = {"type": note_type}

    # Search
    results = store.search(query_embedding, limit=limit, where=where)

    if not results:
        click.echo("No results found.")
        return

    # Display results
    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        distance = result["distance"]
        similarity = 1 - distance  # Cosine distance to similarity

        click.echo(f"{'─' * 60}")
        click.echo(f"[{i}] {meta['file_path']}")
        if meta.get("heading"):
            click.echo(f"    Section: {meta['heading']}")
        click.echo(f"    Type: {meta.get('type', 'note')} | Similarity: {similarity:.2%}")
        click.echo()

        # Show truncated content
        content = result["content"]
        if len(content) > 300:
            content = content[:300] + "..."
        click.echo(f"    {content}")
        click.echo()

    embedder.close()


@main.command()
@click.pass_context
def stats(ctx):
    """Show index statistics."""
    data_path = ctx.obj["data"]
    store = VectorStore(data_path=data_path)

    stats = store.get_stats()
    click.echo(f"Collection: {stats['collection']}")
    click.echo(f"Documents: {stats['count']}")
    click.echo(f"Data path: {stats['data_path']}")


if __name__ == "__main__":
    main()
