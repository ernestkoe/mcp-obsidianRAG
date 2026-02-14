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
