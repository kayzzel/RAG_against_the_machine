from .chunking import Chunking
from .models import Chunk

from pathlib import Path
from typing import Any, Iterator

import bm25s


class Indexer:
    """Builds and persists a BM25 index over chunked repository files."""

    def __init__(self, chunker: Chunking) -> None:
        self.__chunker = chunker

    def build_index(
                    self,
                    repo_path: str
                ) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
        """Walk the repo, chunk every included file, and build a BM25 index.

        Returns the fitted BM25 index and the corpus list backing it
        (each corpus entry is a plain dict, since bm25s persists corpus
        entries as JSON — pydantic models aren't directly serializable
        this way, so we convert Chunk -> dict here).
        """
        chunks = self.__collect_chunks(repo_path)
        corpus = [chunk.model_dump() for chunk in chunks]  # pydantic -> dict
        texts = [c["content"] for c in corpus]

        tokenized = bm25s.tokenize(texts, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(tokenized)
        return retriever, corpus

    def save_index(
                    self,
                    retriever: bm25s.BM25,
                    corpus: list[dict[str, Any]],
                    save_dir: str
                ) -> None:
        """Persist a built BM25 index and its backing corpus to disk.

        Args:
            retriever: A fitted BM25 index (from build_index).
            corpus: The list of chunk dicts backing the index, saved
                alongside it so a later process can reconstruct
                MinimalSource results without re-reading or re-chunking
                the repository.
            save_dir: Directory to write the index files into; created if
                it does not already exist.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        retriever.save(save_dir, corpus=corpus)

    def __collect_chunks(self, repo_path: str) -> list[Chunk]:
        """Read and chunk every included file in the repository.

        For each file yielded by _iter_included_files, reads its contents
        and dispatches to the appropriate chunking strategy based on file
        extension (Python AST-based chunking for .py, header-based chunking
        for .md). Files that cannot be decoded or read are skipped rather
        than causing the whole indexing run to fail.

        Args:
            repo_path: Path to the root of the extracted vLLM repository.

        Returns:
            A flat list of all Chunk objects produced across every included
            file, ready to be converted into a BM25 corpus.
        """
        chunks: list[Chunk] = []

        try:
            filesPath = self.__iter_included_files(repo_path)
        except ValueError as err:
            raise ValueError(
                    f"failed to collect chunks from {repo_path}: {err}"
                             ) from err

        for filePath in filesPath:
            try:
                chunks += self.__chunker.chunk_file(str(filePath))
            except ValueError as err:
                print(f"WARNING: {err}")

        return chunks

    def __walk_included_files(
                             self,
                             vllm_path: Path,
                             tests_path: Path,
                             docs_path: Path
                          ) -> Iterator[Path]:
        yield from vllm_path.glob("**/*.py")
        yield from tests_path.glob("**/*.py")
        yield from docs_path.glob("**/*.md")

    def __iter_included_files(self, repo_path: str) -> Iterator[Path]:
        """Yield paths to files under repo_path that should be indexed.

        Walks the repository recursively and filters down to the files
        judged useful for answering questions about the codebase (per
        §V.8 of the subject: "Index all the files you judge useful in
        the repository"). Excludes directories/files that are unlikely
        to contain retrieval-relevant content (e.g. test suites, CI
        configuration, build tooling) to keep the index focused and
        within the indexing-time budget.

        Args:
            repo_path: Path to the root of the extracted vLLM repository.

        Yields:
            Path objects for each file that should be passed to chunking,
            restricted to file types with an implemented chunking strategy
            (.py and .md).
        """
        repo = Path(repo_path)

        if not repo.is_dir():
            raise ValueError(
                    "repo_path does not exist "
                    f"or is not a directory: {repo_path}"
                 )

        vllm_path = repo / "vllm"
        if not vllm_path.is_dir():
            raise ValueError(f"expected directory not found: {vllm_path}")

        tests_path = repo / "tests"
        if not tests_path.is_dir():
            raise ValueError(f"expected directory not found: {tests_path}")

        docs_path = repo / "docs"
        if not docs_path.is_dir():
            raise ValueError(f"expected directory not found: {docs_path}")

        return self.__walk_included_files(
                                        vllm_path,
                                        tests_path,
                                        docs_path
                                    )
