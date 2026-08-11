"""Retrieval module — loads a persisted BM25 index and answers queries."""

from pathlib import Path
from .models import MinimalSource

from typing import Any

import bm25s


class Retriever:
    """Loads a persisted BM25 index and corpus, and answers search queries.

    Instances are meant to be constructed once per process (loading the
    index is the expensive part) and reused across many .search() calls,
    matching the "index once, search many times" pipeline design: the
    Indexer writes data/processed/ once via `index`, and any later
    `search` / `search_dataset` / `answer` / `answer_dataset` invocation
    -- potentially a fresh process each time -- reloads it here rather
    than re-chunking the repository.

    Attributes:
        (private) __retriever: the loaded bm25s.BM25 index, with its
            corpus attached via load_corpus=True at load time.
    """

    def __init__(self, index_dir: str) -> None:
        """Load a previously built BM25 index and its corpus from disk.

        Args:
            index_dir: Directory containing the persisted index, as
                written by Indexer.save_index (e.g. "data/processed").

        Raises:
            ValueError: If index_dir does not exist, is not a
                directory, or does not contain a loadable index.
        """
        if not Path(index_dir).is_dir():
            raise ValueError(f"index directory not found: {index_dir}")
        self.index_dir = index_dir
        try:
            self.__retriever = bm25s.BM25.load(index_dir, load_corpus=True)
        except Exception as err:
            raise ValueError(
                f"failed to load index from {index_dir}: {err}"
            ) from err

    def search(self, query: str, k: int = 10) -> list[MinimalSource]:
        """Search the index for a single query.

        Tokenizes the query with the same scheme used at indexing time
        (bm25s.tokenize with English stopwords) so query-time and
        index-time tokens are comparable, scores every chunk, and
        returns the top-k as MinimalSource objects.

        Degenerate inputs are handled gracefully rather than raising:
        - an empty or whitespace-only query returns []
        - k <= 0 returns []
        - k larger than the corpus size is clamped down automatically

        Args:
            query: The natural-language search query.
            k: Number of top results to retrieve.

        Returns:
            Up to k MinimalSource results, ranked by relevance
            (highest-scoring first).
        """
        if k <= 0 or not query or not query.strip():
            return []

        query_token = bm25s.tokenize(
                [query],
                stopwords="en",
                show_progress=False
            )
        k_actual = min(k, len(self.__retriever.corpus))
        results, _ = self.__retriever.retrieve(
                query_token,
                k=k_actual,
                show_progress=False
            )  # second param is score

        return [self.__chunk_to_source(hit) for hit in results[0]]

    def __chunk_to_source(self, chunk: dict[str, Any]) -> MinimalSource:
        """Convert one raw corpus entry into a MinimalSource.

        This is the single conversion point between the internal Chunk
        representation persisted in the corpus (start_index/end_index,
        content, chunk_id, chunk_type) and the public MinimalSource
        schema (file_path/first_character_index/last_character_index)
        that every downstream stage -- and the moulinette -- actually
        consumes. Keeping this mapping in exactly one place means a
        future field rename only needs to change here.

        Args:
            chunk: One entry from the loaded bm25s corpus, shaped like
                a serialized Chunk (dict, not the pydantic model,
                since bm25s persists corpus entries as plain JSON).

        Returns:
            The equivalent MinimalSource.

        Raises:
            ValueError: If chunk is not a dict with the required
                keys (file_path, start_index, end_index) of the
                expected types (str, int, int).
        """

        if not isinstance(chunk, dict):
            raise ValueError(
                f"chunk must be a dict, got {type(chunk).__name__}"
            )

        missing = [
            key for key in ("file_path", "start_index", "end_index")
            if key not in chunk
        ]
        if missing:
            raise ValueError(
                f"chunk is missing required key(s): {', '.join(missing)}"
            )

        expected = {
            "file_path": str,
            "start_index": int,
            "end_index": int,
        }
        for key, key_type in expected.items():
            if not isinstance(chunk[key], key_type):
                raise ValueError(
                    f"chunk[{key!r}] must be of type "
                    f"{key_type.__name__}, got "
                    f"{type(chunk[key]).__name__}"
                )

        return MinimalSource(
                file_path=chunk["file_path"],
                first_character_index=chunk["start_index"],
                last_character_index=chunk["end_index"]
            )
