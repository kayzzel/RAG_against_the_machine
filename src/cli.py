from .indexing import Indexer
from .chunking import Chunking
from .retrieval import MinimalSource, Retriever
from .utils import getDataset, write_json_to_file

from typing import Any

import sys
import os


class CLI:
    def index(
        self, max_chunk_size: int = 2000
    ) -> None:
        """Index the vLLM repository at data/raw/vllm-0.10.1 and persist
        the resulting index under data/processed.

        Args:
            max_chunk_size: Maximum chunk size in characters.
        """
        if max_chunk_size <= 0 or max_chunk_size > 2000:
            print(
                "ERROR: max_chunk_size must be > 0 and <= 2000",
                file=sys.stderr,
            )
            sys.exit(1)

        repo_path: str = "data/raw/vllm-0.10.1"
        save_path: str = "data/processed"

        indexer = Indexer(chunker=Chunking(max_chunk_size=max_chunk_size))
        retriever, corpus = indexer.build_index(repo_path)
        indexer.save_index(retriever, corpus, save_path)

    def search(self, query: str, k: int = 10) -> None:
        """Search the indexed repository for a single query.

        Args:
            query: The search query.
            k: Number of results to retrieve.
        """
        index_dir: str = "data/processed"

        retriever: Retriever = Retriever(index_dir)
        top_chunks: list[MinimalSource] = retriever.search(query, k)

        for chunk in top_chunks:
            print(
                f"{chunk.file_path} "
                f"[{chunk.first_character_index}:{chunk.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Batch search over a JSON dataset and save results.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: Number of results per question.
            save_directory: Directory to save search results.
        """
        try:
            dataset: dict[Any, Any] = getDataset(dataset_path)
            retriever: Retriever = Retriever("data/processed")
        except ValueError as err:
            print(f"ERROR: {err}")
            return

        search_results: dict[str, Any] = {
                "search_results": [],
                "k": k
                }

        for question in dataset["rag_questions"]:
            top_chunks: list[MinimalSource] = retriever.search(
                    question["question"],
                    k
                )
            search_results["search_results"].append(
                    {
                        "question_id": question["question_id"],
                        "question": question["question"],
                        "retrieved_sources": [
                            source.model_dump() for source in top_chunks
                        ]
                    }
                )

        output_file: str = os.path.join(
                save_directory,
                os.path.basename(dataset_path)
            )

        write_json_to_file(output_file, search_results)

    def answer(self, question: str, k: int = 10) -> None:
        """Answer a single question end-to-end (search + generate).

        Args:
            question: The question to answer.
            k: Number of sources to retrieve.
        """
        raise NotImplementedError

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for a pre-computed search-results file.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            save_directory: Directory to save results with answers.
        """
        raise NotImplementedError

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
    ) -> None:
        """Compute recall@k against ground truth (local dev only).

        Args:
            student_answer_path: Path to student search results JSON.
            dataset_path: Path to the ground truth dataset JSON.
            k: Number of results per question.
            max_context_length: Maximum context length in tokens.
        """
        raise NotImplementedError
