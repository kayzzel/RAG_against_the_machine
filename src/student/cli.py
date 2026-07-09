import json
from pathlib import Path


class CLI:
    def index(
        self, repo_path: str = "data/raw", max_chunk_size: int = 2000
    ) -> None:
        """Index the repository at REPO_PATH.

        Args:
            repo_path: Path to the repository to index.
            max_chunk_size: Maximum chunk size in characters.
        """
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> None:
        """Search the indexed repository for a single query.

        Args:
            query: The search query.
            k: Number of results to retrieve.
        """
        raise NotImplementedError

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
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        try:
            with open(path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in dataset: {e}")
        raise NotImplementedError

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
