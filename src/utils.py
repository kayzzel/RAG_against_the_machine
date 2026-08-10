from pathlib import Path
from typing import Any

import json
import os
import tempfile


def getDataset(dataset_path: str) -> dict[Any, Any]:
    """Load and validate a RAG dataset JSON file.

    The dataset must match the following format:

    {
        "rag_questions": [
            {
                "question_id": "q1",
                "question": "How to configure OpenAI server?"
            },
            ...
        ]
    }

    Args:
        dataset_path: Path to the dataset JSON file.

    Returns:
        The loaded dataset as a dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is malformed or does not match the
            expected dataset format.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise ValueError(f"Dataset not found: {dataset_path}")
    try:
        with open(path) as f:
            data: Any = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON in dataset: {err}")
    except Exception as err:
        raise ValueError(f"Could't open the dataset: {err}")

    if not isinstance(data, dict):
        raise ValueError(
            "Invalid dataset format: expected a JSON object at the "
            f"top level, got {type(data).__name__}"
        )

    rag_questions = data.get("rag_questions")
    if "rag_questions" not in data:
        raise ValueError(
            "Invalid dataset format: missing required key "
            "'rag_questions'"
        )
    if not isinstance(rag_questions, list):
        raise ValueError(
            "Invalid dataset format: 'rag_questions' must be a list, "
            f"got {type(rag_questions).__name__}"
        )

    if len(rag_questions) == 0:
        raise ValueError("No questions provided in the dataset")
    for i, question in enumerate(rag_questions):
        if not isinstance(question, dict):
            raise ValueError(
                "Invalid dataset format: each entry in 'rag_questions' "
                "must be an object, "
                f"got {type(question).__name__} at index {i}"
            )
        if set(question) != {"question_id", "question"}:
            raise ValueError(
                "Invalid dataset format: each entry in 'rag_questions' "
                "must contain exactly the keys 'question_id' and "
                f"'question', got {sorted(question)} at index {i}"
            )
        for key in ("question_id", "question"):
            if not isinstance(question[key], str):
                raise ValueError(
                    "Invalid dataset format: field "
                    f"'{key}' must be a string, got "
                    f"{type(question[key]).__name__} at index {i}"
                )

    return data


def write_json_to_file(filename: str, data: dict[Any, Any]) -> None:
    """Securely write a dict to a JSON file.

    The write is atomic: the content is first written to a temporary
    file in the same directory, then moved into place with os.replace.
    A crash or error mid-write can therefore never leave a truncated
    or corrupt file at the destination. Missing parent directories are
    created automatically.

    Args:
        filename: Path to the target output file.
        data: The dict to serialize to JSON.

    Raises:
        ValueError: If data cannot be serialized or the file cannot
            be written.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=path.name,
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        os.replace(temp_name, path)
    except (TypeError, ValueError, OSError) as err:
        raise ValueError(
            f"{err.__class__.__name__} Error: {err}"
        ) from err
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
