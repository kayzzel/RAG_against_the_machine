"""Pydantic v2 models for RAG system - Data validation and type safety.

All models follow Pydantic v2 best practices with minimal configuration.
Config is only added when actually needed for validation behavior.
"""

from typing import List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """Represents a source location in the codebase.

    Attributes:
        file_path: Path to the source file (e.g., "vllm/core/scheduler.py")
        first_character_index: Starting position of chunk in file
        last_character_index: Ending position of chunk in file
    """
    model_config = {"extra": "forbid"}

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represents a question without an answer.

    Attributes:
        question_id: Unique identifier (auto-generated UUID if not provided)
        question: The actual question text
    """
    model_config = {"extra": "forbid"}

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this question",
    )
    question: str = Field(..., description="The question text")


class AnsweredQuestion(UnansweredQuestion):
    """Extends UnansweredQuestion with answer and source citations."""
    model_config = {"extra": "forbid"}

    sources: List[MinimalSource] = Field(
        ...,
        description="Source locations this answer is based on",
    )
    answer: str = Field(..., description="The answer text")


class RagDataset(BaseModel):
    """Container for a dataset of RAG questions."""
    model_config = {"extra": "forbid"}

    rag_questions: List[AnsweredQuestion | UnansweredQuestion] = Field(
        ...,
        description="List of questions in the dataset",
    )


class MinimalSearchResults(BaseModel):
    """Results from a single search query.

    Attributes:
        question_id: Reference to the question that was asked
        question: The question text
        retrieved_sources: List of MinimalSource objects found in search
    """
    model_config = {"extra": "forbid"}

    question_id: str = Field(..., description="ID of the question")
    question: str = Field(..., description="The question text")
    retrieved_sources: List[MinimalSource] = Field(
        default_factory=list,
        description="Sources retrieved for this question"
    )


class MinimalAnswer(MinimalSearchResults):
    """Search results with a generated answer.

    Extends MinimalSearchResults by adding the generated answer.

    Attributes:
        answer: LLM-generated answer based on retrieved sources
    """
    model_config = {"extra": "forbid"}

    answer: str = Field(..., description="Generated answer to the question")


class StudentSearchResults(BaseModel):
    """Batch of search results for multiple questions.

    Attributes:
        search_results: List of MinimalSearchResults (one per question)
        k: Number of results retrieved per question
    """
    model_config = {"extra": "forbid"}

    search_results: List[MinimalSearchResults] = Field(
        ...,
        description="List of search results"
    )
    k: int = Field(
        ...,
        description="Number of results per question",
        ge=1
    )


class StudentSearchResultsAndAnswer(BaseModel):
    """Batch of search results with generated answers.

    Attributes:
        search_results: List of MinimalAnswer (question + sources + answer)
        k: Number of results requested per question
    """
    model_config = {"extra": "forbid"}

    search_results: List[MinimalAnswer] = Field(
        ...,
        description="List of search results with answers",
    )
    k: int = Field(
        ...,
        description="Number of results per question",
        ge=1,
    )


class Chunk(BaseModel):
    """Internal representation of a text chunk from the codebase.

    This model is used internally during ingestion and indexing.
    It's not part of the public API output.

    Attributes:
        chunk_id: Unique identifier for this chunk
        content: The actual text content
        file_path: Which file this chunk came from
        start_index: Character position in original file
        end_index: Character position in original file
        chunk_type: Either "python", "text", "code", "markdown", or "basic"
    """
    model_config = {"extra": "forbid"}

    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique chunk identifier"
    )
    content: str = Field(..., description="The chunk text content")
    file_path: str = Field(..., description="Path to the source file")
    start_index: int = Field(
        ...,
        description="Starting character index in file",
        ge=0
    )
    end_index: int = Field(
        ...,
        description="Ending character index in file",
        ge=0
    )
    chunk_type: str = Field(
        ...,
        description=(
            "Type of chunk: 'python', 'text', 'code', 'markdown', "
            "or 'basic'"
        ),
        pattern="^(python|text|code|markdown|basic)$"
    )


class IndexMetadata(BaseModel):
    """Metadata about a built index.

    This model stores information about the created index.
    Useful for debugging and logging.

    Attributes:
        total_chunks: Number of chunks in the index
        file_count: Number of unique files indexed
        index_type: Type of index used (e.g., "bm25")
        created_at: ISO format timestamp when index was created
    """
    model_config = {"extra": "forbid"}

    total_chunks: int = Field(
        ...,
        description="Total number of chunks",
        ge=0
    )
    file_count: int = Field(
        ...,
        description="Number of unique files",
        ge=0
    )
    index_type: str = Field(..., description="Type of index (e.g., 'bm25')")
    created_at: datetime = Field(..., description="ISO 8601 timestamp")
