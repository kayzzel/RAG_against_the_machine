"""Pydantic v2 models for RAG system - Data validation and type safety.

All models follow Pydantic v2 best practices with minimal configuration.
Config is only added when actually needed for validation behavior.
"""

from typing import List
from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """Represents a source location in the codebase.

    Attributes:
        file_path: Path to the source file (e.g., "vllm/core/scheduler.py")
        first_character_index: Starting position of chunk in file
        last_character_index: Ending position of chunk in file
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represents a question without an answer.

    Attributes:
        question_id: Unique identifier (auto-generated UUID if not provided)
        question: The actual question text
    """
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this question"
    )
    question: str = Field(..., description="The question text")


class AnsweredQuestion(UnansweredQuestion):
    """Extends UnansweredQuestion with answer and source citations.

    Attributes:
        sources: List of MinimalSource objects showing where answer comes from
        answer: The answer text
    """
    sources: List[MinimalSource] = Field(
        ...,
        description="Source locations this answer is based on"
    )
    answer: str = Field(..., description="The answer text")


class RagDataset(BaseModel):
    """Container for a dataset of RAG questions.

    Attributes:
        rag_questions: List of questions (answered or unanswered)
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion] = Field(
        ...,
        description="List of questions in the dataset"
    )


class MinimalSearchResults(BaseModel):
    """Results from a single search query.

    Attributes:
        question_id: Reference to the question that was asked
        question: The question text
        retrieved_sources: List of MinimalSource objects found in search
    """
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
    answer: str = Field(..., description="Generated answer to the question")


class StudentSearchResults(BaseModel):
    """Batch of search results for multiple questions.

    Attributes:
        search_results: List of MinimalSearchResults (one per question)
        k: Number of results retrieved per question
    """
    search_results: List[MinimalSearchResults] = Field(
        ...,
        description="List of search results"
    )
    k: int = Field(
        ...,
        description="Number of results per question",
        ge=1
    )


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Batch of search results with generated answers.

    Overrides the search_results field to contain MinimalAnswer objects
    instead of MinimalSearchResults.

    Attributes:
        search_results: List of MinimalAnswer (overrides parent)
        k: Number of results per question
    """
    search_results: List[MinimalAnswer] = Field(  # type: ignore[assignment]
        ...,
        description="List of search results with answers"
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
        chunk_type: Either "python" or "text"
    """
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
        description="Type of chunk: 'python' or 'text'",
        pattern="^(python|text)$"
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
    created_at: str = Field(..., description="ISO 8601 timestamp")
