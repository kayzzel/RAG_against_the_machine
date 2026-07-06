"""Pydantic models for RAG system data validation and type safety."""

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

    class Config:
        """Pydantic config."""
        frozen = False  # Allow mutation if needed


class UnansweredQuestion(BaseModel):
    """Represents a question without an answer.

    Attributes:
        question_id: Unique identifier (auto-generated UUID)
        question: The actual question text
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "question_id": "550e8400-e29b-41d4-a716-446655440000",
                "question": "How to configure OpenAI server?"
            }
        }


class AnsweredQuestion(UnansweredQuestion):
    """Extends UnansweredQuestion with answer and sources.

    Attributes:
        sources: List of MinimalSource objects indicating where answer comes
                 from
        answer: The answer text
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Container for multiple RAG questions.

    Attributes:
        rag_questions: List of questions (answered or unanswered)
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "description":
                "Dataset containing RAG questions with optional answers"
        }


class MinimalSearchResults(BaseModel):
    """Results from a search query.

    Attributes:
        question_id: Reference to the question
        question: The question text
        retrieved_sources: List of sources found in search
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "description": "Search results for a single question"
        }


class MinimalAnswer(MinimalSearchResults):
    """Search results with generated answer.

    Attributes:
        answer: LLM-generated answer based on retrieved sources
    """
    answer: str


class StudentSearchResults(BaseModel):
    """Batch of search results.

    Attributes:
        search_results: List of MinimalSearchResults
        k: Number of results retrieved per question
    """
    search_results: List[MinimalSearchResults]
    k: int

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "description": "Batch of search results with k parameter"
        }


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Batch of search results with answers.

    Attributes:
        search_results: List of MinimalAnswer (overrides parent)
        k: Number of results retrieved per question
    """
    search_results: List[MinimalAnswer]  # Overrides parent type


class Chunk(BaseModel):
    """Internal representation of a text chunk.

    Attributes:
        chunk_id: Unique identifier for this chunk
        content: The actual text content
        file_path: Which file this chunk came from
        start_index: Character position in original file
        end_index: Character position in original file
        chunk_type: Either "python" or "text"
    """
    chunk_id: str
    content: str
    file_path: str
    start_index: int
    end_index: int
    chunk_type: str  # "python" or "text"

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class IndexMetadata(BaseModel):
    """Metadata about the index.

    Attributes:
        total_chunks: Number of chunks in index
        file_count: Number of unique files indexed
        index_type: Type of index (e.g., "bm25")
        created_at: When index was created
    """
    total_chunks: int
    file_count: int
    index_type: str
    created_at: str
