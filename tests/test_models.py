"""Tests for Pydantic models."""

import pytest
from src.student.models import (
    MinimalSource,
    UnansweredQuestion,
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalAnswer,
)


def test_minimal_source_creation() -> None:
    """Test MinimalSource model validation."""
    source = MinimalSource(
        file_path="vllm/core/scheduler.py",
        first_character_index=100,
        last_character_index=500
    )
    assert source.file_path == "vllm/core/scheduler.py"
    assert source.first_character_index == 100
    assert source.last_character_index == 500


def test_unanswered_question_auto_id() -> None:
    """Test that question_id is auto-generated."""
    q1 = UnansweredQuestion(question="What is RAG?")
    q2 = UnansweredQuestion(question="What is RAG?")

    assert q1.question_id != q2.question_id
    assert len(q1.question_id) > 0


def test_answered_question_creation() -> None:
    """Test AnsweredQuestion with sources."""
    source = MinimalSource(
        file_path="docs/rag.md",
        first_character_index=0,
        last_character_index=100
    )

    question = AnsweredQuestion(
        question="What is RAG?",
        sources=[source],
        answer="RAG is Retrieval-Augmented Generation."
    )

    assert len(question.sources) == 1
    assert question.sources[0].file_path == "docs/rag.md"
    assert question.answer == "RAG is Retrieval-Augmented Generation."


def test_minimal_search_results_validation() -> None:
    """Test MinimalSearchResults model."""
    results = MinimalSearchResults(
        question_id="test-id",
        question="How to use?",
        retrieved_sources=[]
    )

    assert results.question_id == "test-id"
    assert len(results.retrieved_sources) == 0
