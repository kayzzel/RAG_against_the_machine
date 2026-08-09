"""Tests for Pydantic models - Comprehensive test suite.

Run with: pytest tests/test_models.py -v
"""

import pytest
import json
from pydantic import ValidationError
import uuid

from src.models import (
    MinimalSource,
    UnansweredQuestion,
    AnsweredQuestion,
    RagDataset,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    Chunk,
    IndexMetadata,
)


class TestMinimalSource:
    def test_creation_basic(self) -> None:
        source = MinimalSource(file_path="vllm/core/scheduler.py", first_character_index=100, last_character_index=500)
        assert source.file_path == "vllm/core/scheduler.py"

    def test_serialization(self) -> None:
        source = MinimalSource(file_path="test.py", first_character_index=0, last_character_index=100)
        data = source.model_dump()
        assert data == {"file_path": "test.py", "first_character_index": 0, "last_character_index": 100}

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            MinimalSource(file_path="test.py", first_character_index=0, last_character_index=100, extra_field="x")

    def test_validation_missing_field(self) -> None:
        with pytest.raises(ValidationError):
            MinimalSource(file_path="test.py")


class TestUnansweredQuestion:
    def test_creation_with_auto_id(self) -> None:
        q = UnansweredQuestion(question="What is RAG?")
        uuid.UUID(q.question_id)

    def test_serialization(self) -> None:
        q = UnansweredQuestion(question="Test?")
        data = q.model_dump()
        assert set(data.keys()) == {"question_id", "question"}

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            UnansweredQuestion(question="Test?", type="unanswered")


class TestAnsweredQuestion:
    def test_creation_with_sources(self) -> None:
        sources = [MinimalSource(file_path="docs.md", first_character_index=0, last_character_index=100)]
        q = AnsweredQuestion(question="What is RAG?", sources=sources, answer="RAG is...")
        assert len(q.sources) == 1

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            AnsweredQuestion(question="Test?", sources=[], answer="A.", type="answered")


class TestRagDataset:
    def test_creation_mixed_questions_no_type_field(self) -> None:
        """RagDataset must correctly disambiguate the union from plain
        dicts shaped exactly like real ground-truth JSON -- i.e. with
        NO 'type' key present, since the subject's data format never
        includes one."""
        unanswered_data = {"question_id": "q1", "question": "Test?"}
        answered_data = {
            "question_id": "q2",
            "question": "Test?",
            "sources": [],
            "answer": "Answer.",
        }
        dataset = RagDataset(rag_questions=[unanswered_data, answered_data])
        assert isinstance(dataset.rag_questions[0], UnansweredQuestion)
        assert isinstance(dataset.rag_questions[1], AnsweredQuestion)
        assert not isinstance(dataset.rag_questions[0], AnsweredQuestion)

    def test_serialization_roundtrip(self) -> None:
        questions = [
            UnansweredQuestion(question="Q1?"),
            AnsweredQuestion(
                question="Q2?",
                sources=[MinimalSource(file_path="f.py", first_character_index=0, last_character_index=50)],
                answer="Ans.",
            ),
        ]
        dataset = RagDataset(rag_questions=questions)
        json_str = dataset.model_dump_json()
        dataset2 = RagDataset(**json.loads(json_str))
        assert len(dataset2.rag_questions) == 2
        assert isinstance(dataset2.rag_questions[1], AnsweredQuestion)


class TestMinimalSearchResults:
    def test_creation(self) -> None:
        result = MinimalSearchResults(question_id="q1", question="How to?", retrieved_sources=[])
        assert result.question_id == "q1"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            MinimalSearchResults(question_id="q1", question="Q?", extras="not allowed")


class TestMinimalAnswer:
    def test_creation(self) -> None:
        answer = MinimalAnswer(question_id="q1", question="What is RAG?", retrieved_sources=[], answer="RAG is...")
        assert answer.answer == "RAG is..."

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            MinimalAnswer(question_id="q1", question="Q?", retrieved_sources=[], answer="A.", extra="x")


class TestStudentSearchResults:
    def test_k_validation(self) -> None:
        StudentSearchResults(search_results=[], k=1)
        with pytest.raises(ValidationError):
            StudentSearchResults(search_results=[], k=0)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            StudentSearchResults(search_results=[], k=10, extra_metadata="x")


class TestStudentSearchResultsAndAnswer:
    def test_creation_has_k(self) -> None:
        answers = [MinimalAnswer(question_id="q1", question="Q1?", retrieved_sources=[], answer="A1.")]
        batch = StudentSearchResultsAndAnswer(search_results=answers, k=10)
        assert batch.k == 10

    def test_serialization_includes_k(self) -> None:
        answers = [MinimalAnswer(question_id="q1", question="Q?", retrieved_sources=[], answer="A.")]
        batch = StudentSearchResultsAndAnswer(search_results=answers, k=5)
        data = json.loads(batch.model_dump_json())
        assert data["k"] == 5
        batch2 = StudentSearchResultsAndAnswer(**data)
        assert batch2.k == 5

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            StudentSearchResultsAndAnswer(search_results=[], k=10, unknown_field="x")


class TestChunk:
    def test_chunk_type_validation(self) -> None:
        Chunk(content="test", file_path="test.py", start_index=0, end_index=4, chunk_type="python")
        with pytest.raises(ValidationError):
            Chunk(content="test", file_path="test.js", start_index=0, end_index=4, chunk_type="javascript")

    def test_index_constraints(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(content="test", file_path="test.py", start_index=-1, end_index=4, chunk_type="python")

    def test_offset_content_consistency(self) -> None:
        original = "def hello(): pass\n    return 42\n"
        chunk = Chunk(content="def hello(): pass", file_path="test.py", start_index=0, end_index=17, chunk_type="python")
        assert original[chunk.start_index:chunk.end_index] == chunk.content


class TestIndexMetadata:
    def test_creation_and_datetime_coercion(self) -> None:
        from datetime import datetime
        metadata = IndexMetadata(total_chunks=1000, file_count=50, index_type="bm25", created_at="2024-01-15T10:30:00")
        assert isinstance(metadata.created_at, datetime)

    def test_constraints(self) -> None:
        with pytest.raises(ValidationError):
            IndexMetadata(total_chunks=-1, file_count=50, index_type="bm25", created_at="2024-01-15T10:30:00")


class TestIntegration:
    def test_full_pipeline_serialization(self) -> None:
        sources = [MinimalSource(file_path="docs/api.md", first_character_index=100, last_character_index=200)]
        answer = MinimalAnswer(question_id="q1", question="How to use the API?", retrieved_sources=sources, answer="...")
        batch = StudentSearchResultsAndAnswer(search_results=[answer], k=10)
        data = json.loads(batch.model_dump_json())
        assert data["k"] == 10
        batch2 = StudentSearchResultsAndAnswer(**data)
        assert batch2.search_results[0].question_id == "q1"

    def test_dataset_with_mix_types_no_type_field(self) -> None:
        """End-to-end: build a RagDataset from raw dicts (as loaded from
        a real JSON file) with no 'type' key, mirroring exactly how
        search_dataset/evaluate will load ground-truth data."""
        raw = {
            "rag_questions": [
                {"question_id": "q1", "question": "Q1?"},
                {
                    "question_id": "q2",
                    "question": "Q2?",
                    "sources": [{"file_path": "f.py", "first_character_index": 0, "last_character_index": 50}],
                    "answer": "A.",
                },
            ]
        }
        dataset = RagDataset(**raw)
        assert isinstance(dataset.rag_questions[0], UnansweredQuestion)
        assert isinstance(dataset.rag_questions[1], AnsweredQuestion)

    def test_extra_fields_forbidden_in_output_chain(self) -> None:
        with pytest.raises(ValidationError):
            StudentSearchResultsAndAnswer(
                search_results=[
                    MinimalAnswer(
                        question_id="q1", question="Q?",
                        retrieved_sources=[MinimalSource(file_path="f.py", first_character_index=0, last_character_index=50, bogus="field")],
                        answer="A.",
                    )
                ],
                k=10,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
