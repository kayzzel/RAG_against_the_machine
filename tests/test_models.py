"""Tests for Pydantic models - Comprehensive test suite.

Run with: pytest tests/test_models.py -v
"""

import pytest
import json
from pydantic import ValidationError
import uuid

from student.models import (
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


# ============================================================================
# MINIMAL SOURCE TESTS
# ============================================================================

class TestMinimalSource:
    """Tests for MinimalSource model."""

    def test_creation_basic(self) -> None:
        """Test basic MinimalSource creation."""
        source = MinimalSource(
            file_path="vllm/core/scheduler.py",
            first_character_index=100,
            last_character_index=500
        )

        assert source.file_path == "vllm/core/scheduler.py"
        assert source.first_character_index == 100
        assert source.last_character_index == 500

    def test_serialization(self) -> None:
        """Test JSON serialization."""
        source = MinimalSource(
            file_path="test.py",
            first_character_index=0,
            last_character_index=100
        )

        # To dict
        data = source.model_dump()
        assert data == {
            "file_path": "test.py",
            "first_character_index": 0,
            "last_character_index": 100
        }

        # To JSON string
        json_str = source.model_dump_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["file_path"] == "test.py"

    def test_deserialization(self) -> None:
        """Test creating from dict/JSON."""
        data = {
            "file_path": "vllm/core.py",
            "first_character_index": 50,
            "last_character_index": 150
        }

        source = MinimalSource(**data)
        assert source.file_path == "vllm/core.py"

    def test_validation_missing_field(self) -> None:
        """Test validation fails with missing required field."""
        with pytest.raises(ValidationError) as exc_info:
            MinimalSource(file_path="test.py")  # Missing indices

        # Check error message
        errors = exc_info.value.errors()
        assert any(e['loc'] == ('first_character_index',) for e in errors)
        assert any(e['loc'] == ('last_character_index',) for e in errors)

    def test_validation_wrong_type(self) -> None:
        """Test validation fails with wrong types."""
        with pytest.raises(ValidationError):
            MinimalSource(
                file_path="test.py",
                first_character_index="not an int",  # ❌ Should be int
                last_character_index=100
            )


# ============================================================================
# QUESTION MODELS TESTS
# ============================================================================

class TestUnansweredQuestion:
    """Tests for UnansweredQuestion model."""

    def test_creation_with_auto_id(self) -> None:
        """Test that question_id is auto-generated."""
        q = UnansweredQuestion(question="What is RAG?")

        assert q.question == "What is RAG?"
        assert q.question_id is not None
        assert len(q.question_id) > 0

        # Should be valid UUID
        uuid.UUID(q.question_id)

    def test_creation_with_custom_id(self) -> None:
        """Test providing custom question_id."""
        custom_id = "q-12345"
        q = UnansweredQuestion(
            question_id=custom_id,
            question="Custom question"
        )

        assert q.question_id == custom_id

    def test_unique_ids(self) -> None:
        """Test that each question gets unique ID."""
        q1 = UnansweredQuestion(question="Question 1")
        q2 = UnansweredQuestion(question="Question 2")
        q3 = UnansweredQuestion(question="Question 1")  # Same question

        # All should have different IDs
        assert q1.question_id != q2.question_id
        assert q1.question_id != q3.question_id
        assert q2.question_id != q3.question_id

    def test_validation(self) -> None:
        """Test validation."""
        # Missing question
        with pytest.raises(ValidationError):
            UnansweredQuestion()

    def test_serialization(self) -> None:
        """Test JSON serialization."""
        q = UnansweredQuestion(question="Test?")

        # To dict
        data = q.model_dump()
        assert "question_id" in data
        assert data["question"] == "Test?"

        # Round-trip
        q2 = UnansweredQuestion(**data)
        assert q2.question_id == q.question_id


class TestAnsweredQuestion:
    """Tests for AnsweredQuestion model."""

    def test_creation_with_sources(self) -> None:
        """Test creating with sources and answer."""
        sources = [
            MinimalSource(
                file_path="docs.md",
                first_character_index=0,
                last_character_index=100
            ),
            MinimalSource(
                file_path="code.py",
                first_character_index=50,
                last_character_index=150
            )
        ]

        q = AnsweredQuestion(
            question="What is RAG?",
            sources=sources,
            answer="RAG is Retrieval-Augmented Generation."
        )

        assert q.question == "What is RAG?"
        assert len(q.sources) == 2
        assert q.answer == "RAG is Retrieval-Augmented Generation."
        assert q.sources[0].file_path == "docs.md"

    def test_inherits_from_unanswered(self) -> None:
        """Test that AnsweredQuestion inherits question_id generation."""
        q = AnsweredQuestion(
            question="Test?",
            sources=[],
            answer="Test."
        )

        # Should have auto-generated question_id from parent
        assert q.question_id is not None
        uuid.UUID(q.question_id)

    def test_serialization(self) -> None:
        """Test JSON serialization with sources."""
        sources = [
            MinimalSource(
                file_path="test.py",
                first_character_index=0,
                last_character_index=50
            )
        ]

        q = AnsweredQuestion(
            question="Test?",
            sources=sources,
            answer="Answer."
        )

        # Serialize
        data = q.model_dump()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["file_path"] == "test.py"

        # Deserialize
        q2 = AnsweredQuestion(**data)
        assert q2.sources[0].file_path == "test.py"


# ============================================================================
# DATASET TESTS
# ============================================================================

class TestRagDataset:
    """Tests for RagDataset model."""

    def test_creation_mixed_questions(self) -> None:
        """Test dataset with both answered and unanswered questions."""
        questions = [
            UnansweredQuestion(question="Q1?"),
            AnsweredQuestion(
                question="Q2?",
                sources=[MinimalSource(file_path="f.py", first_character_index=0, last_character_index=50)],
                answer="Answer."
            ),
            UnansweredQuestion(question="Q3?"),
        ]

        dataset = RagDataset(rag_questions=questions)

        assert len(dataset.rag_questions) == 3
        assert isinstance(dataset.rag_questions[0], UnansweredQuestion)
        assert isinstance(dataset.rag_questions[1], AnsweredQuestion)

    def test_serialization(self) -> None:
        """Test JSON serialization of entire dataset."""
        questions = [
            UnansweredQuestion(question="Q1?"),
            AnsweredQuestion(
                question="Q2?",
                sources=[MinimalSource(file_path="f.py", first_character_index=0, last_character_index=50)],
                answer="Ans."
            ),
        ]

        dataset = RagDataset(rag_questions=questions)

        # Serialize
        json_str = dataset.model_dump_json()
        data = json.loads(json_str)

        # Deserialize
        dataset2 = RagDataset(**data)
        assert len(dataset2.rag_questions) == 2


# ============================================================================
# SEARCH RESULT TESTS
# ============================================================================

class TestMinimalSearchResults:
    """Tests for MinimalSearchResults model."""

    def test_creation(self) -> None:
        """Test basic creation."""
        sources = [
            MinimalSource(file_path="file1.py", first_character_index=0, last_character_index=100),
            MinimalSource(file_path="file2.py", first_character_index=50, last_character_index=150),
        ]

        result = MinimalSearchResults(
            question_id="q1",
            question="How to?",
            retrieved_sources=sources
        )

        assert result.question_id == "q1"
        assert len(result.retrieved_sources) == 2

    def test_empty_sources(self) -> None:
        """Test with no sources (empty search result)."""
        result = MinimalSearchResults(
            question_id="q1",
            question="Test?",
            retrieved_sources=[]
        )

        assert len(result.retrieved_sources) == 0

    def test_serialization(self) -> None:
        """Test JSON serialization."""
        sources = [MinimalSource(file_path="test.py", first_character_index=0, last_character_index=50)]
        result = MinimalSearchResults(
            question_id="q1",
            question="Q?",
            retrieved_sources=sources
        )

        data = result.model_dump()
        assert data["question_id"] == "q1"
        assert len(data["retrieved_sources"]) == 1


class TestMinimalAnswer:
    """Tests for MinimalAnswer model."""

    def test_creation(self) -> None:
        """Test creating answer with sources."""
        sources = [MinimalSource(file_path="doc.md", first_character_index=0, last_character_index=100)]

        answer = MinimalAnswer(
            question_id="q1",
            question="What is RAG?",
            retrieved_sources=sources,
            answer="RAG is..."
        )

        assert answer.question == "What is RAG?"
        assert answer.answer == "RAG is..."
        assert len(answer.retrieved_sources) == 1

    def test_inherits_search_results(self) -> None:
        """Test that MinimalAnswer extends MinimalSearchResults."""
        sources = [MinimalSource(file_path="test.py", first_character_index=0, last_character_index=50)]

        answer = MinimalAnswer(
            question_id="q1",
            question="Q?",
            retrieved_sources=sources,
            answer="A."
        )

        # Should have all MinimalSearchResults fields
        assert answer.question_id == "q1"
        assert answer.question == "Q?"
        assert answer.retrieved_sources == sources
        # Plus answer field
        assert answer.answer == "A."


# ============================================================================
# BATCH RESULT TESTS
# ============================================================================

class TestStudentSearchResults:
    """Tests for StudentSearchResults model."""

    def test_creation(self) -> None:
        """Test batch of search results."""
        results = [
            MinimalSearchResults(
                question_id="q1",
                question="Q1?",
                retrieved_sources=[MinimalSource(file_path="f.py", first_character_index=0, last_character_index=50)]
            ),
            MinimalSearchResults(
                question_id="q2",
                question="Q2?",
                retrieved_sources=[MinimalSource(file_path="f.py", first_character_index=100, last_character_index=150)]
            ),
        ]

        batch = StudentSearchResults(search_results=results, k=10)

        assert len(batch.search_results) == 2
        assert batch.k == 10

    def test_k_validation(self) -> None:
        """Test that k must be >= 1."""
        # Valid
        batch = StudentSearchResults(search_results=[], k=1)
        assert batch.k == 1

        # Invalid
        with pytest.raises(ValidationError):
            StudentSearchResults(search_results=[], k=0)  # ❌ k < 1

        with pytest.raises(ValidationError):
            StudentSearchResults(search_results=[], k=-5)  # ❌ k < 1


class TestStudentSearchResultsAndAnswer:
    """Tests for StudentSearchResultsAndAnswer model."""

    def test_creation(self) -> None:
        """Test batch with answers."""
        answers = [
            MinimalAnswer(
                question_id="q1",
                question="Q1?",
                retrieved_sources=[MinimalSource(file_path="f.py", first_character_index=0, last_character_index=50)],
                answer="A1."
            ),
            MinimalAnswer(
                question_id="q2",
                question="Q2?",
                retrieved_sources=[MinimalSource(file_path="f.py", first_character_index=100, last_character_index=150)],
                answer="A2."
            ),
        ]

        batch = StudentSearchResultsAndAnswer(search_results=answers, k=10)

        assert len(batch.search_results) == 2
        assert batch.search_results[0].answer == "A1."
        assert batch.search_results[1].answer == "A2."

    def test_serialization(self) -> None:
        """Test JSON serialization of batch with answers."""
        answers = [
            MinimalAnswer(
                question_id="q1",
                question="Q?",
                retrieved_sources=[MinimalSource(file_path="f.py", first_character_index=0, last_character_index=50)],
                answer="A."
            ),
        ]

        batch = StudentSearchResultsAndAnswer(search_results=answers, k=5)

        # Serialize and deserialize
        json_str = batch.model_dump_json()
        data = json.loads(json_str)
        batch2 = StudentSearchResultsAndAnswer(**data)

        assert batch2.k == 5
        assert batch2.search_results[0].answer == "A."


# ============================================================================
# INTERNAL MODEL TESTS
# ============================================================================

class TestChunk:
    """Tests for Chunk model."""

    def test_creation_python(self) -> None:
        """Test creating Python chunk."""
        chunk = Chunk(
            chunk_id="c1",
            content="def hello(): pass",
            file_path="hello.py",
            start_index=0,
            end_index=17,
            chunk_type="python"
        )

        assert chunk.content == "def hello(): pass"
        assert chunk.chunk_type == "python"
        assert chunk.start_index == 0

    def test_creation_text(self) -> None:
        """Test creating text chunk."""
        chunk = Chunk(
            chunk_id="c2",
            content="This is documentation",
            file_path="README.md",
            start_index=10,
            end_index=32,
            chunk_type="text"
        )

        assert chunk.chunk_type == "text"

    def test_auto_chunk_id(self) -> None:
        """Test auto-generated chunk_id."""
        c1 = Chunk(
            content="test",
            file_path="test.py",
            start_index=0,
            end_index=4,
            chunk_type="python"
        )

        c2 = Chunk(
            content="test",
            file_path="test.py",
            start_index=0,
            end_index=4,
            chunk_type="python"
        )

        # Different auto-generated IDs
        assert c1.chunk_id != c2.chunk_id
        assert len(c1.chunk_id) > 0

    def test_chunk_type_validation(self) -> None:
        """Test chunk_type must be 'python' or 'text'."""
        # Valid
        Chunk(
            content="test",
            file_path="test.py",
            start_index=0,
            end_index=4,
            chunk_type="python"
        )

        Chunk(
            content="test",
            file_path="test.md",
            start_index=0,
            end_index=4,
            chunk_type="text"
        )

        # Invalid
        with pytest.raises(ValidationError):
            Chunk(
                content="test",
                file_path="test.js",
                start_index=0,
                end_index=4,
                chunk_type="javascript"  # ❌ Not allowed
            )

    def test_index_constraints(self) -> None:
        """Test that indices must be >= 0."""
        # Valid
        Chunk(
            content="test",
            file_path="test.py",
            start_index=0,
            end_index=4,
            chunk_type="python"
        )

        # Invalid - negative index
        with pytest.raises(ValidationError):
            Chunk(
                content="test",
                file_path="test.py",
                start_index=-1,
                end_index=4,
                chunk_type="python"
            )


class TestIndexMetadata:
    """Tests for IndexMetadata model."""

    def test_creation(self) -> None:
        """Test creating metadata."""
        metadata = IndexMetadata(
            total_chunks=1000,
            file_count=50,
            index_type="bm25",
            created_at="2024-01-15T10:30:00"
        )

        assert metadata.total_chunks == 1000
        assert metadata.file_count == 50
        assert metadata.index_type == "bm25"

    def test_constraints(self) -> None:
        """Test count constraints (must be >= 0)."""
        # Valid
        IndexMetadata(
            total_chunks=0,
            file_count=0,
            index_type="bm25",
            created_at="2024-01-15T10:30:00"
        )

        # Invalid - negative count
        with pytest.raises(ValidationError):
            IndexMetadata(
                total_chunks=-1,
                file_count=50,
                index_type="bm25",
                created_at="2024-01-15T10:30:00"
            )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across multiple models."""

    def test_full_pipeline_serialization(self) -> None:
        """Test serialization of a complete RAG result."""
        # Create a complete answer with sources
        sources = [
            MinimalSource(file_path="docs/api.md", first_character_index=100, last_character_index=200),
            MinimalSource(file_path="src/api.py", first_character_index=50, last_character_index=150),
        ]

        answer = MinimalAnswer(
            question_id="q1",
            question="How to use the API?",
            retrieved_sources=sources,
            answer="The API can be used by..."
        )

        # Create a batch
        batch = StudentSearchResultsAndAnswer(
            search_results=[answer],
            k=10
        )

        # Serialize to JSON
        json_str = batch.model_dump_json(indent=2)

        # Parse and verify
        data = json.loads(json_str)
        assert data["k"] == 10
        assert len(data["search_results"]) == 1
        assert data[
                "search_results"
                    ][0]["answer"] == "The API can be used by..."
        assert len(data["search_results"][0]["retrieved_sources"]) == 2

        # Deserialize back
        batch2 = StudentSearchResultsAndAnswer(**data)
        assert batch2.search_results[0].question_id == "q1"

    def test_dataset_with_mix_types(self) -> None:
        """Test dataset with both answered and unanswered questions."""
        dataset = RagDataset(
            rag_questions=[
                UnansweredQuestion(question="Q1?"),
                AnsweredQuestion(
                    question="Q2?",
                    sources=[MinimalSource(file_path="f.py", first_character_index=0, last_character_index=50)],
                    answer="A."
                ),
                UnansweredQuestion(question="Q3?"),
            ]
        )

        # Serialize
        json_str = dataset.model_dump_json()

        # Deserialize
        dataset2 = RagDataset(**json.loads(json_str))

        assert len(dataset2.rag_questions) == 3
        assert isinstance(dataset2.rag_questions[0], UnansweredQuestion)
        assert isinstance(dataset2.rag_questions[1], AnsweredQuestion)


if __name__ == "__main__":
    # Run with: pytest tests/test_models.py -v
    pytest.main([__file__, "-v"])
