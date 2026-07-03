"""RAG Against the Machine - Retrieval-Augmented Generation System."""

__version__ = "1.6"
__author__ = "gabach"

from src.student.models import (
    MinimalSource,
    UnansweredQuestion,
    AnsweredQuestion,
    RagDataset,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

__all__ = [
    "MinimalSource",
    "UnansweredQuestion",
    "AnsweredQuestion",
    "RagDataset",
    "MinimalSearchResults",
    "MinimalAnswer",
    "StudentSearchResults",
    "StudentSearchResultsAndAnswer",
]
