"""RAG Against the Machine - Retrieval-Augmented Generation System."""

__version__ = "1.0"
__author__ = "gabach"

from .models import (
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

__all__ = [
    "MinimalSource",
    "UnansweredQuestion",
    "AnsweredQuestion",
    "RagDataset",
    "MinimalSearchResults",
    "MinimalAnswer",
    "StudentSearchResults",
    "StudentSearchResultsAndAnswer",
    "Chunk",
    "IndexMetadata",
]
