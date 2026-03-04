"""
Base strategy interface that all extractors must implement.
Every strategy returns a normalized ExtractedDocument.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models import ExtractedDocument


class BaseExtractor(ABC):
    """Abstract base class for all extraction strategies."""

    name: str = "base"

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedDocument:
        """
        Extract content from a document and return normalized ExtractedDocument.
        Must always produce a result — never raise on partial failure.
        """

    @abstractmethod
    def confidence(self, doc: ExtractedDocument) -> float:
        """
        Compute a confidence score [0.0, 1.0] for the extraction result.
        Used by ExtractionRouter's escalation guard.
        """
