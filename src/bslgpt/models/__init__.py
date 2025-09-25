"""Data models for BSL GPT application."""

from .document import Document, DocumentStatus
from .query import Query, QueryResult

__all__ = ["Document", "DocumentStatus", "Query", "QueryResult"]