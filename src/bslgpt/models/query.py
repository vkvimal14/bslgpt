"""Query models for storing user queries and responses."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.sql import func
from .document import Base


class Query(Base):
    """Query model for storing user queries and responses."""
    
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    processed_question = Column(Text, nullable=True)
    
    # Response data
    response = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=False)  # in seconds
    
    # Document references
    documents_used = Column(JSON, nullable=True)  # List of document IDs/filenames
    
    # User and session info
    user_id = Column(String(100), nullable=True)
    session_id = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self) -> str:
        return f"<Query(id={self.id}, question='{self.question[:50]}...', created_at='{self.created_at}')>"
    
    def to_dict(self) -> dict:
        """Convert query to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "response": self.response,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "documents_used": self.documents_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class QueryRequest:
    """Data class for incoming query requests."""
    
    question: str
    files: List[str] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class QueryResult:
    """Data class for query results."""
    
    answer: str
    confidence_score: Optional[float] = None
    processing_time: float = 0.0
    documents_used: List[str] = field(default_factory=list)
    individual_answers: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "documents_used": self.documents_used,
            "error": self.error,
        }


@dataclass
class ChatMessage:
    """Data class for chat messages."""
    
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }