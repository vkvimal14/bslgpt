"""Document model for storing PDF and document information."""

from datetime import datetime
from enum import Enum
from typing import Optional
from dataclasses import dataclass
from sqlalchemy import Column, Integer, String, Text, DateTime, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class DocumentStatus(Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class Document(Base):
    """Document model for storing PDF and document metadata."""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, unique=True)
    original_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    
    # Content and processing
    content = Column(Text, nullable=True)
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.PENDING)
    error_message = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime, nullable=True)
    
    # Search and indexing
    word_count = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status.value}')>"
    
    def to_dict(self) -> dict:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "original_name": self.original_name,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "status": self.status.value,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "word_count": self.word_count,
            "page_count": self.page_count,
        }


@dataclass
class DocumentInfo:
    """Data class for document information without SQLAlchemy dependencies."""
    
    filename: str
    content: str
    file_path: str
    file_size: int
    word_count: int
    page_count: int
    status: DocumentStatus
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None