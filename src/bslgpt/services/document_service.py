"""
Document Service for handling PDF processing and management.
"""

import os
import glob
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pdfplumber
import docx2txt
from PyPDF2 import PdfReader
import structlog

from ..config import Config
from ..models.document import DocumentInfo, DocumentStatus


class DocumentService:
    """Service for handling document processing and management."""
    
    def __init__(self, config: Config):
        """Initialize the document service."""
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.context_file = os.path.join(config.PDF_FOLDER, "document_contexts.pkl")
        self._document_cache: Dict[str, DocumentInfo] = {}
        
        # Ensure PDF folder exists
        os.makedirs(config.PDF_FOLDER, exist_ok=True)
        
        self.logger.info(f"Document service initialized with folder: {config.PDF_FOLDER}")
    
    def initialize_documents(self) -> Dict[str, int]:
        """
        Initialize and process all documents in the PDF folder.
        
        Returns:
            Dictionary with processing statistics
        """
        self.logger.info("Starting document initialization")
        
        # Find all supported files
        supported_files = self._find_supported_files()
        
        if not supported_files:
            self.logger.warning("No supported files found in PDF folder")
            return {"processed": 0, "errors": 0, "cached": 0}
        
        # Load existing contexts
        existing_contexts = self._load_cached_contexts()
        
        processed = 0
        errors = 0
        cached = 0
        
        for file_path in supported_files:
            filename = os.path.basename(file_path)
            
            try:
                # Check if file needs reprocessing
                if self._needs_reprocessing(file_path, existing_contexts):
                    self.logger.info(f"Processing file: {filename}")
                    doc_info = self._extract_document_content(file_path)
                    self._document_cache[filename] = doc_info
                    processed += 1
                else:
                    self.logger.debug(f"Using cached content for: {filename}")
                    self._document_cache[filename] = existing_contexts[filename]
                    cached += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
                errors += 1
                # Store error information
                error_doc = DocumentInfo(
                    filename=filename,
                    content="",
                    file_path=file_path,
                    file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    word_count=0,
                    page_count=0,
                    status=DocumentStatus.ERROR,
                    error_message=str(e),
                    created_at=datetime.now()
                )
                self._document_cache[filename] = error_doc
        
        # Save contexts to cache
        self._save_contexts_to_cache()
        
        stats = {"processed": processed, "errors": errors, "cached": cached}
        self.logger.info(f"Document initialization completed: {stats}")
        
        return stats
    
    def get_document_content(self, filename: str) -> Optional[str]:
        """
        Get the content of a specific document.
        
        Args:
            filename: Name of the document file
            
        Returns:
            Document content or None if not found
        """
        if filename in self._document_cache:
            doc_info = self._document_cache[filename]
            if doc_info.status == DocumentStatus.COMPLETED:
                return doc_info.content
        
        return None
    
    def get_all_documents(self) -> Dict[str, str]:
        """
        Get all document contents.
        
        Returns:
            Dictionary mapping filename to content for successfully processed documents
        """
        result = {}
        for filename, doc_info in self._document_cache.items():
            if doc_info.status == DocumentStatus.COMPLETED and doc_info.content:
                result[filename] = doc_info.content
        
        return result
    
    def get_document_info(self, filename: str) -> Optional[DocumentInfo]:
        """
        Get document information.
        
        Args:
            filename: Name of the document file
            
        Returns:
            DocumentInfo object or None if not found
        """
        return self._document_cache.get(filename)
    
    def list_documents(self) -> List[Dict[str, any]]:
        """
        List all documents with their metadata.
        
        Returns:
            List of document metadata dictionaries
        """
        documents = []
        for filename, doc_info in self._document_cache.items():
            documents.append({
                "filename": filename,
                "file_size": doc_info.file_size,
                "word_count": doc_info.word_count,
                "page_count": doc_info.page_count,
                "status": doc_info.status.value,
                "error_message": doc_info.error_message,
                "created_at": doc_info.created_at.isoformat() if doc_info.created_at else None,
                "processed_at": doc_info.processed_at.isoformat() if doc_info.processed_at else None,
            })
        
        return documents
    
    def _find_supported_files(self) -> List[str]:
        """Find all supported files in the PDF folder."""
        supported_files = []
        
        for extension in self.config.ALLOWED_EXTENSIONS:
            pattern = os.path.join(self.config.PDF_FOLDER, f"*.{extension}")
            files = glob.glob(pattern, recursive=False)
            supported_files.extend(files)
        
        return supported_files
    
    def _extract_document_content(self, file_path: str) -> DocumentInfo:
        """
        Extract content from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentInfo object with extracted content
        """
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_extension = os.path.splitext(filename)[1].lower()
        
        content = ""
        page_count = 0
        
        try:
            if file_extension == '.pdf':
                content, page_count = self._extract_pdf_content(file_path)
            elif file_extension == '.docx':
                content = self._extract_docx_content(file_path)
                page_count = 1  # Approximate for DOCX
            elif file_extension == '.txt':
                content = self._extract_txt_content(file_path)
                page_count = 1
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            word_count = len(content.split()) if content else 0
            
            return DocumentInfo(
                filename=filename,
                content=content,
                file_path=file_path,
                file_size=file_size,
                word_count=word_count,
                page_count=page_count,
                status=DocumentStatus.COMPLETED,
                created_at=datetime.now(),
                processed_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract content from {filename}: {e}")
            return DocumentInfo(
                filename=filename,
                content="",
                file_path=file_path,
                file_size=file_size,
                word_count=0,
                page_count=0,
                status=DocumentStatus.ERROR,
                error_message=str(e),
                created_at=datetime.now()
            )
    
    def _extract_pdf_content(self, file_path: str) -> Tuple[str, int]:
        """Extract content from PDF file using multiple methods."""
        content = ""
        page_count = 0
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
        except Exception as e:
            self.logger.warning(f"pdfplumber failed for {file_path}, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                reader = PdfReader(file_path)
                page_count = len(reader.pages)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
            except Exception as e2:
                raise Exception(f"Both PDF extraction methods failed: pdfplumber: {e}, PyPDF2: {e2}")
        
        return content.strip(), page_count
    
    def _extract_docx_content(self, file_path: str) -> str:
        """Extract content from DOCX file."""
        return docx2txt.process(file_path)
    
    def _extract_txt_content(self, file_path: str) -> str:
        """Extract content from TXT file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _load_cached_contexts(self) -> Dict[str, DocumentInfo]:
        """Load cached document contexts from pickle file."""
        try:
            if os.path.exists(self.context_file):
                with open(self.context_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Convert old format to new format if necessary
                if cached_data and isinstance(list(cached_data.values())[0], str):
                    # Old format: {filename: content}
                    converted = {}
                    for filename, content in cached_data.items():
                        file_path = os.path.join(self.config.PDF_FOLDER, filename)
                        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                        
                        converted[filename] = DocumentInfo(
                            filename=filename,
                            content=content,
                            file_path=file_path,
                            file_size=file_size,
                            word_count=len(content.split()) if content else 0,
                            page_count=1,  # Default for legacy data
                            status=DocumentStatus.COMPLETED,
                            created_at=datetime.now()
                        )
                    return converted
                
                return cached_data
        except Exception as e:
            self.logger.warning(f"Failed to load cached contexts: {e}")
        
        return {}
    
    def _save_contexts_to_cache(self):
        """Save document contexts to pickle file."""
        try:
            with open(self.context_file, 'wb') as f:
                pickle.dump(self._document_cache, f)
        except Exception as e:
            self.logger.error(f"Failed to save contexts to cache: {e}")
    
    def _needs_reprocessing(self, file_path: str, existing_contexts: Dict[str, DocumentInfo]) -> bool:
        """Check if a file needs reprocessing based on modification time."""
        filename = os.path.basename(file_path)
        
        if filename not in existing_contexts:
            return True
        
        try:
            file_mtime = os.path.getmtime(file_path)
            cached_doc = existing_contexts[filename]
            
            # If we have processed_at time, compare with file modification time
            if cached_doc.processed_at:
                cached_time = cached_doc.processed_at.timestamp()
                return file_mtime > cached_time
            else:
                # No processed_at time, needs reprocessing
                return True
                
        except (OSError, AttributeError):
            # If we can't get file modification time, reprocess
            return True
    
    def get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of a file for change detection."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""