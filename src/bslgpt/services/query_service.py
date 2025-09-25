"""
Query Service for handling user queries and coordinating responses.
"""

import time
import re
from typing import List, Optional, Dict, Any
import structlog

from ..models.query import QueryRequest, QueryResult, ChatMessage
from .ai_service import AIService
from .document_service import DocumentService


class QueryService:
    """Service for processing user queries and coordinating AI responses."""
    
    def __init__(self, ai_service: AIService, document_service: DocumentService):
        """Initialize the query service."""
        self.ai_service = ai_service
        self.document_service = document_service
        self.logger = structlog.get_logger(__name__)
    
    def process_query(self, question: str, requested_files: Optional[List[str]] = None) -> QueryResult:
        """
        Process a user query and return a comprehensive result.
        
        Args:
            question: The user's question
            requested_files: Optional list of specific files to query
            
        Returns:
            QueryResult object with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Preprocess the question
            processed_question = self._preprocess_question(question)
            
            # Check for general queries first
            general_response = self.ai_service.process_general_query(processed_question)
            if general_response:
                processing_time = time.time() - start_time
                return QueryResult(
                    answer=general_response,
                    processing_time=processing_time,
                    confidence_score=0.9  # High confidence for general responses
                )
            
            # Get document contents
            all_documents = self.document_service.get_all_documents()
            
            if not all_documents:
                return QueryResult(
                    answer="I don't have any documents loaded to answer your question. Please ensure documents are properly initialized.",
                    error="No documents available",
                    processing_time=time.time() - start_time
                )
            
            # Filter documents if specific files are requested
            target_documents = self._filter_documents(all_documents, requested_files)
            
            if not target_documents:
                return QueryResult(
                    answer="The requested documents are not available or could not be processed.",
                    error="Requested documents not found",
                    processing_time=time.time() - start_time
                )
            
            # Query individual documents
            individual_answers = []
            documents_used = []
            
            for filename, content in target_documents.items():
                try:
                    answer = self.ai_service.query_single_document(content, question)
                    if answer and "error" not in answer.lower():
                        individual_answers.append(answer)
                        documents_used.append(filename)
                        self.logger.debug(f"Successfully queried document: {filename}")
                    else:
                        self.logger.warning(f"No relevant information found in: {filename}")
                except Exception as e:
                    self.logger.error(f"Error querying document {filename}: {e}")
                    continue
            
            if not individual_answers:
                return QueryResult(
                    answer="I couldn't find relevant information in the available documents for your question.",
                    processing_time=time.time() - start_time,
                    documents_used=list(target_documents.keys())
                )
            
            # Combine answers if multiple documents were queried
            if len(individual_answers) == 1:
                final_answer = individual_answers[0]
            else:
                final_answer = self.ai_service.combine_responses(individual_answers, question)
            
            # Calculate confidence score
            confidence_score = self.ai_service.calculate_confidence_score(
                question, final_answer, len(documents_used)
            )
            
            processing_time = time.time() - start_time
            
            return QueryResult(
                answer=final_answer,
                confidence_score=confidence_score,
                processing_time=processing_time,
                documents_used=documents_used,
                individual_answers=individual_answers
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            processing_time = time.time() - start_time
            return QueryResult(
                answer="I encountered an error while processing your question. Please try again.",
                error=str(e),
                processing_time=processing_time
            )
    
    def process_chat_query(self, request: QueryRequest) -> QueryResult:
        """
        Process a query with chat context.
        
        Args:
            request: QueryRequest object with question and context
            
        Returns:
            QueryResult object
        """
        # For now, we'll process it as a regular query
        # In the future, we can add chat history context
        return self.process_query(request.question, request.files)
    
    def _preprocess_question(self, question: str) -> str:
        """
        Preprocess the user question to improve matching and understanding.
        
        Args:
            question: The original question
            
        Returns:
            Preprocessed question
        """
        # Convert to lowercase for processing
        processed = question.lower().strip()
        
        # Remove special characters but keep essential punctuation
        processed = re.sub(r'[^\w\s\?\!\.\,\-]', '', processed)
        
        # Replace common abbreviations and synonyms
        replacements = {
            "bsl": "bokaro steel plant",
            "bsp": "bokaro steel plant", 
            "sail": "steel authority of india limited",
            "pcp": "purchase contract procedure",
            "gem": "government e-marketplace",
            "tc": "terms and conditions",
            "gst": "goods and services tax",
            "tender": "procurement tender",
            "vendor": "supplier",
            "po": "purchase order",
        }
        
        for abbrev, full_form in replacements.items():
            processed = processed.replace(abbrev, full_form)
        
        return processed
    
    def _filter_documents(self, all_documents: Dict[str, str], 
                         requested_files: Optional[List[str]]) -> Dict[str, str]:
        """
        Filter documents based on requested files.
        
        Args:
            all_documents: Dictionary of all available documents
            requested_files: Optional list of requested filenames
            
        Returns:
            Filtered dictionary of documents
        """
        if not requested_files:
            return all_documents
        
        filtered = {}
        for filename in requested_files:
            if filename in all_documents:
                filtered[filename] = all_documents[filename]
            else:
                # Try partial matching
                for doc_name, content in all_documents.items():
                    if filename.lower() in doc_name.lower():
                        filtered[doc_name] = content
                        break
        
        return filtered if filtered else all_documents
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """
        Get query suggestions based on partial input.
        
        Args:
            partial_query: Partial query string
            
        Returns:
            List of suggested completions
        """
        # This is a simple implementation - in production you might want
        # to use more sophisticated methods like search indices
        
        common_queries = [
            "What is the procurement procedure for BSL?",
            "What are the payment terms in BSL contracts?",
            "What is the penalty clause for delayed delivery?",
            "What are the quality requirements for materials?",
            "What is the tender process for BSL?",
            "What are the vendor registration requirements?",
            "What is the GST procedure for BSL purchases?",
            "What are the delivery terms for BSL orders?",
            "What is the inspection process for received materials?",
            "What are the performance guarantee requirements?",
        ]
        
        # Filter suggestions based on partial query
        if not partial_query or len(partial_query) < 3:
            return common_queries[:5]
        
        lower_partial = partial_query.lower()
        suggestions = [
            query for query in common_queries 
            if lower_partial in query.lower()
        ]
        
        return suggestions[:5]
    
    def validate_query(self, question: str) -> Dict[str, Any]:
        """
        Validate a user query.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check minimum length
        if len(question.strip()) < 3:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Question is too short. Please provide more details.")
        
        # Check maximum length
        if len(question) > 1000:
            validation_result["warnings"].append("Very long questions might not be processed effectively.")
        
        # Check for potentially problematic content
        problematic_patterns = [
            r'\b(hack|exploit|bypass)\b',
            r'\b(admin|password|login)\b',
            r'\b(delete|remove|destroy)\b'
        ]
        
        for pattern in problematic_patterns:
            if re.search(pattern, question.lower()):
                validation_result["warnings"].append("Your question might contain sensitive terms.")
                break
        
        # Suggest improvements
        if not question.strip().endswith('?'):
            validation_result["suggestions"].append("Consider ending your question with a question mark for better clarity.")
        
        return validation_result