"""
AI Service for handling Gemini AI interactions.
"""

import time
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import structlog
from ..config import Config
from ..models.query import QueryResult


class AIService:
    """Service for handling AI interactions with Google Gemini."""
    
    def __init__(self, config: Config):
        """Initialize the AI service with configuration."""
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Configure Gemini AI
        genai.configure(api_key=config.GEMINI_API_KEY)
        
        # Set up generation configuration
        self.generation_config = {
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "top_k": config.TOP_K,
            "max_output_tokens": config.MAX_OUTPUT_TOKENS,
            "response_mime_type": "text/plain",
        }
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config=self.generation_config,
        )
        
        self.logger.info(f"AI Service initialized with model: {config.GEMINI_MODEL}")
    
    def query_single_document(self, document_content: str, question: str) -> str:
        """
        Query a single document with a question.
        
        Args:
            document_content: The text content of the document
            question: The user's question
            
        Returns:
            The AI's response
        """
        try:
            chat_session = self.model.start_chat(history=[])
            
            prompt = self._build_single_document_prompt(document_content, question)
            
            response = chat_session.send_message(prompt)
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"Error querying single document: {e}")
            return f"Error processing document: {str(e)}"
    
    def combine_responses(self, individual_responses: List[str], question: str) -> str:
        """
        Combine multiple document responses into a final answer.
        
        Args:
            individual_responses: List of responses from individual documents
            question: The original user question
            
        Returns:
            Combined and synthesized response
        """
        if not individual_responses:
            return "No documents were processed for this query."
        
        if len(individual_responses) == 1:
            return individual_responses[0]
        
        try:
            chat_session = self.model.start_chat(history=[])
            
            prompt = self._build_combination_prompt(individual_responses, question)
            
            response = chat_session.send_message(prompt)
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"Error combining responses: {e}")
            return "Error synthesizing responses from multiple documents."
    
    def process_general_query(self, question: str) -> Optional[str]:
        """
        Process general queries that don't require document context.
        
        Args:
            question: The user's question
            
        Returns:
            Response for general queries, or None if not a general query
        """
        # Normalize question
        normalized = question.lower().strip()
        
        # General responses mapping
        general_responses = {
            "hello": "Hello! I'm BSL GPT, your AI assistant for Bokaro Steel Plant documentation. How can I help you today?",
            "hi": "Hi there! I'm here to help you with questions about BSL documents. What would you like to know?",
            "how are you": "I'm functioning well and ready to assist you with your BSL documentation queries!",
            "what's your name": "I'm BSL GPT, an AI assistant specialized in Bokaro Steel Plant documentation.",
            "what can you do": "I can help you find information from BSL documents, answer questions about procedures, policies, and general information related to Bokaro Steel Plant.",
            "thank you": "You're welcome! Feel free to ask if you have any other questions.",
            "thanks": "You're welcome! I'm here whenever you need help with BSL documentation.",
            "hey": "Hey there! Ready to help you with BSL documentation questions.",
        }
        
        # Check for exact matches first
        if normalized in general_responses:
            return general_responses[normalized]
        
        # Check for partial matches
        for key, response in general_responses.items():
            if key in normalized:
                return response
        
        return None
    
    def _build_single_document_prompt(self, document_content: str, question: str) -> str:
        """Build prompt for querying a single document."""
        return f"""You are an AI assistant specialized in analyzing Bokaro Steel Plant (BSL) documentation. 

Below is content from a BSL document:

{document_content}

User Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the document above
2. If the document doesn't contain relevant information, clearly state that
3. Be specific and cite relevant sections when possible
4. Use professional language appropriate for industrial documentation
5. If asked about BSL, BSP, or Bokaro Steel Plant, refer to the information in the document

Please provide a comprehensive answer:"""
    
    def _build_combination_prompt(self, responses: List[str], question: str) -> str:
        """Build prompt for combining multiple document responses."""
        combined_responses = "\n\n---\n\n".join([
            f"Response from Document {i+1}:\n{response}" 
            for i, response in enumerate(responses)
        ])
        
        return f"""I am an AI assistant for Bokaro Steel Plant (BSL) documentation. I have received multiple responses from different documents for the same question.

Original Question: {question}

Multiple Document Responses:
{combined_responses}

Instructions:
1. Synthesize the information from all responses into a comprehensive, coherent answer
2. Remove any contradictions by prioritizing the most recent or authoritative information
3. Combine complementary information from different sources
4. Maintain professional language appropriate for industrial documentation
5. If responses conflict, acknowledge the conflict and provide the most likely correct information
6. Present the final answer as if it comes from a single, authoritative source

Please provide a final, synthesized answer:"""
    
    def calculate_confidence_score(self, question: str, answer: str, document_count: int) -> float:
        """
        Calculate a confidence score for the answer.
        
        This is a simple heuristic-based approach. In a production system,
        you might want to use more sophisticated methods.
        
        Args:
            question: The user's question
            answer: The generated answer
            document_count: Number of documents that contributed to the answer
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_score = 0.5
        
        # Boost confidence if multiple documents were used
        if document_count > 1:
            base_score += 0.2
        
        # Boost confidence for longer, more detailed answers
        if len(answer) > 200:
            base_score += 0.1
        
        # Reduce confidence if answer contains uncertainty words
        uncertainty_words = ["might", "could", "possibly", "maybe", "unclear", "not sure"]
        uncertainty_count = sum(1 for word in uncertainty_words if word in answer.lower())
        base_score -= uncertainty_count * 0.1
        
        # Ensure score is within bounds
        return max(0.1, min(0.9, base_score))