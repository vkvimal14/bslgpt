"""
BSL GPT - AI-Powered Document Query System
==========================================

A modern Flask-based web application for intelligent document processing
and question-answering using Google's Gemini AI.
"""

__version__ = "2.0.0"
__author__ = "BSL GPT Team"
__email__ = "support@bslgpt.com"

from .app import create_app

__all__ = ["create_app"]