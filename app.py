#!/usr/bin/env python3
"""
Modern BSL GPT Application Entry Point
=====================================

This is the main entry point for the modernized BSL GPT application.
Run this file to start the Flask server.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.bslgpt.app import main

if __name__ == '__main__':
    main()