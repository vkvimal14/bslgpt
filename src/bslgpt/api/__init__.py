"""API blueprints for BSL GPT application."""

from flask import Blueprint

# Create the main API blueprint
api_bp = Blueprint('api', __name__)

# Import routes to register them
from . import routes

__all__ = ['api_bp']