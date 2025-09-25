"""Error handlers for Flask application."""

from flask import Flask, jsonify, request
import structlog
from werkzeug.exceptions import HTTPException


def register_error_handlers(app: Flask) -> None:
    """
    Register error handlers for the Flask application.
    
    Args:
        app: Flask application instance
    """
    logger = structlog.get_logger(__name__)
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        logger.warning(f"404 error: {request.url}")
        return jsonify({
            'error': 'Not Found',
            'message': 'The requested resource was not found',
            'status_code': 404
        }), 404
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 errors."""
        logger.warning(f"400 error: {request.url} - {error}")
        return jsonify({
            'error': 'Bad Request',
            'message': 'The request was invalid or cannot be served',
            'status_code': 400
        }), 400
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logger.error(f"500 error: {request.url} - {error}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An internal server error occurred',
            'status_code': 500
        }), 500
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle all HTTP exceptions."""
        logger.warning(f"HTTP {error.code} error: {request.url} - {error}")
        return jsonify({
            'error': error.name,
            'message': error.description,
            'status_code': error.code
        }), error.code
    
    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        """Handle all other exceptions."""
        logger.error(f"Unhandled exception: {request.url} - {error}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500