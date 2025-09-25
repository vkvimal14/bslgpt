"""
Main Flask application factory for BSL GPT.
"""

import os
import logging
from typing import Optional

from flask import Flask, request, jsonify, render_template
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    import logging as structlog

from .config import get_config, Config
from .services import DocumentService, AIService, QueryService
from .api import api_bp
from .utils.logging import setup_logging
from .utils.error_handlers import register_error_handlers


def create_app(config: Optional[Config] = None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config: Optional configuration object. If None, loads from environment.
        
    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__, 
                template_folder='../../templates',
                static_folder='../../static')
    
    # Load configuration
    if config is None:
        config = get_config()
    
    app.config.from_object(config)
    
    # Setup logging
    setup_logging(config.LOG_LEVEL)
    if STRUCTLOG_AVAILABLE:
        logger = structlog.get_logger(__name__)
    else:
        logger = logging.getLogger(__name__)
    
    # Enable CORS if available
    if CORS_AVAILABLE:
        CORS(app)
    
    # Initialize services
    try:
        document_service = DocumentService(config)
        ai_service = AIService(config)
        query_service = QueryService(ai_service, document_service)
        
        # Store services in app context
        app.document_service = document_service
        app.ai_service = ai_service
        app.query_service = query_service
        
        logger.info("Services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register API blueprint
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'version': '2.0.0',
            'services': {
                'document_service': 'active',
                'ai_service': 'active',
                'query_service': 'active'
            }
        })
    
    # Main web interface
    @app.route('/')
    def index():
        """Serve the main web interface."""
        return render_template('index.html')
    
    # Legacy compatibility routes
    @app.route('/api/init', methods=['POST'])
    def legacy_init():
        """Legacy initialization endpoint for backward compatibility."""
        try:
            result = app.document_service.initialize_documents()
            return jsonify({'message': f'Initialized {result["processed"]} documents successfully'})
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return jsonify({'error': 'Failed to initialize documents'}), 500
    
    @app.route('/api/query', methods=['POST'])
    def legacy_query():
        """Legacy query endpoint for backward compatibility."""
        try:
            data = request.get_json()
            if not data or 'question' not in data:
                return jsonify({'error': 'Missing question parameter'}), 400
            
            question = data['question']
            files = data.get('files', [])
            
            result = app.query_service.process_query(question, files)
            
            if result.error:
                return jsonify({'error': result.error}), 500
            
            return jsonify({'answer': result.answer})
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return jsonify({'error': 'Failed to process query'}), 500
    
    logger.info("BSL GPT application created successfully")
    return app


def main():
    """Main entry point for running the application."""
    config = get_config()
    app = create_app(config)
    
    # Ensure PDF folder exists
    os.makedirs(config.PDF_FOLDER, exist_ok=True)
    
    # Initialize documents on startup
    try:
        app.document_service.initialize_documents()
        print(f"✅ Application initialized successfully")
        print(f"📁 PDF folder: {config.PDF_FOLDER}")
        print(f"🤖 AI Model: {config.GEMINI_MODEL}")
        print(f"🌐 Starting server on http://0.0.0.0:5000")
    except Exception as e:
        print(f"❌ Failed to initialize application: {e}")
        return
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=config.FLASK_DEBUG
    )


if __name__ == '__main__':
    main()