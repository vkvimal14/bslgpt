"""API routes for BSL GPT application."""

from flask import request, jsonify, current_app
from . import api_bp
import structlog


logger = structlog.get_logger(__name__)


@api_bp.route('/documents', methods=['GET'])
def list_documents():
    """List all available documents."""
    try:
        documents = current_app.document_service.list_documents()
        return jsonify({
            'documents': documents,
            'count': len(documents)
        })
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({'error': 'Failed to list documents'}), 500


@api_bp.route('/documents/<filename>', methods=['GET'])
def get_document(filename):
    """Get information about a specific document."""
    try:
        doc_info = current_app.document_service.get_document_info(filename)
        if not doc_info:
            return jsonify({'error': 'Document not found'}), 404
        
        return jsonify({
            'filename': doc_info.filename,
            'file_size': doc_info.file_size,
            'word_count': doc_info.word_count,
            'page_count': doc_info.page_count,
            'status': doc_info.status.value,
            'error_message': doc_info.error_message,
            'created_at': doc_info.created_at.isoformat() if doc_info.created_at else None,
            'processed_at': doc_info.processed_at.isoformat() if doc_info.processed_at else None,
        })
    except Exception as e:
        logger.error(f"Error getting document {filename}: {e}")
        return jsonify({'error': 'Failed to get document information'}), 500


@api_bp.route('/query/suggestions', methods=['GET'])
def get_query_suggestions():
    """Get query suggestions based on partial input."""
    try:
        partial_query = request.args.get('q', '')
        suggestions = current_app.query_service.get_query_suggestions(partial_query)
        return jsonify({'suggestions': suggestions})
    except Exception as e:
        logger.error(f"Error getting query suggestions: {e}")
        return jsonify({'error': 'Failed to get suggestions'}), 500


@api_bp.route('/query/validate', methods=['POST'])
def validate_query():
    """Validate a user query."""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question parameter'}), 400
        
        question = data['question']
        validation_result = current_app.query_service.validate_query(question)
        return jsonify(validation_result)
    except Exception as e:
        logger.error(f"Error validating query: {e}")
        return jsonify({'error': 'Failed to validate query'}), 500


@api_bp.route('/search', methods=['POST'])
def advanced_search():
    """Advanced search endpoint with filters and options."""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question parameter'}), 400
        
        question = data['question']
        files = data.get('files', [])
        include_metadata = data.get('include_metadata', False)
        
        result = current_app.query_service.process_query(question, files)
        
        response_data = result.to_dict()
        
        if include_metadata:
            # Add additional metadata
            response_data['metadata'] = {
                'query_length': len(question),
                'files_requested': files,
                'files_processed': len(result.documents_used),
                'processing_time_ms': int(result.processing_time * 1000)
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in advanced search: {e}")
        return jsonify({'error': 'Search failed'}), 500


@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        documents = current_app.document_service.list_documents()
        
        stats = {
            'documents': {
                'total': len(documents),
                'processed': len([d for d in documents if d['status'] == 'completed']),
                'errors': len([d for d in documents if d['status'] == 'error']),
                'total_size': sum(d['file_size'] for d in documents),
                'total_words': sum(d['word_count'] for d in documents),
                'total_pages': sum(d['page_count'] for d in documents),
            },
            'system': {
                'version': '2.0.0',
                'ai_model': current_app.config.get('GEMINI_MODEL', 'gemini-2.0-flash-exp'),
                'pdf_folder': current_app.config.get('PDF_FOLDER', './pdf'),
            }
        }
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500