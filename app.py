from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import base64
import os
from queries import (
    get_silenced_complaints,
    demographic_breakdown,
    geographic_breakdown,
    complaint_type_analysis,
    temporal_decay_analysis,
    similarity_search
)
# Lazy imports to avoid slow startup - imported inside functions
# from agentic_ai import generate_bias_report
# from multimodal import process_image_complaint, process_complaint_batch

# Initialize Flask app with static folder
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)  # Enable CORS for frontend access

print("=" * 60)
print(" SILENCE INDEX - API SERVER")
print("=" * 60)
print("Starting Flask server...")

@app.route('/')
def index():
    """Serve the dashboard"""
    return send_from_directory('static', 'index.html')

# ENDPOINT 1: OVERALL STATISTICS
@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get overall statistics about complaints.
    
    Returns:
        JSON with total complaints, silence rate, avg score, etc.
    """
    try:
        # Get silenced complaints
        silenced = get_silenced_complaints(threshold=70, limit=10000)
        total_silenced = len(silenced)
        
        # Calculate stats from silenced data
        if silenced:
            avg_silence = sum(c['silence_score'] for c in silenced) / len(silenced)
            avg_days = sum(c['days_in_system'] for c in silenced) / len(silenced)
        else:
            avg_silence = 0
            avg_days = 0
        
        # Total complaints (hardcoded as we know it's 10,000)
        total_complaints = 10000
        silence_rate = (total_silenced / total_complaints) * 100
        
        return jsonify({
            'success': True,
            'data': {
                'total_complaints': total_complaints,
                'total_silenced': total_silenced,
                'silence_rate': round(silence_rate, 1),
                'avg_silence_score': round(avg_silence, 1),
                'avg_days_in_system': round(avg_days, 0)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ENDPOINT 2: DEMOGRAPHIC BREAKDOWN
@app.route('/api/demographic-silence', methods=['GET'])
def get_demographic_silence():
    """
    Get silence score breakdown by demographics.
    
    Returns:
        JSON with breakdowns by gender, caste, and income
    """
    try:
        demographics = demographic_breakdown()
        
        return jsonify({
            'success': True,
            'data': demographics
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ENDPOINT 3: GEOGRAPHIC BREAKDOWN
@app.route('/api/geographic-silence', methods=['GET'])
def get_geographic_silence():
    """
    Get silence score breakdown by geography.
    
    Query params:
        - top_n: Number of top wards to return (default: 10)
    
    Returns:
        JSON with ward-level analysis
    """
    try:
        top_n = request.args.get('top_n', default=10, type=int)
        geography = geographic_breakdown(top_n=top_n)
        
        return jsonify({
            'success': True,
            'data': geography
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ENDPOINT 4: COMPLAINT TYPE ANALYSIS
@app.route('/api/complaint-types', methods=['GET'])
def get_complaint_types():
    """
    Get silence score breakdown by complaint category.
    
    Returns:
        JSON with category-level analysis
    """
    try:
        categories = complaint_type_analysis()
        
        return jsonify({
            'success': True,
            'data': categories
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ENDPOINT 5: TEMPORAL DECAY ANALYSIS
@app.route('/api/temporal-decay', methods=['GET'])
def get_temporal_decay():
    """
    Get silence score growth over time.
    
    Returns:
        JSON with time buckets and silence scores
    """
    try:
        temporal = temporal_decay_analysis()
        
        return jsonify({
            'success': True,
            'data': temporal
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ENDPOINT 6: SIMILARITY SEARCH
@app.route('/api/search', methods=['POST'])
def search_complaints():
    """
    Search for similar complaints using semantic search.
    
    Request body:
        {
            "query": "water supply problem",
            "top_k": 20,
            "silence_threshold": 70  (optional)
        }
    
    Returns:
        JSON with similar complaints
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "query" in request body'
            }), 400
        
        query_text = data['query']
        top_k = data.get('top_k', 20)
        silence_threshold = data.get('silence_threshold', None)
        
        results = similarity_search(
            query_text=query_text,
            top_k=top_k,
            silence_threshold=silence_threshold
        )
        
        return jsonify({
            'success': True,
            'data': {
                'query': query_text,
                'total_results': len(results),
                'results': results
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# HEALTH CHECK & ROOT
@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API info"""
    return jsonify({
        'name': 'Silence Index API',
        'version': '1.0',
        'status': 'running',
        'endpoints': [
            'GET  /api/stats',
            'GET  /api/demographic-silence',
            'GET  /api/geographic-silence',
            'GET  /api/complaint-types',
            'GET  /api/temporal-decay',
            'POST /api/search',
            'POST /api/agent/investigate',
            'POST /api/multimodal/process'
        ]
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'silence-index-api'
    })

@app.route('/api/agent/investigate', methods=['POST'])
def agent_investigate():
    """
    Trigger autonomous AI investigation.
    Agent will analyze all bias patterns and generate report.
    """
    try:
        from agentic_ai import generate_bias_report
        report = generate_bias_report()
        
        return jsonify({
            'success': True,
            'data': {
                'report': report,
                'report_file': 'AI_BIAS_REPORT.txt'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================
# CHAT ENDPOINTS
# ============================================================

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """
    Send a message to the AI chat agent.
    
    Request:
        - message: User's message
        - session_id: Optional session ID (creates new if not provided)
    
    Returns:
        - response: AI response text
        - chart: Optional Chart.js config
        - tools_used: List of data tools used
        - session_id: Session ID for continuing conversation
    """
    try:
        from chat_agent import chat, create_session
        
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "message" in request body'
            }), 400
        
        message = data['message']
        session_id = data.get('session_id')
        
        # Create new session if not provided
        if not session_id:
            session_id = create_session()
        
        # Get AI response
        result = chat(message, session_id)
        
        return jsonify({
            'success': True,
            'data': {
                'response': result['response'],
                'chart': result.get('chart'),
                'tools_used': result.get('tools_used', []),
                'session_id': session_id
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat/sessions', methods=['GET'])
def get_sessions():
    """Get list of all chat sessions"""
    try:
        from chat_agent import list_sessions
        sessions = list_sessions()
        return jsonify({
            'success': True,
            'data': sessions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Get chat history for a session"""
    try:
        from chat_agent import get_session_history
        history = get_session_history(session_id)
        return jsonify({
            'success': True,
            'data': history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat/session/new', methods=['POST'])
def new_session():
    """Create a new chat session"""
    try:
        from chat_agent import create_session
        data = request.get_json() or {}
        name = data.get('name')
        session_id = create_session(name)
        return jsonify({
            'success': True,
            'data': {
                'session_id': session_id,
                'name': name or f"Investigation {session_id}"
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat/search', methods=['POST'])
def search_history():
    """Semantic search across chat history"""
    try:
        from chat_agent import search_chat_history
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "query" in request body'
            }), 400
        
        results = search_chat_history(
            query=data['query'],
            session_id=data.get('session_id'),
            limit=data.get('limit', 10)
        )
        
        return jsonify({
            'success': True,
            'data': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/multimodal/process', methods=['POST'])
def process_multimodal_complaint():
    """
    Process multimodal complaint (text + optional image).
    
    Request:
        - text: Complaint text
        - image: Base64 encoded image (optional)
    """
    try:
        # Lazy import to avoid slow startup
        from multimodal import process_image_complaint
        
        data = request.get_json()
        text = data.get('text')
        image_b64 = data.get('image')
        
        if image_b64:
            # Save temp image
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                f.write(base64.b64decode(image_b64))
                temp_path = f.name
            
            result = process_image_complaint(temp_path, text)
            os.unlink(temp_path)  # Clean up
        else:
            # Text only
            from multimodal import get_embedding_model
            embedding_model = get_embedding_model()
            embedding = embedding_model.encode(text)
            result = {
                'text': text,
                'embedding': embedding.tolist(),
                'modality': 'text-only'
            }
        
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ERROR HANDLERS
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# RUN SERVER
if __name__ == '__main__':
    print("\n[OK] Flask app initialized")
    print("[OK] CORS enabled")
    print("[OK] 13 API endpoints registered")
    print("\n" + "=" * 60)
    print(" SENTINEL - Silence Index API Server")
    print("=" * 60)
    print("\nAnalytics Endpoints:")
    print("  GET  /api/stats")
    print("  GET  /api/demographic-silence")
    print("  GET  /api/geographic-silence")
    print("  GET  /api/complaint-types")
    print("  GET  /api/temporal-decay")
    print("  POST /api/search")
    print("\nAI Chat Endpoints:")
    print("  POST /api/chat                 - Send message")
    print("  GET  /api/chat/sessions        - List sessions")
    print("  GET  /api/chat/history/<id>    - Get session history")
    print("  POST /api/chat/session/new     - Create session")
    print("  POST /api/chat/search          - Search history")
    print("\nOther Endpoints:")
    print("  POST /api/agent/investigate")
    print("  POST /api/multimodal/process")
    print("\nServer: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
