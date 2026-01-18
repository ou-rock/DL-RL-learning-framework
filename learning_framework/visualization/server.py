"""Local web server for interactive visualizations

Production-ready server with:
- CORS support for cross-origin requests
- User authentication via X-User-ID header
- Progress tracking API endpoints
- PostgreSQL integration for persistent storage
"""

import argparse
import json
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs

# Database imports (optional - graceful fallback if unavailable)
try:
    from ..db import init_pool, close_pool, ensure_schema
    from ..db.models import UserRepository, ProgressRepository, ReviewRepository
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    init_pool = close_pool = ensure_schema = None
    UserRepository = ProgressRepository = ReviewRepository = None


def get_allowed_origins() -> set:
    """Get allowed CORS origins from environment"""
    origins = os.environ.get('ALLOWED_ORIGINS', '')
    if origins:
        return set(o.strip() for o in origins.split(','))
    # Default: allow all in development
    return {'*'}


class VisualizationHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for visualization server"""

    # Class variables set by server
    data_provider = None
    static_dir = None
    allowed_origins = {'*'}

    def __init__(self, *args, **kwargs):
        # Set directory for static files
        self.directory = str(self.static_dir) if self.static_dir else '.'
        super().__init__(*args, directory=self.directory, **kwargs)

    def _get_cors_origin(self) -> str:
        """Get appropriate CORS origin header value"""
        origin = self.headers.get('Origin', '')
        if '*' in self.allowed_origins:
            return '*'
        if origin in self.allowed_origins:
            return origin
        return ''

    def _get_user_id(self) -> Optional[str]:
        """Extract user ID from X-User-ID header"""
        return self.headers.get('X-User-ID')

    def _send_cors_headers(self):
        """Send CORS headers for preflight and actual requests"""
        origin = self._get_cors_origin()
        if origin:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-User-ID')
            self.send_header('Access-Control-Max-Age', '86400')

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)
        path = parsed.path

        # API endpoints
        if path == '/health':
            self._send_json({'status': 'ok', 'db_available': DB_AVAILABLE})
        elif path == '/api/concepts':
            self._handle_concepts()
        elif path.startswith('/api/concept/'):
            concept_slug = path.split('/')[-1]
            self._handle_concept_detail(concept_slug)
        elif path == '/api/graph':
            self._handle_graph_data()
        elif path.startswith('/api/viz/'):
            parts = path.split('/')
            if len(parts) >= 4:
                concept_slug = parts[3]
                viz_name = parts[4] if len(parts) > 4 else 'main_visualization'
                self._handle_visualization(concept_slug, viz_name)
        elif path == '/api/progress':
            self._handle_progress_get()
        elif path == '/api/reviews':
            self._handle_reviews_get()
        else:
            # Serve static files
            super().do_GET()

    def do_POST(self):
        """Handle POST requests"""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/progress':
            self._handle_progress_post()
        elif path == '/api/reviews':
            self._handle_review_post()
        else:
            self._send_json({'error': 'Not found'}, 404)

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response with CORS headers"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _read_json_body(self) -> Optional[Dict[str, Any]]:
        """Read and parse JSON request body"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                return None
            body = self.rfile.read(content_length)
            return json.loads(body.decode('utf-8'))
        except (json.JSONDecodeError, ValueError):
            return None

    def _load_static_concepts(self):
        """Load concepts from static JSON file when no data_provider"""
        try:
            static_file = Path(self.static_dir) / 'data' / 'concepts.json'
            if static_file.exists():
                with open(static_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('concepts', [])
        except Exception as e:
            print(f"Failed to load static concepts: {e}")
        return []

    def _load_static_concept(self, slug: str):
        """Load a specific concept from static JSON"""
        try:
            # Try concept-specific file first
            concept_file = Path(self.static_dir) / 'data' / 'concepts' / f'{slug}.json'
            if concept_file.exists():
                with open(concept_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            # Fallback to concepts list
            concepts = self._load_static_concepts()
            for c in concepts:
                if c.get('slug') == slug:
                    return c
        except Exception as e:
            print(f"Failed to load static concept {slug}: {e}")
        return None

    def _load_static_graph(self, concept: str = None):
        """Load graph data from static JSON"""
        try:
            # Try concept-specific graph first
            if concept:
                concept_file = Path(self.static_dir) / 'data' / 'concepts' / f'{concept}.json'
                if concept_file.exists():
                    with open(concept_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'graph' in data:
                            return data['graph']
            # Fallback to default graph
            graph_file = Path(self.static_dir) / 'data' / 'graph.json'
            if graph_file.exists():
                with open(graph_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Failed to load static graph: {e}")
        return {'nodes': [], 'edges': []}

    def _handle_concepts(self):
        """Return list of available concepts with user progress"""
        user_id = self._get_user_id()

        if self.data_provider:
            concepts = self.data_provider.get_concepts()
        else:
            # Fallback: load from static JSON file
            concepts = self._load_static_concepts()

        # Attach user progress if available
        progress_map = {}
        if user_id and DB_AVAILABLE and ProgressRepository:
            UserRepository.ensure_user(user_id)
            progress_list = ProgressRepository.get_progress(user_id)
            progress_map = {p['concept']: p for p in progress_list}

        # Merge progress into concepts
        for concept in concepts:
            slug = concept.get('slug', concept.get('id', ''))
            if slug in progress_map:
                concept['progress'] = progress_map[slug]

        self._send_json({'concepts': concepts})

    def _handle_concept_detail(self, concept_slug: str):
        """Return concept detail with visualization info"""
        if self.data_provider:
            concept = self.data_provider.get_concept_detail(concept_slug)
        else:
            # Fallback: load from static JSON
            concept = self._load_static_concept(concept_slug)

        if concept:
            self._send_json(concept)
        else:
            self._send_json({'error': 'Concept not found'}, 404)

    def _handle_graph_data(self):
        """Return computational graph data for viewer"""
        query = parse_qs(urlparse(self.path).query)
        concept = query.get('concept', [None])[0]

        if self.data_provider:
            graph_data = self.data_provider.get_graph_data(concept)
        else:
            # Fallback: load from static JSON
            graph_data = self._load_static_graph(concept)

        self._send_json(graph_data)

    def _handle_visualization(self, concept_slug: str, viz_name: str):
        """Return visualization data as JSON"""
        if self.data_provider:
            viz_data = self.data_provider.get_visualization_data(concept_slug, viz_name)
        else:
            # Fallback: return graph visualization from static data
            graph_data = self._load_static_graph(concept_slug)
            viz_data = {'type': 'graph', **graph_data}

        self._send_json(viz_data)

    def _handle_progress_get(self):
        """Get user progress for all or specific concept"""
        user_id = self._get_user_id()
        if not user_id:
            self._send_json({'error': 'X-User-ID header required'}, 401)
            return

        if not DB_AVAILABLE or not ProgressRepository:
            self._send_json({'error': 'Database not available'}, 503)
            return

        query = parse_qs(urlparse(self.path).query)
        concept = query.get('concept', [None])[0]

        progress = ProgressRepository.get_progress(user_id, concept)
        self._send_json({'progress': progress})

    def _handle_progress_post(self):
        """Update user progress after quiz completion"""
        user_id = self._get_user_id()
        if not user_id:
            self._send_json({'error': 'X-User-ID header required'}, 401)
            return

        if not DB_AVAILABLE or not ProgressRepository:
            self._send_json({'error': 'Database not available'}, 503)
            return

        body = self._read_json_body()
        if not body:
            self._send_json({'error': 'Invalid JSON body'}, 400)
            return

        concept = body.get('concept')
        score = body.get('score')

        if not concept or score is None:
            self._send_json({'error': 'concept and score required'}, 400)
            return

        # Ensure user exists
        UserRepository.ensure_user(user_id)

        # Update progress
        success = ProgressRepository.update_progress(user_id, concept, float(score))
        if success:
            # Also schedule spaced repetition review
            quality = 5 if score >= 0.9 else (4 if score >= 0.7 else (3 if score >= 0.5 else 2))
            ReviewRepository.schedule_review(user_id, concept, quality)
            self._send_json({'success': True})
        else:
            self._send_json({'error': 'Failed to update progress'}, 500)

    def _handle_reviews_get(self):
        """Get concepts due for review"""
        user_id = self._get_user_id()
        if not user_id:
            self._send_json({'error': 'X-User-ID header required'}, 401)
            return

        if not DB_AVAILABLE or not ReviewRepository:
            self._send_json({'error': 'Database not available'}, 503)
            return

        reviews = ReviewRepository.get_due_reviews(user_id)
        self._send_json({'reviews': reviews})

    def _handle_review_post(self):
        """Record review result and schedule next"""
        user_id = self._get_user_id()
        if not user_id:
            self._send_json({'error': 'X-User-ID header required'}, 401)
            return

        if not DB_AVAILABLE or not ReviewRepository:
            self._send_json({'error': 'Database not available'}, 503)
            return

        body = self._read_json_body()
        if not body:
            self._send_json({'error': 'Invalid JSON body'}, 400)
            return

        concept = body.get('concept')
        quality = body.get('quality')  # 0-5 scale

        if not concept or quality is None:
            self._send_json({'error': 'concept and quality required'}, 400)
            return

        success = ReviewRepository.schedule_review(user_id, concept, int(quality))
        if success:
            self._send_json({'success': True})
        else:
            self._send_json({'error': 'Failed to schedule review'}, 500)

    def log_message(self, format, *args):
        """Log requests (enabled for production debugging)"""
        print(f"[{self.log_date_time_string()}] {format % args}")


class VisualizationServer:
    """Local web server for interactive visualizations"""

    def __init__(self, port: int = 8080, data_provider=None, host: str = '0.0.0.0'):
        """Initialize visualization server

        Args:
            port: Port to listen on
            data_provider: Data provider for concept/visualization data
            host: Host to bind to (0.0.0.0 for production, localhost for dev)
        """
        self.port = port
        self.host = host
        self.data_provider = data_provider
        self.server = None
        self.thread = None

        # Set static files directory
        self.static_dir = Path(__file__).parent / 'web'
        self.static_dir.mkdir(exist_ok=True)

    def start(self, blocking: bool = True):
        """Start the server

        Args:
            blocking: If True, block until stopped. If False, run in thread.
        """
        # Initialize database if available
        if DB_AVAILABLE and init_pool:
            if init_pool():
                ensure_schema()
                print("Database initialized successfully")
            else:
                print("Running without database (local mode)")

        # Configure handler class variables
        VisualizationHandler.data_provider = self.data_provider
        VisualizationHandler.static_dir = self.static_dir
        VisualizationHandler.allowed_origins = get_allowed_origins()

        self.server = HTTPServer((self.host, self.port), VisualizationHandler)
        print(f"Server started at http://{self.host}:{self.port}")

        if blocking:
            try:
                self.server.serve_forever()
            finally:
                self.stop()
        else:
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the server and cleanup"""
        if self.server:
            self.server.shutdown()
            self.server = None
        if DB_AVAILABLE and close_pool:
            close_pool()


def main():
    """CLI entry point for running the server"""
    parser = argparse.ArgumentParser(description='Learning Framework Visualization Server')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8080)),
                        help='Port to listen on (default: $PORT or 8080)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--dev', action='store_true',
                        help='Run in development mode (localhost only)')
    args = parser.parse_args()

    host = 'localhost' if args.dev else args.host

    server = VisualizationServer(port=args.port, host=host)
    server.start(blocking=True)


if __name__ == '__main__':
    main()
