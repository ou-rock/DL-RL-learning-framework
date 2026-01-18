"""Local web server for interactive visualizations"""

import json
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs


class VisualizationHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for visualization server"""

    # Class variable set by server
    data_provider = None
    static_dir = None

    def __init__(self, *args, **kwargs):
        # Set directory for static files
        self.directory = str(self.static_dir) if self.static_dir else '.'
        super().__init__(*args, directory=self.directory, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)
        path = parsed.path

        # API endpoints
        if path == '/health':
            self._send_json({'status': 'ok'})
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
        else:
            # Serve static files
            super().do_GET()

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _handle_concepts(self):
        """Return list of available concepts"""
        if self.data_provider:
            concepts = self.data_provider.get_concepts()
        else:
            concepts = []
        self._send_json({'concepts': concepts})

    def _handle_concept_detail(self, concept_slug: str):
        """Return concept detail with visualization info"""
        if self.data_provider:
            concept = self.data_provider.get_concept_detail(concept_slug)
            if concept:
                self._send_json(concept)
            else:
                self._send_json({'error': 'Concept not found'}, 404)
        else:
            self._send_json({'error': 'Data provider not configured'}, 500)

    def _handle_graph_data(self):
        """Return computational graph data for viewer"""
        query = parse_qs(urlparse(self.path).query)
        concept = query.get('concept', [None])[0]

        if self.data_provider:
            graph_data = self.data_provider.get_graph_data(concept)
            self._send_json(graph_data)
        else:
            self._send_json({'nodes': [], 'edges': []})

    def _handle_visualization(self, concept_slug: str, viz_name: str):
        """Return visualization data as JSON"""
        if self.data_provider:
            viz_data = self.data_provider.get_visualization_data(concept_slug, viz_name)
            self._send_json(viz_data)
        else:
            self._send_json({'error': 'Data provider not configured'}, 500)

    def log_message(self, format, *args):
        """Suppress logging for cleaner output"""
        pass


class VisualizationServer:
    """Local web server for interactive visualizations"""

    def __init__(self, port: int = 8080, data_provider=None):
        """Initialize visualization server

        Args:
            port: Port to listen on
            data_provider: Data provider for concept/visualization data
        """
        self.port = port
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
        # Configure handler class variables
        VisualizationHandler.data_provider = self.data_provider
        VisualizationHandler.static_dir = self.static_dir

        self.server = HTTPServer(('localhost', self.port), VisualizationHandler)

        if blocking:
            self.server.serve_forever()
        else:
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the server"""
        if self.server:
            self.server.shutdown()
            self.server = None
