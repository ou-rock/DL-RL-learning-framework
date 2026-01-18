import pytest
import requests
import threading
import time
import socket
from learning_framework.visualization.server import VisualizationServer


def wait_for_server(port, timeout=5):
    """Wait for server to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f'http://localhost:{port}/health', timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
    return False


def find_free_port():
    """Find a free port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    """Fixture that provides a running server and cleans up after"""
    port = find_free_port()
    srv = VisualizationServer(port=port)
    thread = threading.Thread(target=srv.start, daemon=True)
    thread.start()
    wait_for_server(port)
    yield srv, port
    srv.stop()
    time.sleep(0.1)  # Allow socket to close


def test_server_starts_and_stops(server):
    """Test visualization server starts and stops cleanly"""
    srv, port = server

    # Check server is running
    response = requests.get(f'http://localhost:{port}/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_server_serves_static_files(server):
    """Test server serves static HTML/JS files"""
    srv, port = server

    # Request index page
    response = requests.get(f'http://localhost:{port}/')
    assert response.status_code == 200
    assert 'text/html' in response.headers['Content-Type']


def test_server_provides_concept_data(server):
    """Test server provides concept data as JSON"""
    srv, port = server

    # Request concept list
    response = requests.get(f'http://localhost:{port}/api/concepts')
    assert response.status_code == 200
    data = response.json()
    assert 'concepts' in data
