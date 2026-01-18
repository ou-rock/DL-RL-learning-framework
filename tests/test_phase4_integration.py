"""Integration tests for Phase 4: Interactive Visualization"""

import pytest
import tempfile
import threading
import time
import socket
import requests
from pathlib import Path
from learning_framework.visualization import (
    VisualizationServer,
    VisualizationDataProvider,
    ComputationalGraphBuilder,
    ParameterExplorer,
    TrainingMonitor
)


def find_free_port():
    """Find a free port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


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


def test_full_visualization_workflow():
    """Test complete visualization workflow"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Setup mock concept
        concept_dir = data_dir / 'test_concept'
        concept_dir.mkdir()
        (concept_dir / 'concept.json').write_text('''
        {"name": "Test", "slug": "test_concept", "topic": "test"}
        ''')

        # Create data provider
        provider = VisualizationDataProvider(data_dir)

        # Verify concept discovery
        concepts = provider.get_concepts()
        assert len(concepts) == 1

        # Create server
        port = find_free_port()
        server = VisualizationServer(port=port, data_provider=provider)

        # Start server in background
        thread = threading.Thread(target=server.start, kwargs={'blocking': True}, daemon=True)
        thread.start()
        wait_for_server(port)

        try:
            # Test API endpoints
            response = requests.get(f'http://localhost:{port}/health')
            assert response.status_code == 200

            response = requests.get(f'http://localhost:{port}/api/concepts')
            assert response.status_code == 200
            assert len(response.json()['concepts']) == 1
        finally:
            # Stop server
            server.stop()
            time.sleep(0.1)


def test_graph_builder_integration():
    """Test computational graph builder integration"""
    builder = ComputationalGraphBuilder()

    graph = builder.build_feedforward_graph(
        input_dim=10,
        hidden_dims=[32, 16],
        output_dim=5,
        activations=['relu', 'relu', 'softmax'],
        batch_size=32,
        include_gradients=True
    )

    # Verify structure
    assert len(graph['nodes']) > 0
    assert len(graph['edges']) > 0

    # Verify gradient edges
    grad_edges = [e for e in graph['edges'] if e['type'] == 'gradient']
    assert len(grad_edges) > 0


def test_parameter_explorer_integration():
    """Test parameter explorer with loss computation"""
    import numpy as np

    explorer = ParameterExplorer()

    explorer.define_parameters({
        'learning_rate': {'min': 0.001, 'max': 1.0, 'default': 0.01, 'scale': 'log'}
    })

    def loss_fn(params):
        lr = params['learning_rate']
        return (np.log10(lr) + 2) ** 2

    explorer.set_loss_function(loss_fn)

    effect = explorer.compute_effect('learning_rate', num_points=20)

    assert len(effect['values']) == 20
    assert len(effect['losses']) == 20
    assert min(effect['losses']) < max(effect['losses'])


def test_training_monitor_integration():
    """Test training monitor full workflow"""
    monitor = TrainingMonitor(total_epochs=100)

    # Simulate training
    for epoch in range(50):
        loss = 1.0 / (epoch + 1)
        accuracy = 1.0 - loss
        monitor.record(epoch=epoch, loss=loss, accuracy=accuracy)

    # Verify outputs
    json_data = monitor.to_json()

    assert json_data['current']['epoch'] == 49
    assert json_data['total_epochs'] == 100
    assert len(json_data['history']['loss']) == 50

    # Verify time estimation
    remaining = monitor.estimate_remaining_time()
    assert remaining >= 0
