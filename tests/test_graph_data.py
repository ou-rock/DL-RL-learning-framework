import pytest
import numpy as np
from learning_framework.visualization.graph_data import ComputationalGraphBuilder


def test_graph_builder_simple_network():
    """Test graph builder creates nodes for simple network"""
    builder = ComputationalGraphBuilder()

    # Simulate a simple 2-layer network
    graph_data = builder.build_feedforward_graph(
        input_dim=10,
        hidden_dims=[32, 16],
        output_dim=5
    )

    assert 'nodes' in graph_data
    assert 'edges' in graph_data

    # Check nodes exist for each layer
    node_ids = [n['id'] for n in graph_data['nodes']]
    assert 'input' in node_ids
    assert 'hidden_0' in node_ids
    assert 'hidden_1' in node_ids
    assert 'output' in node_ids


def test_graph_builder_includes_operations():
    """Test graph includes operation nodes (matmul, activation)"""
    builder = ComputationalGraphBuilder()

    graph_data = builder.build_feedforward_graph(
        input_dim=10,
        hidden_dims=[32],
        output_dim=5,
        activations=['relu', 'softmax']
    )

    node_types = [n['type'] for n in graph_data['nodes']]
    assert 'matmul' in node_types
    assert 'relu' in node_types


def test_graph_builder_computes_tensor_shapes():
    """Test graph includes tensor shapes"""
    builder = ComputationalGraphBuilder()

    graph_data = builder.build_feedforward_graph(
        input_dim=10,
        hidden_dims=[32],
        output_dim=5,
        batch_size=16
    )

    input_node = next(n for n in graph_data['nodes'] if n['id'] == 'input')
    assert input_node['shape'] == [16, 10]


def test_graph_builder_backprop_mode():
    """Test graph builder includes gradient flow"""
    builder = ComputationalGraphBuilder()

    graph_data = builder.build_feedforward_graph(
        input_dim=10,
        hidden_dims=[32],
        output_dim=5,
        include_gradients=True
    )

    # Check gradient edges exist
    grad_edges = [e for e in graph_data['edges'] if e.get('type') == 'gradient']
    assert len(grad_edges) > 0
