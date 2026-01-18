import pytest
import tempfile
from pathlib import Path
from learning_framework.visualization.data_provider import VisualizationDataProvider


def test_data_provider_loads_concepts():
    """Test data provider loads concepts from data directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create mock concept
        concept_dir = data_dir / 'gradient_descent'
        concept_dir.mkdir()
        (concept_dir / 'concept.json').write_text('''
        {
            "name": "Gradient Descent",
            "slug": "gradient_descent",
            "topic": "optimization"
        }
        ''')

        provider = VisualizationDataProvider(data_dir)
        concepts = provider.get_concepts()

        assert len(concepts) == 1
        assert concepts[0]['slug'] == 'gradient_descent'


def test_data_provider_returns_concept_detail():
    """Test data provider returns concept detail with visualizations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        concept_dir = data_dir / 'backprop'
        concept_dir.mkdir()
        (concept_dir / 'concept.json').write_text('''
        {
            "name": "Backpropagation",
            "slug": "backprop",
            "topic": "neural_networks"
        }
        ''')
        (concept_dir / 'visualize.py').write_text('''
def main_visualization():
    """Main viz for backprop"""
    pass
        ''')

        provider = VisualizationDataProvider(data_dir)
        detail = provider.get_concept_detail('backprop')

        assert detail['name'] == 'Backpropagation'
        assert 'visualizations' in detail


def test_data_provider_generates_graph_data():
    """Test data provider generates computational graph data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        provider = VisualizationDataProvider(data_dir)
        graph_data = provider.get_graph_data('backprop')

        assert 'nodes' in graph_data
        assert 'edges' in graph_data
