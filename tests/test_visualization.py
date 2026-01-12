import pytest
import tempfile
from pathlib import Path
from learning_framework.visualization.renderer import VisualizationRenderer


def test_renderer_discovers_viz_functions():
    """Test renderer discovers functions in visualize.py"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test visualization
        viz_dir = Path(tmpdir) / 'data' / 'test_concept'
        viz_dir.mkdir(parents=True)

        viz_code = '''
import matplotlib.pyplot as plt

def main_visualization():
    """Primary test visualization"""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    return fig

def secondary_viz():
    """Secondary visualization"""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [2, 1])
    return fig
'''
        (viz_dir / 'visualize.py').write_text(viz_code)

        # Discover functions
        renderer = VisualizationRenderer(base_path=Path(tmpdir) / 'data')
        functions = renderer.get_available_visualizations('test_concept')

        assert len(functions) >= 2
        names = [f['name'] for f in functions]
        assert 'main_visualization' in names
        assert 'secondary_viz' in names


def test_renderer_executes_visualization():
    """Test renderer executes viz function and returns figure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        viz_dir = Path(tmpdir) / 'data' / 'test_concept'
        viz_dir.mkdir(parents=True)

        viz_code = '''
import matplotlib.pyplot as plt

def test_viz():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    return fig
'''
        (viz_dir / 'visualize.py').write_text(viz_code)

        # Render
        renderer = VisualizationRenderer(base_path=Path(tmpdir) / 'data')
        fig = renderer.execute_visualization('test_concept', 'test_viz')

        assert fig is not None
        assert hasattr(fig, 'savefig')  # Is a matplotlib figure
