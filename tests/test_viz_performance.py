"""Performance tests for visualization rendering"""
import pytest
import time
from pathlib import Path


def test_visualization_renders_under_2_seconds():
    """Visualizations should render in under 2 seconds"""
    from learning_framework.visualization.renderer import VisualizationRenderer

    renderer = VisualizationRenderer()

    # Get available visualizations
    viz_list = renderer.get_available_visualizations("gradient_descent")

    if not viz_list:
        pytest.skip("No visualizations available for gradient_descent")

    # Measure render time
    start = time.perf_counter()
    fig = renderer.execute_visualization("gradient_descent", viz_list[0]['function'])
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"Visualization took {elapsed:.2f}s, should be < 2.0s"


def test_multiple_visualizations_reuse_setup():
    """Multiple visualizations should share setup work"""
    from learning_framework.visualization.renderer import VisualizationRenderer

    renderer = VisualizationRenderer()

    viz_list = renderer.get_available_visualizations("activation_functions")

    if len(viz_list) < 2:
        pytest.skip("Need at least 2 visualizations for this test")

    # First render
    start = time.perf_counter()
    renderer.execute_visualization("activation_functions", viz_list[0]['function'])
    first_time = time.perf_counter() - start

    # Second render should be similar or faster
    start = time.perf_counter()
    renderer.execute_visualization("activation_functions", viz_list[1]['function'])
    second_time = time.perf_counter() - start

    # Second should not be significantly slower
    assert second_time < first_time * 2, "Second render should not be much slower"


def test_get_visualizations_is_cached():
    """get_available_visualizations should be cached"""
    from learning_framework.visualization.renderer import VisualizationRenderer

    # Clear any existing cache
    VisualizationRenderer.clear_cache()

    renderer = VisualizationRenderer()

    # First call
    start = time.perf_counter()
    viz_list1 = renderer.get_available_visualizations("gradient_descent")
    first_time = time.perf_counter() - start

    # Second call (should be cached)
    start = time.perf_counter()
    viz_list2 = renderer.get_available_visualizations("gradient_descent")
    second_time = time.perf_counter() - start

    # Results should be the same
    assert viz_list1 == viz_list2

    # Second call should be faster or similar (caching)
    assert second_time <= first_time * 1.5


def test_module_loading_is_cached():
    """Module loading should be cached across instances"""
    from learning_framework.visualization.renderer import VisualizationRenderer

    # Clear cache
    VisualizationRenderer.clear_cache()

    # First instance loads module
    renderer1 = VisualizationRenderer()
    viz_list = renderer1.get_available_visualizations("backpropagation")

    if not viz_list:
        pytest.skip("No visualizations available for backpropagation")

    start = time.perf_counter()
    renderer1.execute_visualization("backpropagation", viz_list[0]['function'])
    first_time = time.perf_counter() - start

    # Second instance should reuse cached module
    renderer2 = VisualizationRenderer()
    start = time.perf_counter()
    renderer2.execute_visualization("backpropagation", viz_list[0]['function'])
    second_time = time.perf_counter() - start

    # Module should be cached, so similar or faster
    assert second_time <= first_time * 1.5
