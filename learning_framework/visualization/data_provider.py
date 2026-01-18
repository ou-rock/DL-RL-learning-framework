"""Data provider for visualization server API"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from learning_framework.visualization.graph_data import ComputationalGraphBuilder
from learning_framework.visualization.renderer import VisualizationRenderer


class VisualizationDataProvider:
    """Provides data for visualization API endpoints"""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize data provider

        Args:
            data_dir: Directory containing concept data
        """
        if data_dir is None:
            data_dir = Path.cwd() / 'data'

        self.data_dir = Path(data_dir)
        self.graph_builder = ComputationalGraphBuilder()
        self.renderer = VisualizationRenderer(self.data_dir)

    def get_concepts(self) -> List[Dict[str, Any]]:
        """Get list of available concepts

        Returns:
            List of concept summaries
        """
        concepts = []

        if not self.data_dir.exists():
            return concepts

        for concept_dir in self.data_dir.iterdir():
            if not concept_dir.is_dir():
                continue

            concept_file = concept_dir / 'concept.json'
            if concept_file.exists():
                try:
                    with open(concept_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    concepts.append({
                        'slug': data.get('slug', concept_dir.name),
                        'name': data.get('name', concept_dir.name),
                        'topic': data.get('topic', 'unknown'),
                        'status': data.get('status', 'skeleton')
                    })
                except Exception:
                    continue

        return concepts

    def get_concept_detail(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get detailed concept information

        Args:
            slug: Concept identifier

        Returns:
            Concept detail with visualizations
        """
        concept_dir = self.data_dir / slug
        concept_file = concept_dir / 'concept.json'

        if not concept_file.exists():
            return None

        with open(concept_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get available visualizations
        viz_file = concept_dir / 'visualize.py'
        visualizations = []

        if viz_file.exists():
            try:
                visualizations = self.renderer.get_available_visualizations(slug)
            except Exception:
                pass

        return {
            **data,
            'visualizations': visualizations,
            'has_quiz': (concept_dir / 'quiz_mc.json').exists(),
            'has_challenge': (concept_dir / 'challenge.py').exists()
        }

    def get_graph_data(self, concept: Optional[str] = None) -> Dict[str, Any]:
        """Get computational graph data for visualization

        Args:
            concept: Optional concept to get specific graph

        Returns:
            Graph data with nodes and edges
        """
        # Generate example graph based on concept
        graph_configs = {
            'backprop': {
                'input_dim': 10,
                'hidden_dims': [32, 16],
                'output_dim': 5,
                'include_gradients': True
            },
            'gradient_descent': {
                'input_dim': 5,
                'hidden_dims': [10],
                'output_dim': 2,
                'include_gradients': False
            },
            'default': {
                'input_dim': 8,
                'hidden_dims': [16, 8],
                'output_dim': 4,
                'include_gradients': False
            }
        }

        config = graph_configs.get(concept, graph_configs['default'])

        return self.graph_builder.build_feedforward_graph(**config)

    def get_visualization_data(
        self,
        concept: str,
        viz_name: str = 'main_visualization'
    ) -> Dict[str, Any]:
        """Get visualization data as JSON

        Args:
            concept: Concept identifier
            viz_name: Visualization function name

        Returns:
            Visualization data for rendering
        """
        # For now, return graph-type visualization
        if viz_name == 'computational_graph' or concept in ['backprop', 'backpropagation']:
            return {
                'type': 'graph',
                **self.get_graph_data(concept)
            }

        # Default: return parameter exploration data
        return {
            'type': 'params',
            'parameters': self._get_default_params(concept),
            'concept': concept,
            'viz_name': viz_name
        }

    def _get_default_params(self, concept: str) -> List[Dict[str, Any]]:
        """Get default parameters for a concept

        Args:
            concept: Concept identifier

        Returns:
            List of parameter definitions
        """
        params_by_concept = {
            'gradient_descent': [
                {'name': 'learning_rate', 'label': 'Learning Rate', 'min': 0.001, 'max': 1.0, 'value': 0.01, 'scale': 'log'},
                {'name': 'momentum', 'label': 'Momentum', 'min': 0.0, 'max': 0.99, 'value': 0.9, 'scale': 'linear'}
            ],
            'sgd': [
                {'name': 'learning_rate', 'label': 'Learning Rate', 'min': 0.0001, 'max': 0.1, 'value': 0.01, 'scale': 'log'},
                {'name': 'batch_size', 'label': 'Batch Size', 'min': 8, 'max': 256, 'value': 32, 'scale': 'linear'}
            ],
            'default': [
                {'name': 'learning_rate', 'label': 'Learning Rate', 'min': 0.001, 'max': 1.0, 'value': 0.01, 'scale': 'log'}
            ]
        }

        return params_by_concept.get(concept, params_by_concept['default'])
