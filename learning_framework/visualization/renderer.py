"""Visualization rendering and execution"""

import importlib.util
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class VisualizationRenderer:
    """Renders matplotlib visualizations from concept visualize.py files"""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize renderer

        Args:
            base_path: Base data directory (default: data/)
        """
        if base_path is None:
            base_path = Path.cwd() / 'data'

        self.base_path = Path(base_path)

    def get_available_visualizations(self, concept_slug: str) -> List[Dict[str, str]]:
        """Discover visualization functions in visualize.py

        Args:
            concept_slug: Concept identifier

        Returns:
            List of {name, description} dicts
        """
        viz_path = self.base_path / concept_slug / 'visualize.py'

        if not viz_path.exists():
            return []

        # Import module
        module = self._import_viz_module(concept_slug)

        # Find all functions
        functions = []
        for name, obj in inspect.getmembers(module):
            if name.startswith('_'):
                continue
            if inspect.isfunction(obj):
                doc = obj.__doc__ or "No description"
                description = doc.strip().split('\n')[0]  # First line only
                functions.append({
                    'name': name,
                    'description': description
                })

        return functions

    def execute_visualization(
        self,
        concept_slug: str,
        function_name: str = 'main_visualization'
    ):
        """Execute visualization function and return figure

        Args:
            concept_slug: Concept identifier
            function_name: Visualization function name

        Returns:
            matplotlib Figure object

        Raises:
            FileNotFoundError: If visualize.py doesn't exist
            AttributeError: If function not found
        """
        viz_path = self.base_path / concept_slug / 'visualize.py'

        if not viz_path.exists():
            raise FileNotFoundError(f"No visualization for {concept_slug}")

        # Import and execute
        module = self._import_viz_module(concept_slug)

        if not hasattr(module, function_name):
            raise AttributeError(
                f"Function '{function_name}' not found in {concept_slug}/visualize.py"
            )

        viz_function = getattr(module, function_name)
        fig = viz_function()

        return fig

    def _import_viz_module(self, concept_slug: str):
        """Dynamically import visualize.py module

        Args:
            concept_slug: Concept identifier

        Returns:
            Imported module
        """
        viz_path = self.base_path / concept_slug / 'visualize.py'

        spec = importlib.util.spec_from_file_location(
            f"viz_{concept_slug}",
            viz_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module
