"""Material indexer for discovering learning resources"""

import re
from pathlib import Path
from typing import Dict, List, Any


class MaterialIndexer:
    """Scans directories to discover and index learning materials"""

    # Common ML/DL keywords to detect
    ML_KEYWORDS = [
        'backprop', 'gradient', 'loss', 'optimization', 'optimizer',
        'neural', 'network', 'layer', 'activation', 'sigmoid', 'relu',
        'convolution', 'pooling', 'dropout', 'batch_norm',
        'q_learning', 'policy', 'reward', 'agent', 'environment',
        'reinforcement', 'dqn', 'actor', 'critic'
    ]

    # Patterns for chapter detection
    CHAPTER_PATTERNS = [
        r'^ch\d+$',          # ch01, ch02, etc.
        r'^chapter\d+$',     # chapter01, chapter02
        r'^step\d+$',        # step01, step02 (for dezero)
    ]

    def scan_directory(self, root_path: Path) -> Dict[str, Any]:
        """Scan directory for learning materials

        Args:
            root_path: Root directory to scan

        Returns:
            Dictionary with scan results
        """
        root_path = Path(root_path)

        results = {
            'root': str(root_path),
            'chapters': [],
            'files': [],
            'keywords': set(),
        }

        # Scan directory structure
        for item in root_path.rglob('*'):
            if item.is_dir():
                # Check for chapter pattern
                if self._is_chapter_dir(item.name):
                    rel_path = item.relative_to(root_path)
                    results['chapters'].append(str(rel_path))

            elif item.is_file():
                rel_path = item.relative_to(root_path)
                results['files'].append(str(rel_path))

                # Extract keywords from Python files
                if item.suffix == '.py':
                    keywords = self._extract_keywords(item)
                    results['keywords'].update(keywords)

        # Convert set to list for JSON serialization
        results['keywords'] = list(results['keywords'])

        return results

    def _is_chapter_dir(self, dirname: str) -> bool:
        """Check if directory name matches chapter pattern

        Args:
            dirname: Directory name

        Returns:
            True if matches chapter pattern
        """
        dirname_lower = dirname.lower()
        return any(
            re.match(pattern, dirname_lower)
            for pattern in self.CHAPTER_PATTERNS
        )

    def _extract_keywords(self, file_path: Path) -> List[str]:
        """Extract ML keywords from Python file

        Args:
            file_path: Path to Python file

        Returns:
            List of found keywords
        """
        try:
            content = file_path.read_text(encoding='utf-8').lower()
        except Exception:
            return []

        found_keywords = []
        for keyword in self.ML_KEYWORDS:
            if keyword in content:
                found_keywords.append(keyword)

        return found_keywords
