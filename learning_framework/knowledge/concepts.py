"""Concept loading and registry management"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_concept(concept_slug: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load concept from concept.json file

    Args:
        concept_slug: Concept identifier (e.g., 'backpropagation')
        base_path: Base data directory (default: data/)

    Returns:
        Concept dictionary

    Raises:
        FileNotFoundError: If concept.json doesn't exist
    """
    if base_path is None:
        base_path = Path.cwd() / 'data'

    concept_path = base_path / concept_slug / 'concept.json'

    if not concept_path.exists():
        raise FileNotFoundError(f"Concept not found: {concept_slug}")

    with open(concept_path, 'r', encoding='utf-8') as f:
        concept = json.load(f)

    return concept


class ConceptRegistry:
    """Registry of all concepts in the system

    Maintains master list and provides querying by topic, difficulty, etc.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize concept registry

        Args:
            base_path: Base data directory (default: data/)
        """
        if base_path is None:
            base_path = Path.cwd() / 'data'

        self.base_path = Path(base_path)
        self.registry_path = self.base_path / 'concepts.json'
        self._concepts = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load concepts registry from concepts.json"""
        if not self.registry_path.exists():
            return {'version': '0.2.0', 'concepts': {}, 'topics': {}}

        with open(self.registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save(self):
        """Save registry to disk"""
        self.base_path.mkdir(parents=True, exist_ok=True)

        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self._concepts, f, indent=2, ensure_ascii=False)

    def register(
        self,
        concept_slug: str,
        topic: str,
        difficulty: str,
        status: str = 'skeleton'
    ):
        """Register a concept in the registry

        Args:
            concept_slug: Concept identifier
            topic: Topic category
            difficulty: Difficulty level
            status: 'complete' or 'skeleton'
        """
        if 'concepts' not in self._concepts:
            self._concepts['concepts'] = {}

        self._concepts['concepts'][concept_slug] = {
            'status': status,
            'topic': topic,
            'difficulty': difficulty
        }

        # Update topic index
        if 'topics' not in self._concepts:
            self._concepts['topics'] = {}

        if topic not in self._concepts['topics']:
            self._concepts['topics'][topic] = []

        if concept_slug not in self._concepts['topics'][topic]:
            self._concepts['topics'][topic].append(concept_slug)

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered concepts

        Returns:
            Dictionary of concept_slug -> metadata
        """
        return self._concepts.get('concepts', {})

    def get_by_topic(self, topic: str) -> List[str]:
        """Get concept slugs for a specific topic

        Args:
            topic: Topic name

        Returns:
            List of concept slugs in that topic
        """
        return self._concepts.get('topics', {}).get(topic, [])

    def get_topics(self) -> List[str]:
        """Get list of all topics

        Returns:
            List of topic names
        """
        return list(self._concepts.get('topics', {}).keys())
