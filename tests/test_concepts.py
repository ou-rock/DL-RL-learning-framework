import pytest
import tempfile
import json
from pathlib import Path
from learning_framework.knowledge.concepts import ConceptRegistry, load_concept


def test_load_concept_from_file():
    """Test loading concept from concept.json"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test concept
        concept_dir = Path(tmpdir) / 'data' / 'test_concept'
        concept_dir.mkdir(parents=True)

        concept_data = {
            'name': 'Test Concept',
            'slug': 'test_concept',
            'topic': 'testing',
            'difficulty': 'easy',
            'status': 'complete',
            'prerequisites': [],
            'description': 'A test concept',
            'materials': {},
            'tags': ['test']
        }

        (concept_dir / 'concept.json').write_text(json.dumps(concept_data, indent=2))

        # Load concept
        concept = load_concept('test_concept', base_path=Path(tmpdir) / 'data')

        assert concept['name'] == 'Test Concept'
        assert concept['slug'] == 'test_concept'
        assert concept['difficulty'] == 'easy'


def test_concept_registry_tracks_all_concepts():
    """Test registry maintains list of all concepts"""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ConceptRegistry(base_path=Path(tmpdir) / 'data')

        # Add concepts
        registry.register('gradient_descent', topic='optimization', difficulty='beginner')
        registry.register('backprop', topic='neural_networks', difficulty='intermediate')

        # Get all concepts
        all_concepts = registry.get_all()

        assert len(all_concepts) == 2
        assert 'gradient_descent' in all_concepts
        assert 'backprop' in all_concepts


def test_concept_registry_groups_by_topic():
    """Test getting concepts grouped by topic"""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ConceptRegistry(base_path=Path(tmpdir) / 'data')

        registry.register('gradient_descent', topic='optimization', difficulty='beginner')
        registry.register('backprop', topic='neural_networks', difficulty='intermediate')
        registry.register('sgd', topic='optimization', difficulty='beginner')

        # Get by topic
        optimization = registry.get_by_topic('optimization')

        assert len(optimization) == 2
        assert 'gradient_descent' in optimization
        assert 'sgd' in optimization
