"""Tests for challenge management system"""

import pytest
import tempfile
from pathlib import Path
from learning_framework.assessment.challenge import ChallengeManager


def test_challenge_manager_loads_challenge():
    """Test challenge manager loads challenge file and extracts description"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create challenge directory
        challenge_dir = Path(tmpdir) / 'challenges'
        challenge_dir.mkdir(parents=True)

        # Create a fill-type challenge
        challenge_content = '''"""
Implement a simple linear layer.

Complete the forward pass by computing: output = input @ weights + bias
"""

import numpy as np

class LinearLayer:
    def __init__(self, input_dim: int, output_dim: int):
        """Initialize layer with random weights and zero bias"""
        # TODO: Initialize self.weights and self.bias
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: output = x @ weights + bias"""
        # TODO: Implement forward pass
        pass
'''
        challenge_file = challenge_dir / 'linear_layer_fill.py'
        challenge_file.write_text(challenge_content)

        # Load challenge
        manager = ChallengeManager(challenges_path=challenge_dir)
        challenge = manager.load_challenge('linear_layer_fill.py')

        assert challenge is not None
        assert challenge['name'] == 'linear_layer'
        assert challenge['type'] == 'fill'
        assert 'Implement a simple linear layer' in challenge['description']
        assert challenge['file_path'] == challenge_file


def test_challenge_manager_detects_challenge_type():
    """Test challenge manager correctly detects challenge type from filename"""
    with tempfile.TemporaryDirectory() as tmpdir:
        challenge_dir = Path(tmpdir) / 'challenges'
        challenge_dir.mkdir(parents=True)

        # Create different types of challenges
        for challenge_type in ['fill', 'scratch', 'debug']:
            challenge_file = challenge_dir / f'test_{challenge_type}.py'
            challenge_file.write_text('"""Test challenge"""\\npass')

        manager = ChallengeManager(challenges_path=challenge_dir)

        # Test each type
        fill_challenge = manager.load_challenge('test_fill.py')
        assert fill_challenge['type'] == 'fill'

        scratch_challenge = manager.load_challenge('test_scratch.py')
        assert scratch_challenge['type'] == 'scratch'

        debug_challenge = manager.load_challenge('test_debug.py')
        assert debug_challenge['type'] == 'debug'


def test_challenge_manager_lists_challenges():
    """Test challenge manager lists all available challenges"""
    with tempfile.TemporaryDirectory() as tmpdir:
        challenge_dir = Path(tmpdir) / 'challenges'
        challenge_dir.mkdir(parents=True)

        # Create multiple challenges
        challenges = [
            ('neural_net_fill.py', '"""Build neural network"""'),
            ('gradient_descent_scratch.py', '"""Implement from scratch"""'),
            ('backprop_debug.py', '"""Fix backprop bug"""')
        ]

        for filename, content in challenges:
            (challenge_dir / filename).write_text(content)

        manager = ChallengeManager(challenges_path=challenge_dir)
        all_challenges = manager.list_challenges()

        assert len(all_challenges) == 3
        assert any(c['name'] == 'neural_net' for c in all_challenges)
        assert any(c['name'] == 'gradient_descent' for c in all_challenges)
        assert any(c['name'] == 'backprop' for c in all_challenges)


def test_challenge_manager_filters_by_type():
    """Test challenge manager filters challenges by type"""
    with tempfile.TemporaryDirectory() as tmpdir:
        challenge_dir = Path(tmpdir) / 'challenges'
        challenge_dir.mkdir(parents=True)

        # Create challenges of different types
        challenges = [
            ('layer1_fill.py', '"""Fill implementation"""'),
            ('layer2_fill.py', '"""Fill implementation"""'),
            ('network_scratch.py', '"""Build from scratch"""'),
            ('optimizer_debug.py', '"""Fix bug"""')
        ]

        for filename, content in challenges:
            (challenge_dir / filename).write_text(content)

        manager = ChallengeManager(challenges_path=challenge_dir)

        # Test filtering
        fill_challenges = manager.get_challenges_by_type('fill')
        assert len(fill_challenges) == 2
        assert all(c['type'] == 'fill' for c in fill_challenges)

        scratch_challenges = manager.get_challenges_by_type('scratch')
        assert len(scratch_challenges) == 1
        assert scratch_challenges[0]['name'] == 'network'

        debug_challenges = manager.get_challenges_by_type('debug')
        assert len(debug_challenges) == 1
        assert debug_challenges[0]['name'] == 'optimizer'


def test_challenge_manager_copies_to_workspace():
    """Test challenge manager copies challenge file to workspace"""
    with tempfile.TemporaryDirectory() as tmpdir:
        challenge_dir = Path(tmpdir) / 'challenges'
        challenge_dir.mkdir(parents=True)
        workspace_dir = Path(tmpdir) / 'workspace'
        workspace_dir.mkdir(parents=True)

        # Create challenge
        challenge_content = '"""Test challenge"""\\nimport numpy as np\\npass'
        challenge_file = challenge_dir / 'test_fill.py'
        challenge_file.write_text(challenge_content)

        manager = ChallengeManager(challenges_path=challenge_dir)

        # Copy to workspace
        workspace_file = manager.copy_to_workspace('test_fill.py', workspace_dir)

        assert workspace_file.exists()
        assert workspace_file.parent == workspace_dir
        assert workspace_file.name == 'test_fill.py'
        assert workspace_file.read_text() == challenge_content


def test_challenge_manager_handles_missing_description():
    """Test challenge manager handles files without docstring"""
    with tempfile.TemporaryDirectory() as tmpdir:
        challenge_dir = Path(tmpdir) / 'challenges'
        challenge_dir.mkdir(parents=True)

        # Create challenge without docstring
        challenge_file = challenge_dir / 'no_doc_fill.py'
        challenge_file.write_text('import numpy as np\\npass')

        manager = ChallengeManager(challenges_path=challenge_dir)
        challenge = manager.load_challenge('no_doc_fill.py')

        assert challenge is not None
        assert challenge['description'] == ''


def test_challenge_manager_handles_nonexistent_file():
    """Test challenge manager handles loading nonexistent file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        challenge_dir = Path(tmpdir) / 'challenges'
        challenge_dir.mkdir(parents=True)

        manager = ChallengeManager(challenges_path=challenge_dir)
        challenge = manager.load_challenge('nonexistent.py')

        assert challenge is None


def test_challenge_manager_default_path():
    """Test challenge manager uses default path if not specified"""
    # This should create manager with default path (data/challenges/)
    manager = ChallengeManager()

    # Should not raise error, even if directory doesn't exist
    challenges = manager.list_challenges()
    assert isinstance(challenges, list)
