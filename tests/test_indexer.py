import pytest
import tempfile
from pathlib import Path
from learning_framework.knowledge.indexer import MaterialIndexer


def test_indexer_scans_directories():
    """Test indexer scans directories and finds files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test directory structure
        base = Path(tmpdir)
        (base / 'ch01').mkdir()
        (base / 'ch01' / 'README.md').write_text('# Chapter 1: Introduction')
        (base / 'ch01' / 'example.py').write_text('# Example code')
        (base / 'ch02').mkdir()
        (base / 'ch02' / 'README.md').write_text('# Chapter 2: Basics')

        indexer = MaterialIndexer()
        results = indexer.scan_directory(base)

        assert len(results['chapters']) == 2
        assert 'ch01' in results['chapters']
        assert 'ch02' in results['chapters']
        assert len(results['files']) >= 3


def test_indexer_detects_chapter_pattern():
    """Test indexer detects chapter naming patterns"""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / 'ch01').mkdir()
        (base / 'ch02').mkdir()
        (base / 'chapter03').mkdir()
        (base / 'other').mkdir()

        indexer = MaterialIndexer()
        results = indexer.scan_directory(base)

        chapters = results['chapters']
        assert 'ch01' in chapters
        assert 'ch02' in chapters
        assert 'chapter03' in chapters
        assert 'other' not in chapters


def test_indexer_finds_python_files():
    """Test indexer finds Python files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / 'script.py').write_text('print("hello")')
        (base / 'train.py').write_text('# training script')
        (base / 'README.md').write_text('# Readme')

        indexer = MaterialIndexer()
        results = indexer.scan_directory(base)

        py_files = [f for f in results['files'] if f.endswith('.py')]
        assert len(py_files) == 2


def test_indexer_extracts_ml_keywords():
    """Test indexer extracts ML keywords from Python files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # File with ML keywords
        (base / 'backprop.py').write_text('''
def backpropagation(network, loss):
    """Compute gradients via backprop"""
    gradients = compute_gradients(loss)
    return gradients
        ''')

        indexer = MaterialIndexer()
        results = indexer.scan_directory(base)

        keywords = results.get('keywords', [])
        assert 'backpropagation' in keywords or 'backprop' in keywords
        assert 'gradients' in keywords or 'gradient' in keywords
