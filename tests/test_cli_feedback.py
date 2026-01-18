"""Tests for feedback CLI commands"""
import pytest
from click.testing import CliRunner
from learning_framework.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_feedback_help(runner):
    """feedback command shows help"""
    result = runner.invoke(cli, ['feedback', '--help'])
    assert result.exit_code == 0
    assert 'feedback' in result.output.lower()


def test_feedback_bug_command(runner):
    """Can submit bug via CLI"""
    result = runner.invoke(cli, [
        'feedback', 'bug',
        '--title', 'Test bug',
        '--description', 'Test description'
    ])
    assert result.exit_code == 0
    assert 'FB-' in result.output


def test_feedback_feature_command(runner):
    """Can submit feature request via CLI"""
    result = runner.invoke(cli, [
        'feedback', 'feature',
        '--title', 'Test feature',
        '--description', 'Test description'
    ])
    assert result.exit_code == 0
    assert 'FB-' in result.output


def test_feedback_usability_command(runner):
    """Can submit usability feedback via CLI"""
    result = runner.invoke(cli, [
        'feedback', 'usability',
        '--title', 'Test usability',
        '--description', 'Test description'
    ])
    assert result.exit_code == 0
    assert 'FB-' in result.output


def test_feedback_list_command(runner):
    """Can list feedback"""
    # Submit one first
    runner.invoke(cli, [
        'feedback', 'bug',
        '--title', 'List test bug',
        '--description', 'For testing list'
    ])

    result = runner.invoke(cli, ['feedback', 'list'])
    assert result.exit_code == 0


def test_feedback_export_command(runner, tmp_path):
    """Can export feedback to file"""
    # Submit one first
    runner.invoke(cli, [
        'feedback', 'bug',
        '--title', 'Export test',
        '--description', 'For testing export'
    ])

    export_file = tmp_path / "export.json"
    result = runner.invoke(cli, ['feedback', 'export', str(export_file)])
    assert result.exit_code == 0
    assert export_file.exists()
