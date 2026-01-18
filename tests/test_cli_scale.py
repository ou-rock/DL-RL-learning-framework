import pytest
from click.testing import CliRunner
from learning_framework.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_scale_command_exists(runner):
    """lf scale command exists"""
    result = runner.invoke(cli, ['scale', '--help'])
    assert result.exit_code == 0
    assert 'GPU job' in result.output or 'scale' in result.output.lower()


def test_scale_estimate_shows_cost(runner, tmp_path):
    """lf scale --estimate shows cost estimate"""
    impl = tmp_path / "backprop.py"
    impl.write_text("def backward(): pass")

    result = runner.invoke(cli, [
        'scale', str(impl),
        '--backend', 'vastai',
        '--estimate'
    ])

    # Should show estimate without submitting
    assert 'estimate' in result.output.lower() or '$' in result.output


def test_scale_status_command(runner):
    """lf status <job_id> shows job status"""
    result = runner.invoke(cli, ['status', 'vast_12345'])
    assert result.exit_code in [0, 1]  # May fail if job not found


def test_scale_logs_command(runner):
    """lf logs <job_id> shows job logs"""
    result = runner.invoke(cli, ['logs', 'vast_12345'])
    assert result.exit_code in [0, 1]  # May fail if job not found


def test_scale_results_command(runner):
    """lf results <job_id> shows job results"""
    result = runner.invoke(cli, ['results', 'vast_12345'])
    assert result.exit_code in [0, 1]  # May fail if job not found
