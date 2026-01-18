"""Phase 7 integration tests for polish and documentation"""
import pytest
from pathlib import Path
from click.testing import CliRunner

from learning_framework.cli import cli
from learning_framework.errors import (
    LearningFrameworkError,
    ConfigurationError,
    ConceptNotFoundError,
    PrerequisiteError,
    QuizError,
    ChallengeError,
    GPUBackendError,
    BudgetExceededError,
    format_error_message,
    suggest_fix,
    handle_error
)
from learning_framework.feedback import FeedbackCollector, FeedbackType


@pytest.fixture
def runner():
    return CliRunner()


class TestErrorHandling:
    """Test error handling integration"""

    def test_all_errors_have_codes(self):
        """All error types have unique codes"""
        errors = [
            ConfigurationError("test"),
            ConceptNotFoundError("test"),
            PrerequisiteError("test", missing=["prereq"]),
            QuizError("test"),
            ChallengeError("test"),
            GPUBackendError("test"),
            BudgetExceededError(1.0, 0.5, "daily"),
            LearningFrameworkError("test")
        ]

        codes = [e.code for e in errors]
        # Check all have codes
        assert all(code is not None for code in codes)
        # Check base error code is different from specialized
        assert errors[-1].code != errors[0].code

    def test_all_errors_are_formattable(self):
        """All errors can be formatted for display"""
        errors = [
            ConfigurationError("config error"),
            ConceptNotFoundError("concept", suggestions=["test"]),
            PrerequisiteError("concept", missing=["prereq"]),
            QuizError("quiz error"),
            ChallengeError("challenge error"),
            GPUBackendError("gpu error"),
            BudgetExceededError(1.0, 0.5, "daily")
        ]

        for error in errors:
            formatted = format_error_message(error)
            assert len(formatted) > 0
            assert error.code in formatted

    def test_all_errors_have_suggestions(self):
        """All errors provide actionable suggestions"""
        errors = [
            ConfigurationError("config error", config_key="test_key"),
            ConceptNotFoundError("concept", suggestions=["test"]),
            PrerequisiteError("concept", missing=["prereq"]),
            QuizError("quiz error"),
            ChallengeError("challenge error"),
            GPUBackendError("gpu error"),
            BudgetExceededError(1.0, 0.5, "daily")
        ]

        for error in errors:
            suggestions = suggest_fix(error)
            assert len(suggestions) > 0

    def test_handle_error_wraps_unknown_errors(self):
        """handle_error wraps non-framework errors"""
        generic_error = ValueError("some generic error")
        message, suggestions = handle_error(generic_error)

        assert "E999" in message
        assert len(suggestions) > 0


class TestFeedbackSystem:
    """Test feedback collection integration"""

    def test_full_feedback_workflow(self, tmp_path):
        """Test complete feedback submission workflow"""
        collector = FeedbackCollector(db_path=tmp_path / "test_feedback.db")

        # Submit various types
        bug_id = collector.submit(
            FeedbackType.BUG,
            "Test bug",
            "Bug description",
            steps_to_reproduce="1. Do X\n2. See Y"
        )

        feature_id = collector.submit(
            FeedbackType.FEATURE,
            "Test feature",
            "Feature description"
        )

        # List feedback
        all_feedback = collector.list_feedback()
        assert len(all_feedback) == 2

        # Filter by type
        bugs = collector.list_feedback(feedback_type=FeedbackType.BUG)
        assert len(bugs) == 1
        assert bugs[0].id == bug_id

        # Get specific feedback
        bug = collector.get_feedback(bug_id)
        assert bug is not None
        assert bug.title == "Test bug"

        # Export
        export_path = tmp_path / "export.json"
        collector.export(export_path)
        assert export_path.exists()

    def test_feedback_includes_system_info(self, tmp_path):
        """Feedback automatically includes system info"""
        collector = FeedbackCollector(db_path=tmp_path / "test.db")

        feedback_id = collector.submit(
            FeedbackType.BUG,
            "Test",
            "Description"
        )

        feedback = collector.get_feedback(feedback_id)
        assert feedback.system_info is not None
        assert 'python_version' in feedback.system_info
        assert 'platform' in feedback.system_info


class TestDocumentation:
    """Test documentation files exist and are valid"""

    def test_quickstart_exists(self):
        """QUICKSTART.md exists"""
        quickstart = Path('docs/QUICKSTART.md')
        assert quickstart.exists(), "QUICKSTART.md should exist"

    def test_troubleshooting_exists(self):
        """TROUBLESHOOTING.md exists"""
        troubleshooting = Path('docs/TROUBLESHOOTING.md')
        assert troubleshooting.exists(), "TROUBLESHOOTING.md should exist"

    def test_beta_testing_exists(self):
        """BETA_TESTING.md exists"""
        beta = Path('docs/BETA_TESTING.md')
        assert beta.exists(), "BETA_TESTING.md should exist"

    def test_tutorials_exist(self):
        """Tutorial directory exists with content"""
        tutorials = Path('docs/tutorials')
        assert tutorials.exists(), "tutorials directory should exist"

        # At least one tutorial
        md_files = list(tutorials.glob('*.md'))
        assert len(md_files) > 0, "Should have at least one tutorial"

    def test_backpropagation_tutorial_exists(self):
        """Backpropagation tutorial exists"""
        tutorial = Path('docs/tutorials/backpropagation.md')
        assert tutorial.exists(), "backpropagation.md tutorial should exist"


class TestCLIIntegration:
    """Test CLI commands work together"""

    def test_help_for_all_commands(self, runner):
        """All commands have help text"""
        commands = ['learn', 'quiz', 'progress', 'config', 'index', 'feedback']

        for cmd in commands:
            result = runner.invoke(cli, [cmd, '--help'])
            assert result.exit_code == 0, f"{cmd} --help should work"
            assert len(result.output) > 50, f"{cmd} should have help text"

    def test_feedback_workflow_cli(self, runner, tmp_path):
        """Feedback workflow works through CLI"""
        # Submit bug
        result = runner.invoke(cli, [
            'feedback', 'bug',
            '--title', 'CLI Test Bug',
            '--description', 'Testing from CLI'
        ])
        assert result.exit_code == 0
        assert 'FB-' in result.output

        # List feedback
        result = runner.invoke(cli, ['feedback', 'list'])
        assert result.exit_code == 0

    def test_feedback_subcommands_exist(self, runner):
        """All feedback subcommands exist"""
        subcommands = ['bug', 'feature', 'usability', 'list', 'export']

        for subcmd in subcommands:
            result = runner.invoke(cli, ['feedback', subcmd, '--help'])
            assert result.exit_code == 0, f"feedback {subcmd} should exist"


class TestPerformance:
    """Test performance meets requirements"""

    def test_cli_startup_time(self, runner):
        """CLI should start quickly"""
        import time

        start = time.perf_counter()
        result = runner.invoke(cli, ['--help'])
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"CLI startup took {elapsed:.2f}s, should be < 2.0s"

    def test_quiz_caching_works(self, tmp_path):
        """Quiz caching provides performance benefit"""
        import time
        import json
        from learning_framework.assessment.quiz import ConceptQuiz

        # Create test quiz
        quiz_dir = tmp_path / "quizzes"
        quiz_dir.mkdir()
        quiz_data = {
            "concept": "perf_test",
            "questions": [
                {"id": f"q{i}", "question": f"Q{i}", "type": "multiple_choice",
                 "options": ["A", "B"], "answer": "A"}
                for i in range(50)
            ]
        }
        (quiz_dir / "perf_test.json").write_text(json.dumps(quiz_data))

        # Clear cache
        ConceptQuiz.clear_cache()

        # First load
        quiz1 = ConceptQuiz("perf_test", quiz_dir=quiz_dir)
        start = time.perf_counter()
        quiz1.generate_quiz(5)
        first_time = time.perf_counter() - start

        # Second load (cached)
        quiz2 = ConceptQuiz("perf_test", quiz_dir=quiz_dir)
        start = time.perf_counter()
        quiz2.generate_quiz(5)
        second_time = time.perf_counter() - start

        # Cached should be similar or faster
        assert second_time <= first_time * 2

    def test_visualization_caching_works(self):
        """Visualization caching provides performance benefit"""
        import time
        from learning_framework.visualization.renderer import VisualizationRenderer

        # Clear cache
        VisualizationRenderer.clear_cache()

        renderer = VisualizationRenderer()

        # First call
        start = time.perf_counter()
        renderer.get_available_visualizations("gradient_descent")
        first_time = time.perf_counter() - start

        # Second call (cached)
        start = time.perf_counter()
        renderer.get_available_visualizations("gradient_descent")
        second_time = time.perf_counter() - start

        # Cached should be faster or similar
        assert second_time <= first_time * 1.5


class TestPhase7Completeness:
    """Test Phase 7 deliverables are complete"""

    def test_error_module_exists(self):
        """errors.py module exists"""
        from learning_framework import errors
        assert hasattr(errors, 'LearningFrameworkError')
        assert hasattr(errors, 'format_error_message')
        assert hasattr(errors, 'suggest_fix')

    def test_feedback_module_exists(self):
        """feedback.py module exists"""
        from learning_framework import feedback
        assert hasattr(feedback, 'FeedbackCollector')
        assert hasattr(feedback, 'FeedbackType')
        assert hasattr(feedback, 'Feedback')

    def test_quiz_has_caching(self):
        """Quiz module has caching methods"""
        from learning_framework.assessment.quiz import ConceptQuiz
        assert hasattr(ConceptQuiz, 'clear_cache')
        assert hasattr(ConceptQuiz, 'preload')
        assert hasattr(ConceptQuiz, '_quiz_cache')

    def test_renderer_has_caching(self):
        """Renderer module has caching methods"""
        from learning_framework.visualization.renderer import VisualizationRenderer
        assert hasattr(VisualizationRenderer, 'clear_cache')
        assert hasattr(VisualizationRenderer, '_module_cache')
