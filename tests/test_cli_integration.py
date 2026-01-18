"""Integration tests for CLI commands - testing actual execution paths

These tests verify that CLI commands work end-to-end, catching API mismatches
between components that unit tests might miss.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from click.testing import CliRunner
from learning_framework.cli import cli


@pytest.fixture
def test_environment(tmp_path):
    """Set up a complete test environment with concept data"""
    # Create data directory structure
    data_dir = tmp_path / 'data'
    data_dir.mkdir()

    # Create concepts.json registry
    concepts_registry = {
        "version": "0.2.0",
        "concepts": {
            "test_concept": {
                "status": "complete",
                "topic": "test_topic",
                "difficulty": "beginner"
            }
        },
        "topics": {
            "test_topic": ["test_concept"]
        }
    }
    with open(data_dir / 'concepts.json', 'w') as f:
        json.dump(concepts_registry, f)

    # Create individual concept directory and concept.json
    concept_dir = data_dir / 'test_concept'
    concept_dir.mkdir()

    concept_data = {
        "name": "Test Concept",
        "slug": "test_concept",
        "topic": "test_topic",
        "difficulty": "beginner",
        "status": "complete",
        "prerequisites": [],
        "description": "A test concept for integration testing.",
        "explanation": "This is the explanation text.",
        "key_points": ["Point 1", "Point 2"],
        "materials": {
            "explanation": "",
            "code_examples": []
        }
    }
    with open(concept_dir / 'concept.json', 'w') as f:
        json.dump(concept_data, f)

    # Create quiz questions
    quiz_data = {
        "questions": [
            {
                "id": "q1",
                "type": "multiple_choice",
                "question": "What is the test answer?",
                "options": ["Wrong", "Correct", "Also wrong", "Still wrong"],
                "answer": "Correct",
                "explanation": "Correct is the right answer."
            }
        ]
    }
    with open(concept_dir / 'quiz.json', 'w') as f:
        json.dump(quiz_data, f)

    # Create user_data directory for progress database
    user_data_dir = tmp_path / 'user_data'
    user_data_dir.mkdir()

    # Store original cwd and change to test directory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    yield tmp_path

    # Restore original cwd
    os.chdir(original_cwd)


class TestLearnCommand:
    """Integration tests for 'lf learn' command"""

    def test_learn_shows_concept_list(self, test_environment):
        """Test that learn command shows available concepts"""
        runner = CliRunner()
        # Input 'q' to quit the selection menu
        result = runner.invoke(cli, ['learn'], input='q\n')

        # Should show the topic and concept
        assert 'test_topic' in result.output
        assert 'Test Concept' in result.output

    def test_learn_with_valid_concept(self, test_environment):
        """Test learn command with a specific valid concept"""
        runner = CliRunner()
        # Input '5' to exit the learning menu (Back to concept selection)
        result = runner.invoke(cli, ['learn', '--concept', 'test_concept'], input='5\n')

        # Should show concept name and description
        assert 'Test Concept' in result.output
        assert 'test concept for integration testing' in result.output.lower()
        # Should show menu options
        assert 'Read explanation' in result.output
        assert 'Take quiz' in result.output

    def test_learn_with_invalid_concept(self, test_environment):
        """Test learn command with non-existent concept"""
        runner = CliRunner()
        result = runner.invoke(cli, ['learn', '--concept', 'nonexistent'])

        # Should show error
        assert result.exit_code != 0 or 'not found' in result.output.lower() or 'error' in result.output.lower()

    def test_learn_read_explanation(self, test_environment):
        """Test reading explanation in learn mode"""
        runner = CliRunner()
        # Select option 1 (Read explanation) then 5 (Exit)
        result = runner.invoke(cli, ['learn', '--concept', 'test_concept'], input='1\n5\n')

        # Should show explanation content
        assert 'explanation text' in result.output.lower()


class TestQuizCommand:
    """Integration tests for 'lf quiz' command"""

    def test_quiz_daily_review_empty(self, test_environment):
        """Test quiz command with no due items"""
        runner = CliRunner()
        result = runner.invoke(cli, ['quiz'])

        # Should show no items due
        assert 'No items due for review' in result.output
        assert result.exit_code == 0

    def test_quiz_with_specific_concept(self, test_environment):
        """Test quiz command with specific concept executes without API errors"""
        runner = CliRunner()
        # Answer the first question (option 2 is correct)
        result = runner.invoke(cli, ['quiz', '--concept', 'test_concept'], input='2\n')

        # Should execute without TypeError or API mismatch errors
        # May show "No quiz questions" if quiz.json format doesn't match exactly
        assert 'TypeError' not in result.output
        assert 'missing' not in result.output.lower() or 'No quiz' in result.output
        assert result.exit_code == 0


class TestProgressCommand:
    """Integration tests for 'lf progress' command"""

    def test_progress_command_runs(self, test_environment):
        """Test that progress command executes without errors"""
        runner = CliRunner()
        result = runner.invoke(cli, ['progress'])

        # Currently shows "Not yet implemented" but should not crash
        assert result.exit_code == 0


class TestConfigCommand:
    """Integration tests for 'lf config' command"""

    def test_config_displays_settings(self, test_environment):
        """Test that config command shows current settings"""
        runner = CliRunner()
        result = runner.invoke(cli, ['config'])

        assert result.exit_code == 0
        assert 'Configuration' in result.output


class TestIndexCommand:
    """Integration tests for 'lf index' command"""

    def test_index_command_runs(self, test_environment):
        """Test that index command executes"""
        runner = CliRunner()
        result = runner.invoke(cli, ['index'])

        # May show "no materials directories" but should not crash
        assert result.exit_code == 0


class TestAnswerGraderIntegration:
    """Tests that verify AnswerGrader API is used correctly"""

    def test_grader_has_check_answer_method(self):
        """Ensure CLI uses the correct grader method name"""
        from learning_framework.assessment import AnswerGrader

        grader = AnswerGrader()
        # CLI should use check_answer, not grade
        assert hasattr(grader, 'check_answer'), "AnswerGrader should have check_answer method"
        assert callable(grader.check_answer), "check_answer should be callable"


class TestComponentIntegration:
    """Tests that verify correct integration between components"""

    def test_knowledge_graph_initialization(self, test_environment):
        """Test that KnowledgeGraph is initialized correctly in CLI context"""
        from learning_framework.knowledge import KnowledgeGraph

        # Should not raise TypeError about wrong argument type
        graph = KnowledgeGraph()
        result = graph.check_prerequisites('test_concept')

        # Should return a dict with expected keys
        assert isinstance(result, dict)
        assert 'ready' in result
        assert 'missing' in result

    def test_spaced_repetition_scheduler_initialization(self, test_environment):
        """Test that SpacedRepetitionScheduler is initialized correctly"""
        from learning_framework.assessment import SpacedRepetitionScheduler

        # Should not raise TypeError about wrong argument type
        scheduler = SpacedRepetitionScheduler()
        due_items = scheduler.get_due_items()

        # Should return a list
        assert isinstance(due_items, list)

    def test_concept_registry_api(self, test_environment):
        """Test ConceptRegistry API matches CLI usage"""
        from learning_framework.knowledge import ConceptRegistry

        registry = ConceptRegistry()

        # get_all should return dict of {slug: metadata}
        all_concepts = registry.get_all()
        assert isinstance(all_concepts, dict)
        assert 'test_concept' in all_concepts

        # get_topics should return list of topic names
        topics = registry.get_topics()
        assert isinstance(topics, list)
        assert 'test_topic' in topics

        # get_by_topic should return list of slugs
        topic_concepts = registry.get_by_topic('test_topic')
        assert isinstance(topic_concepts, list)
        assert 'test_concept' in topic_concepts
