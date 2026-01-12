import pytest
import tempfile
import json
from pathlib import Path
from learning_framework.assessment.quiz import ConceptQuiz
from learning_framework.assessment.grader import AnswerGrader


def test_quiz_loads_questions():
    """Test quiz loads MC and fill-in-blank questions"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create concept with quiz files
        concept_dir = Path(tmpdir) / 'data' / 'test_concept'
        concept_dir.mkdir(parents=True)

        concept_data = {
            'name': 'Test',
            'slug': 'test_concept',
            'topic': 'testing',
            'difficulty': 'easy',
            'status': 'complete',
            'prerequisites': []
        }
        (concept_dir / 'concept.json').write_text(json.dumps(concept_data))

        quiz_mc = {
            'questions': [
                {
                    'id': 'mc_001',
                    'type': 'multiple_choice',
                    'question': 'Test question?',
                    'options': ['A', 'B', 'C', 'D'],
                    'correct_index': 1,
                    'explanation': 'Test explanation'
                }
            ]
        }
        (concept_dir / 'quiz_mc.json').write_text(json.dumps(quiz_mc))

        quiz_fb = {
            'questions': [
                {
                    'id': 'fb_001',
                    'type': 'fill_blank',
                    'question': 'Fill ___',
                    'answer': 'blank',
                    'alternatives': ['blanks']
                }
            ]
        }
        (concept_dir / 'quiz_fillblank.json').write_text(json.dumps(quiz_fb))

        # Load quiz
        quiz = ConceptQuiz('test_concept', base_path=Path(tmpdir) / 'data')

        assert len(quiz.mc_questions) == 1
        assert len(quiz.fb_questions) == 1


def test_quiz_generates_mixed_questions():
    """Test quiz generates mix of MC and fill-blank"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup (similar to above, with more questions)
        concept_dir = Path(tmpdir) / 'data' / 'test_concept'
        concept_dir.mkdir(parents=True)

        concept_data = {'name': 'Test', 'slug': 'test_concept', 'topic': 'test',
                       'difficulty': 'easy', 'prerequisites': []}
        (concept_dir / 'concept.json').write_text(json.dumps(concept_data))

        # Create 10 MC questions
        mc_questions = [
            {'id': f'mc_{i}', 'type': 'multiple_choice', 'question': f'Q{i}',
             'options': ['A', 'B'], 'correct_index': 0, 'explanation': 'Test'}
            for i in range(10)
        ]
        (concept_dir / 'quiz_mc.json').write_text(
            json.dumps({'questions': mc_questions}))

        # Create 5 fill-blank questions
        fb_questions = [
            {'id': f'fb_{i}', 'type': 'fill_blank', 'question': f'Fill {i}',
             'answer': 'test', 'alternatives': []}
            for i in range(5)
        ]
        (concept_dir / 'quiz_fillblank.json').write_text(
            json.dumps({'questions': fb_questions}))

        # Generate quiz
        quiz = ConceptQuiz('test_concept', base_path=Path(tmpdir) / 'data')
        questions = quiz.generate_quiz(num_questions=10, mix_types=True)

        assert len(questions) == 10

        # Should have both types
        mc_count = sum(1 for q in questions if q['type'] == 'multiple_choice')
        fb_count = sum(1 for q in questions if q['type'] == 'fill_blank')

        assert mc_count > 0
        assert fb_count > 0


def test_grader_checks_multiple_choice():
    """Test grading multiple choice answers"""
    grader = AnswerGrader()

    question = {
        'type': 'multiple_choice',
        'correct_index': 2
    }

    assert grader.check_answer(question, 2) == True
    assert grader.check_answer(question, 0) == False


def test_grader_checks_fill_blank_with_alternatives():
    """Test grading fill-in-blank with alternatives"""
    grader = AnswerGrader()

    question = {
        'type': 'fill_blank',
        'answer': 'gradient',
        'alternatives': ['gradients', 'grad', '∇']
    }

    assert grader.check_answer(question, 'gradient') == True
    assert grader.check_answer(question, 'Gradient') == True  # Case insensitive
    assert grader.check_answer(question, 'gradients') == True  # Alternative
    assert grader.check_answer(question, 'wrong') == False
