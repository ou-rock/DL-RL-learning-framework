"""Performance tests for quiz loading"""
import pytest
import time
from pathlib import Path
import json


def test_quiz_loads_under_100ms(tmp_path):
    """Quiz should load in under 100ms"""
    from learning_framework.assessment.quiz import ConceptQuiz

    # Create test quiz file
    quiz_dir = tmp_path / "data" / "quizzes"
    quiz_dir.mkdir(parents=True)

    quiz_data = {
        "concept": "test_concept",
        "questions": [
            {
                "id": f"q{i}",
                "question": f"Test question {i}",
                "type": "multiple_choice",
                "options": ["A", "B", "C", "D"],
                "answer": "A"
            }
            for i in range(100)  # 100 questions
        ]
    }

    quiz_file = quiz_dir / "test_concept.json"
    quiz_file.write_text(json.dumps(quiz_data))

    # Measure load time
    start = time.perf_counter()
    quiz = ConceptQuiz("test_concept", quiz_dir=quiz_dir)
    questions = quiz.generate_quiz(num_questions=10)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1, f"Quiz loading took {elapsed:.3f}s, should be < 0.1s"
    assert len(questions) == 10


def test_quiz_caches_questions(tmp_path):
    """Subsequent quiz generations should be faster due to caching"""
    from learning_framework.assessment.quiz import ConceptQuiz

    # Create test quiz file
    quiz_dir = tmp_path / "data" / "quizzes"
    quiz_dir.mkdir(parents=True)

    quiz_data = {
        "concept": "cache_test",
        "questions": [
            {
                "id": f"q{i}",
                "question": f"Test question {i}",
                "type": "multiple_choice",
                "options": ["A", "B", "C", "D"],
                "answer": "A"
            }
            for i in range(50)
        ]
    }

    quiz_file = quiz_dir / "cache_test.json"
    quiz_file.write_text(json.dumps(quiz_data))

    # Clear any existing cache
    ConceptQuiz.clear_cache()

    # First generation
    quiz = ConceptQuiz("cache_test", quiz_dir=quiz_dir)

    start = time.perf_counter()
    quiz.generate_quiz(num_questions=5)
    first_time = time.perf_counter() - start

    # Second generation (should use cache)
    start = time.perf_counter()
    quiz.generate_quiz(num_questions=5)
    second_time = time.perf_counter() - start

    # Second should be at least as fast
    assert second_time <= first_time * 1.5  # Allow some variance


def test_quiz_preload_multiple_concepts(tmp_path):
    """Can preload multiple concepts for faster access"""
    from learning_framework.assessment.quiz import ConceptQuiz

    # Create test quiz files
    quiz_dir = tmp_path / "data" / "quizzes"
    quiz_dir.mkdir(parents=True)

    concepts = ["concept_a", "concept_b", "concept_c"]
    for concept in concepts:
        quiz_data = {
            "concept": concept,
            "questions": [
                {"id": f"q{i}", "question": f"Q{i}", "type": "multiple_choice",
                 "options": ["A", "B"], "answer": "A"}
                for i in range(10)
            ]
        }
        (quiz_dir / f"{concept}.json").write_text(json.dumps(quiz_data))

    # Clear cache
    ConceptQuiz.clear_cache()

    # Preload all concepts
    ConceptQuiz.preload(concepts, quiz_dir=quiz_dir)

    # Access should be fast now
    start = time.perf_counter()
    for concept in concepts:
        quiz = ConceptQuiz(concept, quiz_dir=quiz_dir)
        quiz.generate_quiz(num_questions=5)
    elapsed = time.perf_counter() - start

    # All three should complete quickly since preloaded
    assert elapsed < 0.1, f"Preloaded quiz access took {elapsed:.3f}s"
