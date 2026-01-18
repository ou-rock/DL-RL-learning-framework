"""Quiz engine for ML concepts with caching for performance"""

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from learning_framework.knowledge.concepts import load_concept
from learning_framework.assessment.spaced_repetition import SpacedRepetitionScheduler
from learning_framework.assessment.grader import AnswerGrader
from learning_framework.progress import ProgressDatabase


class ConceptQuiz:
    """Quiz engine for a specific concept with caching"""

    # Class-level cache for quiz data
    _quiz_cache: Dict[str, Dict] = {}

    def __init__(
        self,
        concept_slug: str,
        base_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        quiz_dir: Optional[Path] = None
    ):
        """Initialize quiz for a concept

        Args:
            concept_slug: Concept identifier
            base_path: Base data directory
            db_path: Database path
            quiz_dir: Directory containing quiz JSON files (for testing)
        """
        if base_path is None:
            base_path = Path.cwd() / 'data'
        if db_path is None:
            db_path = Path.cwd() / 'user_data' / 'progress.db'

        self.concept_slug = concept_slug
        self.base_path = Path(base_path)
        self.quiz_dir = quiz_dir  # Custom quiz directory for testing
        self._questions = None  # Lazy-loaded

        # Try to load concept, but allow quiz to work without it
        try:
            self.concept = load_concept(concept_slug, base_path)
        except Exception:
            self.concept = {'name': concept_slug, 'topic': 'unknown', 'difficulty': 1}

        # Lazy initialization of database resources
        self._scheduler = None
        self._grader = None
        self._db = None
        self._db_path = db_path

    @property
    def questions(self) -> List[Dict]:
        """Lazy-load questions with caching"""
        if self._questions is None:
            self._questions = self._load_all_questions()
        return self._questions

    @property
    def mc_questions(self) -> List[Dict]:
        """Get multiple choice questions (backward compatibility)"""
        return [q for q in self.questions if q.get('type') == 'multiple_choice']

    @property
    def fb_questions(self) -> List[Dict]:
        """Get fill-in-blank questions (backward compatibility)"""
        return [q for q in self.questions if q.get('type') == 'fill_blank']

    @property
    def scheduler(self):
        """Lazy-load scheduler"""
        if self._scheduler is None:
            self._scheduler = SpacedRepetitionScheduler(self._db_path)
        return self._scheduler

    @property
    def grader(self):
        """Lazy-load grader"""
        if self._grader is None:
            self._grader = AnswerGrader()
        return self._grader

    @property
    def db(self):
        """Lazy-load database"""
        if self._db is None:
            self._db = ProgressDatabase(self._db_path)
        return self._db

    def _load_all_questions(self) -> List[Dict]:
        """Load questions from cache or file"""
        # Determine quiz directory
        if self.quiz_dir:
            quiz_base = self.quiz_dir
            cache_key = f"{self.quiz_dir}:{self.concept_slug}"
        else:
            quiz_base = self.base_path / self.concept_slug
            cache_key = f"{self.base_path}:{self.concept_slug}"

        # Check class-level cache first
        if cache_key in ConceptQuiz._quiz_cache:
            return ConceptQuiz._quiz_cache[cache_key]['questions']

        questions = []

        # Try loading from single quiz file (test format)
        if self.quiz_dir:
            quiz_file = self.quiz_dir / f"{self.concept_slug}.json"
            if quiz_file.exists():
                with open(quiz_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    questions = data.get('questions', [])
        else:
            # Load from separate MC and fill-blank files (original format)
            questions.extend(self._load_questions_file('quiz_mc.json'))
            questions.extend(self._load_questions_file('quiz_fillblank.json'))

        # Store in cache
        ConceptQuiz._quiz_cache[cache_key] = {
            'questions': questions,
            'loaded_at': time.time()
        }

        return questions

    def _load_questions_file(self, filename: str) -> List[Dict[str, Any]]:
        """Load questions from a specific JSON file

        Args:
            filename: Quiz JSON filename

        Returns:
            List of question dictionaries
        """
        quiz_path = self.base_path / self.concept_slug / filename

        if not quiz_path.exists():
            return []

        with open(quiz_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('questions', [])

    def generate_quiz(
        self,
        num_questions: int = 10,
        mix_types: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate quiz with mixed question types

        Args:
            num_questions: Total number of questions
            mix_types: Mix MC and fill-in-blank (70/30 split)

        Returns:
            List of selected questions (randomized)
        """
        available = self.questions.copy()

        if not available:
            return []

        if mix_types:
            mc_questions = self.mc_questions
            fb_questions = self.fb_questions

            if mc_questions and fb_questions:
                # 70% multiple choice, 30% fill-in-blank
                num_mc = int(num_questions * 0.7)
                num_fb = num_questions - num_mc

                # Select questions
                mc_selected = random.sample(
                    mc_questions,
                    min(num_mc, len(mc_questions))
                )
                fb_selected = random.sample(
                    fb_questions,
                    min(num_fb, len(fb_questions))
                )

                selected = mc_selected + fb_selected
            else:
                # Only one type available
                selected = random.sample(
                    available,
                    min(num_questions, len(available))
                )
        else:
            selected = random.sample(
                available,
                min(num_questions, len(available))
            )

        # Shuffle for random order
        random.shuffle(selected)
        return selected[:num_questions]

    @classmethod
    def clear_cache(cls):
        """Clear the question cache"""
        cls._quiz_cache.clear()

    @classmethod
    def preload(cls, concepts: List[str], quiz_dir: Optional[Path] = None):
        """Preload quiz data for multiple concepts

        Args:
            concepts: List of concept slugs to preload
            quiz_dir: Directory containing quiz files
        """
        for concept in concepts:
            quiz = cls(concept, quiz_dir=quiz_dir)
            _ = quiz.questions  # Trigger load

    def grade_answer(
        self,
        question_id: str,
        user_answer: Any,
        correct: bool
    ):
        """Grade answer and update spaced repetition schedule

        Args:
            question_id: Question identifier
            user_answer: User's answer
            correct: Whether answer was correct
        """
        # Get current interval
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT review_interval
            FROM quiz_results
            WHERE question_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (question_id,))
        row = cursor.fetchone()
        current_interval = row['review_interval'] if row else 1

        # Calculate next review
        next_review, new_interval = self.scheduler.calculate_next_review(
            question_id, correct, current_interval
        )

        # Get concept ID
        concept_data = self.db.get_concept(self.concept_slug)
        if not concept_data:
            # Create concept if doesn't exist
            self.db.add_concept(
                self.concept_slug,
                self.concept['topic'],
                self.concept['difficulty']
            )
            concept_data = self.db.get_concept(self.concept_slug)

        # Record result
        cursor.execute("""
            INSERT INTO quiz_results
            (concept_id, question_id, correct, timestamp, next_review, review_interval)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            concept_data['id'],
            question_id,
            int(correct),
            datetime.now(),
            next_review.date(),
            new_interval
        ))
        self.db.conn.commit()

    def close(self):
        """Close resources"""
        self.scheduler.close()
        self.db.close()
