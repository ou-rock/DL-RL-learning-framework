"""Quiz engine for ML concepts"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from learning_framework.knowledge.concepts import load_concept
from learning_framework.assessment.spaced_repetition import SpacedRepetitionScheduler
from learning_framework.assessment.grader import AnswerGrader
from learning_framework.progress import ProgressDatabase


class ConceptQuiz:
    """Quiz engine for a specific concept"""

    def __init__(
        self,
        concept_slug: str,
        base_path: Optional[Path] = None,
        db_path: Optional[Path] = None
    ):
        """Initialize quiz for a concept

        Args:
            concept_slug: Concept identifier
            base_path: Base data directory
            db_path: Database path
        """
        if base_path is None:
            base_path = Path.cwd() / 'data'
        if db_path is None:
            db_path = Path.cwd() / 'user_data' / 'progress.db'

        self.concept = load_concept(concept_slug, base_path)
        self.concept_slug = concept_slug
        self.base_path = Path(base_path)

        # Load quiz questions
        self.mc_questions = self._load_questions('quiz_mc.json')
        self.fb_questions = self._load_questions('quiz_fillblank.json')

        # Initialize helpers
        self.scheduler = SpacedRepetitionScheduler(db_path)
        self.grader = AnswerGrader()
        self.db = ProgressDatabase(db_path)

    def _load_questions(self, filename: str) -> List[Dict[str, Any]]:
        """Load questions from JSON file

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
        questions = []

        if mix_types and self.mc_questions and self.fb_questions:
            # 70% multiple choice, 30% fill-in-blank
            num_mc = int(num_questions * 0.7)
            num_fb = num_questions - num_mc

            # Select questions
            mc_selected = random.sample(
                self.mc_questions,
                min(num_mc, len(self.mc_questions))
            )
            fb_selected = random.sample(
                self.fb_questions,
                min(num_fb, len(self.fb_questions))
            )

            questions = mc_selected + fb_selected
        else:
            # Use only available type
            available = self.mc_questions + self.fb_questions
            questions = random.sample(
                available,
                min(num_questions, len(available))
            )

        # Shuffle for random order
        random.shuffle(questions)
        return questions

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
