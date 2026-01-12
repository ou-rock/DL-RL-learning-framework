"""Answer grading logic for quizzes"""

from typing import Any, Dict


class AnswerGrader:
    """Grades quiz answers for different question types"""

    def check_answer(self, question: Dict[str, Any], user_answer: Any) -> bool:
        """Check if user's answer is correct

        Args:
            question: Question dictionary
            user_answer: User's answer

        Returns:
            True if correct, False otherwise
        """
        question_type = question.get('type')

        if question_type == 'multiple_choice':
            return self._check_multiple_choice(question, user_answer)
        elif question_type == 'fill_blank':
            return self._check_fill_blank(question, user_answer)
        else:
            raise ValueError(f"Unknown question type: {question_type}")

    def _check_multiple_choice(self, question: Dict, user_answer: int) -> bool:
        """Check multiple choice answer

        Args:
            question: Question with 'correct_index'
            user_answer: Selected option index (0-based)

        Returns:
            True if matches correct_index
        """
        return user_answer == question['correct_index']

    def _check_fill_blank(self, question: Dict, user_answer: str) -> bool:
        """Check fill-in-blank answer with fuzzy matching

        Args:
            question: Question with 'answer' and 'alternatives'
            user_answer: User's typed answer

        Returns:
            True if matches answer or any alternative
        """
        user_clean = str(user_answer).strip().lower()

        # Check main answer
        if user_clean == question['answer'].lower():
            return True

        # Check alternatives
        for alternative in question.get('alternatives', []):
            if user_clean == alternative.lower():
                return True

        return False
