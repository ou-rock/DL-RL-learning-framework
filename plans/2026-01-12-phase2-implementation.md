# Phase 2: Learning & Assessment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build quiz system with spaced repetition, knowledge graph with concept auto-discovery, and visualization engine for 5 core DL/RL concepts.

**Architecture:** Extends Phase 1 infrastructure with assessment module (quiz + spaced repetition), knowledge module (concept detection + graph), and visualization module (matplotlib rendering). Implements 5 complete concepts (Gradient Descent, Backpropagation, Loss Functions, Activation Functions, Q-Learning) with quizzes and visualizations.

**Tech Stack:** Python 3.10+, matplotlib, numpy, existing Phase 1 (Click, Rich, SQLite, pytest)

---

## Task 1: Spaced Repetition Module (Extract from Vocab Quiz)

**Files:**
- Create: `learning_framework/assessment/__init__.py`
- Create: `learning_framework/assessment/spaced_repetition.py`
- Create: `tests/test_spaced_repetition.py`
- Reference: `skills/tools/quiz_today.py` (existing vocab quiz)

**Step 1: Write failing test for SM-2 algorithm**

Create `tests/test_spaced_repetition.py`:

```python
import pytest
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
from learning_framework.assessment.spaced_repetition import SpacedRepetitionScheduler


def test_sm2_correct_answer_increases_interval():
    """Test SM-2: correct answer increases review interval"""
    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = SpacedRepetitionScheduler(db_path=Path(tmpdir) / 'test.db')

        # Correct answer: 1 day → ~3 days (1 * 2.5 = 2.5)
        next_review, new_interval = scheduler.calculate_next_review(
            item_id='test_001',
            correct=True,
            current_interval=1
        )

        assert new_interval == 2  # int(1 * 2.5) = 2
        assert (next_review - datetime.now()).days >= 1


def test_sm2_incorrect_answer_resets_interval():
    """Test SM-2: incorrect answer resets to day 1"""
    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = SpacedRepetitionScheduler(db_path=Path(tmpdir) / 'test.db')

        # Incorrect answer: any interval → 1 day
        next_review, new_interval = scheduler.calculate_next_review(
            item_id='test_001',
            correct=False,
            current_interval=14
        )

        assert new_interval == 1
        assert (next_review - datetime.now()).days == 0  # Tomorrow


def test_get_due_items_returns_items_due_today():
    """Test retrieving items due for review"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'test.db'

        # Setup: Create database with quiz results
        from learning_framework.progress import ProgressDatabase
        db = ProgressDatabase(db_path)
        db.add_concept('test_concept', 'test_topic', 'easy')

        # Add a quiz result that's due today
        concept = db.get_concept('test_concept')
        db.conn.execute("""
            INSERT INTO quiz_results
            (concept_id, question_id, correct, timestamp, next_review, review_interval)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (concept['id'], 'q_001', 1, datetime.now(),
              datetime.now().date(), 1))
        db.conn.commit()
        db.close()

        # Test: Get due items
        scheduler = SpacedRepetitionScheduler(db_path=db_path)
        due_items = scheduler.get_due_items(tier='quiz')

        assert len(due_items) > 0
        assert any(item['question_id'] == 'q_001' for item in due_items)
```

**Step 2: Run test to verify it fails**

```bash
cd learning-framework-dev
python -m pytest tests/test_spaced_repetition.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'learning_framework.assessment'"

**Step 3: Create assessment package**

Create `learning_framework/assessment/__init__.py`:

```python
"""Assessment system: quizzes, spaced repetition, grading"""

from learning_framework.assessment.spaced_repetition import SpacedRepetitionScheduler

__all__ = ['SpacedRepetitionScheduler']
```

**Step 4: Implement SpacedRepetitionScheduler**

Create `learning_framework/assessment/spaced_repetition.py`:

```python
"""Spaced repetition scheduler using SM-2 algorithm

Extracted from vocab quiz system for reusability across
both vocabulary learning and ML concept quizzes.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from learning_framework.progress import ProgressDatabase


class SpacedRepetitionScheduler:
    """SM-2 spaced repetition algorithm for optimal review scheduling

    Works for any learning items: German vocab, ML concepts, etc.

    Algorithm:
    - Correct answer: interval *= 2.5
    - Incorrect answer: reset to 1 day
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize scheduler

        Args:
            db_path: Path to progress database (default: user_data/progress.db)
        """
        if db_path is None:
            db_path = Path.cwd() / 'user_data' / 'progress.db'

        self.db = ProgressDatabase(db_path)

    def calculate_next_review(
        self,
        item_id: str,
        correct: bool,
        current_interval: int = 1
    ) -> Tuple[datetime, int]:
        """Calculate next review date using SM-2 algorithm

        Args:
            item_id: Question ID or item identifier
            correct: Whether answer was correct
            current_interval: Current interval in days

        Returns:
            (next_review_date, new_interval_days)
        """
        if correct:
            # Correct answer: increase interval
            new_interval = max(1, int(current_interval * 2.5))
        else:
            # Incorrect answer: reset to day 1
            new_interval = 1

        # Calculate next review date
        next_review = datetime.now() + timedelta(days=new_interval)

        return next_review, new_interval

    def get_due_items(
        self,
        concept: Optional[str] = None,
        tier: str = 'quiz'
    ) -> List[Dict[str, Any]]:
        """Get items due for review today

        Args:
            concept: Filter by concept slug (optional)
            tier: 'quiz' or 'implementation'

        Returns:
            List of due items with metadata
        """
        query = """
            SELECT
                qr.id,
                qr.question_id,
                qr.next_review,
                qr.review_interval,
                c.slug as concept_slug,
                c.name as concept_name
            FROM quiz_results qr
            JOIN concepts c ON qr.concept_id = c.id
            WHERE qr.next_review <= DATE('now')
        """

        params = []

        if concept:
            query += " AND c.slug = ?"
            params.append(concept)

        query += " ORDER BY qr.next_review ASC"

        cursor = self.db.conn.cursor()
        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append(dict(row))

        return results

    def close(self):
        """Close database connection"""
        self.db.close()
```

**Step 5: Update ProgressDatabase schema for spaced repetition**

Modify `learning_framework/progress/database.py`:

Add to `SCHEMA` constant:

```python
# In the SCHEMA string, add these columns to quiz_results table:
CREATE TABLE IF NOT EXISTS quiz_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id INTEGER NOT NULL,
    question_id TEXT NOT NULL,
    correct BOOLEAN NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    next_review DATE,                    -- NEW
    review_interval INTEGER DEFAULT 1,   -- NEW
    FOREIGN KEY (concept_id) REFERENCES concepts(id)
);

CREATE INDEX IF NOT EXISTS idx_next_review ON quiz_results(next_review);  -- NEW
```

**Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_spaced_repetition.py -v
```

Expected: PASS (all 3 tests pass)

**Step 7: Commit**

```bash
git add learning_framework/assessment/ learning_framework/progress/database.py tests/test_spaced_repetition.py
git commit -m "feat: add spaced repetition scheduler with SM-2 algorithm"
```

---

## Task 2: Concept Loading & Knowledge Graph Foundation

**Files:**
- Create: `learning_framework/knowledge/concepts.py`
- Create: `learning_framework/knowledge/graph.py`
- Create: `tests/test_concepts.py`
- Modify: `learning_framework/knowledge/__init__.py`

**Step 1: Write failing test for concept loading**

Create `tests/test_concepts.py`:

```python
import pytest
import tempfile
import json
from pathlib import Path
from learning_framework.knowledge.concepts import ConceptRegistry, load_concept


def test_load_concept_from_file():
    """Test loading concept from concept.json"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test concept
        concept_dir = Path(tmpdir) / 'data' / 'test_concept'
        concept_dir.mkdir(parents=True)

        concept_data = {
            'name': 'Test Concept',
            'slug': 'test_concept',
            'topic': 'testing',
            'difficulty': 'easy',
            'status': 'complete',
            'prerequisites': [],
            'description': 'A test concept',
            'materials': {},
            'tags': ['test']
        }

        (concept_dir / 'concept.json').write_text(json.dumps(concept_data, indent=2))

        # Load concept
        concept = load_concept('test_concept', base_path=Path(tmpdir) / 'data')

        assert concept['name'] == 'Test Concept'
        assert concept['slug'] == 'test_concept'
        assert concept['difficulty'] == 'easy'


def test_concept_registry_tracks_all_concepts():
    """Test registry maintains list of all concepts"""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ConceptRegistry(base_path=Path(tmpdir) / 'data')

        # Add concepts
        registry.register('gradient_descent', topic='optimization', difficulty='beginner')
        registry.register('backprop', topic='neural_networks', difficulty='intermediate')

        # Get all concepts
        all_concepts = registry.get_all()

        assert len(all_concepts) == 2
        assert 'gradient_descent' in all_concepts
        assert 'backprop' in all_concepts


def test_concept_registry_groups_by_topic():
    """Test getting concepts grouped by topic"""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ConceptRegistry(base_path=Path(tmpdir) / 'data')

        registry.register('gradient_descent', topic='optimization', difficulty='beginner')
        registry.register('backprop', topic='neural_networks', difficulty='intermediate')
        registry.register('sgd', topic='optimization', difficulty='beginner')

        # Get by topic
        optimization = registry.get_by_topic('optimization')

        assert len(optimization) == 2
        assert 'gradient_descent' in optimization
        assert 'sgd' in optimization
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_concepts.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement concept loading**

Create `learning_framework/knowledge/concepts.py`:

```python
"""Concept loading and registry management"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_concept(concept_slug: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load concept from concept.json file

    Args:
        concept_slug: Concept identifier (e.g., 'backpropagation')
        base_path: Base data directory (default: data/)

    Returns:
        Concept dictionary

    Raises:
        FileNotFoundError: If concept.json doesn't exist
    """
    if base_path is None:
        base_path = Path.cwd() / 'data'

    concept_path = base_path / concept_slug / 'concept.json'

    if not concept_path.exists():
        raise FileNotFoundError(f"Concept not found: {concept_slug}")

    with open(concept_path, 'r', encoding='utf-8') as f:
        concept = json.load(f)

    return concept


class ConceptRegistry:
    """Registry of all concepts in the system

    Maintains master list and provides querying by topic, difficulty, etc.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize concept registry

        Args:
            base_path: Base data directory (default: data/)
        """
        if base_path is None:
            base_path = Path.cwd() / 'data'

        self.base_path = Path(base_path)
        self.registry_path = self.base_path / 'concepts.json'
        self._concepts = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load concepts registry from concepts.json"""
        if not self.registry_path.exists():
            return {'version': '0.2.0', 'concepts': {}, 'topics': {}}

        with open(self.registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save(self):
        """Save registry to disk"""
        self.base_path.mkdir(parents=True, exist_ok=True)

        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self._concepts, f, indent=2, ensure_ascii=False)

    def register(
        self,
        concept_slug: str,
        topic: str,
        difficulty: str,
        status: str = 'skeleton'
    ):
        """Register a concept in the registry

        Args:
            concept_slug: Concept identifier
            topic: Topic category
            difficulty: Difficulty level
            status: 'complete' or 'skeleton'
        """
        if 'concepts' not in self._concepts:
            self._concepts['concepts'] = {}

        self._concepts['concepts'][concept_slug] = {
            'status': status,
            'topic': topic,
            'difficulty': difficulty
        }

        # Update topic index
        if 'topics' not in self._concepts:
            self._concepts['topics'] = {}

        if topic not in self._concepts['topics']:
            self._concepts['topics'][topic] = []

        if concept_slug not in self._concepts['topics'][topic]:
            self._concepts['topics'][topic].append(concept_slug)

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered concepts

        Returns:
            Dictionary of concept_slug -> metadata
        """
        return self._concepts.get('concepts', {})

    def get_by_topic(self, topic: str) -> List[str]:
        """Get concept slugs for a specific topic

        Args:
            topic: Topic name

        Returns:
            List of concept slugs in that topic
        """
        return self._concepts.get('topics', {}).get(topic, [])

    def get_topics(self) -> List[str]:
        """Get list of all topics

        Returns:
            List of topic names
        """
        return list(self._concepts.get('topics', {}).keys())
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_concepts.py -v
```

Expected: PASS (all 3 tests pass)

**Step 5: Create KnowledgeGraph with prerequisite checking**

Create `learning_framework/knowledge/graph.py`:

```python
"""Knowledge graph with prerequisite relationships"""

from typing import Dict, List, Any, Optional
from pathlib import Path

from learning_framework.knowledge.concepts import load_concept
from learning_framework.progress import ProgressDatabase


class KnowledgeGraph:
    """Knowledge graph managing concept prerequisites and dependencies"""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize knowledge graph

        Args:
            db_path: Path to progress database
        """
        if db_path is None:
            db_path = Path.cwd() / 'user_data' / 'progress.db'

        self.db = ProgressDatabase(db_path)
        self.base_path = Path.cwd() / 'data'

    def check_prerequisites(self, concept_slug: str) -> Dict[str, Any]:
        """Check if prerequisites are mastered

        Args:
            concept_slug: Concept to check

        Returns:
            {
                'ready': bool,
                'missing': [list of unmastered prerequisites],
                'warning': str (if soft warning needed)
            }
        """
        concept = load_concept(concept_slug, self.base_path)
        prerequisites = concept.get('prerequisites', [])

        if not prerequisites:
            return {'ready': True, 'missing': [], 'warning': None}

        missing = []

        for prereq_slug in prerequisites:
            prereq_data = self.db.get_concept(prereq_slug)

            if not prereq_data or not prereq_data.get('quiz_passed'):
                missing.append(prereq_slug)

        warning = None
        if missing:
            prereq_names = [load_concept(slug, self.base_path)['name']
                          for slug in missing]
            warning = f"Recommended prerequisites: {', '.join(prereq_names)}"

        return {
            'ready': len(missing) == 0,
            'missing': missing,
            'warning': warning
        }

    def get_learning_path(self, concept_slug: str) -> List[str]:
        """Get ordered list of concepts to learn (including prerequisites)

        Args:
            concept_slug: Target concept

        Returns:
            Ordered list of concept slugs (prerequisites first)
        """
        visited = set()
        path = []

        def dfs(slug):
            if slug in visited:
                return
            visited.add(slug)

            concept = load_concept(slug, self.base_path)
            for prereq in concept.get('prerequisites', []):
                dfs(prereq)

            path.append(slug)

        dfs(concept_slug)
        return path

    def close(self):
        """Close database connection"""
        self.db.close()
```

**Step 6: Update knowledge package __init__.py**

Modify `learning_framework/knowledge/__init__.py`:

```python
"""Knowledge graph and material indexing"""

from learning_framework.knowledge.indexer import MaterialIndexer
from learning_framework.knowledge.concepts import ConceptRegistry, load_concept
from learning_framework.knowledge.graph import KnowledgeGraph

__all__ = ['MaterialIndexer', 'ConceptRegistry', 'load_concept', 'KnowledgeGraph']
```

**Step 7: Commit**

```bash
git add learning_framework/knowledge/ tests/test_concepts.py
git commit -m "feat: add concept loading and knowledge graph with prerequisites"
```

---

## Task 3: Quiz System Core (Question Loading & Grading)

**Files:**
- Create: `learning_framework/assessment/quiz.py`
- Create: `learning_framework/assessment/grader.py`
- Create: `tests/test_quiz.py`
- Modify: `learning_framework/assessment/__init__.py`

**Step 1: Write failing tests for quiz system**

Create `tests/test_quiz.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_quiz.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement AnswerGrader**

Create `learning_framework/assessment/grader.py`:

```python
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
```

**Step 4: Implement ConceptQuiz**

Create `learning_framework/assessment/quiz.py`:

```python
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
```

**Step 5: Update assessment __init__.py**

Modify `learning_framework/assessment/__init__.py`:

```python
"""Assessment system: quizzes, spaced repetition, grading"""

from learning_framework.assessment.spaced_repetition import SpacedRepetitionScheduler
from learning_framework.assessment.quiz import ConceptQuiz
from learning_framework.assessment.grader import AnswerGrader

__all__ = ['SpacedRepetitionScheduler', 'ConceptQuiz', 'AnswerGrader']
```

**Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_quiz.py -v
```

Expected: PASS (all 5 tests pass)

**Step 7: Commit**

```bash
git add learning_framework/assessment/ tests/test_quiz.py
git commit -m "feat: add quiz system with MC and fill-in-blank questions"
```

---

## Task 4: Create 5 Complete Concepts with Quiz Content

**Files:**
- Create: `data/gradient_descent/concept.json`
- Create: `data/gradient_descent/quiz_mc.json`
- Create: `data/gradient_descent/quiz_fillblank.json`
- Create: `data/backpropagation/concept.json`
- Create: `data/backpropagation/quiz_mc.json`
- Create: `data/backpropagation/quiz_fillblank.json`
- Create: `data/loss_functions/concept.json`
- Create: `data/loss_functions/quiz_mc.json`
- Create: `data/loss_functions/quiz_fillblank.json`
- Create: `data/activation_functions/concept.json`
- Create: `data/activation_functions/quiz_mc.json`
- Create: `data/activation_functions/quiz_fillblank.json`
- Create: `data/q_learning/concept.json`
- Create: `data/q_learning/quiz_mc.json`
- Create: `data/q_learning/quiz_fillblank.json`
- Create: `data/concepts.json` (master registry)
- Create: `data/templates/concept.json.template`
- Create: `data/templates/quiz_mc.json.template`
- Create: `data/templates/quiz_fillblank.json.template`

**Step 1: Create data directory structure**

```bash
cd learning-framework-dev
mkdir -p data/{gradient_descent,backpropagation,loss_functions,activation_functions,q_learning,templates}
```

**Step 2: Create Gradient Descent concept**

Create `data/gradient_descent/concept.json`:

```json
{
  "name": "Gradient Descent",
  "slug": "gradient_descent",
  "topic": "optimization",
  "difficulty": "beginner",
  "status": "complete",
  "prerequisites": [],
  "description": "Iterative optimization algorithm that follows the negative gradient to find local minima of a function. Foundation of neural network training.",
  "materials": {
    "explanation": "",
    "code_examples": []
  },
  "estimated_time_minutes": 30,
  "tags": ["optimization", "fundamentals", "algorithms"]
}
```

Create `data/gradient_descent/quiz_mc.json`:

```json
{
  "version": "1.0",
  "questions": [
    {
      "id": "gd_mc_001",
      "type": "multiple_choice",
      "question": "What direction does gradient descent move in parameter space?",
      "options": [
        "Direction of steepest increase",
        "Direction of steepest decrease",
        "Random direction",
        "Perpendicular to the gradient"
      ],
      "correct_index": 1,
      "explanation": "Gradient descent moves in the negative gradient direction (steepest decrease) to minimize the loss function.",
      "difficulty": 1,
      "tags": ["direction", "gradient"]
    },
    {
      "id": "gd_mc_002",
      "question": "What happens if the learning rate is too large?",
      "options": [
        "Convergence is slower but guaranteed",
        "The algorithm may oscillate or diverge",
        "Better generalization is achieved",
        "Training becomes faster and more stable"
      ],
      "correct_index": 1,
      "explanation": "Too large learning rate causes overshooting, leading to oscillation around the minimum or divergence away from it.",
      "difficulty": 2,
      "tags": ["learning_rate", "hyperparameters"]
    },
    {
      "id": "gd_mc_003",
      "question": "What is the purpose of the learning rate α in gradient descent?",
      "options": [
        "To increase the gradient magnitude",
        "To control the step size of parameter updates",
        "To add randomness to the optimization",
        "To regularize the model"
      ],
      "correct_index": 1,
      "explanation": "Learning rate α controls how large each step is. θ = θ - α∇J(θ). Too small = slow convergence, too large = instability.",
      "difficulty": 1,
      "tags": ["learning_rate", "parameters"]
    },
    {
      "id": "gd_mc_004",
      "question": "Gradient descent is guaranteed to find the global minimum when:",
      "options": [
        "The loss function is convex",
        "The learning rate is very small",
        "The model has many parameters",
        "We use momentum"
      ],
      "correct_index": 0,
      "explanation": "For convex functions, any local minimum is also the global minimum, so gradient descent will find it. For non-convex functions, it may get stuck in local minima.",
      "difficulty": 2,
      "tags": ["convexity", "convergence"]
    },
    {
      "id": "gd_mc_005",
      "question": "How do we know when to stop gradient descent?",
      "options": [
        "When loss reaches exactly zero",
        "After a fixed number of iterations",
        "When gradient magnitude is close to zero or loss stops decreasing",
        "When learning rate becomes zero"
      ],
      "correct_index": 2,
      "explanation": "Convergence criteria: gradient ≈ 0 (at minimum) or loss stops improving. We rarely reach exact zero loss.",
      "difficulty": 2,
      "tags": ["convergence", "stopping_criteria"]
    },
    {
      "id": "gd_mc_006",
      "question": "What does the gradient ∇J(θ) represent?",
      "options": [
        "The minimum value of the loss",
        "The direction and magnitude of steepest increase",
        "The learning rate",
        "The number of iterations needed"
      ],
      "correct_index": 1,
      "explanation": "Gradient is a vector pointing in direction of steepest increase. Its magnitude indicates how steep the slope is.",
      "difficulty": 1,
      "tags": ["gradient", "calculus"]
    },
    {
      "id": "gd_mc_007",
      "question": "Why is it called 'gradient descent' and not 'gradient ascent'?",
      "options": [
        "We want to minimize loss (go down the slope)",
        "We want to maximize accuracy (go up the slope)",
        "It's just convention, both work the same",
        "Descent is faster than ascent"
      ],
      "correct_index": 0,
      "explanation": "We minimize loss by going down (descending) the loss surface. Gradient ascent would maximize the function instead.",
      "difficulty": 1,
      "tags": ["terminology", "optimization"]
    },
    {
      "id": "gd_mc_008",
      "question": "In batch gradient descent, what is updated after each iteration?",
      "options": [
        "Only one randomly selected parameter",
        "All parameters using gradient computed on entire dataset",
        "All parameters using gradient from one sample",
        "The learning rate"
      ],
      "correct_index": 1,
      "explanation": "Batch GD computes gradient using ALL training data, then updates all parameters. Contrast with SGD (one sample) or mini-batch (subset).",
      "difficulty": 2,
      "tags": ["batch_gd", "variants"]
    },
    {
      "id": "gd_mc_009",
      "question": "What is the main disadvantage of batch gradient descent?",
      "options": [
        "Less accurate than other methods",
        "Cannot escape local minima",
        "Computationally expensive for large datasets",
        "Requires tuning many hyperparameters"
      ],
      "correct_index": 2,
      "explanation": "Batch GD must process entire dataset per iteration, which is slow and memory-intensive for large datasets (millions of examples).",
      "difficulty": 2,
      "tags": ["batch_gd", "computational_cost"]
    },
    {
      "id": "gd_mc_010",
      "question": "What happens at a local minimum in gradient descent?",
      "options": [
        "Gradient becomes zero and updates stop",
        "Loss becomes zero",
        "Learning rate increases automatically",
        "Algorithm restarts from random position"
      ],
      "correct_index": 0,
      "explanation": "At local minimum, gradient = 0, so θ = θ - α×0 = θ (no change). Algorithm stops making progress unless we add momentum or noise.",
      "difficulty": 2,
      "tags": ["local_minimum", "gradient"]
    }
  ]
}
```

Create `data/gradient_descent/quiz_fillblank.json`:

```json
{
  "version": "1.0",
  "questions": [
    {
      "id": "gd_fb_001",
      "type": "fill_blank",
      "question": "The gradient descent update rule is: θ = θ - α × ___",
      "answer": "∇J(θ)",
      "alternatives": ["gradient", "∇J", "grad J", "dJ/dθ"],
      "explanation": "θ = θ - α∇J(θ) where α is learning rate and ∇J(θ) is the gradient of loss with respect to parameters.",
      "difficulty": 1,
      "tags": ["equation", "update_rule"]
    },
    {
      "id": "gd_fb_002",
      "question": "Gradient descent is guaranteed to converge to global minimum for ___ functions.",
      "answer": "convex",
      "alternatives": ["convex functions"],
      "explanation": "Convex functions have a single bowl-shaped minimum. Any local minimum is the global minimum.",
      "difficulty": 2,
      "tags": ["convexity", "convergence"]
    },
    {
      "id": "gd_fb_003",
      "question": "The hyperparameter that controls step size in gradient descent is called the ___.",
      "answer": "learning rate",
      "alternatives": ["learning-rate", "step size", "α", "alpha"],
      "explanation": "Learning rate (α) determines how big each update step is: θ = θ - α∇J(θ).",
      "difficulty": 1,
      "tags": ["hyperparameters", "learning_rate"]
    },
    {
      "id": "gd_fb_004",
      "question": "If the gradient is zero, we are at a ___ of the loss function.",
      "answer": "critical point",
      "alternatives": ["stationary point", "minimum", "maximum", "extremum"],
      "explanation": "∇J = 0 indicates a critical/stationary point (could be minimum, maximum, or saddle point).",
      "difficulty": 2,
      "tags": ["gradient", "critical_points"]
    },
    {
      "id": "gd_fb_005",
      "question": "Gradient descent moves in the direction ___ to the gradient vector.",
      "answer": "opposite",
      "alternatives": ["negative", "reverse", "inverse", "contrary"],
      "explanation": "We subtract the gradient (θ = θ - α∇J), moving opposite to gradient direction to go downhill.",
      "difficulty": 1,
      "tags": ["direction", "gradient"]
    }
  ]
}
```

**Step 3: Create Backpropagation concept** (abbreviated for space)

Create `data/backpropagation/concept.json`:

```json
{
  "name": "Backpropagation",
  "slug": "backpropagation",
  "topic": "neural_networks",
  "difficulty": "intermediate",
  "status": "complete",
  "prerequisites": ["gradient_descent"],
  "description": "Algorithm for efficiently computing gradients in neural networks using the chain rule. Enables training of deep networks by propagating errors backward through layers.",
  "materials": {
    "explanation": "",
    "code_examples": []
  },
  "estimated_time_minutes": 45,
  "tags": ["neural_networks", "gradients", "training"]
}
```

Create `data/backpropagation/quiz_mc.json` with 10 questions (abbreviated example):

```json
{
  "version": "1.0",
  "questions": [
    {
      "id": "bp_mc_001",
      "type": "multiple_choice",
      "question": "What mathematical rule makes backpropagation possible?",
      "options": [
        "Product rule",
        "Chain rule",
        "Power rule",
        "Quotient rule"
      ],
      "correct_index": 1,
      "explanation": "Chain rule allows us to decompose complex derivatives: dL/dw = dL/dy × dy/dx × dx/dw.",
      "difficulty": 2,
      "tags": ["chain_rule", "calculus"]
    },
    {
      "id": "bp_mc_002",
      "question": "Why is it called 'backpropagation'?",
      "options": [
        "It goes back in time",
        "It propagates errors backward from output to input",
        "It backtracks when wrong",
        "It's the reverse of forward propagation"
      ],
      "correct_index": 1,
      "explanation": "Forward pass computes outputs. Backward pass computes gradients by propagating errors from output layers back to input layers.",
      "difficulty": 1,
      "tags": ["terminology", "direction"]
    }
  ]
}
```

(Create remaining 8 MC questions and 5 fill-blank questions similarly)

**Step 4: Create remaining 3 concepts** (Loss Functions, Activation Functions, Q-Learning)

Follow same pattern for:
- `data/loss_functions/` (10 MC + 5 fill-blank)
- `data/activation_functions/` (10 MC + 5 fill-blank)
- `data/q_learning/` (10 MC + 5 fill-blank)

**Step 5: Create master registry**

Create `data/concepts.json`:

```json
{
  "version": "0.2.0",
  "concepts": {
    "gradient_descent": {
      "status": "complete",
      "topic": "optimization",
      "difficulty": "beginner"
    },
    "backpropagation": {
      "status": "complete",
      "topic": "neural_networks",
      "difficulty": "intermediate"
    },
    "loss_functions": {
      "status": "complete",
      "topic": "neural_networks",
      "difficulty": "beginner"
    },
    "activation_functions": {
      "status": "complete",
      "topic": "neural_networks",
      "difficulty": "beginner"
    },
    "q_learning": {
      "status": "complete",
      "topic": "reinforcement_learning",
      "difficulty": "intermediate"
    }
  },
  "topics": {
    "optimization": ["gradient_descent"],
    "neural_networks": ["backpropagation", "loss_functions", "activation_functions"],
    "reinforcement_learning": ["q_learning"]
  }
}
```

**Step 6: Create templates for easy expansion**

Create `data/templates/concept.json.template`:

```json
{
  "name": "[CONCEPT_NAME]",
  "slug": "[concept_slug]",
  "topic": "[topic]",
  "difficulty": "[beginner|intermediate|advanced]",
  "status": "skeleton",
  "prerequisites": [],
  "description": "[Brief description of what this concept is]",
  "materials": {
    "explanation": "",
    "code_examples": []
  },
  "estimated_time_minutes": 30,
  "tags": []
}
```

**Step 7: Commit**

```bash
git add data/
git commit -m "feat: add 5 complete concepts with quiz content (75 questions total)"
```

---

## Task 5: Visualization Engine Foundation

**Files:**
- Create: `learning_framework/visualization/__init__.py`
- Create: `learning_framework/visualization/renderer.py`
- Create: `learning_framework/visualization/display.py`
- Create: `tests/test_visualization.py`

**Step 1: Write failing tests for visualization**

Create `tests/test_visualization.py`:

```python
import pytest
import tempfile
from pathlib import Path
from learning_framework.visualization.renderer import VisualizationRenderer


def test_renderer_discovers_viz_functions():
    """Test renderer discovers functions in visualize.py"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test visualization
        viz_dir = Path(tmpdir) / 'data' / 'test_concept'
        viz_dir.mkdir(parents=True)

        viz_code = '''
import matplotlib.pyplot as plt

def main_visualization():
    """Primary test visualization"""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    return fig

def secondary_viz():
    """Secondary visualization"""
    fig, ax = plt.subplots()
    ax.plot([1, 2], [2, 1])
    return fig
'''
        (viz_dir / 'visualize.py').write_text(viz_code)

        # Discover functions
        renderer = VisualizationRenderer(base_path=Path(tmpdir) / 'data')
        functions = renderer.get_available_visualizations('test_concept')

        assert len(functions) >= 2
        names = [f['name'] for f in functions]
        assert 'main_visualization' in names
        assert 'secondary_viz' in names


def test_renderer_executes_visualization():
    """Test renderer executes viz function and returns figure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        viz_dir = Path(tmpdir) / 'data' / 'test_concept'
        viz_dir.mkdir(parents=True)

        viz_code = '''
import matplotlib.pyplot as plt

def test_viz():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    return fig
'''
        (viz_dir / 'visualize.py').write_text(viz_code)

        # Render
        renderer = VisualizationRenderer(base_path=Path(tmpdir) / 'data')
        fig = renderer.execute_visualization('test_concept', 'test_viz')

        assert fig is not None
        assert hasattr(fig, 'savefig')  # Is a matplotlib figure
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_visualization.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create visualization package**

Create `learning_framework/visualization/__init__.py`:

```python
"""Visualization engine for concept visualizations"""

from learning_framework.visualization.renderer import VisualizationRenderer

__all__ = ['VisualizationRenderer']
```

**Step 4: Implement VisualizationRenderer**

Create `learning_framework/visualization/renderer.py`:

```python
"""Visualization rendering and execution"""

import importlib.util
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class VisualizationRenderer:
    """Renders matplotlib visualizations from concept visualize.py files"""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize renderer

        Args:
            base_path: Base data directory (default: data/)
        """
        if base_path is None:
            base_path = Path.cwd() / 'data'

        self.base_path = Path(base_path)

    def get_available_visualizations(self, concept_slug: str) -> List[Dict[str, str]]:
        """Discover visualization functions in visualize.py

        Args:
            concept_slug: Concept identifier

        Returns:
            List of {name, description} dicts
        """
        viz_path = self.base_path / concept_slug / 'visualize.py'

        if not viz_path.exists():
            return []

        # Import module
        module = self._import_viz_module(concept_slug)

        # Find all functions
        functions = []
        for name, obj in inspect.getmembers(module):
            if name.startswith('_'):
                continue
            if inspect.isfunction(obj):
                doc = obj.__doc__ or "No description"
                description = doc.strip().split('\n')[0]  # First line only
                functions.append({
                    'name': name,
                    'description': description
                })

        return functions

    def execute_visualization(
        self,
        concept_slug: str,
        function_name: str = 'main_visualization'
    ):
        """Execute visualization function and return figure

        Args:
            concept_slug: Concept identifier
            function_name: Visualization function name

        Returns:
            matplotlib Figure object

        Raises:
            FileNotFoundError: If visualize.py doesn't exist
            AttributeError: If function not found
        """
        viz_path = self.base_path / concept_slug / 'visualize.py'

        if not viz_path.exists():
            raise FileNotFoundError(f"No visualization for {concept_slug}")

        # Import and execute
        module = self._import_viz_module(concept_slug)

        if not hasattr(module, function_name):
            raise AttributeError(
                f"Function '{function_name}' not found in {concept_slug}/visualize.py"
            )

        viz_function = getattr(module, function_name)
        fig = viz_function()

        return fig

    def _import_viz_module(self, concept_slug: str):
        """Dynamically import visualize.py module

        Args:
            concept_slug: Concept identifier

        Returns:
            Imported module
        """
        viz_path = self.base_path / concept_slug / 'visualize.py'

        spec = importlib.util.spec_from_file_location(
            f"viz_{concept_slug}",
            viz_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module
```

**Step 5: Implement DisplayManager for output modes**

Create `learning_framework/visualization/display.py`:

```python
"""Display management for visualizations (browser, terminal, file)"""

import webbrowser
import subprocess
import shutil
from pathlib import Path
from typing import Optional


class DisplayManager:
    """Manages different visualization output modes"""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize display manager

        Args:
            output_dir: Directory for saved visualizations
        """
        if output_dir is None:
            output_dir = Path.cwd() / 'user_data' / 'visualizations'

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def show(self, fig, concept_slug: str, mode: str = 'browser') -> Optional[str]:
        """Display figure using specified mode

        Args:
            fig: Matplotlib figure
            concept_slug: Concept identifier (for filename)
            mode: 'browser', 'terminal', 'file', or 'interactive'

        Returns:
            Path to saved file (if applicable)
        """
        if mode == 'browser':
            return self._show_browser(fig, concept_slug)
        elif mode == 'terminal':
            return self._show_terminal(fig, concept_slug)
        elif mode == 'file':
            return self._save_file(fig, concept_slug)
        elif mode == 'interactive':
            return self._show_interactive(fig)
        else:
            raise ValueError(f"Unknown display mode: {mode}")

    def _show_browser(self, fig, concept_slug: str) -> str:
        """Save PNG and open in browser"""
        path = self.output_dir / f"{concept_slug}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')

        # Open in browser
        webbrowser.open(f"file://{path.absolute()}")

        return str(path)

    def _show_terminal(self, fig, concept_slug: str) -> str:
        """Display in terminal using imgcat or sixel"""
        path = self.output_dir / f"{concept_slug}.png"
        fig.savefig(path, dpi=100, bbox_inches='tight')

        # Try imgcat (iTerm2)
        if shutil.which('imgcat'):
            subprocess.run(['imgcat', str(path)])
        # Try img2sixel
        elif shutil.which('img2sixel'):
            subprocess.run(['img2sixel', str(path)])
        else:
            print(f"Terminal image display not available.")
            print(f"Saved to: {path}")
            print("Install imgcat (iTerm2) or img2sixel for terminal display")

        return str(path)

    def _save_file(self, fig, concept_slug: str) -> str:
        """Save to file without displaying"""
        path = self.output_dir / f"{concept_slug}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return str(path)

    def _show_interactive(self, fig) -> None:
        """Show interactive matplotlib window"""
        import matplotlib
        matplotlib.use('TkAgg')  # Switch to interactive backend
        import matplotlib.pyplot as plt

        plt.show()
        return None
```

**Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_visualization.py -v
```

Expected: PASS

**Step 7: Commit**

```bash
git add learning_framework/visualization/ tests/test_visualization.py
git commit -m "feat: add visualization engine with multiple output modes"
```

---

## Task 6: Create Visualizations for 5 Concepts

**Files:**
- Create: `data/gradient_descent/visualize.py`
- Create: `data/backpropagation/visualize.py`
- Create: `data/loss_functions/visualize.py`
- Create: `data/activation_functions/visualize.py`
- Create: `data/q_learning/visualize.py`
- Create: `data/templates/visualize.py.template`

**Step 1: Create Gradient Descent visualization**

Create `data/gradient_descent/visualize.py`:

```python
"""Visualizations for Gradient Descent concept"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main_visualization():
    """3D loss surface with gradient descent convergence path

    Shows how gradient descent navigates a convex loss surface
    to find the minimum.
    """
    fig = plt.figure(figsize=(14, 6))

    # Create 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')

    # Generate loss surface (simple quadratic bowl)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Convex function

    # Plot surface
    ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')

    # Gradient descent path (learning rate = 0.2)
    path_x = [4.0]
    path_y = [4.0]
    lr = 0.2

    for _ in range(10):
        x_curr = path_x[-1]
        y_curr = path_y[-1]
        # Gradient of x² + y² is (2x, 2y)
        grad_x = 2 * x_curr
        grad_y = 2 * y_curr
        # Update
        path_x.append(x_curr - lr * grad_x)
        path_y.append(y_curr - lr * grad_y)

    path_z = [x**2 + y**2 for x, y in zip(path_x, path_y)]

    ax1.plot(path_x, path_y, path_z, 'r-o', linewidth=2, markersize=6,
             label='GD Path (α=0.2)')

    ax1.set_xlabel('θ₁')
    ax1.set_ylabel('θ₂')
    ax1.set_zlabel('Loss J(θ)')
    ax1.set_title('Gradient Descent on 3D Loss Surface')
    ax1.legend()

    # 2D contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(path_x, path_y, 'r-o', linewidth=2, markersize=6,
             label='GD Path')
    ax2.scatter([0], [0], color='red', s=100, marker='*',
                label='Minimum', zorder=5)
    ax2.set_xlabel('θ₁')
    ax2.set_ylabel('θ₂')
    ax2.set_title('Contour View')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def learning_rate_comparison():
    """Compare convergence with different learning rates

    Demonstrates effect of learning rate on convergence speed.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    learning_rates = [0.01, 0.1, 0.5, 0.9]

    for ax, lr in zip(axes.flat, learning_rates):
        # Generate loss surface
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2

        # Gradient descent with this learning rate
        path_x = [4.0]
        path_y = [4.0]

        for _ in range(20):
            x_curr = path_x[-1]
            y_curr = path_y[-1]
            grad_x = 2 * x_curr
            grad_y = 2 * y_curr
            path_x.append(x_curr - lr * grad_x)
            path_y.append(y_curr - lr * grad_y)

            # Break if diverging
            if abs(path_x[-1]) > 10 or abs(path_y[-1]) > 10:
                break

        # Plot
        contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
        ax.plot(path_x, path_y, 'r-o', linewidth=1.5, markersize=4)
        ax.scatter([0], [0], color='red', s=100, marker='*', zorder=5)

        # Determine behavior
        if abs(path_x[-1]) > 10:
            behavior = "Diverges"
            color = 'red'
        elif len(path_x) < 20 and abs(path_x[-1]) < 0.01:
            behavior = "Fast convergence"
            color = 'green'
        else:
            behavior = "Slow convergence"
            color = 'orange'

        ax.set_title(f'α = {lr}: {behavior}', color=color, fontweight='bold')
        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Test standalone
    fig = main_visualization()
    plt.show()
```

**Step 2: Create Backpropagation visualization**

Create `data/backpropagation/visualize.py`:

```python
"""Visualizations for Backpropagation concept"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np


def main_visualization():
    """Computational graph showing forward and backward passes

    Demonstrates how information flows forward and gradients flow backward.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define node positions
    nodes = {
        'x': (1, 4),
        'w1': (1, 2),
        'z1': (3, 3),
        'σ': (5, 3),
        'a1': (7, 3),
        'w2': (7, 1),
        'z2': (9, 2),
        'ŷ': (11, 2),
        'y': (11, 4),
        'L': (13, 3)
    }

    # Draw nodes
    for name, (x, y) in nodes.items():
        circle = Circle((x, y), 0.3, color='lightblue', ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=12,
                fontweight='bold', zorder=4)

    # Forward pass arrows (green, solid)
    forward_edges = [
        ('x', 'z1', 'multiply'),
        ('w1', 'z1', 'multiply'),
        ('z1', 'σ', 'sigmoid'),
        ('σ', 'a1', ''),
        ('a1', 'z2', 'multiply'),
        ('w2', 'z2', 'multiply'),
        ('z2', 'ŷ', ''),
        ('ŷ', 'L', 'loss'),
        ('y', 'L', 'target'),
    ]

    for src, dst, label in forward_edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=20, linewidth=2,
            color='green', alpha=0.7, zorder=2
        )
        ax.add_patch(arrow)

        # Add label
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.3, label, fontsize=8,
                   color='darkgreen', ha='center')

    # Backward pass arrows (red, dashed)
    backward_edges = [
        ('L', 'ŷ', '∂L/∂ŷ'),
        ('ŷ', 'z2', '∂L/∂z2'),
        ('z2', 'w2', '∂L/∂w2'),
        ('z2', 'a1', '∂L/∂a1'),
        ('a1', 'σ', '∂L/∂σ'),
        ('σ', 'z1', '∂L/∂z1'),
        ('z1', 'w1', '∂L/∂w1'),
    ]

    for src, dst, label in backward_edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=15, linewidth=1.5,
            color='red', alpha=0.6, linestyle='--', zorder=1
        )
        ax.add_patch(arrow)

        # Add gradient label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y - 0.3, label, fontsize=7,
               color='darkred', ha='center', style='italic')

    # Legend
    ax.text(7, 6.5, 'Forward Pass (green, solid)', color='green',
           fontsize=12, fontweight='bold')
    ax.text(7, 6, 'Backward Pass (red, dashed)', color='red',
           fontsize=12, fontweight='bold')

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Backpropagation: Forward & Backward Passes',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def gradient_flow():
    """Gradient magnitude through layers (vanishing gradient demo)

    Compares gradient flow with sigmoid vs ReLU activation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = ['Output', 'Layer 4', 'Layer 3', 'Layer 2', 'Layer 1', 'Input']
    x = range(len(layers))

    # Sigmoid: gradients vanish exponentially
    sigmoid_grads = [1.0, 0.25, 0.0625, 0.0156, 0.0039, 0.00098]

    # ReLU: gradients stay relatively constant
    relu_grads = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7]

    # Plot
    ax.bar([i - 0.2 for i in x], sigmoid_grads, width=0.4,
           label='Sigmoid', color='orange', alpha=0.7)
    ax.bar([i + 0.2 for i in x], relu_grads, width=0.4,
           label='ReLU', color='green', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Gradient Flow Through Network Layers\n(Vanishing Gradient Problem)',
                fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation
    ax.annotate('Sigmoid gradients\nvanish rapidly!',
               xy=(4, 0.01), xytext=(3, 0.2),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=11, color='red', fontweight='bold')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = main_visualization()
    plt.show()
```

**Step 3: Create visualizations for remaining 3 concepts**

(Create similar visualization files for Loss Functions, Activation Functions, Q-Learning - abbreviated for space)

**Step 4: Create visualization template**

Create `data/templates/visualize.py.template`:

```python
"""
Visualization for [CONCEPT_NAME]

Functions:
- main_visualization(): Primary visualization (required)
"""

import numpy as np
import matplotlib.pyplot as plt


def main_visualization():
    """[Brief description of what this visualization shows]

    Explain what insight this provides for understanding the concept.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # TODO: Add your visualization code here

    ax.set_title('[CONCEPT_NAME] Visualization', fontsize=14, fontweight='bold')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Test standalone
    fig = main_visualization()
    plt.show()
```

**Step 5: Test visualizations**

```bash
cd learning-framework-dev
python data/gradient_descent/visualize.py
python data/backpropagation/visualize.py
```

Expected: Displays visualizations in matplotlib window

**Step 6: Commit**

```bash
git add data/*/visualize.py data/templates/
git commit -m "feat: add visualizations for 5 core concepts"
```

---

## Task 7: Enhanced CLI with Learn and Quiz Commands

**Files:**
- Modify: `learning_framework/cli.py`
- Create: `tests/test_cli_learn.py`

**Step 1: Write test for enhanced learn command**

Create `tests/test_cli_learn.py`:

```python
import pytest
from click.testing import CliRunner
from learning_framework.cli import cli


def test_learn_command_exists():
    """Test learn command is available"""
    runner = CliRunner()
    result = runner.invoke(cli, ['learn', '--help'])
    assert result.exit_code == 0
    assert 'Learning' in result.output or 'concept' in result.output


def test_quiz_command_with_concept():
    """Test quiz command accepts concept option"""
    runner = CliRunner()
    result = runner.invoke(cli, ['quiz', '--help'])
    assert result.exit_code == 0
    assert '--concept' in result.output or 'concept' in result.output
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_cli_learn.py -v
```

Expected: PASS (learn/quiz already exist as placeholders from Phase 1)

**Step 3: Enhance learn command**

Modify `learning_framework/cli.py`:

Add imports at top:

```python
from learning_framework.knowledge import load_concept, KnowledgeGraph, ConceptRegistry
from learning_framework.assessment import ConceptQuiz
from learning_framework.visualization import VisualizationRenderer
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
```

Replace placeholder `learn` command with:

```python
@cli.command()
@click.argument('concept', required=False)
@click.pass_context
def learn(ctx, concept):
    """Start interactive learning session for a concept

    If no concept specified, shows available concepts to choose from.
    """
    config = ctx.obj['config']

    if not concept:
        # Show available concepts
        registry = ConceptRegistry()
        topics = registry.get_topics()

        console.print("\n[bold cyan]Available Learning Topics[/bold cyan]\n")

        for topic in topics:
            concepts = registry.get_by_topic(topic)
            console.print(f"\n[bold]{topic.replace('_', ' ').title()}:[/bold]")

            for concept_slug in concepts:
                try:
                    c = load_concept(concept_slug)
                    status = "✓" if c.get('status') == 'complete' else "○"
                    console.print(f"  {status} {concept_slug} - {c['name']}")
                except:
                    console.print(f"  ○ {concept_slug}")

        console.print("\n[dim]Usage: lf learn <concept_slug>[/dim]")
        return

    # Load specific concept
    try:
        concept_data = load_concept(concept)
    except FileNotFoundError:
        console.print(f"[red]Concept not found: {concept}[/red]")
        return

    # Check prerequisites
    graph = KnowledgeGraph()
    prereq_check = graph.check_prerequisites(concept)
    graph.close()

    # Display concept info
    console.print(Panel(
        f"[bold]{concept_data['name']}[/bold]\n"
        f"Topic: {concept_data['topic']} | Difficulty: {concept_data['difficulty']}\n"
        f"{concept_data['description']}",
        title="Learning Session",
        border_style="cyan"
    ))

    # Show prerequisite warning if needed
    if prereq_check['warning']:
        console.print(f"\n[yellow]⚠️  {prereq_check['warning']}[/yellow]")
        if not click.confirm("Continue anyway?", default=True):
            return

    # Interactive menu
    while True:
        console.print("\n[bold]What would you like to do?[/bold]\n")
        console.print("  1. 📖 Read Explanation")
        console.print("  2. 🎨 View Visualization")
        console.print("  3. ❓ Take Quiz")
        console.print("  4. 📊 View Progress")
        console.print("  5. 🔙 Back\n")

        choice = click.prompt("Select option [1-5]", type=int)

        if choice == 1:
            _show_explanation(concept_data)
        elif choice == 2:
            _show_visualization(concept, config)
        elif choice == 3:
            _take_quiz(concept)
        elif choice == 4:
            _show_progress(concept)
        elif choice == 5:
            break
        else:
            console.print("[red]Invalid choice[/red]")


def _show_explanation(concept_data):
    """Show concept explanation"""
    console.print(f"\n[bold cyan]{concept_data['name']}[/bold cyan]\n")
    console.print(concept_data['description'])

    if concept_data.get('materials', {}).get('explanation'):
        console.print(f"\n[dim]See also: {concept_data['materials']['explanation']}[/dim]")


def _show_visualization(concept_slug, config):
    """Show concept visualization"""
    renderer = VisualizationRenderer()

    try:
        viz_functions = renderer.get_available_visualizations(concept_slug)

        if not viz_functions:
            console.print("[yellow]No visualizations available yet[/yellow]")
            return

        console.print("\n[bold]Available Visualizations:[/bold]\n")
        for i, func in enumerate(viz_functions, 1):
            console.print(f"  {i}. {func['name']}: {func['description']}")

        choice = click.prompt("\nSelect visualization", type=int, default=1)
        func_name = viz_functions[choice - 1]['name']

        # Render
        from learning_framework.visualization.display import DisplayManager
        display = DisplayManager()

        fig = renderer.execute_visualization(concept_slug, func_name)
        output_mode = config.get('visualization.output_mode', 'browser')
        path = display.show(fig, concept_slug, mode=output_mode)

        if path:
            console.print(f"[green]Visualization displayed: {path}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def _take_quiz(concept_slug):
    """Take concept quiz"""
    from learning_framework.assessment import ConceptQuiz, AnswerGrader

    try:
        quiz = ConceptQuiz(concept_slug)
        grader = AnswerGrader()

        console.print(f"\n[bold cyan]Quiz: {quiz.concept['name']}[/bold cyan]\n")

        questions = quiz.generate_quiz(num_questions=10, mix_types=True)

        if not questions:
            console.print("[yellow]No quiz questions available yet[/yellow]")
            quiz.close()
            return

        score = 0

        for i, q in enumerate(questions, 1):
            console.print(f"\n{'='*60}")
            console.print(f"Question {i}/{len(questions)}")
            console.print(f"{'='*60}\n")

            console.print(q['question'])

            if q['type'] == 'multiple_choice':
                for j, option in enumerate(q['options'], 1):
                    console.print(f"  {j}. {option}")

                answer = click.prompt("Your answer", type=int) - 1
                correct = grader.check_answer(q, answer)

            elif q['type'] == 'fill_blank':
                answer = click.prompt("Your answer", type=str)
                correct = grader.check_answer(q, answer)

            # Show result
            if correct:
                console.print("[green]✓ Correct![/green]")
                score += 1
            else:
                console.print("[red]✗ Incorrect[/red]")
                if 'explanation' in q:
                    console.print(f"[yellow]{q['explanation']}[/yellow]")

            # Update spaced repetition
            quiz.grade_answer(q['id'], answer, correct)

        # Final score
        percentage = (score / len(questions)) * 100
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Final Score: {score}/{len(questions)} ({percentage:.1f}%)[/bold]")

        # Update mastery if passed
        if percentage >= 80:
            quiz.db.update_concept_mastery(concept_slug, quiz_passed=True)
            console.print("\n[green]🎉 Quiz tier mastered![/green]")

        quiz.close()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


def _show_progress(concept_slug):
    """Show concept progress"""
    from learning_framework.progress import ProgressDatabase

    db = ProgressDatabase()
    mastery = db.get_concept(concept_slug)

    if not mastery:
        console.print("[yellow]No progress recorded yet[/yellow]")
        db.close()
        return

    table = Table(title=f"Progress: {concept_slug}")
    table.add_column("Tier", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    quiz_status = "✓ Passed" if mastery['quiz_passed'] else "○ Not Passed"
    quiz_details = f"Last: {mastery['last_reviewed']}" if mastery.get('last_reviewed') else "Never attempted"
    table.add_row("Quiz", quiz_status, quiz_details)

    impl_status = "✓ Passed" if mastery.get('implementation_passed') else "○ Not Passed"
    table.add_row("Implementation", impl_status, "Coming in Phase 3")

    gpu_status = "✓ Validated" if mastery.get('gpu_validated') else "○ Not Validated"
    table.add_row("GPU Scaling", gpu_status, "Coming in Phase 5")

    console.print(table)
    db.close()
```

**Step 4: Enhance quiz command for daily reviews**

Replace placeholder `quiz` command:

```python
@cli.command()
@click.option('--daily', is_flag=True, help='Daily review (spaced repetition)')
@click.option('--concept', help='Quiz on specific concept')
@click.pass_context
def quiz(ctx, daily, concept):
    """Take concept quiz or daily review"""

    from learning_framework.assessment import SpacedRepetitionScheduler

    if daily:
        # Daily review workflow
        scheduler = SpacedRepetitionScheduler()
        due_items = scheduler.get_due_items(tier='quiz')
        scheduler.close()

        if not due_items:
            console.print("[green]No reviews due today! 🎉[/green]")
            return

        console.print(f"\n[cyan]Daily Review - {len(due_items)} questions due[/cyan]\n")

        # Group by concept
        by_concept = {}
        for item in due_items:
            concept_slug = item['concept_slug']
            if concept_slug not in by_concept:
                by_concept[concept_slug] = []
            by_concept[concept_slug].append(item)

        # Quiz each concept
        for concept_slug, items in by_concept.items():
            console.print(f"\n{'='*60}")
            console.print(f"[bold]{concept_slug}[/bold] ({len(items)} questions)")
            console.print(f"{'='*60}\n")

            # Take quiz on these specific questions
            _take_quiz(concept_slug)

    elif concept:
        # Specific concept quiz
        _take_quiz(concept)

    else:
        console.print("[yellow]Use --daily for daily review or --concept <name> for specific concept[/yellow]")
```

**Step 5: Test enhanced CLI**

```bash
cd learning-framework-dev
python -m learning_framework.cli learn
python -m learning_framework.cli learn gradient_descent
python -m learning_framework.cli quiz --concept gradient_descent
```

Expected: Interactive menus work, quiz functions properly

**Step 6: Run tests**

```bash
python -m pytest tests/test_cli_learn.py -v
```

Expected: PASS

**Step 7: Commit**

```bash
git add learning_framework/cli.py tests/test_cli_learn.py
git commit -m "feat: enhance CLI with interactive learn and quiz commands"
```

---

## Task 8: Configuration Updates & Documentation

**Files:**
- Modify: `user_data/config-example.yaml`
- Modify: `README.md`
- Create: `docs/PHASE2_USAGE.md`

**Step 1: Update configuration example**

Modify `user_data/config-example.yaml`:

Add Phase 2 configuration:

```yaml
# Phase 1 config...
materials_directories:
  - "D:/ourock-test/ourock-test/DeepLearning"
  - "D:/ourock-test/ourock-test/RL"

daily_gpu_budget: 5.0
max_job_cost: 1.0
editor: "code"

# Phase 2: Visualization Settings
visualization:
  output_mode: "browser"        # browser, terminal, file, interactive
  save_to_disk: true            # Keep copies in user_data/visualizations/
  terminal_protocol: "auto"     # auto, imgcat, sixel, none

# Phase 2: Quiz Settings
quiz:
  questions_per_session: 10
  passing_score: 0.8            # 80% to pass quiz tier
  mix_question_types: true      # Mix MC + fill-in-blank
  show_explanations: true       # Show explanations for wrong answers

# Phase 2: Spaced Repetition
spaced_repetition:
  enabled: true
  quiz_intervals: [1, 3, 7, 14, 30]        # days
  implementation_intervals: [30, 60, 90]   # days (Phase 3)
  reset_on_fail: true           # Reset to day 1 on wrong answer

# Phase 2: Learning Settings
learning:
  show_prerequisite_warnings: true
  allow_skip_prerequisites: true  # Soft warnings only
```

**Step 2: Update README**

Modify `README.md`:

Update Phase 2 section:

```markdown
## Phase 2 Complete ✓

- [x] Knowledge graph with concept auto-discovery
- [x] Quiz system (MC + fill-in-blank)
- [x] Spaced repetition (SM-2 algorithm)
- [x] Visualization engine (matplotlib)
- [x] 5 complete concepts with 75 quiz questions
- [x] Interactive learning flow

## Quick Start

### 1. Learn a Concept

```bash
lf learn                    # Show available concepts
lf learn gradient_descent   # Interactive learning session
```

### 2. Take Quizzes

```bash
lf quiz --concept gradient_descent  # Quiz on specific concept
lf quiz --daily                     # Daily review (spaced repetition)
```

### 3. View Visualizations

Visualizations are shown automatically during learning sessions,
or can be triggered directly:

```bash
lf visualize gradient_descent
```

### 4. Track Progress

```bash
lf progress                # Overall progress
lf progress --topic neural_networks
```

## Available Concepts

**Complete (with quizzes + visualizations):**
- Gradient Descent
- Backpropagation
- Loss Functions
- Activation Functions
- Q-Learning

**Coming Soon:** Phase 3 Implementation Challenges
```

**Step 3: Create Phase 2 usage guide**

Create `docs/PHASE2_USAGE.md`:

```markdown
# Phase 2 Usage Guide

Complete guide to using the Learning & Assessment system.

## Learning Workflow

### 1. Explore Available Concepts

```bash
lf learn
```

Shows all concepts grouped by topic, with status indicators:
- ✓ Complete (full quiz + visualization)
- ○ Skeleton (detected but no content yet)

### 2. Start Learning Session

```bash
lf learn gradient_descent
```

Interactive menu offers:
1. Read Explanation - View concept description
2. View Visualization - Interactive visualizations
3. Take Quiz - 10-question quiz (MC + fill-in-blank)
4. View Progress - See mastery status
5. Back - Return to concept selection

### 3. Take Quiz

Within learning session, select option 3, or directly:

```bash
lf quiz --concept gradient_descent
```

- 10 questions (7 MC, 3 fill-in-blank)
- Immediate feedback with explanations
- Pass threshold: 80%
- Auto-updates spaced repetition schedule

### 4. Daily Review

```bash
lf quiz --daily
```

Reviews questions due today based on spaced repetition:
- Correct answer → interval increases (1d → 3d → 7d → 14d → 30d)
- Wrong answer → reset to 1 day

## Visualizations

Each concept has custom visualizations:

**Gradient Descent:**
- 3D loss surface with convergence path
- Learning rate comparison

**Backpropagation:**
- Computational graph (forward + backward passes)
- Gradient flow through layers

**Loss Functions:**
- 3D loss landscapes
- Contour plots

**Activation Functions:**
- Function shape comparisons
- Derivative plots

**Q-Learning:**
- Q-value heatmaps
- Policy visualization

### Output Modes

Configure in `user_data/config.yaml`:

```yaml
visualization:
  output_mode: "browser"  # Options: browser, terminal, file, interactive
```

- **browser**: Save PNG and auto-open in browser
- **terminal**: Display in terminal (requires imgcat or sixel)
- **file**: Save to user_data/visualizations/
- **interactive**: Matplotlib window

## Progress Tracking

### Three-Tier Mastery System

Each concept has three mastery tiers:

1. **Quiz** (Phase 2) - Pass quiz with ≥80%
2. **Implementation** (Phase 3) - Implement algorithm correctly
3. **GPU Scaling** (Phase 5) - Validate on real-world dataset

### View Progress

```bash
lf progress                          # Overall statistics
lf progress --topic neural_networks  # Topic-specific
```

Shows:
- Concepts mastered per tier
- Next review dates
- Quiz statistics

## Spaced Repetition Schedule

Automatic review scheduling using SM-2 algorithm:

| Attempt | Result    | Next Review |
|---------|-----------|-------------|
| 1       | Correct   | 1 day       |
| 2       | Correct   | 3 days      |
| 3       | Correct   | 7 days      |
| 4       | Incorrect | 1 day (reset)|
| 5       | Correct   | 3 days      |

## Adding Your Own Content

### Expand Skeleton Concepts

1. Navigate to `data/<concept_slug>/`
2. Edit `quiz_mc.json` - Add multiple choice questions
3. Edit `quiz_fillblank.json` - Add fill-in-blank questions
4. Create `visualize.py` - Add visualization functions

Use templates in `data/templates/` as starting point.

### Question Format

**Multiple Choice:**

```json
{
  "id": "unique_id",
  "type": "multiple_choice",
  "question": "Your question?",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_index": 1,
  "explanation": "Why B is correct...",
  "difficulty": 2,
  "tags": ["topic", "subtopic"]
}
```

**Fill-in-Blank:**

```json
{
  "id": "unique_id",
  "type": "fill_blank",
  "question": "The formula is: x = ___",
  "answer": "correct answer",
  "alternatives": ["also correct", "alternative spelling"],
  "explanation": "Explanation...",
  "difficulty": 1,
  "tags": ["equations"]
}
```

## Troubleshooting

**Quiz shows "No questions available":**
- Check that `quiz_mc.json` and `quiz_fillblank.json` exist
- Verify JSON syntax is valid

**Visualization fails:**
- Ensure matplotlib is installed: `pip install matplotlib`
- Check `visualize.py` has no syntax errors
- Test standalone: `python data/<concept>/visualize.py`

**Daily review shows nothing:**
- Take some quizzes first to populate review schedule
- Check `user_data/progress.db` exists

## Configuration

Edit `user_data/config.yaml`:

```yaml
quiz:
  questions_per_session: 10      # Number of questions per quiz
  passing_score: 0.8             # 80% to pass
  mix_question_types: true       # Mix MC + fill-blank

spaced_repetition:
  enabled: true
  quiz_intervals: [1, 3, 7, 14, 30]  # Days between reviews

learning:
  show_prerequisite_warnings: true
  allow_skip_prerequisites: true     # Soft warnings only
```

## Next Steps

Phase 3 will add:
- Implementation challenges (fill-in-blank → from-scratch → debug)
- Automated test runner
- C++ implementations

Phase 5 will add:
- GPU scaling validation
- Remote job submission
- Cost tracking
```

**Step 4: Commit**

```bash
git add user_data/config-example.yaml README.md docs/PHASE2_USAGE.md
git commit -m "docs: update configuration and add Phase 2 usage guide"
```

---

## Task 9: Integration Testing & Final Verification

**Files:**
- Create: `tests/test_phase2_integration.py`

**Step 1: Write comprehensive integration test**

Create `tests/test_phase2_integration.py`:

```python
"""Integration tests for Phase 2 learning system"""

import pytest
import tempfile
import json
from pathlib import Path
from learning_framework.knowledge import ConceptRegistry, load_concept, KnowledgeGraph
from learning_framework.assessment import ConceptQuiz, SpacedRepetitionScheduler
from learning_framework.visualization import VisualizationRenderer


def test_complete_learning_workflow():
    """Test complete workflow: concept load → quiz → spaced repetition"""

    # Setup: Create test concept
    with tempfile.TemporaryDirectory() as tmpdir:
        concept_dir = Path(tmpdir) / 'data' / 'test_concept'
        concept_dir.mkdir(parents=True)

        # Concept metadata
        concept_data = {
            'name': 'Test Concept',
            'slug': 'test_concept',
            'topic': 'testing',
            'difficulty': 'easy',
            'status': 'complete',
            'prerequisites': []
        }
        (concept_dir / 'concept.json').write_text(json.dumps(concept_data))

        # Quiz questions
        quiz_mc = {
            'questions': [
                {
                    'id': 'q1',
                    'type': 'multiple_choice',
                    'question': 'Test?',
                    'options': ['A', 'B'],
                    'correct_index': 0,
                    'explanation': 'Because A'
                }
            ]
        }
        (concept_dir / 'quiz_mc.json').write_text(json.dumps(quiz_mc))

        quiz_fb = {'questions': []}
        (concept_dir / 'quiz_fillblank.json').write_text(json.dumps(quiz_fb))

        db_path = Path(tmpdir) / 'progress.db'

        # Step 1: Load concept
        concept = load_concept('test_concept', Path(tmpdir) / 'data')
        assert concept['name'] == 'Test Concept'

        # Step 2: Take quiz
        quiz = ConceptQuiz('test_concept',
                          base_path=Path(tmpdir) / 'data',
                          db_path=db_path)

        questions = quiz.generate_quiz(num_questions=1)
        assert len(questions) == 1

        # Simulate answering correctly
        quiz.grade_answer('q1', 0, correct=True)

        # Step 3: Check spaced repetition scheduled
        scheduler = SpacedRepetitionScheduler(db_path)
        due_items = scheduler.get_due_items()

        # Should not be due today (next review is future)
        assert len(due_items) == 0

        quiz.close()
        scheduler.close()


def test_prerequisite_checking():
    """Test knowledge graph prerequisite checking"""

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / 'data'

        # Create two concepts: A (no prereqs) and B (requires A)
        for slug, prereqs in [('concept_a', []), ('concept_b', ['concept_a'])]:
            concept_dir = data_path / slug
            concept_dir.mkdir(parents=True)

            concept_data = {
                'name': slug.replace('_', ' ').title(),
                'slug': slug,
                'topic': 'test',
                'difficulty': 'easy',
                'prerequisites': prereqs
            }
            (concept_dir / 'concept.json').write_text(json.dumps(concept_data))

        db_path = Path(tmpdir) / 'progress.db'

        # Check prerequisites
        graph = KnowledgeGraph(db_path)

        # B requires A (not mastered yet)
        check = graph.check_prerequisites('concept_b')
        assert check['ready'] == False
        assert 'concept_a' in check['missing']
        assert check['warning'] is not None

        # Now mark A as mastered
        from learning_framework.progress import ProgressDatabase
        db = ProgressDatabase(db_path)
        db.add_concept('concept_a', 'test', 'easy')
        db.update_concept_mastery('concept_a', quiz_passed=True)
        db.close()

        # Check again
        check = graph.check_prerequisites('concept_b')
        assert check['ready'] == True
        assert len(check['missing']) == 0

        graph.close()


def test_visualization_system():
    """Test visualization rendering pipeline"""

    with tempfile.TemporaryDirectory() as tmpdir:
        viz_dir = Path(tmpdir) / 'data' / 'test_viz'
        viz_dir.mkdir(parents=True)

        # Create visualization
        viz_code = '''
import matplotlib.pyplot as plt

def test_plot():
    """Test visualization"""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    return fig
'''
        (viz_dir / 'visualize.py').write_text(viz_code)

        # Render
        renderer = VisualizationRenderer(Path(tmpdir) / 'data')

        # Discover functions
        functions = renderer.get_available_visualizations('test_viz')
        assert len(functions) == 1
        assert functions[0]['name'] == 'test_plot'

        # Execute
        fig = renderer.execute_visualization('test_viz', 'test_plot')
        assert fig is not None


def test_quiz_mixed_question_types():
    """Test quiz generates mix of MC and fill-blank"""

    with tempfile.TemporaryDirectory() as tmpdir:
        concept_dir = Path(tmpdir) / 'data' / 'mixed_quiz'
        concept_dir.mkdir(parents=True)

        concept_data = {
            'name': 'Mixed Quiz',
            'slug': 'mixed_quiz',
            'topic': 'test',
            'difficulty': 'easy',
            'prerequisites': []
        }
        (concept_dir / 'concept.json').write_text(json.dumps(concept_data))

        # Create 10 MC questions
        mc_questions = [
            {'id': f'mc_{i}', 'type': 'multiple_choice',
             'question': f'Q{i}', 'options': ['A', 'B'],
             'correct_index': 0, 'explanation': 'Test'}
            for i in range(10)
        ]
        (concept_dir / 'quiz_mc.json').write_text(
            json.dumps({'questions': mc_questions}))

        # Create 5 fill-blank questions
        fb_questions = [
            {'id': f'fb_{i}', 'type': 'fill_blank',
             'question': f'Fill {i}', 'answer': 'test',
             'alternatives': [], 'explanation': 'Test'}
            for i in range(5)
        ]
        (concept_dir / 'quiz_fillblank.json').write_text(
            json.dumps({'questions': fb_questions}))

        # Generate quiz
        quiz = ConceptQuiz('mixed_quiz',
                          base_path=Path(tmpdir) / 'data',
                          db_path=Path(tmpdir) / 'test.db')

        questions = quiz.generate_quiz(num_questions=10, mix_types=True)

        assert len(questions) == 10

        # Should have both types
        mc_count = sum(1 for q in questions if q['type'] == 'multiple_choice')
        fb_count = sum(1 for q in questions if q['type'] == 'fill_blank')

        assert mc_count > 0
        assert fb_count > 0

        quiz.close()
```

**Step 2: Run all Phase 2 tests**

```bash
python -m pytest tests/test_spaced_repetition.py tests/test_concepts.py tests/test_quiz.py tests/test_visualization.py tests/test_phase2_integration.py -v
```

Expected: All tests pass

**Step 3: Run full test suite (Phase 1 + Phase 2)**

```bash
python -m pytest -v
```

Expected: All tests pass

**Step 4: Manual testing checklist**

```bash
# Test learn command
python -m learning_framework.cli learn

# Test specific concept
python -m learning_framework.cli learn gradient_descent

# Test quiz
python -m learning_framework.cli quiz --concept gradient_descent

# Test visualization
python -m learning_framework.cli learn gradient_descent
# Select option 2 (View Visualization)

# Test progress tracking
python -m learning_framework.cli progress
```

**Step 5: Commit**

```bash
git add tests/test_phase2_integration.py
git commit -m "test: add comprehensive Phase 2 integration tests"
```

---

## Phase 2 Completion Checklist

Verify all tasks complete:

```bash
# 1. All tests pass
python -m pytest -v

# 2. All CLI commands work
python -m learning_framework.cli learn
python -m learning_framework.cli learn gradient_descent
python -m learning_framework.cli quiz --concept gradient_descent
python -m learning_framework.cli quiz --daily
python -m learning_framework.cli progress

# 3. Data structure complete
ls data/  # Should show 5 complete + templates
ls data/gradient_descent/  # Should have concept.json, quiz files, visualize.py

# 4. Documentation complete
cat README.md
cat docs/PHASE2_USAGE.md

# 5. Git history clean
git log --oneline
```

**Expected Results:**
- ✓ All tests passing (25+ tests)
- ✓ All CLI commands functional
- ✓ 5 complete concepts with 75 quiz questions
- ✓ Visualizations working
- ✓ Spaced repetition scheduling correctly
- ✓ Documentation complete

---

## Final Commit & Tag

```bash
git add -A
git commit -m "feat: Phase 2 complete - Learning & Assessment system"
git tag -a v0.2.0-phase2 -m "Phase 2: Learning & Assessment Complete"
```

---

## Execution Notes

**Estimated Time:** 15-23 hours total

**Task Breakdown:**
1. Spaced Repetition: 1-2 hours
2. Concept Loading & Graph: 2-3 hours
3. Quiz System: 2-3 hours
4. Create 5 Concepts: 4-6 hours (content writing)
5. Visualization Engine: 2-3 hours
6. Create Visualizations: 3-4 hours
7. Enhanced CLI: 2-3 hours
8. Configuration & Docs: 1-2 hours
9. Testing: 2-3 hours

**Key Principles Applied:**
- **TDD**: All features test-driven
- **DRY**: Reusable SpacedRepetitionScheduler
- **YAGNI**: Only Phase 2 features
- **Frequent commits**: 9 major commits

**Dependencies:**
- matplotlib (visualization)
- numpy (numerical operations)
- All Phase 1 dependencies

**Files Created:** 50+
**Tests Written:** 25+
**Git Commits:** 9 major commits
**Quiz Questions:** 75 (5 concepts × 15 questions each)
**Visualizations:** 10+ (2 per concept)
