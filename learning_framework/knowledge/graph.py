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
