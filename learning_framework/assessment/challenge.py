"""Challenge management for implementation exercises"""

import ast
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional


class ChallengeManager:
    """Manager for loading and organizing implementation challenges

    Challenges are Python files with naming pattern: {name}_{type}.py
    where type is one of: 'fill', 'scratch', 'debug'

    - fill: Fill in missing implementation (TODOs)
    - scratch: Implement from scratch (minimal starter code)
    - debug: Fix broken implementation
    """

    CHALLENGE_TYPES = ['fill', 'scratch', 'debug']

    def __init__(self, challenges_path: Optional[Path] = None):
        """Initialize challenge manager

        Args:
            challenges_path: Directory containing challenge files.
                           Defaults to data/challenges/
        """
        if challenges_path is None:
            challenges_path = Path.cwd() / 'data' / 'challenges'

        self.challenges_path = Path(challenges_path)

    def load_challenge(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a challenge file and extract metadata

        Args:
            filename: Challenge filename (e.g., 'linear_layer_fill.py')

        Returns:
            Challenge dictionary with keys:
                - name: Challenge name (without type suffix)
                - type: Challenge type ('fill', 'scratch', 'debug')
                - description: First docstring in file
                - file_path: Full path to challenge file
            Returns None if file doesn't exist.
        """
        file_path = self.challenges_path / filename

        if not file_path.exists():
            return None

        # Extract name and type from filename
        name, type_ext = self._parse_filename(filename)
        challenge_type = self._extract_type(type_ext)

        # Extract description from first docstring
        description = self._extract_description(file_path)

        return {
            'name': name,
            'type': challenge_type,
            'description': description,
            'file_path': file_path
        }

    def get_challenge_type(self, filename: str) -> Optional[str]:
        """Detect challenge type from filename

        Args:
            filename: Challenge filename

        Returns:
            Challenge type ('fill', 'scratch', 'debug') or None
        """
        _, type_ext = self._parse_filename(filename)
        return self._extract_type(type_ext)

    def list_challenges(self) -> List[Dict[str, Any]]:
        """List all available challenges

        Returns:
            List of challenge dictionaries (see load_challenge for format)
        """
        if not self.challenges_path.exists():
            return []

        challenges = []
        for file_path in self.challenges_path.glob('*.py'):
            challenge = self.load_challenge(file_path.name)
            if challenge:
                challenges.append(challenge)

        return challenges

    def get_challenges_by_type(self, challenge_type: str) -> List[Dict[str, Any]]:
        """Get challenges filtered by type

        Args:
            challenge_type: Type to filter by ('fill', 'scratch', 'debug')

        Returns:
            List of challenge dictionaries matching the type
        """
        all_challenges = self.list_challenges()
        return [c for c in all_challenges if c['type'] == challenge_type]

    def copy_to_workspace(
        self,
        filename: str,
        workspace_path: Path
    ) -> Path:
        """Copy challenge file to workspace directory

        Args:
            filename: Challenge filename
            workspace_path: Target workspace directory

        Returns:
            Path to copied file in workspace
        """
        source_path = self.challenges_path / filename
        target_path = workspace_path / filename

        shutil.copy2(source_path, target_path)
        return target_path

    def _parse_filename(self, filename: str) -> tuple[str, str]:
        """Parse filename into name and type_ext

        Args:
            filename: Challenge filename (e.g., 'linear_layer_fill.py')

        Returns:
            Tuple of (name, type_ext) e.g., ('linear_layer', 'fill')
        """
        # Remove .py extension
        name_with_type = filename.rsplit('.', 1)[0]

        # Split by last underscore to separate name and type
        parts = name_with_type.rsplit('_', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return name_with_type, ''

    def _extract_type(self, type_str: str) -> str:
        """Extract challenge type from type string

        Args:
            type_str: Type string from filename

        Returns:
            Challenge type if valid, empty string otherwise
        """
        if type_str in self.CHALLENGE_TYPES:
            return type_str
        return ''

    def _extract_description(self, file_path: Path) -> str:
        """Extract description from first docstring in file

        Args:
            file_path: Path to Python file

        Returns:
            First docstring content, or empty string if none found
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)

            # Get first docstring (module or first element)
            if (tree.body and
                isinstance(tree.body[0], ast.Expr) and
                isinstance(tree.body[0].value, ast.Constant) and
                isinstance(tree.body[0].value.value, str)):
                return tree.body[0].value.value.strip()

            return ''
        except Exception:
            return ''
