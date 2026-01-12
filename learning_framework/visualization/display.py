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
