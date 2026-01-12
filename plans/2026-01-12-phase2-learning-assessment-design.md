# Phase 2: Learning & Assessment System - Design Document

**Date:** 2026-01-12
**Author:** LH
**Status:** Ready for Implementation
**Depends on:** Phase 1 (Core Infrastructure) ✓

---

## Executive Summary

Phase 2 builds the learning and assessment system on top of Phase 1's infrastructure. It creates an interactive learning experience with quiz-based assessment, spaced repetition, and concept visualizations. The system ships with 5 fully-implemented concepts and 25+ skeleton concept definitions auto-discovered from existing materials.

### Core Objectives

1. **Knowledge Graph** - Build prerequisite relationships among ~30 detected concepts
2. **Quiz System** - Multiple choice + fill-in-blank questions with spaced repetition
3. **Spaced Repetition** - SM-2 algorithm extracted from vocab quiz for reusability
4. **Static Visualizations** - Custom matplotlib visualizations per concept
5. **Interactive Learning Flow** - Menu-driven interface for flexible learning

---

## System Architecture

### Component Overview

Phase 2 adds three major subsystems:

**1. Knowledge Graph & Concept Registry**
- Auto-discovers ~30 concepts from existing DeepLearning/ and RL/ materials
- Creates skeleton `concept.json` files for all detected concepts
- Fully implements 5 priority concepts with complete quiz content
- Soft prerequisite warnings (doesn't block access)
- Master registry in `data/concepts.json`

**2. Quiz System with Spaced Repetition**
- Extracted `SpacedRepetitionScheduler` from vocab quiz (SM-2 algorithm)
- Two question types: Multiple choice + Fill-in-blank
- Tier-based review intervals:
  - Quiz: 1 → 3 → 7 → 14 → 30 days
  - Implementation: 30 → 60 → 90 days (monthly reviews)
  - GPU: One-time validation (no repetition)
- 5-10 seed questions per fully-implemented concept
- Question templates for easy expansion

**3. Static Visualization Engine**
- Matplotlib-based visualizations tailored per concept
- Multiple output modes: browser, terminal, file
- Custom visualization scripts per concept
- Examples: loss surfaces, computational graphs, activation shapes

### Integration with Phase 1

- Uses `ProgressDatabase` to track quiz results and review schedules
- Uses `MaterialIndexer` to detect concepts and find code examples
- Uses `ConfigManager` for visualization and quiz preferences
- Extends CLI with enhanced `lf learn` and `lf quiz` commands

---

## Detailed Component Design

### Component 1: Knowledge Graph & Concept Registry

**Auto-Discovery Process:**

When `lf index` runs:
1. Scans DeepLearning/ and RL/ folders (Phase 1 indexer)
2. Detects concepts from keywords, chapter titles, file names
3. Auto-generates skeleton `concept.json` for each detected concept
4. Categorizes by topic (neural_networks, deep_learning, reinforcement_learning)
5. Infers prerequisites based on chapter order and common patterns
6. Updates master registry `data/concepts.json`

**Concept Directory Structure:**

```
data/
├── concepts.json                    # Master registry (~30 concepts)
│
├── gradient_descent/                # Fully implemented (1/5)
│   ├── concept.json                # Complete metadata
│   ├── quiz_mc.json                # 10 multiple choice questions
│   ├── quiz_fillblank.json         # 5 fill-in-blank questions
│   ├── visualize.py                # Loss surface + convergence path
│   └── notes.md                    # User's personal notes (optional)
│
├── backpropagation/                 # Fully implemented (2/5)
│   ├── concept.json
│   ├── quiz_mc.json
│   ├── quiz_fillblank.json
│   ├── visualize.py                # Computational graph
│   └── notes.md
│
├── loss_functions/                  # Fully implemented (3/5)
├── activation_functions/            # Fully implemented (4/5)
├── q_learning/                      # Fully implemented (5/5)
│
├── batch_normalization/             # Skeleton (auto-detected)
│   ├── concept.json                # Basic metadata only
│   └── quiz_template.json          # Empty template for user
│
├── [25+ more skeleton concepts...]
│
└── templates/
    ├── concept.json.template
    ├── quiz_mc.json.template
    ├── quiz_fillblank.json.template
    └── visualize.py.template
```

**Concept Metadata Schema (concept.json):**

```json
{
  "name": "Backpropagation",
  "slug": "backpropagation",
  "topic": "neural_networks",
  "difficulty": "intermediate",
  "status": "complete",
  "prerequisites": ["gradient_descent", "chain_rule"],
  "description": "Algorithm for computing gradients using reverse-mode autodiff",
  "materials": {
    "explanation": "materials/DeepLearning/深度学习入门/ch05/README.md",
    "code_examples": [
      "materials/DeepLearning/深度学习入门/ch05/train_neuralnet.py",
      "materials/DeepLearning/dezero自制框架/step10.py"
    ]
  },
  "estimated_time_minutes": 45,
  "tags": ["fundamentals", "optimization", "gradients"]
}
```

**Master Registry (concepts.json):**

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
    "batch_normalization": {
      "status": "skeleton",
      "topic": "deep_learning",
      "difficulty": "intermediate"
    }
  },
  "topics": {
    "neural_networks": ["gradient_descent", "backpropagation", "activation_functions"],
    "deep_learning": ["cnns", "rnns", "batch_normalization"],
    "reinforcement_learning": ["q_learning", "policy_gradients", "bellman_equation"]
  }
}
```

**Prerequisite Checking:**

```python
class KnowledgeGraph:
    def check_prerequisites(self, concept_slug):
        """Check if prerequisites are mastered

        Returns:
            {
                'ready': bool,
                'missing': [list of unmastered prerequisites],
                'warning': str (if soft warning needed)
            }
        """
        concept = load_concept(concept_slug)
        missing = []

        for prereq_slug in concept['prerequisites']:
            prereq_mastery = progress_db.get_concept(prereq_slug)
            if not prereq_mastery or not prereq_mastery['quiz_passed']:
                missing.append(prereq_slug)

        return {
            'ready': len(missing) == 0,
            'missing': missing,
            'warning': f"Recommended: {', '.join(missing)}" if missing else None
        }
```

**5 Fully-Implemented Concepts:**

1. **Gradient Descent**
   - Topic: Optimization
   - Prerequisites: None (foundational)
   - Visualizations: 3D loss surface, convergence path
   - Quiz: 10 MC + 5 fill-in-blank

2. **Backpropagation**
   - Topic: Neural Networks
   - Prerequisites: Gradient Descent
   - Visualizations: Computational graph, gradient flow
   - Quiz: 10 MC + 5 fill-in-blank

3. **Loss Functions**
   - Topic: Neural Networks
   - Prerequisites: None
   - Visualizations: 3D loss landscapes (MSE, cross-entropy)
   - Quiz: 10 MC + 5 fill-in-blank

4. **Activation Functions**
   - Topic: Neural Networks
   - Prerequisites: None
   - Visualizations: Function shapes, derivative comparisons
   - Quiz: 10 MC + 5 fill-in-blank

5. **Q-Learning**
   - Topic: Reinforcement Learning
   - Prerequisites: None
   - Visualizations: Q-value heatmap, policy arrows
   - Quiz: 10 MC + 5 fill-in-blank

---

### Component 2: Quiz System with Spaced Repetition

**Shared Spaced Repetition Module:**

Extracted from vocab quiz system into reusable component.

**File: `learning_framework/assessment/spaced_repetition.py`**

```python
class SpacedRepetitionScheduler:
    """SM-2 algorithm for optimal review scheduling

    Works for both German vocab AND ML concepts
    """

    def __init__(self, db_path=None):
        self.db = ProgressDatabase(db_path)

    def calculate_next_review(self, item_id, correct, current_interval=1):
        """Calculate next review date using SM-2

        Args:
            item_id: Question ID or concept ID
            correct: Whether answer was correct
            current_interval: Current interval in days

        Returns:
            (next_review_date, new_interval_days)

        SM-2 Algorithm:
        - Correct: interval *= 2.5 (1 → 3 → 7 → 14 → 30...)
        - Incorrect: reset to 1 day
        """
        if correct:
            new_interval = int(current_interval * 2.5)
            next_review = datetime.now() + timedelta(days=new_interval)
        else:
            new_interval = 1
            next_review = datetime.now() + timedelta(days=1)

        return next_review, new_interval

    def get_due_items(self, concept=None, tier='quiz'):
        """Get items due for review today

        Args:
            concept: Filter by concept slug (optional)
            tier: 'quiz' or 'implementation'

        Returns:
            List of due question/implementation IDs
        """
        query = """
            SELECT qr.*, c.slug as concept_slug
            FROM quiz_results qr
            JOIN concepts c ON qr.concept_id = c.id
            WHERE qr.next_review <= DATE('now')
        """

        if concept:
            query += " AND c.slug = ?"
            params = (concept,)
        else:
            params = ()

        cursor = self.db.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()
```

**Quiz Question Types:**

**Multiple Choice Questions (quiz_mc.json):**

```json
{
  "version": "1.0",
  "questions": [
    {
      "id": "backprop_mc_001",
      "type": "multiple_choice",
      "question": "Why does vanishing gradient happen with sigmoid activation?",
      "options": [
        "Sigmoid outputs are between 0 and 1",
        "Sigmoid derivative maximum is 0.25",
        "Sigmoid is not differentiable at 0",
        "Sigmoid grows exponentially"
      ],
      "correct_index": 1,
      "explanation": "Sigmoid derivative max is 0.25. When backpropagating through many layers, gradients are multiplied: 0.25 × 0.25 × ... → vanishes to near zero.",
      "difficulty": 2,
      "tags": ["gradients", "activations", "vanishing_gradient"]
    },
    {
      "id": "backprop_mc_002",
      "question": "What does the chain rule allow us to do in backpropagation?",
      "options": [
        "Calculate loss faster",
        "Decompose complex derivatives into simpler parts",
        "Avoid computing gradients",
        "Make networks deeper"
      ],
      "correct_index": 1,
      "explanation": "Chain rule: dL/dw = dL/dy × dy/dx × dx/dw. We can compute each part separately and multiply them.",
      "difficulty": 2,
      "tags": ["chain_rule", "calculus"]
    }
  ]
}
```

**Fill-in-Blank Questions (quiz_fillblank.json):**

```json
{
  "version": "1.0",
  "questions": [
    {
      "id": "backprop_fb_001",
      "type": "fill_blank",
      "question": "The gradient descent update rule is: θ = θ - α × ___",
      "answer": "∇J(θ)",
      "alternatives": ["gradient", "∇J", "dJ/dθ", "grad J"],
      "explanation": "Gradient ∇J(θ) points in direction of steepest increase, so we subtract it (with learning rate α) to minimize loss.",
      "difficulty": 1,
      "tags": ["optimization", "equations", "gradient_descent"]
    },
    {
      "id": "backprop_fb_002",
      "question": "In backpropagation, gradients flow ___ through the network (forward/backward).",
      "answer": "backward",
      "alternatives": ["backwards", "in reverse"],
      "explanation": "Forward pass computes outputs. Backward pass computes gradients from output to input.",
      "difficulty": 1,
      "tags": ["backpropagation", "direction"]
    }
  ]
}
```

**Quiz Engine (`learning_framework/assessment/quiz.py`):**

```python
class ConceptQuiz:
    """Quiz engine for ML concepts"""

    def __init__(self, concept_slug):
        self.concept = load_concept(concept_slug)
        self.mc_questions = load_json(f"data/{concept_slug}/quiz_mc.json")
        self.fb_questions = load_json(f"data/{concept_slug}/quiz_fillblank.json")
        self.scheduler = SpacedRepetitionScheduler()

    def generate_quiz(self, num_questions=10, mix_types=True):
        """Generate quiz with mixed question types

        Args:
            num_questions: Total questions
            mix_types: Mix MC and fill-in-blank

        Returns:
            List of question dicts
        """
        questions = []

        # Get due reviews first
        due_items = self.scheduler.get_due_items(
            concept=self.concept['slug'],
            tier='quiz'
        )

        # Mix: 70% MC, 30% fill-in-blank
        if mix_types:
            num_mc = int(num_questions * 0.7)
            num_fb = num_questions - num_mc

            mc_pool = self.mc_questions['questions']
            fb_pool = self.fb_questions['questions']

            # Prioritize due reviews
            mc_selected = self._select_questions(mc_pool, num_mc, due_items)
            fb_selected = self._select_questions(fb_pool, num_fb, due_items)

            questions = mc_selected + fb_selected
        else:
            questions = random.sample(self.mc_questions['questions'], num_questions)

        random.shuffle(questions)
        return questions

    def grade_answer(self, question_id, user_answer, correct):
        """Grade answer and update spaced repetition schedule

        Args:
            question_id: Question identifier
            user_answer: User's answer
            correct: Whether answer was correct
        """
        # Get current interval from database
        current_interval = self._get_interval(question_id) or 1

        # Calculate next review
        next_review, new_interval = self.scheduler.calculate_next_review(
            question_id, correct, current_interval
        )

        # Update database
        self.db.conn.execute("""
            INSERT INTO quiz_results (
                concept_id, question_id, correct, timestamp,
                next_review, review_interval
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            self.concept['id'], question_id, int(correct),
            datetime.now(), next_review, new_interval
        ))
        self.db.conn.commit()

    def check_fill_blank(self, question, user_answer):
        """Check fill-in-blank answer with fuzzy matching

        Args:
            question: Question dict
            user_answer: User's typed answer

        Returns:
            bool: Whether answer is correct
        """
        user_clean = user_answer.strip().lower()

        # Check exact answer
        if user_clean == question['answer'].lower():
            return True

        # Check alternatives
        for alt in question.get('alternatives', []):
            if user_clean == alt.lower():
                return True

        return False
```

**Tier-Based Review Schedule:**

| Tier | Initial Interval | Success Multiplier | Frequency | Notes |
|------|------------------|-------------------|-----------|-------|
| Quiz | 1 day | 2.5x | Daily when due | Ensures concept retention |
| Implementation | 30 days | 2.0x | Monthly+ | Keeps coding skills fresh |
| GPU | ∞ | N/A | One-time | Too expensive to repeat |

**Example Review Progression:**

```
Quiz Question "backprop_mc_001":
Day 1: Answered correctly → Next review: Day 3
Day 3: Answered correctly → Next review: Day 10 (3 × 2.5 = 7.5 ≈ 7)
Day 10: Answered incorrectly → Next review: Day 11 (reset to 1)
Day 11: Answered correctly → Next review: Day 14
Day 14: Answered correctly → Next review: Day 44 (14 × 2.5 = 35)
```

---

### Component 3: Static Visualization Engine

**Visualization Architecture:**

Each concept gets custom visualization scripts tailored to explain that specific concept most effectively.

**File: `learning_framework/visualization/renderer.py`**

```python
class VisualizationRenderer:
    """Renders matplotlib visualizations with multiple output modes"""

    def __init__(self):
        self.config = ConfigManager()

    def render(self, concept_slug, function='main_visualization', output=None):
        """Load and execute visualization script

        Args:
            concept_slug: Concept identifier
            function: Visualization function name
            output: 'browser', 'terminal', 'file', or from config

        Returns:
            Path to saved visualization (if applicable)
        """
        viz_path = Path(f"data/{concept_slug}/visualize.py")

        if not viz_path.exists():
            raise FileNotFoundError(f"No visualization for {concept_slug}")

        # Import visualization module dynamically
        spec = importlib.util.spec_from_file_location("viz_module", viz_path)
        viz_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(viz_module)

        # Get requested function
        viz_function = getattr(viz_module, function)

        # Execute to get matplotlib figure
        fig = viz_function()

        # Display based on output mode
        output_mode = output or self.config.get('visualization.output_mode', 'browser')
        return self._display(fig, concept_slug, output_mode)

    def _display(self, fig, concept_slug, mode):
        """Display figure based on output mode"""
        if mode == 'browser':
            # Save PNG and auto-open
            path = f"user_data/visualizations/{concept_slug}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            webbrowser.open(f"file://{Path(path).absolute()}")
            return path

        elif mode == 'terminal':
            # Try imgcat (iTerm2) or sixel
            path = f"/tmp/{concept_slug}.png"
            fig.savefig(path, dpi=100)

            if self._has_imgcat():
                subprocess.run(['imgcat', path])
            elif self._has_sixel():
                subprocess.run(['img2sixel', path])
            else:
                print("Terminal image display not available. Use --output browser")

            return path

        elif mode == 'file':
            path = f"user_data/visualizations/{concept_slug}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {path}")
            return path

        elif mode == 'interactive':
            plt.show()
            return None

    def get_available_visualizations(self, concept_slug):
        """Discover visualization functions in visualize.py

        Returns:
            List of {name, description} dicts
        """
        viz_path = Path(f"data/{concept_slug}/visualize.py")

        if not viz_path.exists():
            return []

        # Parse file to find functions with docstrings
        spec = importlib.util.spec_from_file_location("viz_module", viz_path)
        viz_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(viz_module)

        functions = []
        for name in dir(viz_module):
            if name.startswith('_'):
                continue

            obj = getattr(viz_module, name)
            if callable(obj):
                doc = obj.__doc__ or "No description"
                functions.append({
                    'name': name,
                    'description': doc.strip().split('\n')[0]
                })

        return functions
```

**Example Visualizations:**

**1. Gradient Descent (`data/gradient_descent/visualize.py`):**

```python
"""Visualizations for Gradient Descent concept"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main_visualization():
    """3D loss surface with gradient descent convergence path"""
    fig = plt.figure(figsize=(12, 5))

    # Create 3D loss surface
    ax1 = fig.add_subplot(121, projection='3d')

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Simple convex function

    ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')

    # Gradient descent path
    path_x = [4, 3.2, 2.56, 2.048, 1.6384, 1.31072]
    path_y = [4, 3.2, 2.56, 2.048, 1.6384, 1.31072]
    path_z = [x**2 + y**2 for x, y in zip(path_x, path_y)]

    ax1.plot(path_x, path_y, path_z, 'r-o', linewidth=2, markersize=6, label='GD Path')

    ax1.set_xlabel('θ₁')
    ax1.set_ylabel('θ₂')
    ax1.set_zlabel('Loss J(θ)')
    ax1.set_title('Gradient Descent on Loss Surface')
    ax1.legend()

    # 2D contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(path_x, path_y, 'r-o', linewidth=2, markersize=6, label='GD Path')
    ax2.set_xlabel('θ₁')
    ax2.set_ylabel('θ₂')
    ax2.set_title('Contour View (Learning Rate α=0.2)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def compare_learning_rates():
    """Compare convergence with different learning rates"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    learning_rates = [0.01, 0.1, 0.5, 0.9]

    for ax, lr in zip(axes.flat, learning_rates):
        # Simulate gradient descent with different learning rates
        # Plot convergence paths
        ax.set_title(f'Learning Rate α = {lr}')

    plt.tight_layout()
    return fig
```

**2. Backpropagation (`data/backpropagation/visualize.py`):**

```python
"""Visualizations for Backpropagation concept"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def main_visualization():
    """Computational graph with forward and backward passes"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Node positions
    nodes = {
        'x': (1, 4), 'w1': (1, 2),
        'z1': (3, 3),
        'a1': (5, 3),
        'w2': (5, 1),
        'z2': (7, 2),
        'a2': (9, 2),
        'y': (9, 4),
        'L': (11, 3)
    }

    # Draw nodes
    for name, (x, y) in nodes.items():
        circle = plt.Circle((x, y), 0.3, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold')

    # Forward pass arrows (solid)
    forward_edges = [
        ('x', 'z1'), ('w1', 'z1'), ('z1', 'a1'),
        ('a1', 'z2'), ('w2', 'z2'), ('z2', 'a2'),
        ('a2', 'L'), ('y', 'L')
    ]

    for src, dst in forward_edges:
        arrow = FancyArrowPatch(
            nodes[src], nodes[dst],
            arrowstyle='->', mutation_scale=20, linewidth=2,
            color='green', alpha=0.7
        )
        ax.add_patch(arrow)

    # Backward pass arrows (dashed)
    backward_edges = [
        ('L', 'a2'), ('a2', 'z2'), ('z2', 'w2'),
        ('z2', 'a1'), ('a1', 'z1'), ('z1', 'w1')
    ]

    for src, dst in backward_edges:
        arrow = FancyArrowPatch(
            nodes[src], nodes[dst],
            arrowstyle='->', mutation_scale=20, linewidth=2,
            color='red', alpha=0.7, linestyle='--'
        )
        ax.add_patch(arrow)

    # Labels
    ax.text(6, 5, 'Forward Pass (solid green)', color='green', fontsize=12, fontweight='bold')
    ax.text(6, 0.5, 'Backward Pass (dashed red)', color='red', fontsize=12, fontweight='bold')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Backpropagation: Forward and Backward Passes', fontsize=16, fontweight='bold')

    plt.tight_layout()
    return fig

def gradient_flow():
    """Gradient magnitude through layers (vanishing gradient demo)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = ['Output', 'Layer 4', 'Layer 3', 'Layer 2', 'Layer 1', 'Input']

    # Sigmoid activation (vanishing)
    sigmoid_grads = [1.0, 0.25, 0.0625, 0.0156, 0.0039, 0.00098]

    # ReLU activation (better)
    relu_grads = [1.0, 0.8, 0.7, 0.65, 0.6, 0.55]

    x = range(len(layers))

    ax.bar([i - 0.2 for i in x], sigmoid_grads, width=0.4, label='Sigmoid', color='orange', alpha=0.7)
    ax.bar([i + 0.2 for i in x], relu_grads, width=0.4, label='ReLU', color='green', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Gradient Flow Through Network Layers')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig
```

**Visualization Template (`data/templates/visualize.py.template`):**

```python
"""
Visualization for [CONCEPT_NAME]

This file should contain one or more visualization functions.
Each function should return a matplotlib figure.

Functions:
- main_visualization(): Primary visualization (required)
- [optional_viz_2](): Additional visualization (optional)
"""

import numpy as np
import matplotlib.pyplot as plt

def main_visualization():
    """[Brief description of what this visualization shows]

    Explain what insight this visualization provides for understanding
    the concept.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # TODO: Add your visualization code here
    # Example:
    # x = np.linspace(0, 10, 100)
    # y = np.sin(x)
    # ax.plot(x, y)

    ax.set_title('[CONCEPT_NAME] Visualization')
    ax.set_xlabel('X axis label')
    ax.set_ylabel('Y axis label')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Add more visualization functions as needed

if __name__ == '__main__':
    # Test visualization standalone
    fig = main_visualization()
    plt.show()
```

---

### Component 4: Learning Flow & CLI Integration

**Enhanced `lf learn` Command:**

Menu-driven interface with full user control:

```
╔══════════════════════════════════════════════════════════════╗
║  Learning: Backpropagation                                   ║
║  Topic: Neural Networks  |  Difficulty: Intermediate         ║
╚══════════════════════════════════════════════════════════════╝

Prerequisites: Gradient Descent ✓, Chain Rule ⚠️ (not mastered)

⚠️  Warning: Chain Rule is recommended before Backpropagation
   Continue anyway? [y/N]

What would you like to do?

  1. 📖 Read Explanation (from your materials)
  2. 🎨 View Visualization
  3. ❓ Take Quiz (10 questions)
  4. 📊 View Progress
  5. 📝 Add Personal Notes
  6. 🔙 Back to Topics

  [1-6 or q to quit]: _
```

**Menu Options Implementation:**

**Option 1 - Read Explanation:**
- Display explanation from indexed materials
- Show code examples with syntax highlighting
- Offer to open files in editor

**Option 2 - View Visualization:**
- List available visualization functions
- Render chosen visualization
- Support multiple output modes

**Option 3 - Take Quiz:**
- Mix due reviews + new questions
- 70% multiple choice, 30% fill-in-blank
- Show explanations for wrong answers
- Update spaced repetition schedule
- Mark quiz tier as passed if score ≥ 80%

**Option 4 - View Progress:**
- Show mastery status (quiz, implementation, GPU tiers)
- Display next review dates
- Show quiz statistics (attempts, best score)

**Option 5 - Add Personal Notes:**
- Open `data/{concept}/notes.md` in configured editor
- Persistent personal notes per concept

**Enhanced `lf quiz --daily`:**

Daily review workflow:

```bash
lf quiz --daily

╔══════════════════════════════════════════════════════════════╗
║  Daily Review - 2026-01-12                                   ║
╚══════════════════════════════════════════════════════════════╝

📅 23 questions due for review across 5 concepts

Reviewing: Gradient Descent (5 questions)
════════════════════════════════════════════════════════════════

Q1. [Multiple Choice]
What happens if learning rate is too large?

  1. Convergence is slower
  2. Oscillation or divergence
  3. Better generalization
  4. Vanishing gradients

Your answer [1-4]: 2
✓ Correct!

[... continues through all due questions ...]

════════════════════════════════════════════════════════════════
Summary
════════════════════════════════════════════════════════════════
Reviewed: 5 concepts, 23 questions
Score: 20/23 (87%)

Next review: 2026-01-15 (3 concepts due)
```

---

### Component 5: Data Flow & File Structure

**Complete Phase 2 Directory Structure:**

```
learning-framework-dev/
├── learning_framework/
│   ├── __init__.py
│   ├── cli.py                          # Enhanced: learn, quiz commands
│   ├── config.py                       # (Phase 1)
│   │
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── indexer.py                  # Enhanced: concept detection
│   │   ├── graph.py                    # NEW: Knowledge graph
│   │   └── concepts.py                 # NEW: Concept loading
│   │
│   ├── assessment/
│   │   ├── __init__.py                 # NEW
│   │   ├── spaced_repetition.py       # NEW: SM-2 algorithm
│   │   ├── quiz.py                     # NEW: Quiz engine
│   │   └── grader.py                   # NEW: Answer grading
│   │
│   ├── visualization/
│   │   ├── __init__.py                 # NEW
│   │   ├── renderer.py                 # NEW: Matplotlib rendering
│   │   └── display.py                  # NEW: Output modes
│   │
│   └── progress/                       # (Phase 1)
│       ├── __init__.py
│       ├── database.py                 # Enhanced schema
│       └── tracker.py
│
├── data/
│   ├── concepts.json                   # Master registry
│   ├── gradient_descent/               # Complete (1/5)
│   ├── backpropagation/                # Complete (2/5)
│   ├── loss_functions/                 # Complete (3/5)
│   ├── activation_functions/           # Complete (4/5)
│   ├── q_learning/                     # Complete (5/5)
│   ├── [25+ skeleton concepts]/
│   └── templates/
│
├── user_data/
│   ├── progress.db                     # Enhanced schema
│   ├── config.yaml                     # Enhanced config
│   └── visualizations/                 # NEW: Saved images
│
└── tests/
    ├── test_quiz.py                    # NEW
    ├── test_spaced_repetition.py       # NEW
    ├── test_knowledge_graph.py         # NEW
    ├── test_visualization.py           # NEW
    └── [Phase 1 tests...]
```

**Key Data Flows:**

**Flow 1: Material Indexing → Concept Discovery**
```
lf index
→ Scan DeepLearning/, RL/ folders
→ Extract chapter names, keywords
→ Auto-generate skeleton concept.json files
→ Update data/concepts.json registry
→ Build dependency graph
```

**Flow 2: Learning a Concept**
```
lf learn backpropagation
→ Load concept from data/backpropagation/concept.json
→ Check prerequisites (soft warning)
→ Display menu
→ User selects option (read/viz/quiz/progress/notes)
→ Execute selected flow
```

**Flow 3: Taking a Quiz**
```
Select: [3] Take Quiz
→ Load quiz questions from quiz_mc.json + quiz_fillblank.json
→ Get due reviews from progress.db
→ Generate mixed quiz (reviews + new questions)
→ For each question:
  - Display question
  - Get answer
  - Grade (correct/incorrect)
  - Show explanation if wrong
  - Update spaced repetition schedule
→ Calculate final score
→ If ≥80%: Mark quiz_passed = 1
```

**Flow 4: Daily Review**
```
lf quiz --daily
→ Query progress.db for due questions
→ Group by concept
→ For each concept with due questions:
  - Quiz on due items
  - Update review schedules
→ Display summary statistics
```

**Database Schema Enhancements:**

```sql
-- Add to concepts table (from Phase 1)
ALTER TABLE concepts ADD COLUMN first_visit BOOLEAN DEFAULT 1;
ALTER TABLE concepts ADD COLUMN total_quiz_attempts INTEGER DEFAULT 0;
ALTER TABLE concepts ADD COLUMN best_quiz_score REAL DEFAULT 0.0;

-- Add columns to quiz_results (from Phase 1)
ALTER TABLE quiz_results ADD COLUMN next_review DATE;
ALTER TABLE quiz_results ADD COLUMN review_interval INTEGER DEFAULT 1;

-- Create indexes for performance
CREATE INDEX idx_next_review ON quiz_results(next_review);
CREATE INDEX idx_concept_due ON concepts(next_review);

-- Track visualization views
CREATE TABLE IF NOT EXISTS visualizations_viewed (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id INTEGER NOT NULL,
    visualization_name TEXT NOT NULL,
    viewed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (concept_id) REFERENCES concepts(id)
);
```

**Configuration Enhancements (user_data/config.yaml):**

```yaml
# Phase 1 config...
materials_directories:
  - "D:/ourock-test/ourock-test/DeepLearning"
  - "D:/ourock-test/ourock-test/RL"

daily_gpu_budget: 5.0
editor: "code"

# Phase 2 additions:
visualization:
  output_mode: "browser"        # browser, terminal, file, interactive
  save_to_disk: true            # Keep copies in user_data/
  terminal_protocol: "auto"     # auto, imgcat, sixel, none

quiz:
  questions_per_session: 10
  passing_score: 0.8            # 80% to pass
  mix_question_types: true      # MC + fill-in-blank
  show_explanations: true       # Show explanations for wrong answers

spaced_repetition:
  enabled: true
  quiz_intervals: [1, 3, 7, 14, 30]        # days
  implementation_intervals: [30, 60, 90]   # days
  reset_on_fail: true           # Reset to day 1 on wrong answer

learning:
  show_prerequisite_warnings: true
  allow_skip_prerequisites: true  # Soft warnings only
```

---

## Implementation Phases

### Task Breakdown

Phase 2 breaks into 7 major tasks:

1. **Knowledge Graph & Concept Detection** (2-3 hours)
   - Enhance MaterialIndexer to detect concepts
   - Build KnowledgeGraph class with prerequisite checking
   - Create skeleton concept.json files for detected concepts
   - Implement concept loading and registry

2. **Spaced Repetition Module** (1-2 hours)
   - Extract SM-2 algorithm from vocab quiz
   - Create reusable SpacedRepetitionScheduler
   - Database schema updates for review tracking
   - Unit tests for scheduling logic

3. **Quiz System Core** (2-3 hours)
   - ConceptQuiz class with question generation
   - Answer grading (MC + fill-in-blank)
   - Integration with spaced repetition
   - Database updates for quiz results

4. **Create 5 Fully-Implemented Concepts** (4-6 hours)
   - Write 15 quiz questions each (MC + fill-in-blank)
   - Create visualizations for each concept
   - Write concept metadata and descriptions
   - Test quiz flow for each concept

5. **Visualization Engine** (2-3 hours)
   - VisualizationRenderer with multiple output modes
   - DisplayManager for browser/terminal/file output
   - Template creation for new visualizations
   - Integration with CLI

6. **CLI Enhancement** (2-3 hours)
   - Enhanced `lf learn` with menu system
   - Enhanced `lf quiz --daily` with reviews
   - Progress display formatting
   - Notes editing integration

7. **Testing & Documentation** (2-3 hours)
   - Unit tests for all new components
   - Integration tests for learning flow
   - README updates
   - Example workflows documentation

**Total Estimated Time:** 15-23 hours

---

## Success Criteria

### Functional Requirements

- ✅ Auto-discover ~30 concepts from existing materials
- ✅ 5 concepts fully implemented (quiz + visualization)
- ✅ Quiz system with 2 question types works smoothly
- ✅ Spaced repetition schedules reviews correctly
- ✅ Visualizations render in multiple output modes
- ✅ Menu-driven learning flow is intuitive
- ✅ Daily review workflow is efficient

### Quality Requirements

- ✅ All quiz questions have clear explanations
- ✅ Visualizations are informative and attractive
- ✅ Prerequisite warnings are helpful not annoying
- ✅ Code is well-tested (>80% coverage for new components)
- ✅ Documentation is complete and accurate

### User Experience

- ✅ Can start learning immediately (5 complete concepts)
- ✅ Can expand to more concepts easily (templates provided)
- ✅ Daily reviews take < 10 minutes
- ✅ Progress is always visible
- ✅ System feels complete (not a prototype)

---

## Risk Mitigation

### Technical Risks

**Risk:** Visualization rendering fails on some platforms
**Mitigation:** Multiple output modes, graceful fallbacks, clear error messages

**Risk:** Spaced repetition logic has bugs
**Mitigation:** Extract from proven vocab quiz system, extensive unit tests

**Risk:** Auto-concept detection creates poor quality skeletons
**Mitigation:** Manual review of auto-generated concepts, easy editing

### Content Risks

**Risk:** Quiz questions have errors or poor explanations
**Mitigation:** Manual authoring for 5 core concepts, user can edit JSON files

**Risk:** Visualizations don't clarify concepts well
**Mitigation:** Iterative refinement, multiple visualization types per concept

### User Experience Risks

**Risk:** Menu-driven interface feels clunky
**Mitigation:** Rich formatting, keyboard shortcuts, fast navigation

**Risk:** Too many concepts feel overwhelming
**Mitigation:** Clear status indicators (complete vs skeleton), topic grouping

---

## Future Enhancements (Phase 3+)

**Phase 3 will add:**
- Implementation challenges (fill-in-blank → from-scratch → debug)
- Automated test runner with numerical gradient checking
- C++ basic matrix operations

**Later phases:**
- Interactive visualizations (web-based)
- GPU backend for scaling
- Advanced C++ implementations
- More question types (ordering, matching, etc.)

---

## Conclusion

Phase 2 transforms the framework from infrastructure into a working learning system. By shipping with 5 complete concepts and 25+ skeleton definitions, users can start learning immediately while having a clear path to expand the system. The extracted spaced repetition module makes the system reusable beyond just this project, and the visualization engine provides the visual understanding critical for mastering DL/RL concepts.

**Dependencies:** Phase 1 Complete ✓
**Next Steps:** Create detailed implementation plan → Execute in isolated worktree
**Estimated Completion:** 15-23 hours of development
