# Learning Framework User Guide

## Getting Started

### First-Time Setup

1. **Install the framework**
   ```bash
   pip install -e .
   ```

2. **Create your configuration**
   ```bash
   cp user_data/config-example.yaml user_data/config.yaml
   ```

3. **Verify installation**
   ```bash
   lf --version
   lf config
   ```

## Learning Workflow

### Step 1: Browse Available Concepts

```bash
lf learn
```

This will show you all available concepts organized by topic. You can select a concept to start learning.

**Example Output:**
```
Available Concepts

Deep Learning
  1. Gradient Descent
  2. Backpropagation
  3. Loss Functions
  4. Activation Functions

Reinforcement Learning
  5. Q-Learning

Select concept number (or 'q' to quit):
```

### Step 2: Learn a Concept

After selecting a concept, you'll see an interactive menu:

```
Learning: Gradient Descent
An optimization algorithm that iteratively updates parameters...

1. Read explanation
2. View visualization
3. Take quiz
4. View progress
5. Back to concept selection
```

**Option 1: Read Explanation**
- Displays the concept explanation
- Shows key points to remember
- Best for first-time learning

**Option 2: View Visualization**
- Launches interactive matplotlib visualizations
- Opens in your default browser
- Helps build intuition visually

**Option 3: Take Quiz**
- 5 questions mixing multiple-choice and fill-in-blank
- Immediate feedback with explanations
- Results saved for spaced repetition

**Option 4: View Progress**
- Shows quiz attempts, accuracy, and correct count
- Displays items due for review
- Tracks your learning journey

### Step 3: Daily Review

The framework uses spaced repetition to optimize learning. Run daily reviews:

```bash
lf quiz
```

**Example Output:**
```
Daily Review: 8 items due

  • gradient_descent: 3 items
  • backpropagation: 5 items

Start daily review? [Y/n]:
```

This will quiz you on items that are due for review based on the SM-2 algorithm.

### Step 4: Quiz on Specific Concept

To practice a specific concept:

```bash
lf quiz --concept gradient_descent
```

This bypasses the daily review and quizzes you on the specified concept.

## Understanding Prerequisites

Some concepts have prerequisites. If you try to learn a concept without completing its prerequisites, you'll see:

```
⚠️  Prerequisites

Prerequisites needed:
  - gradient_descent
  - loss_functions

Continue anyway? [y/N]:
```

You can choose to continue or go back and learn the prerequisites first.

## Progress Tracking

Your progress is automatically tracked in `user_data/progress.db`. The system tracks:

- Quiz attempts
- Correct/incorrect answers
- Review intervals (spaced repetition)
- Next review dates

View your overall progress:

```bash
lf progress
```

## Spaced Repetition System

The framework uses the SM-2 algorithm for optimal review scheduling:

- **Correct answer**: Review interval increases (1 day → 6 days → 15 days → ...)
- **Incorrect answer**: Review interval resets to 1 day

This ensures you review material just before you're likely to forget it.

## Tips for Effective Learning

1. **Start with prerequisites**: Follow the prerequisite chain for best understanding
2. **Use all three modes**: Read → Visualize → Quiz for comprehensive learning
3. **Do daily reviews**: Consistency is key for spaced repetition
4. **Don't skip explanations**: When you get a quiz answer wrong, read the explanation
5. **Aim for 80%+**: If your quiz score is below 80%, review the explanation again

## Adding Your Own Materials

You can index your own learning materials:

1. **Configure materials directories** in `user_data/config.yaml`:
   ```yaml
   materials_directories:
     - "/path/to/your/DeepLearning"
     - "/path/to/your/RL"
   ```

2. **Run indexer**:
   ```bash
   lf index
   ```

This will scan your directories and create an index in `materials/index.json`.

## Creating Custom Concepts

To add your own concepts:

1. **Copy the template**:
   ```bash
   cp -r data/templates data/my_concept
   ```

2. **Edit files**:
   - `concept.json`: Metadata and explanation
   - `quiz_mc.json`: Multiple choice questions (10 recommended)
   - `quiz_fill.json`: Fill-in-blank questions (5 recommended)
   - `visualize.py`: Visualization code

3. **Register in `data/concepts.json`**:
   ```json
   {
     "concepts": [
       {
         "slug": "my_concept",
         "name": "My Concept",
         "topic": "My Topic",
         "description": "Brief description",
         "prerequisites": []
       }
     ]
   }
   ```

4. **Test your concept**:
   ```bash
   lf learn --concept my_concept
   ```

## Troubleshooting

### "concepts.json not found"
Run `lf index` to create the index, or create `data/concepts.json` manually.

### "No quiz questions available"
Ensure `quiz_mc.json` and `quiz_fill.json` exist in the concept directory.

### "Error rendering visualization"
Check that the `visualize.py` file has a `main_visualization()` function.

### Database locked errors
Close any other instances of the framework accessing the database.

## Command Reference

| Command | Description |
|---------|-------------|
| `lf --help` | Show all available commands |
| `lf --version` | Show version |
| `lf config` | View current configuration |
| `lf index` | Index learning materials |
| `lf learn` | Interactive concept selection |
| `lf learn --concept <slug>` | Learn specific concept |
| `lf quiz` | Daily review (spaced repetition) |
| `lf quiz --concept <slug>` | Quiz on specific concept |
| `lf progress` | View learning progress |

## Configuration Options

Edit `user_data/config.yaml`:

```yaml
# Directories containing learning materials
materials_directories:
  - "/path/to/materials"

# Daily GPU budget (for future GPU features)
daily_gpu_budget: 5.0

# Code editor (for future implementation challenges)
editor: "code"
```

## Data Storage

- **Configuration**: `user_data/config.yaml`
- **Progress Database**: `user_data/progress.db`
- **Concept Content**: `data/<concept>/`
- **Material Index**: `materials/index.json`

All data is stored locally on your machine.
