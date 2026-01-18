"Command-line interface for learning framework"

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from pathlib import Path
import json

from learning_framework import __version__
from learning_framework.config import ConfigManager
from learning_framework.knowledge import MaterialIndexer, load_concept, KnowledgeGraph, ConceptRegistry
from learning_framework.assessment import ConceptQuiz, AnswerGrader, SpacedRepetitionScheduler
from learning_framework.visualization import VisualizationRenderer
from learning_framework.progress import ProgressDatabase
from learning_framework.errors import (
    LearningFrameworkError,
    ConfigurationError,
    ConceptNotFoundError,
    PrerequisiteError,
    QuizError,
    ChallengeError,
    GPUBackendError,
    BudgetExceededError,
    format_error_message,
    suggest_fix,
    handle_error
)

console = Console()


def _get_default_db_path() -> Path:
    """Get default path for progress database"""
    return Path.cwd() / 'user_data' / 'progress.db'


def display_error(error: Exception):
    """Display error with suggestions to user"""
    message, suggestions = handle_error(error)
    console.print(f"\n[red]{message}[/red]")

    if suggestions:
        console.print("\n[yellow]Suggestions:[/yellow]")
        for i, suggestion in enumerate(suggestions, 1):
            console.print(f"  {i}. {suggestion}")
    console.print()


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """Interactive DL/RL Mastery Framework

    Learn fundamentals locally, validate at scale on remote GPUs.
    """
    # Store config in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['config'] = ConfigManager()


@cli.command()
@click.option('--concept', help='Concept slug to learn')
@click.pass_context
def learn(ctx, concept):
    """Start interactive learning session"""
    config_mgr = ctx.obj['config']
    
    # Track if we started with a specific concept (single-shot mode)
    initial_concept = concept

    while True:
        try:
            # Initialize components
            registry = ConceptRegistry()  # Auto-loads from data/concepts.json
            graph = KnowledgeGraph()  # Uses default db_path

            # If no concept specified, show selection menu
            if not concept:
                console.print("\n[bold cyan]Available Concepts[/bold cyan]\n")

                topics = registry.get_topics()
                all_concepts = []  # List of (slug, name) tuples

                for topic in topics:
                    console.print(f"[bold yellow]{topic}[/bold yellow]")
                    topic_slugs = registry.get_by_topic(topic)
                    for slug in topic_slugs:
                        # Load full concept data to get name
                        try:
                            concept_data = load_concept(slug)
                            name = concept_data.get('name', slug)
                        except FileNotFoundError:
                            name = slug  # Fallback to slug if concept.json missing
                        all_concepts.append({'slug': slug, 'name': name})
                        idx = len(all_concepts)
                        console.print(f"  {idx}. {name}")

                console.print()
                choice = click.prompt("Select concept number (or 'q' to quit)", type=str)

                if choice.lower() == 'q':
                    return

                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(all_concepts):
                        concept = all_concepts[idx]['slug']
                    else:
                        console.print("[red]Invalid selection[/red]")
                        continue
                except ValueError:
                    console.print("[red]Invalid input[/red]")
                    continue

            # Validate concept exists
            all_slugs = list(registry.get_all().keys())
            if concept not in all_slugs:
                # Find similar concepts
                from difflib import get_close_matches
                suggestions = get_close_matches(concept, all_slugs, n=3, cutoff=0.6)
                raise ConceptNotFoundError(concept, suggestions=suggestions)

            # Load concept
            concept_data = load_concept(concept)
            if not concept_data:
                raise ConceptNotFoundError(concept)

        except LearningFrameworkError as e:
            display_error(e)
            if initial_concept:
                raise SystemExit(1)
            else:
                concept = None
                continue
        except Exception as e:
            display_error(e)
            raise SystemExit(1)

        # Check prerequisites
        prereq_result = graph.check_prerequisites(concept)
        missing = prereq_result.get('missing', [])
        if missing:
            console.print(Panel(
                f"[yellow]Prerequisites needed:[/yellow]\n" +
                "\n".join(f"  - {p}" for p in missing),
                title="⚠️  Prerequisites"
            ))
            if not click.confirm("Continue anyway?"):
                if initial_concept:
                    return
                concept = None
                continue

        # Interactive learning loop
        while True:
            console.print(f"\n[bold cyan]Learning: {concept_data['name']}[/bold cyan]")
            
            # Prefer Markdown description if available (for future use), fallback to plain description
            description = concept_data.get('description', '')
            console.print(f"[dim]{description}[/dim]\n")

            console.print("1. Read explanation")
            console.print("2. View visualization")
            console.print("3. Take quiz")
            console.print("4. View progress")
            console.print("5. Back to concept selection")
            console.print()

            choice = click.prompt("Choose option", type=int, default=1)

            if choice == 1:
                _show_explanation(concept_data)
            elif choice == 2:
                _show_visualization(concept)
            elif choice == 3:
                _take_quiz(concept)
            elif choice == 4:
                _show_progress(concept)
            elif choice == 5:
                break
            else:
                console.print("[yellow]Invalid choice[/yellow]")
        
        # Exit logic
        if initial_concept:
            break
        
        # Reset concept to show menu again
        concept = None


def _show_explanation(concept_data):
    """Display concept explanation"""
    # 1. 如果有 Markdown 内容，优先渲染 MD
    if 'explanation_md' in concept_data:
        console.print(Panel(
            Markdown(concept_data['explanation_md']),
            title=f"📖 {concept_data['name']}",
            border_style="cyan"
        ))
    
    # 2. 否则使用旧方式 (纯文本)
    else:
        console.print(Panel(
            concept_data.get('explanation', 'No explanation available.'),
            title=f"📖 {concept_data['name']}",
            border_style="cyan"
        ))

    if 'key_points' in concept_data:
        console.print("\n[bold]Key Points:[/bold]")
        for point in concept_data['key_points']:
            console.print(f"  • {point}")

    click.pause("\nPress any key to continue...")


def _show_visualization(concept_slug):
    """Show visualization for concept"""
    renderer = VisualizationRenderer()

    try:
        viz_list = renderer.get_available_visualizations(concept_slug)

        if not viz_list:
            console.print("[yellow]No visualizations available for this concept[/yellow]")
            return

        console.print("\n[bold]Available Visualizations:[/bold]")
        for i, viz in enumerate(viz_list, 1):
            console.print(f"{i}. {viz['name']}")

        choice = click.prompt("Select visualization", type=int, default=1)

        if 1 <= choice <= len(viz_list):
            selected = viz_list[choice - 1]
            console.print(f"[cyan]Rendering {selected['name']}...[/cyan]")

            fig = renderer.execute_visualization(concept_slug, selected['function'])

            # Display in browser
            from learning_framework.visualization import DisplayManager
            display = DisplayManager()
            display.show(fig, concept_slug, mode='browser')

            console.print("[green]✓ Visualization displayed[/green]")
        else:
            console.print("[yellow]Invalid choice[/yellow]")

    except Exception as e:
        console.print(f"[red]Error rendering visualization: {e}[/red]")


@cli.command()
@click.option('--concept', help='Specific concept to verify visualization for')
def viz(concept):
    """Launch visualization server or verify static visualization"""
    if not concept:
        # Launch server (default behavior)
        from learning_framework.visualization import VisualizationServer, VisualizationDataProvider
        provider = VisualizationDataProvider()
        server = VisualizationServer(data_provider=provider)
        try:
            server.start()
        except KeyboardInterrupt:
            server.stop()
    else:
        # Verify static visualization (same as learn -> option 2)
        _show_visualization(concept)


@cli.command()
@click.argument('concept_slug')
def visualize(concept_slug):
    """Alias for viz command (legacy support)"""
    _show_visualization(concept_slug)


def _take_quiz(concept_slug):
    """Take quiz for concept"""
    try:
        quiz_engine = ConceptQuiz(concept_slug)
        questions = quiz_engine.generate_quiz(num_questions=5, mix_types=True)

        if not questions:
            console.print("[yellow]No quiz questions available[/yellow]")
            return

        console.print(f"\n[bold cyan]Quiz: {len(questions)} questions[/bold cyan]\n")

        score = 0
        results = []

        for i, q in enumerate(questions, 1):
            console.print(f"\n[bold]Question {i}/{len(questions)}[/bold]")
            console.print(q['question'])

            if q['type'] == 'multiple_choice':
                for j, option in enumerate(q['options'], 1):
                    console.print(f"  {j}. {option}")

                answer_idx = click.prompt("Your answer", type=int)
                user_answer = q['options'][answer_idx - 1] if 1 <= answer_idx <= len(q['options']) else ""
            else:
                user_answer = click.prompt("Your answer", type=str)

            grader = AnswerGrader()
            is_correct = grader.check_answer(q, user_answer)

            results.append({
                'question_id': q.get('id', f'q{i}'),
                'correct': is_correct
            })

            if is_correct:
                console.print("[green]✓ Correct![/green]")
                score += 1
            else:
                console.print(f"[red]✗ Incorrect. Answer: {q['answer']}[/red]")
                if 'explanation' in q:
                    console.print(f"[dim]{q['explanation']}[/dim]")

        # Save results using spaced repetition scheduler
        scheduler = SpacedRepetitionScheduler()  # Creates its own db connection

        for result in results:
            scheduler.calculate_next_review(
                f"{concept_slug}:{result['question_id']}",
                result['correct']
            )

        console.print(f"\n[bold]Score: {score}/{len(questions)}[/bold]")
        percentage = (score / len(questions)) * 100

        if percentage >= 80:
            console.print("[green]Great job! 🎉[/green]")
        elif percentage >= 60:
            console.print("[yellow]Good effort! Keep practicing.[/yellow]")
        else:
            console.print("[red]Review the material and try again.[/red]")

        click.pause("\nPress any key to continue...")

    except Exception as e:
        console.print(f"[red]Error running quiz: {e}[/red]")

def _show_progress(concept_slug):
    """Show learning progress for concept"""
    db = ProgressDatabase(_get_default_db_path())

    try:
        # Get quiz history
        stats = db.get_quiz_stats(concept_slug)
        attempts = stats['attempts']
        correct = stats['correct_count']
        accuracy = stats['accuracy']

        console.print(Panel(
            f"[bold]Attempts:[/bold] {attempts}\n"
            f"[bold]Correct:[/bold] {correct}\n"
            f"[bold]Accuracy:[/bold] {accuracy*100:.1f}%",
            title=f"📊 Progress: {concept_slug}",
            border_style="green"
        ))

        # Get due items
        scheduler = SpacedRepetitionScheduler()  # Creates its own db connection
        due_items = scheduler.get_due_items(concept=concept_slug)

        if due_items:
            console.print(f"\n[yellow]{len(due_items)} items due for review[/yellow]")
        else:
            console.print("\n[green]No items due for review[/green]")

        click.pause("\nPress any key to continue...")

    finally:
        db.close()


@cli.command()
@click.option('--concept', help='Specific concept to quiz on')
@click.option('--num-questions', default=10, help='Number of questions')
@click.pass_context
def quiz(ctx, concept, num_questions):
    """Take concept quiz (daily review or specific concept)"""

    if concept:
        # Quiz on specific concept
        _take_quiz(concept)
    else:
        # Daily review - show due items
        scheduler = SpacedRepetitionScheduler()  # Creates its own db connection
        due_items = scheduler.get_due_items()

        if not due_items:
            console.print("[green]No items due for review today! 🎉[/green]")
            console.print("[dim]Come back tomorrow or use --concept to quiz on a specific topic[/dim]")
            return

        console.print(f"[bold cyan]Daily Review: {len(due_items)} items due[/bold cyan]\n")

        # Group by concept
        by_concept = {}
        for item in due_items:
            concept_name = item['item_id'].split(':')[0]
            by_concept.setdefault(concept_name, []).append(item)

        # Show breakdown
        for concept_name, items in by_concept.items():
            console.print(f"  • {concept_name}: {len(items)} items")

        console.print()

        if click.confirm("Start daily review?", default=True):
            for concept_name in by_concept.keys():
                console.print(f"\n[bold]Reviewing: {concept_name}[/bold]")
                _take_quiz(concept_name)


@cli.command()
def progress():
    """View learning progress"""
    console.print("[cyan]Progress report...[/cyan]")
    console.print("[yellow]Not yet implemented[/yellow]")


@cli.command()
@click.pass_context
def index(ctx):
    """Re-index learning materials"""
    config_mgr = ctx.obj['config']

    console.print("[cyan]Indexing materials...[/cyan]\n")

    materials_dirs = config_mgr.get('materials_directories', [])

    if not materials_dirs:
        console.print("[yellow]No materials directories configured.[/yellow]")
        console.print("[dim]Add directories in user_data/config.yaml[/dim]")
        return

    indexer = MaterialIndexer()
    all_results = {}

    for mat_dir in materials_dirs:
        mat_path = Path(mat_dir)
        if not mat_path.exists():
            console.print(f"[red]Directory not found: {mat_dir}[/red]")
            continue

        console.print(f"Scanning: {mat_dir}")
        results = indexer.scan_directory(mat_path)
        all_results[str(mat_path)] = results

        console.print(f"  Found: {len(results['chapters'])} chapters, "
                     f"{len(results['files'])} files, "
                     f"{len(results['keywords'])} keywords")

    # Save index
    index_path = Path('materials') / 'index.json'
    index_path.parent.mkdir(exist_ok=True)

    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]✓ Index saved to: {index_path}[/green]")


@cli.command()
@click.pass_context
def config(ctx):
    """Configure framework settings"""
    config_mgr = ctx.obj['config']

    console.print("\n[bold cyan]Configuration[/bold cyan]\n")

    # Display current settings
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    all_config = config_mgr.get_all()
    for key, value in sorted(all_config.items()):
        # Mask API keys
        if 'key' in key.lower() and value:
            value = '*' * 8 + value[-4:] if len(str(value)) > 4 else '****'
        table.add_row(key, str(value))

    console.print(table)
    console.print(f"\n[dim]Config file: {config_mgr.config_path}[/dim]")
    console.print("[yellow]Use 'lf config --set key=value' to change settings (not yet implemented)[/yellow]")


@cli.group()
def challenge():
    """Manage implementation challenges"""
    pass

@challenge.command('list')
@click.option('--type', 'challenge_type',
              type=click.Choice(['fill', 'scratch', 'debug', 'all']),
              default='all', help='Filter by challenge type')
def list_challenges(challenge_type):
    """List available implementation challenges"""
    from learning_framework.assessment.challenge import ChallengeManager
    manager = ChallengeManager()
    
    if challenge_type == 'all':
        challenges = manager.list_challenges()
    else:
        challenges = manager.get_challenges_by_type(challenge_type)
        
    if not challenges:
        console.print(f"[yellow]No challenges found[/yellow]")
        return
        
    table = Table(title=f"Available Challenges ({challenge_type})")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Description")
    
    for c in challenges:
        desc = c['description'].split('\n')[0] if c['description'] else "No description"
        table.add_row(c['name'], c['type'], desc)
        
    console.print(table)

@challenge.command()
@click.argument('name')
def start(name):
    """Start a challenge (copies template to workspace)"""
    from learning_framework.assessment.challenge import ChallengeManager
    manager = ChallengeManager()
    
    # Try to find challenge by name or filename
    challenge = manager.load_challenge(name)
    if not challenge:
        # Try appending .py if not present
        if not name.endswith('.py'):
             # Try common types
             for c_type in ['fill', 'scratch']:
                 try_name = f"{name}_{c_type}.py"
                 challenge = manager.load_challenge(try_name)
                 if challenge:
                     break
    
    if not challenge:
        console.print(f"[red]Challenge '{name}' not found[/red]")
        return
        
    # Copy to user_data/implementations
    workspace = Path.cwd() / 'user_data' / 'implementations'
    workspace.mkdir(parents=True, exist_ok=True)
    
    target = manager.copy_to_workspace(challenge['file_path'].name, workspace)
    console.print(f"[green]✓ Challenge started![/green]")
    console.print(f"File created at: {target}")
    console.print(f"[dim]Edit this file to complete the challenge.[/dim]")

@cli.command()
@click.argument('file_path')
def test(file_path):
    """Run tests for an implementation"""
    from learning_framework.assessment.test_runner import TestRunner
    
    target_path = Path(file_path)
    if not target_path.exists():
        # Check in user_data/implementations
        impl_path = Path.cwd() / 'user_data' / 'implementations' / file_path
        if impl_path.exists():
            target_path = impl_path
        else:
            console.print(f"[red]File not found: {file_path}[/red]")
            return
            
    runner = TestRunner()
    
    # Find corresponding test file (convention: tests/test_{name}.py)
    # This logic needs to map implementation back to the challenge test
    # For now, we'll assume the challenge system knows where tests are
    # Simplified: running pytest on the implementation folder to see if it picks up local tests
    # Ideally, we should look up the test in data/challenges/tests/
    
    # Simple runner invocation
    console.print(f"[cyan]Running tests for {target_path.name}...[/cyan]")
    # TODO: Connect to specific test file in data/challenges/tests
    # For this stub, we just run the file itself if it has tests, or fail nicely
    
    # Placeholder for actual test run
    result = runner.run_tests(str(target_path))
    if result['passed']:
        console.print("[green]Tests passed![/green]")
    else:
        console.print("[red]Tests failed[/red]")
        console.print(result['stdout'])

@cli.command()
@click.argument('script_path')
@click.option('--backend', default='vastai', help='GPU backend (e.g. vastai)')
@click.option('--estimate', is_flag=True, help='Estimate cost without running')
def scale(script_path, backend, estimate):
    """Run implementation on remote GPU"""
    console.print(f"[cyan]Scaling to GPU ({backend})...[/cyan]")
    if estimate:
        console.print(f"Estimated cost: $0.15/hr")
    else:
        console.print("[yellow]Job submission not fully configured in this demo[/yellow]")


# ===== Feedback Commands =====

from learning_framework.feedback import FeedbackCollector, FeedbackType


@cli.group()
def feedback():
    """Submit and manage beta testing feedback"""
    pass


@feedback.command()
@click.option('--title', required=True, help='Short title for the bug')
@click.option('--description', required=True, help='Detailed description')
@click.option('--steps', help='Steps to reproduce')
@click.option('--severity', type=click.Choice(['low', 'medium', 'high', 'critical']),
              default='medium', help='Bug severity')
def bug(title, description, steps, severity):
    """Report a bug"""
    collector = FeedbackCollector()

    feedback_id = collector.submit(
        feedback_type=FeedbackType.BUG,
        title=title,
        description=description,
        steps_to_reproduce=steps,
        severity=severity
    )

    console.print(f"[green]✓ Bug reported: {feedback_id}[/green]")
    console.print("[dim]Thank you for your feedback![/dim]")


@feedback.command()
@click.option('--title', required=True, help='Short title for the feature')
@click.option('--description', required=True, help='Detailed description')
def feature(title, description):
    """Request a new feature"""
    collector = FeedbackCollector()

    feedback_id = collector.submit(
        feedback_type=FeedbackType.FEATURE,
        title=title,
        description=description
    )

    console.print(f"[green]✓ Feature request submitted: {feedback_id}[/green]")
    console.print("[dim]Thank you for your suggestion![/dim]")


@feedback.command()
@click.option('--title', required=True, help='Short title')
@click.option('--description', required=True, help='Detailed description')
@click.option('--severity', type=click.Choice(['low', 'medium', 'high']),
              default='medium')
def usability(title, description, severity):
    """Report a usability issue"""
    collector = FeedbackCollector()

    feedback_id = collector.submit(
        feedback_type=FeedbackType.USABILITY,
        title=title,
        description=description,
        severity=severity
    )

    console.print(f"[green]✓ Usability feedback submitted: {feedback_id}[/green]")


@feedback.command('list')
@click.option('--type', 'feedback_type',
              type=click.Choice(['bug', 'feature', 'usability', 'all']),
              default='all', help='Filter by type')
def list_feedback(feedback_type):
    """List submitted feedback"""
    collector = FeedbackCollector()

    type_filter = None
    if feedback_type != 'all':
        type_filter = FeedbackType(feedback_type)

    feedback_list = collector.list_feedback(feedback_type=type_filter)

    if not feedback_list:
        console.print("[yellow]No feedback submitted yet[/yellow]")
        return

    table = Table(title="Submitted Feedback")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Title")
    table.add_column("Status")
    table.add_column("Date")

    for fb in feedback_list:
        table.add_row(
            fb.id,
            fb.feedback_type,
            fb.title[:40] + "..." if len(fb.title) > 40 else fb.title,
            fb.status,
            fb.created_at[:10] if fb.created_at else ""
        )

    console.print(table)


@feedback.command()
@click.argument('output_file', type=click.Path())
def export(output_file):
    """Export feedback to JSON file"""
    collector = FeedbackCollector()
    collector.export(Path(output_file))
    console.print(f"[green]✓ Feedback exported to {output_file}[/green]")


if __name__ == '__main__':
    cli()