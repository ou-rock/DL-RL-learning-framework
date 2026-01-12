"""Command-line interface for learning framework"""

import click
from rich.console import Console
from rich.table import Table
from pathlib import Path
import json

from learning_framework import __version__
from learning_framework.config import ConfigManager
from learning_framework.knowledge import MaterialIndexer

console = Console()


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """Interactive Deep Learning & RL Mastery Framework

    Learn fundamentals locally, validate at scale on remote GPUs.
    """
    # Store config in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['config'] = ConfigManager()


@cli.command()
def learn():
    """Start interactive learning session"""
    console.print("[cyan]Learning session starting...[/cyan]")
    console.print("[yellow]Not yet implemented[/yellow]")


@cli.command()
def quiz():
    """Take concept quiz"""
    console.print("[cyan]Quiz starting...[/cyan]")
    console.print("[yellow]Not yet implemented[/yellow]")


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


if __name__ == '__main__':
    cli()
