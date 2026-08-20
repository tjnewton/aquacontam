"""AquaContam CLI — ``python -m aquacontam <command>``.

Commands::

    python -m aquacontam benchmark --task T1
    python -m aquacontam export --output ./release/
    python -m aquacontam leaderboard validate submissions/my_model.json

The ``webapp`` subcommand registers only when the optional
``aquacontam.webapp`` package is present in the installation.
"""

from __future__ import annotations

import importlib.util
import sys

import click

from aquacontam._constants import DATASET_VERSION


@click.group()
@click.version_option(package_name="aquacontam")
def cli() -> None:
    """AquaContam: ML benchmark for water contamination prediction."""


def _webapp_available() -> bool:
    """True when the optional webapp package ships in this installation."""
    return importlib.util.find_spec("aquacontam.webapp") is not None


if _webapp_available():

    @cli.command()
    @click.option("--port", default=8050, type=int, help="Port to serve on.")
    @click.option("--host", default="127.0.0.1", help="Host to bind to.")
    @click.option("--debug", is_flag=True, help="Enable debug mode.")
    @click.option("--predictions-dir", default=None, help="Path to predictions directory.")
    def webapp(port: int, host: str, debug: bool, predictions_dir: str | None) -> None:
        """Launch the interactive risk map webapp."""
        try:
            from aquacontam.webapp import create_app
        except ModuleNotFoundError as exc:
            if exc.name is not None and exc.name.startswith("aquacontam.webapp"):
                raise click.ClickException(
                    "the interactive webapp is not included in this distribution"
                ) from exc
            raise

        app = create_app(predictions_dir=predictions_dir, debug=debug)
        click.echo(f"Starting AquaContam Risk Map on http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)


@cli.command()
@click.option("--task", default=None, help="Run a specific task (e.g. T1, T2, T4).")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility.")
def benchmark(task: str | None, seed: int) -> None:
    """Run benchmark tasks."""
    from aquacontam._reproducibility import set_seed

    set_seed(seed)

    # Ensure task modules are imported so tasks register themselves
    import aquacontam.benchmark.tasks  # noqa: F401
    from aquacontam.benchmark.registry import list_tasks, run_task

    if task:
        click.echo(f"Running benchmark task: {task}")
        try:
            result = run_task(task, model=None, data=None)
            click.echo(f"  Metrics: {result.metrics}")
        except Exception as exc:
            click.echo(f"  Error: {exc}", err=True)
            click.echo(
                "  Hint: CLI benchmark requires fitted models. "
                "Use the Python API for full benchmark runs.",
                err=True,
            )
            raise SystemExit(1) from None
    else:
        click.echo("Running all benchmark tasks...")
        tasks = list_tasks()
        click.echo(f"  Registered tasks: {[t.name for t in tasks]}")
        click.echo("  (Provide --task to run a specific task with data)")


@cli.command()
@click.option(
    "--output",
    "-o",
    default="aquacontam-dataset-v1.0",
    help="Output directory for the dataset package.",
)
@click.option("--version", default=DATASET_VERSION, help="Dataset version string.")
def export(output: str, version: str) -> None:
    """Package the dataset for distribution."""
    click.echo(f"Exporting dataset to: {output}")
    click.echo(f"  Version: {version}")
    click.echo("  (Provide water quality data via Python API for full export)")
    click.echo("  Example: from aquacontam.export import export_dataset")


@cli.group()
def leaderboard() -> None:
    """Leaderboard management commands."""


@leaderboard.command("validate")
@click.argument("path", type=click.Path(exists=True))
def leaderboard_validate(path: str) -> None:
    """Validate a submission JSON file."""
    from aquacontam.leaderboard.schema import load_submission

    try:
        data = load_submission(path)
        n_results = len(data["results"])
        click.echo(f"Valid submission: {data['model_name']} ({n_results} result(s))")
    except ValueError as exc:
        click.echo(f"Invalid: {exc}", err=True)
        raise SystemExit(1) from None


@leaderboard.command("rank")
@click.argument("directory", type=click.Path(exists=True))
def leaderboard_rank(directory: str) -> None:
    """Rank all submissions in a directory."""
    from aquacontam.leaderboard.ranking import generate_leaderboard_document

    doc = generate_leaderboard_document(directory)
    click.echo(doc)


@leaderboard.command("publish")
@click.argument("directory", type=click.Path(exists=True))
@click.option("-o", "--output", default="LEADERBOARD.md", help="Output file path.")
def leaderboard_publish(directory: str, output: str) -> None:
    """Generate full leaderboard document and write to file."""
    from pathlib import Path

    from aquacontam.leaderboard.ranking import generate_leaderboard_document

    doc = generate_leaderboard_document(directory)
    Path(output).write_text(doc)
    click.echo(f"Leaderboard written to {output}")


def main() -> int:
    """Entry point for the CLI."""
    cli()
    return 0


if __name__ == "__main__":
    sys.exit(main())
