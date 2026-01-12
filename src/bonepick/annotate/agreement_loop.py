import os
from pathlib import Path
from collections import defaultdict
from typing import Any
import hashlib

import click
import msgspec
import smart_open
from lazy_imports import try_import

from bonepick.train.data_utils import compile_jq, FILE_SUFFIXES
from bonepick.cli import PathParamType


with try_import() as extra_dependencies:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from sklearn.metrics import cohen_kappa_score, confusion_matrix


def compute_hash(value: Any) -> str:
    """Compute a hash for any value to use as a key."""
    # Convert value to string and hash it
    value_str = str(value)
    return hashlib.sha256(value_str.encode()).hexdigest()


def load_annotations_from_dataset(
    dataset_path: Path,
    label_expression: str,
    key_expression: str,
) -> dict[str, Any]:
    """Load annotations from a dataset directory.

    Returns a dict mapping hash(key) -> label for each row.
    """
    decoder = msgspec.json.Decoder()
    label_selector = compile_jq(label_expression)
    key_selector = compile_jq(key_expression)

    annotations: dict[str, Any] = {}
    rows = []

    # Walk through all files in the dataset
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = Path(root) / file
            if "".join(file_path.suffixes) not in FILE_SUFFIXES:
                continue

            with smart_open.open(file_path, "rb") as f:  # pyright: ignore
                for line in f:
                    row = decoder.decode(line)

                    # Extract key and label using jq expressions
                    key_value = key_selector(row)
                    label_value = label_selector(row)

                    # Hash the key to use as dict key
                    key_hash = compute_hash(key_value)

                    # Store annotation
                    annotations[key_hash] = label_value
                    rows.append(row)

    return annotations


def compute_agreement_metrics(labels1: list[Any], labels2: list[Any]) -> dict[str, float]:
    """Compute agreement metrics between two sets of labels."""
    # Simple agreement
    total = len(labels1)
    agreements = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2)
    agreement_rate = agreements / total if total > 0 else 0.0

    # Cohen's Kappa
    kappa = cohen_kappa_score(labels1, labels2)

    return {
        "agreement_rate": agreement_rate,
        "cohen_kappa": kappa,
        "total_samples": total,
        "agreements": agreements,
        "disagreements": total - agreements,
    }


def create_confusion_matrix(labels1: list[Any], labels2: list[Any]) -> tuple[Any, list[Any]]:
    """Create confusion matrix for the two label sets."""
    cm = confusion_matrix(labels1, labels2)
    unique_labels = sorted(set(labels1 + labels2))
    return cm, unique_labels


@click.command()
@click.option(
    "--dataset1",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    help="First dataset directory",
)
@click.option(
    "--dataset2",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    help="Second dataset directory",
)
@click.option(
    "--label-expression",
    type=str,
    required=True,
    help="JQ expression to extract label from each row (e.g., '.label' or '.annotation.category')",
)
@click.option(
    "--key-expression",
    type=str,
    required=True,
    help="JQ expression to extract unique key from each row (e.g., '.id' or '.text')",
)
@click.option(
    "--show-confusion-matrix/--no-confusion-matrix",
    is_flag=True,
    default=True,
    help="Show confusion matrix",
)
@click.option(
    "--show-disagreements/--no-disagreements",
    is_flag=True,
    default=False,
    help="Show samples where annotators disagreed",
)
@click.option(
    "--max-disagreements",
    type=int,
    default=10,
    help="Maximum number of disagreement examples to show",
)
def annotation_agreement(
    dataset1: Path,
    dataset2: Path,
    label_expression: str,
    key_expression: str,
    show_confusion_matrix: bool,
    show_disagreements: bool,
    max_disagreements: int,
):
    """Compare annotations between two datasets and compute agreement metrics.

    This command compares annotations from two datasets that contain the same
    samples annotated by different annotators or systems. It computes various
    agreement metrics including simple agreement rate and Cohen's Kappa.

    Examples:

        # Compare two annotation datasets using 'id' as key and 'label' as annotation
        bonepick annotation-agreement \\
            --dataset1 ./annotator1 \\
            --dataset2 ./annotator2 \\
            --label-expression '.label' \\
            --key-expression '.id'

        # Use nested fields and show disagreements
        bonepick annotation-agreement \\
            --dataset1 ./annotator1 \\
            --dataset2 ./annotator2 \\
            --label-expression '.annotation.category' \\
            --key-expression '.metadata.sample_id' \\
            --show-disagreements
    """
    # Check if extra dependencies are installed
    extra_dependencies.check()

    console = Console()

    console.print("\n[bold cyan]Annotation Agreement Analysis[/bold cyan]\n")
    console.print(f"Dataset 1: {dataset1}")
    console.print(f"Dataset 2: {dataset2}")
    console.print(f"Label expression: {label_expression}")
    console.print(f"Key expression: {key_expression}\n")

    # Load annotations from both datasets
    console.print("[yellow]Loading annotations from dataset 1...[/yellow]")
    annotations1 = load_annotations_from_dataset(dataset1, label_expression, key_expression)
    console.print(f"Loaded {len(annotations1):,} annotations from dataset 1\n")

    console.print("[yellow]Loading annotations from dataset 2...[/yellow]")
    annotations2 = load_annotations_from_dataset(dataset2, label_expression, key_expression)
    console.print(f"Loaded {len(annotations2):,} annotations from dataset 2\n")

    # Find intersection of keys
    keys1 = set(annotations1.keys())
    keys2 = set(annotations2.keys())
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1

    # Basic counts
    console.print("[bold]Dataset Coverage:[/bold]")
    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="magenta")

    table.add_row("Samples in dataset 1", f"{len(annotations1):,}")
    table.add_row("Samples in dataset 2", f"{len(annotations2):,}")
    table.add_row("Common samples", f"{len(common_keys):,}")
    table.add_row("Only in dataset 1", f"{len(only_in_1):,}")
    table.add_row("Only in dataset 2", f"{len(only_in_2):,}")

    console.print(table)
    console.print()

    if len(common_keys) == 0:
        console.print("[bold red]No common samples found between datasets![/bold red]")
        console.print("Check that your key expression correctly identifies the same samples in both datasets.")
        return

    # Extract labels for common keys
    labels1 = [annotations1[key] for key in common_keys]
    labels2 = [annotations2[key] for key in common_keys]

    # Compute agreement metrics
    console.print("[yellow]Computing agreement metrics...[/yellow]\n")
    metrics = compute_agreement_metrics(labels1, labels2)

    # Display metrics
    console.print(
        Panel.fit(
            f"[bold green]Agreement Rate:[/bold green] {metrics['agreement_rate']:.2%}\n"
            f"[bold green]Cohen's Kappa:[/bold green] {metrics['cohen_kappa']:.4f}\n"
            f"[bold]Agreements:[/bold] {metrics['agreements']:,} / {metrics['total_samples']:,}\n"
            f"[bold]Disagreements:[/bold] {metrics['disagreements']:,} / {metrics['total_samples']:,}",
            title="[bold cyan]Agreement Metrics[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    # Interpretation of Cohen's Kappa
    kappa = metrics["cohen_kappa"]
    if kappa < 0:
        interpretation = "Poor (less than chance)"
    elif kappa < 0.20:
        interpretation = "Slight"
    elif kappa < 0.40:
        interpretation = "Fair"
    elif kappa < 0.60:
        interpretation = "Moderate"
    elif kappa < 0.80:
        interpretation = "Substantial"
    else:
        interpretation = "Almost Perfect"

    console.print(f"[dim]Cohen's Kappa interpretation: {interpretation}[/dim]\n")

    # Label distribution
    unique_labels = sorted(set(labels1 + labels2))
    console.print("[bold]Label Distribution:[/bold]")

    label_table = Table(show_header=True)
    label_table.add_column("Label", style="cyan")
    label_table.add_column("Dataset 1", justify="right", style="magenta")
    label_table.add_column("Dataset 2", justify="right", style="blue")

    label_counts1 = defaultdict(int)
    label_counts2 = defaultdict(int)
    for l1, l2 in zip(labels1, labels2):
        label_counts1[l1] += 1
        label_counts2[l2] += 1

    for label in unique_labels:
        label_table.add_row(
            str(label),
            f"{label_counts1[label]:,} ({label_counts1[label] / len(labels1):.1%})",
            f"{label_counts2[label]:,} ({label_counts2[label] / len(labels2):.1%})",
        )

    console.print(label_table)
    console.print()

    # Confusion matrix
    if show_confusion_matrix and len(unique_labels) <= 20:
        console.print("[bold]Confusion Matrix:[/bold]")
        console.print("[dim](rows=dataset1, columns=dataset2)[/dim]\n")

        cm, _ = create_confusion_matrix(labels1, labels2)

        # Create table for confusion matrix
        cm_table = Table(show_header=True)
        cm_table.add_column("Dataset 1 \\ Dataset 2", style="cyan")
        for label in unique_labels:
            cm_table.add_column(str(label), justify="right")

        for i, label1 in enumerate(unique_labels):
            row = [str(label1)]
            for j, label2 in enumerate(unique_labels):
                count = cm[i, j]
                # Highlight diagonal (agreements) in green
                if i == j:
                    row.append(f"[bold green]{count:,}[/bold green]")
                elif count > 0:
                    row.append(f"[yellow]{count:,}[/yellow]")
                else:
                    row.append(f"[dim]{count:,}[/dim]")
            cm_table.add_row(*row)

        console.print(cm_table)
        console.print()
    elif show_confusion_matrix and len(unique_labels) > 20:
        console.print(f"[yellow]Skipping confusion matrix (too many labels: {len(unique_labels)})[/yellow]\n")

    # Show disagreement examples
    if show_disagreements:
        console.print(f"[bold]Disagreement Examples (max {max_disagreements}):[/bold]\n")

        disagreements = [
            (key, annotations1[key], annotations2[key])
            for key in common_keys
            if annotations1[key] != annotations2[key]
        ]

        if disagreements:
            for i, (key, label1, label2) in enumerate(disagreements[:max_disagreements]):
                console.print(f"[cyan]Example {i + 1}:[/cyan]")
                console.print(f"  Key hash: {key[:16]}...")
                console.print(f"  Dataset 1: [magenta]{label1}[/magenta]")
                console.print(f"  Dataset 2: [blue]{label2}[/blue]")
                console.print()

            if len(disagreements) > max_disagreements:
                console.print(f"[dim]... and {len(disagreements) - max_disagreements} more disagreements[/dim]\n")
        else:
            console.print("[green]No disagreements found! Perfect agreement.[/green]\n")
