import os
from pathlib import Path
from collections import defaultdict, Counter
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
                    if (key_value := key_selector(row)) is None:
                        raise ValueError(f"Key expression {key_expression} returned None for row {row}")

                    if (label_value := label_selector(row)) is None:
                        raise ValueError(f"Label expression {label_expression} returned None for row {row}")

                    # Hash the key to use as dict key
                    key_hash = compute_hash(key_value)

                    # Store annotation
                    annotations[key_hash] = label_value
                    rows.append(row)

    return annotations


def compute_agreement_metrics(labels1: list[Any], labels2: list[Any], ordinal: bool = False) -> dict[str, float]:
    """Compute agreement metrics between two sets of labels.

    Args:
        labels1: First set of labels
        labels2: Second set of labels
        ordinal: If True, compute ordinal metrics (weighted kappa, MAE, RMSE)
    """
    # Simple agreement
    total = len(labels1)
    agreements = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2)
    agreement_rate = agreements / total if total > 0 else 0.0

    metrics = {
        "agreement_rate": agreement_rate,
        "total_samples": total,
        "agreements": agreements,
        "disagreements": total - agreements,
    }

    if ordinal:
        # Convert labels to numeric if they aren't already
        try:
            labels1_numeric = [float(l) for l in labels1]
            labels2_numeric = [float(l) for l in labels2]
        except (ValueError, TypeError):
            raise ValueError("For ordinal metrics, labels must be numeric or convertible to numeric")

        # Weighted Cohen's Kappa (quadratic weights)
        kappa = cohen_kappa_score(labels1, labels2, weights='quadratic')
        metrics["weighted_kappa"] = kappa

        # Mean Absolute Error
        differences = [abs(l1 - l2) for l1, l2 in zip(labels1_numeric, labels2_numeric)]
        mae = sum(differences) / total if total > 0 else 0.0
        metrics["mae"] = mae

        # Root Mean Squared Error
        squared_diffs = [(l1 - l2) ** 2 for l1, l2 in zip(labels1_numeric, labels2_numeric)]
        rmse = (sum(squared_diffs) / total) ** 0.5 if total > 0 else 0.0
        metrics["rmse"] = rmse

        # Pearson correlation
        mean1 = sum(labels1_numeric) / len(labels1_numeric)
        mean2 = sum(labels2_numeric) / len(labels2_numeric)

        numerator = sum((l1 - mean1) * (l2 - mean2) for l1, l2 in zip(labels1_numeric, labels2_numeric))
        denom1 = (sum((l1 - mean1) ** 2 for l1 in labels1_numeric)) ** 0.5
        denom2 = (sum((l2 - mean2) ** 2 for l2 in labels2_numeric)) ** 0.5

        if denom1 > 0 and denom2 > 0:
            correlation = numerator / (denom1 * denom2)
            metrics["pearson_correlation"] = correlation
        else:
            metrics["pearson_correlation"] = 0.0

    else:
        # Standard Cohen's Kappa (unweighted)
        kappa = cohen_kappa_score(labels1, labels2)
        metrics["cohen_kappa"] = kappa

    return metrics


def create_confusion_matrix(labels1: list[Any], labels2: list[Any]) -> tuple[Any, list[Any]]:
    """Create confusion matrix for the two label sets."""
    cm = confusion_matrix(labels1, labels2)
    unique_labels = sorted(set(labels1 + labels2))
    return cm, unique_labels


def display_difference_histogram(labels1: list[Any], labels2: list[Any], console: Console, max_width: int = 60):
    """Display a histogram of label differences for ordinal data.

    Args:
        labels1: First set of numeric labels
        labels2: Second set of numeric labels
        console: Rich console for output
        max_width: Maximum width of histogram bars
    """
    # Convert to numeric
    labels1_numeric = [float(l) for l in labels1]
    labels2_numeric = [float(l) for l in labels2]

    # Calculate differences (dataset2 - dataset1)
    differences = [l2 - l1 for l1, l2 in zip(labels1_numeric, labels2_numeric)]

    # Count frequency of each difference
    diff_counts = Counter(differences)

    # Sort by difference value
    sorted_diffs = sorted(diff_counts.items())

    if not sorted_diffs:
        console.print("[yellow]No differences to display[/yellow]")
        return

    # Find max count for scaling
    max_count = max(count for _, count in sorted_diffs)

    console.print("[bold]Difference Histogram (Dataset 2 - Dataset 1):[/bold]")
    console.print("[dim]Negative values: Dataset 1 rated higher | Positive values: Dataset 2 rated higher[/dim]\n")

    # Create histogram table
    hist_table = Table(show_header=True, box=None)
    hist_table.add_column("Difference", justify="right", style="cyan", width=12)
    hist_table.add_column("Count", justify="right", style="magenta", width=8)
    hist_table.add_column("Bar", style="blue")

    for diff, count in sorted_diffs:
        # Calculate bar width
        bar_width = int((count / max_count) * max_width) if max_count > 0 else 0
        bar = "█" * bar_width

        # Color the bar based on difference
        if diff == 0:
            bar_colored = f"[bold green]{bar}[/bold green]"
        elif diff > 0:
            bar_colored = f"[blue]{bar}[/blue]"
        else:
            bar_colored = f"[yellow]{bar}[/yellow]"

        # Format difference value
        if diff == int(diff):
            diff_str = f"{int(diff):+d}"
        else:
            diff_str = f"{diff:+.1f}"

        percentage = (count / len(differences)) * 100
        count_str = f"{count:,} ({percentage:.1f}%)"

        hist_table.add_row(diff_str, count_str, bar_colored)

    console.print(hist_table)
    console.print()


@click.command()
@click.option(
    "--dataset-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    multiple=True,
    help="Dataset directory (can be specified multiple times)",
)
@click.option(
    "--label-expression",
    type=str,
    required=True,
    multiple=True,
    help="JQ expression to extract label from each row (e.g., '.label' or '.annotation.category'). Can be specified multiple times if each dataset has a different label expression.",
)
@click.option(
    "--key-expression",
    type=str,
    required=True,
    multiple=True,
    help="JQ expression to extract unique key from each row (e.g., '.id' or '.text'). Can be specified multiple times if each dataset has a different key expression.",
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
@click.option(
    "--ordinal/--no-ordinal",
    is_flag=True,
    default=False,
    help="Treat labels as ordinal (ordered) values. Computes weighted kappa, MAE, RMSE, and shows difference histogram.",
)
def annotation_agreement(
    dataset_dir: tuple[Path, ...],
    label_expression: tuple[str, ...],
    key_expression: tuple[str, ...],
    show_confusion_matrix: bool,
    show_disagreements: bool,
    max_disagreements: int,
    ordinal: bool,
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

    if len(dataset_dir) < 2:
        raise ValueError("At least two dataset directories are required")

    if len(label_expression) != len(dataset_dir):
        if len(label_expression) != 1:
            raise ValueError(
                "If multiple label expressions are provided, "
                "they must be the same length as the number of dataset directories; "
                f"got {len(label_expression)} label expressions for {len(dataset_dir)} dataset directories!"
            )
        else:
            label_expression = label_expression * len(dataset_dir)

    if len(key_expression) != len(dataset_dir):
        if len(key_expression) != 1:
            raise ValueError(
                "If multiple key expressions are provided, "
                "they must be the same length as the number of dataset directories; "
                f"got {len(key_expression)} key expressions for {len(dataset_dir)} dataset directories!"
            )
        else:
            key_expression = key_expression * len(dataset_dir)

    console = Console()

    for i in range(len(dataset_dir) - 1):
        dataset1 = dataset_dir[i]
        dataset2 = dataset_dir[i + 1]
        label_expression1 = label_expression[i]
        label_expression2 = label_expression[i + 1]
        key_expression1 = key_expression[i]
        key_expression2 = key_expression[i + 1]

        console.print("\n[bold cyan]Annotation Agreement Analysis[/bold cyan]\n")
        console.print(
            "[bold cyan]Dataset 1:[/bold cyan]\n"
            f"  - path:       {dataset1}\n"
            f"  - label expr: {label_expression1}\n"
            f"  - key expr:   {key_expression1}\n"
        )
        console.print(
            "[bold cyan]Dataset 2:[/bold cyan]\n"
            f"  - path:       {dataset2}\n"
            f"  - label expr: {label_expression2}\n"
            f"  - key expr:   {key_expression2}\n"
        )

        # Load annotations from both datasets
        console.print("[yellow]Loading annotations from dataset 1...[/yellow]")
        annotations1 = load_annotations_from_dataset(
            dataset_path=dataset1,
            label_expression=label_expression1,
            key_expression=key_expression1,
        )
        console.print(f"Loaded {len(annotations1):,} annotations from dataset 1\n")

        console.print("[yellow]Loading annotations from dataset 2...[/yellow]")
        annotations2 = load_annotations_from_dataset(
            dataset_path=dataset2,
            label_expression=label_expression2,
            key_expression=key_expression2,
        )
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
        metrics = compute_agreement_metrics(labels1, labels2, ordinal=ordinal)

        # Display metrics
        if ordinal:
            metrics_text = (
                f"[bold green]Agreement Rate:[/bold green] {metrics['agreement_rate']:.2%}\n"
                f"[bold green]Weighted Kappa (quadratic):[/bold green] {metrics['weighted_kappa']:.4f}\n"
                f"[bold green]Mean Absolute Error (MAE):[/bold green] {metrics['mae']:.4f}\n"
                f"[bold green]Root Mean Squared Error (RMSE):[/bold green] {metrics['rmse']:.4f}\n"
                f"[bold green]Pearson Correlation:[/bold green] {metrics['pearson_correlation']:.4f}\n"
                f"[bold]Agreements:[/bold] {metrics['agreements']:,} / {metrics['total_samples']:,}\n"
                f"[bold]Disagreements:[/bold] {metrics['disagreements']:,} / {metrics['total_samples']:,}"
            )
        else:
            metrics_text = (
                f"[bold green]Agreement Rate:[/bold green] {metrics['agreement_rate']:.2%}\n"
                f"[bold green]Cohen's Kappa:[/bold green] {metrics['cohen_kappa']:.4f}\n"
                f"[bold]Agreements:[/bold] {metrics['agreements']:,} / {metrics['total_samples']:,}\n"
                f"[bold]Disagreements:[/bold] {metrics['disagreements']:,} / {metrics['total_samples']:,}"
            )

        console.print(
            Panel.fit(
                metrics_text,
                title="[bold cyan]Agreement Metrics[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print()

        # Interpretation of Kappa
        kappa = metrics.get("weighted_kappa") if ordinal else metrics.get("cohen_kappa")
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

        kappa_name = "Weighted Kappa" if ordinal else "Cohen's Kappa"
        console.print(f"[dim]{kappa_name} interpretation: {interpretation}[/dim]\n")

        # Display histogram for ordinal data
        if ordinal:
            display_difference_histogram(labels1, labels2, console)

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
                    console.print(
                        f"[dim]... and {len(disagreements) - max_disagreements} more disagreements[/dim]\n"
                    )
            else:
                console.print("[green]No disagreements found! Perfect agreement.[/green]\n")
