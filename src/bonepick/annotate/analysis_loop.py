import hashlib
import os
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal

import click
import msgspec
import smart_open
from lazy_imports import try_import
from tqdm import tqdm

from bonepick.cli import PathParamType
from bonepick.data.expressions import compile_jq
from bonepick.data.utils import FILE_SUFFIXES

with try_import() as extra_dependencies:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
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
            labels1_numeric = [float(label) for label in labels1]
            labels2_numeric = [float(label) for label in labels2]
        except (ValueError, TypeError):
            raise ValueError("For ordinal metrics, labels must be numeric or convertible to numeric")

        # Weighted Cohen's Kappa (quadratic weights)
        kappa = cohen_kappa_score(labels1, labels2, weights="quadratic")
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
    labels1_numeric = [float(label) for label in labels1]
    labels2_numeric = [float(label) for label in labels2]

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
    "-d",
    "--dataset-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    multiple=True,
    help="Dataset directory (can be specified multiple times)",
)
@click.option(
    "-l",
    "--label-expression",
    type=str,
    required=True,
    multiple=True,
    help="JQ expression to extract label from each row (e.g., '.label' or '.annotation.category'). Can be specified multiple times if each dataset has a different label expression.",
)
@click.option(
    "-k",
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
    """Compute agreement metrics between annotators.

    Compares annotations from two datasets containing the same samples
    annotated by different annotators or systems. Computes various
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
        assert kappa is not None, "Kappa is None, this should never happen!"

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


def _load_labels_and_keys_from_single_file(
    file_path: Path,
    label_expression: str,
    key_expression: str | None,
    label_type: Literal["ordinal", "value", "text"],
) -> tuple[list[Any], list[str]]:
    """Load labels and keys from a single file.

    Args:
        file_path: Path to the JSONL file
        label_expression: JQ expression to extract label from each row
        key_expression: JQ expression to extract key from each row (optional)
        label_type: Type of label ("ordinal" for int, "value" for float, "text" for string)

    Returns:
        Tuple of (labels, keys) where keys are strings
    """
    decoder = msgspec.json.Decoder()
    label_selector = compile_jq(label_expression)
    key_selector = compile_jq(key_expression) if key_expression else None

    labels: list[Any] = []
    keys: list[str] = []

    with smart_open.open(file_path, "rb") as f:  # pyright: ignore
        for line in f:
            row = decoder.decode(line)

            # Extract label using jq expression
            if (label_value := label_selector(row)) is None:
                raise ValueError(f"Label expression {label_expression} returned None for row {row}")

            # Cast label based on type
            if label_type == "ordinal":
                label_value = int(label_value)
            elif label_type == "value":
                label_value = float(label_value)
            else:  # text
                label_value = str(label_value)

            labels.append(label_value)

            # Extract key if expression provided
            if key_selector is not None:
                if (key_value := key_selector(row)) is None:
                    raise ValueError(f"Key expression {key_expression} returned None for row {row}")
                keys.append(str(key_value))

    return labels, keys


def load_labels_and_keys_from_path(
    dataset_path: Path,
    label_expression: str,
    key_expression: str | None,
    label_type: Literal["ordinal", "value", "text"],
    max_workers: int | None = None,
) -> tuple[list[Any], list[str]]:
    """Load labels and keys from a dataset path in parallel.

    Args:
        dataset_path: Path to the dataset directory containing JSONL files
        label_expression: JQ expression to extract label from each row
        key_expression: JQ expression to extract key from each row (optional)
        label_type: Type of label ("ordinal" for int, "value" for float, "text" for string)
        max_workers: Maximum number of parallel workers (defaults to CPU count)

    Returns:
        Tuple of (labels, keys) where keys are strings
    """
    max_workers = max_workers or os.cpu_count() or 1

    # Collect all files
    all_files: list[Path] = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = Path(root) / file
            if "".join(file_path.suffixes) not in FILE_SUFFIXES:
                continue
            all_files.append(file_path)

    if not all_files:
        return [], []

    labels: list[Any] = []
    keys: list[str] = []

    with ExitStack() as stack:
        pool_cls = ProcessPoolExecutor if max_workers > 1 else ThreadPoolExecutor
        pool = stack.enter_context(pool_cls(max_workers=max_workers))
        pbar = stack.enter_context(
            tqdm(total=len(all_files), desc="Loading files", unit=" files", unit_scale=True)
        )

        futures = []
        for file_path in all_files:
            future = pool.submit(
                _load_labels_and_keys_from_single_file,
                file_path=file_path,
                label_expression=label_expression,
                key_expression=key_expression,
                label_type=label_type,
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                file_labels, file_keys = future.result()
                labels.extend(file_labels)
                keys.extend(file_keys)
            except Exception as e:
                for f in futures:
                    f.cancel()
                raise e
            pbar.update(1)

    return labels, keys


def display_label_distribution(
    labels: list[Any],
    label_type: Literal["ordinal", "value", "text"],
    console: Console,
    max_width: int = 50,
):
    """Display label distribution with histogram visualization.

    Args:
        labels: List of labels
        label_type: Type of label
        console: Rich console for output
        max_width: Maximum width of histogram bars
    """
    total = len(labels)
    label_counts = Counter(labels)

    if label_type == "value":
        # For regression values, show statistics instead of counts
        console.print("[bold]Label Statistics:[/bold]")
        labels_float = [float(l) for l in labels]

        stats_table = Table(show_header=True)
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", justify="right", style="magenta")

        stats_table.add_row("Count", f"{total:,}")
        stats_table.add_row("Mean", f"{statistics.mean(labels_float):.4f}")
        stats_table.add_row("Std Dev", f"{statistics.stdev(labels_float):.4f}" if len(labels_float) > 1 else "N/A")
        stats_table.add_row("Min", f"{min(labels_float):.4f}")
        stats_table.add_row("Max", f"{max(labels_float):.4f}")
        stats_table.add_row("Median", f"{statistics.median(labels_float):.4f}")

        # Percentiles
        sorted_labels = sorted(labels_float)
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            idx = int(len(sorted_labels) * p / 100)
            idx = min(idx, len(sorted_labels) - 1)
            stats_table.add_row(f"P{p}", f"{sorted_labels[idx]:.4f}")

        console.print(stats_table)
        console.print()

        # Show histogram buckets for continuous values
        console.print("[bold]Value Distribution (10 buckets):[/bold]")
        min_val, max_val = min(labels_float), max(labels_float)
        bucket_size = (max_val - min_val) / 10 if max_val > min_val else 1
        buckets: Counter[int] = Counter()
        for val in labels_float:
            bucket_idx = min(int((val - min_val) / bucket_size), 9)
            buckets[bucket_idx] += 1

        max_count = max(buckets.values()) if buckets else 1
        hist_table = Table(show_header=True, box=None)
        hist_table.add_column("Range", justify="right", style="cyan", width=20)
        hist_table.add_column("Count", justify="right", style="magenta", width=12)
        hist_table.add_column("Percentile", justify="right", style="green", width=10)
        hist_table.add_column("Bar", style="blue")

        cumulative = 0
        for i in range(10):
            bucket_min = min_val + i * bucket_size
            bucket_max = min_val + (i + 1) * bucket_size
            count = buckets.get(i, 0)
            cumulative += count
            percentile = (cumulative / total) * 100

            bar_width = int((count / max_count) * max_width) if max_count > 0 else 0
            bar = "█" * bar_width

            hist_table.add_row(
                f"[{bucket_min:.2f}, {bucket_max:.2f})",
                f"{count:,} ({count / total:.1%})",
                f"{percentile:.1f}%",
                bar,
            )

        console.print(hist_table)

    else:
        # For ordinal and text labels, show counts per label
        console.print("[bold]Label Distribution:[/bold]")

        # Sort labels appropriately
        if label_type == "ordinal":
            sorted_labels = sorted(label_counts.keys())
        else:
            # Sort by count descending for text labels
            sorted_labels = [l for l, _ in label_counts.most_common()]

        max_count = max(label_counts.values()) if label_counts else 1

        hist_table = Table(show_header=True, box=None)
        hist_table.add_column("Label", justify="right", style="cyan", width=20)
        hist_table.add_column("Count", justify="right", style="magenta", width=12)
        hist_table.add_column("Percentile", justify="right", style="green", width=10)
        hist_table.add_column("Bar", style="blue")

        cumulative = 0
        for label in sorted_labels:
            count = label_counts[label]
            cumulative += count
            percentile = (cumulative / total) * 100

            bar_width = int((count / max_count) * max_width) if max_count > 0 else 0
            bar = "█" * bar_width

            # Truncate long labels
            label_str = str(label)
            if len(label_str) > 18:
                label_str = label_str[:15] + "..."

            hist_table.add_row(
                label_str,
                f"{count:,} ({count / total:.1%})",
                f"{percentile:.1f}%",
                bar,
            )

        console.print(hist_table)

    console.print()


def compute_quantile_bucket_boundaries(sorted_values: list[int], num_buckets: int) -> list[int]:
    """Compute bucket boundaries based on quantiles for roughly equal counts.

    Args:
        sorted_values: Sorted list of values
        num_buckets: Target number of buckets

    Returns:
        List of bucket boundaries (including min and max)
    """
    if not sorted_values:
        return []

    n = len(sorted_values)
    boundaries = [sorted_values[0]]

    for i in range(1, num_buckets):
        # Get value at this quantile
        idx = min(int(n * i / num_buckets), n - 1)
        val = sorted_values[idx]
        # Avoid duplicate boundaries
        if val > boundaries[-1]:
            boundaries.append(val)

    # Always include max + 1 for the final boundary (exclusive)
    boundaries.append(sorted_values[-1] + 1)

    return boundaries


def display_key_length_distribution(
    keys: list[str],
    console: Console,
    num_buckets: int = 10,
    max_width: int = 50,
):
    """Display distribution of key lengths using quantile-based buckets.

    Buckets are sized to have roughly similar counts (within an order of magnitude),
    rather than uniform width.

    Args:
        keys: List of keys (strings)
        console: Rich console for output
        num_buckets: Target number of buckets for the histogram
        max_width: Maximum width of histogram bars
    """
    if not keys:
        console.print("[yellow]No keys to analyze[/yellow]")
        return

    lengths = [len(k) for k in keys]
    total = len(lengths)
    min_len, max_len = min(lengths), max(lengths)

    console.print("[bold]Key Length Statistics:[/bold]")

    stats_table = Table(show_header=True)
    stats_table.add_column("Statistic", style="cyan")
    stats_table.add_column("Value", justify="right", style="magenta")

    stats_table.add_row("Count", f"{total:,}")
    stats_table.add_row("Mean Length", f"{statistics.mean(lengths):.1f}")
    stats_table.add_row("Std Dev", f"{statistics.stdev(lengths):.1f}" if len(lengths) > 1 else "N/A")
    stats_table.add_row("Min Length", f"{min_len:,}")
    stats_table.add_row("Max Length", f"{max_len:,}")
    stats_table.add_row("Median Length", f"{statistics.median(lengths):.1f}")

    # Percentiles
    sorted_lengths = sorted(lengths)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        idx = int(len(sorted_lengths) * p / 100)
        idx = min(idx, len(sorted_lengths) - 1)
        stats_table.add_row(f"P{p}", f"{sorted_lengths[idx]:,}")

    console.print(stats_table)
    console.print()

    # Compute quantile-based bucket boundaries
    boundaries = compute_quantile_bucket_boundaries(sorted_lengths, num_buckets)
    actual_num_buckets = len(boundaries) - 1

    if actual_num_buckets == 0:
        console.print("[yellow]All keys have the same length[/yellow]")
        return

    console.print(f"[bold]Key Length Distribution ({actual_num_buckets} buckets):[/bold]")

    # Count items in each bucket
    bucket_counts: list[int] = [0] * actual_num_buckets
    for length in lengths:
        # Binary search for the right bucket
        for i in range(actual_num_buckets):
            if boundaries[i] <= length < boundaries[i + 1]:
                bucket_counts[i] += 1
                break

    max_count = max(bucket_counts) if bucket_counts else 1
    hist_table = Table(show_header=True, box=None)
    hist_table.add_column("Range", justify="right", style="cyan", width=20)
    hist_table.add_column("Count", justify="right", style="magenta", width=12)
    hist_table.add_column("Percentile", justify="right", style="green", width=10)
    hist_table.add_column("Bar", style="blue")

    cumulative = 0
    for i in range(actual_num_buckets):
        bucket_min = boundaries[i]
        bucket_max = boundaries[i + 1]
        count = bucket_counts[i]
        cumulative += count
        percentile = (cumulative / total) * 100

        bar_width = int((count / max_count) * max_width) if max_count > 0 else 0
        bar = "█" * bar_width

        hist_table.add_row(
            f"[{bucket_min:,}, {bucket_max:,})",
            f"{count:,} ({count / total:.1%})",
            f"{percentile:.1f}%",
            bar,
        )

    console.print(hist_table)
    console.print()


@click.command()
@click.option(
    "-d",
    "--dataset-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    help="Dataset directory containing JSONL files (supports .jsonl, .jsonl.gz, .jsonl.zst)",
)
@click.option(
    "-l",
    "--label-expression",
    type=str,
    required=True,
    help="JQ expression to extract label from each row (e.g., '.label' or '.annotation.score')",
)
@click.option(
    "-k",
    "--key-expression",
    type=str,
    default=None,
    help="JQ expression to extract key from each row (e.g., '.text' or '.content'). If provided, shows key length distribution.",
)
@click.option(
    "-t",
    "--label-type",
    type=click.Choice(["ordinal", "value", "text"]),
    default="ordinal",
    help="Type of label: 'ordinal' (cast to int), 'value' (cast to float for regression), 'text' (cast to string for classification)",
)
@click.option(
    "-b",
    "--key-length-buckets",
    type=int,
    default=10,
    help="Number of buckets for key length distribution histogram (default: 10)",
)
def label_distribution(
    dataset_dir: Path,
    label_expression: str,
    key_expression: str | None,
    label_type: Literal["ordinal", "value", "text"],
    key_length_buckets: int,
):
    """Show label distribution in a dataset.

    Analyzes a dataset directory containing JSONL files and displays
    the distribution of labels with CLI visualizations, including percentiles
    and histogram bars.

    Examples:

        # Analyze ordinal labels (default)
        bonepick label-distribution \\
            --dataset-dir ./annotated_data \\
            --label-expression '.annotation.score'

        # Analyze regression values
        bonepick label-distribution \\
            --dataset-dir ./annotated_data \\
            --label-expression '.quality_score' \\
            --label-type value

        # Analyze text classification labels with key length stats
        bonepick label-distribution \\
            --dataset-dir ./annotated_data \\
            --label-expression '.category' \\
            --key-expression '.text' \\
            --label-type text
    """
    extra_dependencies.check()

    console = Console()

    console.print("\n[bold cyan]Label Distribution Analysis[/bold cyan]\n")
    console.print(
        f"[bold cyan]Dataset:[/bold cyan] {dataset_dir}\n"
        f"[bold cyan]Label Expression:[/bold cyan] {label_expression}\n"
        f"[bold cyan]Label Type:[/bold cyan] {label_type}\n"
    )
    if key_expression:
        console.print(f"[bold cyan]Key Expression:[/bold cyan] {key_expression}\n")

    # Load labels and keys
    labels, keys = load_labels_and_keys_from_path(
        dataset_path=dataset_dir,
        label_expression=label_expression,
        key_expression=key_expression,
        label_type=label_type,
    )
    console.print(f"Loaded {len(labels):,} samples\n")

    if not labels:
        console.print("[bold red]No data found in dataset![/bold red]")
        return

    # Display label distribution
    display_label_distribution(labels, label_type, console)

    # Display key length distribution if key expression provided
    if key_expression and keys:
        display_key_length_distribution(keys, console, num_buckets=key_length_buckets)
