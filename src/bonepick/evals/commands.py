import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from typing import cast as typing_cast

import click
import msgspec
import numpy as np
import smart_open
import yaml
from numpy.typing import NDArray
from tqdm import tqdm

from bonepick.cli import PathParamType
from bonepick.data.expressions import compile_jq
from bonepick.data.utils import FILE_SUFFIXES


def _load_predictions_from_single_file(
    file_path: Path,
    prediction_expression: str,
    label_expression: str,
) -> tuple[list[float], list[Any]]:
    """Load predictions and labels from a single JSONL file.

    Args:
        file_path: Path to the JSONL file
        prediction_expression: JQ expression to extract prediction (0-1 scalar)
        label_expression: JQ expression to extract ordinal label

    Returns:
        Tuple of (predictions, labels)
    """
    decoder = msgspec.json.Decoder()
    prediction_selector = compile_jq(prediction_expression)
    label_selector = compile_jq(label_expression)

    predictions: list[float] = []
    labels: list[Any] = []

    with smart_open.open(file_path, "rb") as f:  # pyright: ignore
        for line in f:
            row = decoder.decode(line)

            # Extract prediction
            pred_value = prediction_selector(row)
            if pred_value is None:
                raise ValueError(f"Prediction expression {prediction_expression} returned None for row")
            predictions.append(float(pred_value))

            # Extract label
            label_value = label_selector(row)
            if label_value is None:
                raise ValueError(f"Label expression {label_expression} returned None for row")
            labels.append(label_value)

    return predictions, labels


def load_predictions_and_labels(
    dataset_dirs: list[Path],
    prediction_expression: str,
    label_expression: str,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """Load predictions and labels from dataset directories.

    Args:
        dataset_dirs: List of dataset directories
        prediction_expression: JQ expression to extract prediction (0-1 scalar)
        label_expression: JQ expression to extract ordinal label
        max_workers: Maximum number of parallel workers

    Returns:
        Tuple of (predictions array, encoded labels array, unique sorted labels)
    """
    max_workers = max_workers or os.cpu_count() or 1

    # Collect all files from all directories
    all_files: list[Path] = []
    for dataset_dir in dataset_dirs:
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                file_path = Path(root) / file
                if "".join(file_path.suffixes) not in FILE_SUFFIXES:
                    continue
                all_files.append(file_path)

    if not all_files:
        raise click.ClickException("No data files found in the specified directories")

    all_predictions: list[float] = []
    all_labels: list[Any] = []

    with ExitStack() as stack:
        pool_cls = ProcessPoolExecutor if max_workers > 1 else ThreadPoolExecutor
        pool = stack.enter_context(pool_cls(max_workers=max_workers))
        pbar = stack.enter_context(
            tqdm(total=len(all_files), desc="Loading files", unit=" files", unit_scale=True)
        )

        futures = []
        for file_path in all_files:
            future = pool.submit(
                _load_predictions_from_single_file,
                file_path=file_path,
                prediction_expression=prediction_expression,
                label_expression=label_expression,
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                preds, labs = future.result()
                all_predictions.extend(preds)
                all_labels.extend(labs)
            except Exception as e:
                for f in futures:
                    f.cancel()
                raise e
            pbar.update(1)

    # Convert labels to ordinal encoding
    # Sort unique labels to establish ordering
    unique_labels = sorted(set(all_labels))
    label_to_rank = {label: rank for rank, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_rank[label] for label in all_labels])

    return np.array(all_predictions), encoded_labels, unique_labels


def compute_auc_with_ties(
    predictions: NDArray[np.float64],
    labels: NDArray[np.int64],
    num_classes: int,
) -> dict[str, float | None]:
    """Compute AUC metrics handling ties in ordinal labels.

    For ordinal labels with ties, we compute:
    1. Macro AUC: Average of one-vs-rest AUCs for each class
    2. Ordinal AUC: Average pairwise AUC between adjacent classes
    3. Weighted AUC: Weighted average based on class frequencies

    Uses the corrected Mann-Whitney U statistic for ties.

    Args:
        predictions: Array of prediction scores (0-1)
        labels: Array of encoded ordinal labels (0, 1, 2, ...)
        num_classes: Number of unique classes

    Returns:
        Dictionary with various AUC metrics
    """
    from scipy import stats

    results: dict[str, float | None] = {}

    # One-vs-rest AUC for each class
    ovr_aucs: list[float] = []
    class_weights: list[int] = []

    for class_idx in range(num_classes):
        binary_labels = (labels >= class_idx).astype(int)

        # Check if both classes are present
        if len(np.unique(binary_labels)) < 2:
            continue

        # Use Mann-Whitney U statistic which handles ties correctly
        pos_preds = predictions[binary_labels == 1]
        neg_preds = predictions[binary_labels == 0]

        if len(pos_preds) == 0 or len(neg_preds) == 0:
            continue

        # Mann-Whitney U with tie correction
        statistic, _ = stats.mannwhitneyu(pos_preds, neg_preds, alternative="greater", method="asymptotic")
        auc = statistic / (len(pos_preds) * len(neg_preds))

        ovr_aucs.append(auc)
        class_weights.append(len(pos_preds))

    if ovr_aucs:
        results["macro_auc"] = float(np.mean(ovr_aucs))
        total_weight = sum(class_weights)
        results["weighted_auc"] = float(sum(auc * w for auc, w in zip(ovr_aucs, class_weights)) / total_weight)
    else:
        results["macro_auc"] = None
        results["weighted_auc"] = None

    # Pairwise AUC between adjacent classes (for ordinal data)
    pairwise_aucs: list[float] = []
    for i in range(num_classes - 1):
        mask = (labels == i) | (labels == i + 1)
        if mask.sum() < 2:
            continue

        subset_preds = predictions[mask]
        subset_labels = (labels[mask] == i + 1).astype(int)

        if len(np.unique(subset_labels)) < 2:
            continue

        pos_preds = subset_preds[subset_labels == 1]
        neg_preds = subset_preds[subset_labels == 0]

        if len(pos_preds) == 0 or len(neg_preds) == 0:
            continue

        statistic, _ = stats.mannwhitneyu(pos_preds, neg_preds, alternative="greater", method="asymptotic")
        auc = statistic / (len(pos_preds) * len(neg_preds))
        pairwise_aucs.append(auc)

    if pairwise_aucs:
        results["ordinal_auc"] = float(np.mean(pairwise_aucs))
    else:
        results["ordinal_auc"] = None

    return results


def compute_rank_correlation_metrics(
    predictions: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> dict[str, float]:
    """Compute rank correlation metrics with proper tie handling.

    Args:
        predictions: Array of prediction scores
        labels: Array of ordinal labels

    Returns:
        Dictionary with correlation metrics
    """
    from scipy import stats

    results: dict[str, float] = {}

    # Spearman correlation (handles ties via average ranks)
    spearman_corr, spearman_p = stats.spearmanr(predictions, labels)
    results["spearman_correlation"] = float(typing_cast(NDArray[np.float64], spearman_corr))
    results["spearman_pvalue"] = float(typing_cast(NDArray[np.float64], spearman_p))

    # Kendall's Tau-b (designed for ties)
    kendall_corr, kendall_p = stats.kendalltau(predictions, labels, method="asymptotic")
    results["kendall_tau_b"] = float(typing_cast(NDArray[np.float64], kendall_corr))
    results["kendall_pvalue"] = float(typing_cast(NDArray[np.float64], kendall_p))

    # Pearson correlation (for comparison)
    pearson_corr, pearson_p = stats.pearsonr(predictions, labels)
    results["pearson_correlation"] = float(typing_cast(NDArray[np.float64], pearson_corr))
    results["pearson_pvalue"] = float(typing_cast(NDArray[np.float64], pearson_p))

    return results


def compute_regression_metrics(
    predictions: NDArray[np.float64],
    labels: NDArray[np.int64],
    num_classes: int,
) -> dict[str, float]:
    """Compute regression-style metrics treating ordinal labels as numeric.

    Normalizes labels to [0, 1] range for comparison with predictions.

    Args:
        predictions: Array of prediction scores (0-1)
        labels: Array of ordinal labels (0 to num_classes-1)
        num_classes: Number of unique classes

    Returns:
        Dictionary with regression metrics
    """
    # Normalize labels to [0, 1] range
    if num_classes > 1:
        normalized_labels = labels / (num_classes - 1)
    else:
        normalized_labels = labels.astype(float)

    # MSE
    mse = float(np.mean((predictions - normalized_labels) ** 2))

    # RMSE
    rmse = float(np.sqrt(mse))

    # MAE
    mae = float(np.mean(np.abs(predictions - normalized_labels)))

    # R-squared
    ss_res = np.sum((normalized_labels - predictions) ** 2)
    ss_tot = np.sum((normalized_labels - np.mean(normalized_labels)) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared,
    }


def compute_calibration_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration metrics for ordinal predictions.

    Args:
        predictions: Array of prediction scores (0-1)
        labels: Array of ordinal labels
        num_classes: Number of unique classes
        num_bins: Number of bins for calibration analysis

    Returns:
        Dictionary with calibration metrics and bin data
    """
    # Normalize labels to [0, 1]
    if num_classes > 1:
        normalized_labels = labels / (num_classes - 1)
    else:
        normalized_labels = labels.astype(float)

    # Bin predictions
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges[1:-1])

    bin_data = []
    ece = 0.0  # Expected Calibration Error

    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue

        bin_preds = predictions[mask]
        bin_labels = normalized_labels[mask]

        mean_pred = float(np.mean(bin_preds))
        mean_label = float(np.mean(bin_labels))
        count = int(mask.sum())

        bin_data.append(
            {
                "bin_start": float(bin_edges[bin_idx]),
                "bin_end": float(bin_edges[bin_idx + 1]),
                "mean_prediction": mean_pred,
                "mean_label": mean_label,
                "count": count,
                "calibration_error": abs(mean_pred - mean_label),
            }
        )

        # Weighted contribution to ECE
        ece += (count / len(predictions)) * abs(mean_pred - mean_label)

    return {
        "expected_calibration_error": float(ece),
        "bins": bin_data,
    }


def compute_per_class_metrics(
    predictions: NDArray[np.float64],
    labels: NDArray[np.int64],
    unique_labels: list[Any],
) -> list[dict[str, Any]]:
    """Compute per-class statistics.

    Args:
        predictions: Array of prediction scores
        labels: Array of encoded labels
        unique_labels: List of original label values

    Returns:
        List of per-class metric dictionaries
    """
    per_class = []

    for class_idx, label_value in enumerate(unique_labels):
        mask = labels == class_idx
        count = int(mask.sum())

        if count == 0:
            continue

        class_preds = predictions[mask]

        per_class.append(
            {
                "label": label_value,
                "count": count,
                "mean_prediction": float(np.mean(class_preds)),
                "std_prediction": float(np.std(class_preds)),
                "min_prediction": float(np.min(class_preds)),
                "max_prediction": float(np.max(class_preds)),
                "median_prediction": float(np.median(class_preds)),
            }
        )

    return per_class


def display_results(
    results: dict[str, Any],
    unique_labels: list[Any],
    show_calibration: bool = True,
    show_histogram: bool = True,
    max_width: int = 50,
) -> None:
    """Display results with CLI visualization.

    Args:
        results: Dictionary with all computed metrics
        unique_labels: List of unique label values
        show_calibration: Whether to show calibration plot
        show_histogram: Whether to show prediction histogram per class
        max_width: Maximum width for histogram bars
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Overall metrics
    console.print("\n[bold cyan]Prediction Evaluation Results[/bold cyan]\n")

    # AUC metrics
    auc_text = ""
    if results.get("macro_auc") is not None:
        auc_text += f"[bold green]Macro AUC:[/bold green] {results['macro_auc']:.4f}\n"
    if results.get("weighted_auc") is not None:
        auc_text += f"[bold green]Weighted AUC:[/bold green] {results['weighted_auc']:.4f}\n"
    if results.get("ordinal_auc") is not None:
        auc_text += f"[bold green]Ordinal AUC (adjacent pairs):[/bold green] {results['ordinal_auc']:.4f}\n"

    # Correlation metrics
    corr_text = (
        f"[bold green]Spearman Correlation:[/bold green] {results['spearman_correlation']:.4f} "
        f"(p={results['spearman_pvalue']:.2e})\n"
        f"[bold green]Kendall's Tau-b:[/bold green] {results['kendall_tau_b']:.4f} "
        f"(p={results['kendall_pvalue']:.2e})\n"
        f"[bold green]Pearson Correlation:[/bold green] {results['pearson_correlation']:.4f} "
        f"(p={results['pearson_pvalue']:.2e})\n"
    )

    # Regression metrics
    reg_text = (
        f"[bold green]MSE:[/bold green] {results['mse']:.6f}\n"
        f"[bold green]RMSE:[/bold green] {results['rmse']:.6f}\n"
        f"[bold green]MAE:[/bold green] {results['mae']:.6f}\n"
        f"[bold green]R-squared:[/bold green] {results['r_squared']:.4f}\n"
    )

    # Calibration
    cal_text = (
        f"[bold green]Expected Calibration Error:[/bold green] {results['expected_calibration_error']:.4f}\n"
    )

    metrics_text = auc_text + "\n" + corr_text + "\n" + reg_text + "\n" + cal_text
    metrics_text += f"\n[bold]Total Samples:[/bold] {results['total_samples']:,}"

    console.print(
        Panel.fit(
            metrics_text,
            title="[bold cyan]Overall Metrics[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    # Per-class table
    console.print("[bold]Per-Class Statistics:[/bold]")
    class_table = Table(show_header=True)
    class_table.add_column("Label", style="cyan")
    class_table.add_column("Count", justify="right", style="magenta")
    class_table.add_column("Mean Pred", justify="right")
    class_table.add_column("Std Pred", justify="right")
    class_table.add_column("Min", justify="right")
    class_table.add_column("Median", justify="right")
    class_table.add_column("Max", justify="right")

    for pc in results["per_class_metrics"]:
        class_table.add_row(
            str(pc["label"]),
            f"{pc['count']:,}",
            f"{pc['mean_prediction']:.4f}",
            f"{pc['std_prediction']:.4f}",
            f"{pc['min_prediction']:.4f}",
            f"{pc['median_prediction']:.4f}",
            f"{pc['max_prediction']:.4f}",
        )

    console.print(class_table)
    console.print()

    # Prediction histogram per class
    if show_histogram:
        console.print("[bold]Prediction Distribution by Class:[/bold]")
        console.print("[dim](Shows mean prediction with std dev range)[/dim]\n")

        # Find global min/max for consistent scale
        all_means = [pc["mean_prediction"] for pc in results["per_class_metrics"]]

        hist_table = Table(show_header=True, box=None)
        hist_table.add_column("Label", justify="right", style="cyan", width=12)
        hist_table.add_column("Mean", justify="right", style="magenta", width=8)
        hist_table.add_column("Distribution", style="blue")

        for pc in results["per_class_metrics"]:
            mean_pos = int(pc["mean_prediction"] * max_width)
            std = pc["std_prediction"]

            # Create bar showing mean with std range
            bar = [" "] * (max_width + 1)

            # Mark std range
            std_low = max(0, int((pc["mean_prediction"] - std) * max_width))
            std_high = min(max_width, int((pc["mean_prediction"] + std) * max_width))
            for i in range(std_low, std_high + 1):
                bar[i] = "-"

            # Mark mean
            bar[mean_pos] = "|"

            bar_str = "".join(bar)
            hist_table.add_row(
                str(pc["label"]),
                f"{pc['mean_prediction']:.3f}",
                f"0[{bar_str}]1",
            )

        console.print(hist_table)
        console.print()

    # Calibration plot
    if show_calibration and results.get("calibration_bins"):
        console.print("[bold]Calibration Plot:[/bold]")
        console.print("[dim](Perfect calibration: mean prediction = mean label)[/dim]\n")

        cal_table = Table(show_header=True, box=None)
        cal_table.add_column("Bin", justify="right", style="cyan", width=15)
        cal_table.add_column("Count", justify="right", style="magenta", width=8)
        cal_table.add_column("Mean Pred", justify="right", width=10)
        cal_table.add_column("Mean Label", justify="right", width=10)
        cal_table.add_column("Error", justify="right", width=8)
        cal_table.add_column("Visualization", style="blue")

        for bin_data in results["calibration_bins"]:
            # Create visual comparison
            pred_pos = int(bin_data["mean_prediction"] * 20)
            label_pos = int(bin_data["mean_label"] * 20)

            vis = [" "] * 21
            vis[label_pos] = "L"
            vis[pred_pos] = "P" if pred_pos != label_pos else "="

            cal_table.add_row(
                f"[{bin_data['bin_start']:.1f}, {bin_data['bin_end']:.1f})",
                f"{bin_data['count']:,}",
                f"{bin_data['mean_prediction']:.3f}",
                f"{bin_data['mean_label']:.3f}",
                f"{bin_data['calibration_error']:.3f}",
                "".join(vis),
            )

        console.print(cal_table)
        console.print("[dim]L=Label, P=Prediction, ==Match[/dim]")
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
    "-p",
    "--prediction-expression",
    type=str,
    required=True,
    help="JQ expression to extract prediction score (0-1 scalar)",
)
@click.option(
    "-l",
    "--label-expression",
    type=str,
    required=True,
    help="JQ expression to extract ordinal label",
)
@click.option(
    "-o",
    "--output-file",
    type=PathParamType(mkdir=False, is_file=False, optional=True),
    default=None,
    help="Output file for results (YAML format)",
)
@click.option(
    "--num-calibration-bins",
    type=int,
    default=10,
    help="Number of bins for calibration analysis",
)
@click.option(
    "--num-proc",
    type=int,
    default=None,
    help="Number of parallel workers (default: CPU count)",
)
@click.option(
    "--show-calibration/--no-calibration",
    is_flag=True,
    default=True,
    help="Show calibration visualization",
)
@click.option(
    "--show-histogram/--no-histogram",
    is_flag=True,
    default=True,
    help="Show prediction histogram by class",
)
def eval_predictions(
    dataset_dir: tuple[Path, ...],
    prediction_expression: str,
    label_expression: str,
    output_file: Path | None,
    num_calibration_bins: int,
    num_proc: int | None,
    show_calibration: bool,
    show_histogram: bool,
):
    """Evaluate predictions against ordinal labels.

    Computes AUC, rank correlation, and regression metrics between
    scalar predictions (0-1) and ordinal gold labels. Handles ties
    in ordinal labels using appropriate statistical methods.

    Metrics computed:
    - AUC: Macro, weighted, and ordinal (adjacent pairs) using Mann-Whitney U
    - Correlation: Spearman, Kendall's Tau-b, Pearson
    - Regression: MSE, RMSE, MAE, R-squared (labels normalized to 0-1)
    - Calibration: Expected Calibration Error with bin analysis

    Examples:

        # Evaluate predictions from a single dataset
        bonepick eval-predictions \\
            -d ./annotated_data \\
            -p '.metadata.classifier.quality_score' \\
            -l '.annotation.rating'

        # Evaluate from multiple directories with output file
        bonepick eval-predictions \\
            -d ./data1 -d ./data2 \\
            -p '.prediction' \\
            -l '.label' \\
            -o results.yaml
    """
    from rich.console import Console

    console = Console()

    console.print("\n[bold cyan]Prediction Evaluation[/bold cyan]\n")
    console.print(f"[bold cyan]Dataset(s):[/bold cyan] {', '.join(str(d) for d in dataset_dir)}")
    console.print(f"[bold cyan]Prediction Expression:[/bold cyan] {prediction_expression}")
    console.print(f"[bold cyan]Label Expression:[/bold cyan] {label_expression}\n")

    # Load data
    predictions, labels, unique_labels = load_predictions_and_labels(
        dataset_dirs=list(dataset_dir),
        prediction_expression=prediction_expression,
        label_expression=label_expression,
        max_workers=num_proc,
    )

    num_classes = len(unique_labels)
    console.print(f"Loaded {len(predictions):,} samples with {num_classes} unique labels: {unique_labels}\n")

    # Validate predictions are in [0, 1] range
    pred_min, pred_max = predictions.min(), predictions.max()
    if pred_min < 0 or pred_max > 1:
        console.print(
            f"[yellow]Warning: Predictions outside [0,1] range: [{pred_min:.4f}, {pred_max:.4f}][/yellow]\n"
        )

    # Compute all metrics
    console.print("[yellow]Computing metrics...[/yellow]\n")

    results: dict[str, Any] = {
        "total_samples": len(predictions),
        "num_classes": num_classes,
        "unique_labels": unique_labels,
        "prediction_range": {"min": float(pred_min), "max": float(pred_max)},
    }

    # AUC metrics with tie handling
    auc_results = compute_auc_with_ties(predictions, labels, num_classes)
    results.update(auc_results)

    # Rank correlation metrics
    corr_results = compute_rank_correlation_metrics(predictions, labels)
    results.update(corr_results)

    # Regression metrics
    reg_results = compute_regression_metrics(predictions, labels, num_classes)
    results.update(reg_results)

    # Calibration metrics
    cal_results = compute_calibration_metrics(predictions, labels, num_classes, num_calibration_bins)
    results["expected_calibration_error"] = cal_results["expected_calibration_error"]
    results["calibration_bins"] = cal_results["bins"]

    # Per-class metrics
    results["per_class_metrics"] = compute_per_class_metrics(predictions, labels, unique_labels)

    # Display results
    display_results(
        results,
        unique_labels,
        show_calibration=show_calibration,
        show_histogram=show_histogram,
    )

    # Save to file if specified
    if output_file is not None:
        output_dict = {
            "dataset_dirs": [str(d) for d in dataset_dir],
            "prediction_expression": prediction_expression,
            "label_expression": label_expression,
            "total_samples": results["total_samples"],
            "num_classes": results["num_classes"],
            "unique_labels": [str(l) for l in results["unique_labels"]],
            "prediction_range": results["prediction_range"],
            "auc_metrics": {
                "macro_auc": results.get("macro_auc"),
                "weighted_auc": results.get("weighted_auc"),
                "ordinal_auc": results.get("ordinal_auc"),
            },
            "correlation_metrics": {
                "spearman_correlation": results["spearman_correlation"],
                "spearman_pvalue": results["spearman_pvalue"],
                "kendall_tau_b": results["kendall_tau_b"],
                "kendall_pvalue": results["kendall_pvalue"],
                "pearson_correlation": results["pearson_correlation"],
                "pearson_pvalue": results["pearson_pvalue"],
            },
            "regression_metrics": {
                "mse": results["mse"],
                "rmse": results["rmse"],
                "mae": results["mae"],
                "r_squared": results["r_squared"],
            },
            "calibration": {
                "expected_calibration_error": results["expected_calibration_error"],
                "bins": results["calibration_bins"],
            },
            "per_class_metrics": [
                {
                    "label": str(pc["label"]),
                    "count": pc["count"],
                    "mean_prediction": round(pc["mean_prediction"], 6),
                    "std_prediction": round(pc["std_prediction"], 6),
                    "min_prediction": round(pc["min_prediction"], 6),
                    "max_prediction": round(pc["max_prediction"], 6),
                    "median_prediction": round(pc["median_prediction"], 6),
                }
                for pc in results["per_class_metrics"]
            ],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(output_dict, f, sort_keys=False, indent=2, allow_unicode=True)

        console.print(f"[green]Results saved to {output_file}[/green]")
