from pathlib import Path
from typing import Any

import click
import numpy as np
import yaml

from bonepick.cli import PathParamType
from bonepick.data.expressions import generate_calibration_jq_expression
from bonepick.evals.display import display_results
from bonepick.evals.modeling import fit_linear_weights, fit_log_linear_weights
from bonepick.evals.utils import (
    compute_auc_with_ties,
    compute_calibration_metrics,
    compute_per_class_metrics,
    compute_rank_correlation_metrics,
    compute_regression_metrics,
    load_predictions_and_labels,
    load_predictions_and_labels_for_calibration,
)


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
def eval_calibration(
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
        bonepick eval-calibration \\
            -d ./annotated_data \\
            -p '.metadata.classifier.quality_score' \\
            -l '.annotation.rating'

        # Evaluate from multiple directories with output file
        bonepick eval-calibration \\
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
    help="JQ expression to extract prediction dict {component_name: value}",
)
@click.option(
    "-l",
    "--label-expression",
    type=str,
    required=True,
    help="JQ expression to extract gold label (ordinal scalar)",
)
@click.option(
    "-m",
    "--model-type",
    type=click.Choice(["linear", "log-linear"]),
    default="linear",
    help="Model type: 'linear' for linear regression, 'log-linear' for logistic. Defaults to 'linear'.",
)
@click.option(
    "-o",
    "--output-file",
    type=PathParamType(mkdir=False, is_file=False, optional=True),
    default=None,
    help="Output file for results (YAML format)",
)
@click.option(
    "--num-proc",
    type=int,
    default=None,
    help="Number of parallel workers (default: CPU count)",
)
def train_calibration(
    dataset_dir: tuple[Path, ...],
    prediction_expression: str,
    label_expression: str,
    model_type: str,
    output_file: Path | None,
    num_proc: int | None,
):
    """Train a calibration model to map prediction components to gold labels.

    Learns weights for each prediction component to approximate gold labels.
    Useful for understanding how different model prediction dimensions
    relate to human annotations or gold scores.

    The prediction expression must return a dict of {component_name: value}.
    The label expression returns a gold scalar which is normalized to [0, 1].
    The model fits weights to minimize error between weighted prediction
    components and normalized gold labels.

    Model types:
    - linear: score = clamp(sum(w_i * pred_i) + bias, 0, 1)
    - log-linear: score = sigmoid(sum(w_i * pred_i) + bias)

    Examples:

        # Train linear model mapping prediction components to gold ratings
        bonepick train-calibration \\
            -d ./annotated_data \\
            -p '.prediction.components' \\
            -l '.annotation.rating' \\
            -m linear

        # Train log-linear model with output file
        bonepick train-calibration \\
            -d ./data \\
            -p '.model_scores' \\
            -l '.gold_label' \\
            -m log-linear \\
            -o calibration_weights.yaml

    Prediction expression must return a dict like:
        {"clarity": 0.8, "relevance": 0.6, "accuracy": 0.9}
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from scipy.special import expit as sigmoid

    console = Console()

    console.print("\n[bold cyan]Calibration Model Training[/bold cyan]\n")
    console.print(f"[bold cyan]Dataset(s):[/bold cyan] {', '.join(str(d) for d in dataset_dir)}")
    console.print(f"[bold cyan]Prediction Expression:[/bold cyan] {prediction_expression}")
    console.print(f"[bold cyan]Label Expression:[/bold cyan] {label_expression}")
    console.print(f"[bold cyan]Model Type:[/bold cyan] {model_type}\n")

    # Load data: prediction_expression extracts dict, label_expression extracts scalar
    prediction_matrix, labels, unique_labels, component_names = load_predictions_and_labels_for_calibration(
        dataset_dirs=list(dataset_dir),
        prediction_expression=prediction_expression,
        label_expression=label_expression,
        max_workers=num_proc,
    )

    n_samples, n_components = prediction_matrix.shape
    num_classes = len(unique_labels)
    console.print(f"Loaded {n_samples:,} samples with {n_components} prediction components: {component_names}\n")
    console.print(f"Gold labels: {num_classes} unique values: {unique_labels}\n")

    # Normalize labels to [0, 1] range
    if num_classes > 1:
        normalized_labels = labels / (num_classes - 1)
    else:
        normalized_labels = labels.astype(float)

    label_min, label_max = normalized_labels.min(), normalized_labels.max()
    console.print(f"Normalized labels range: [{label_min:.4f}, {label_max:.4f}]\n")

    # Fit model
    console.print("[yellow]Fitting calibration model...[/yellow]\n")

    if model_type == "linear":
        weights, bias = fit_linear_weights(normalized_labels, prediction_matrix, component_names)
    else:
        weights, bias = fit_log_linear_weights(normalized_labels, prediction_matrix, component_names)

    # Compute fitted values
    if model_type == "linear":
        fitted = prediction_matrix @ np.array([weights[name] for name in component_names]) + bias
        fitted = np.clip(fitted, 0, 1)
    else:
        logits = prediction_matrix @ np.array([weights[name] for name in component_names]) + bias
        fitted = sigmoid(logits)

    # Compute fit metrics
    mse = float(np.mean((fitted - normalized_labels) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(fitted - normalized_labels)))
    ss_res = np.sum((normalized_labels - fitted) ** 2)
    ss_tot = np.sum((normalized_labels - np.mean(normalized_labels)) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Generate jq expression (uses prediction_expression since that's the dict)
    jq_expr = generate_calibration_jq_expression(weights, bias, model_type, prediction_expression)

    # Display results
    console.print(
        Panel.fit(
            f"[bold green]R-squared:[/bold green] {r_squared:.4f}\n"
            f"[bold green]RMSE:[/bold green] {rmse:.6f}\n"
            f"[bold green]MAE:[/bold green] {mae:.6f}\n"
            f"[bold green]MSE:[/bold green] {mse:.6f}\n"
            f"\n[bold]Bias:[/bold] {bias:.6f}",
            title="[bold cyan]Model Fit Metrics[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    # Weights table
    console.print("[bold]Learned Weights:[/bold]")
    weights_table = Table(show_header=True)
    weights_table.add_column("Component", style="cyan")
    weights_table.add_column("Weight", justify="right", style="magenta")
    weights_table.add_column("Contribution", style="blue")

    # Sort by absolute weight
    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    max_abs_weight = max(abs(w) for _, w in sorted_weights) if sorted_weights else 1.0

    for name, weight in sorted_weights:
        # Visual bar
        bar_len = int(abs(weight) / max_abs_weight * 20) if max_abs_weight > 0 else 0
        if weight >= 0:
            bar = "[green]" + "+" * bar_len + "[/green]"
        else:
            bar = "[red]" + "-" * bar_len + "[/red]"
        weights_table.add_row(name, f"{weight:.6f}", bar)

    console.print(weights_table)
    console.print()

    # JQ expression
    console.print("[bold]JQ Expression:[/bold]")
    console.print(f"[dim]{jq_expr}[/dim]\n")

    # Save to file if specified
    if output_file is not None:
        output_dict = {
            "dataset_dirs": [str(d) for d in dataset_dir],
            "prediction_expression": prediction_expression,
            "label_expression": label_expression,
            "model_type": model_type,
            "total_samples": n_samples,
            "component_names": component_names,
            "unique_labels": [str(l) for l in unique_labels],
            "weights": {name: round(weight, 6) for name, weight in weights.items()},
            "bias": round(bias, 6),
            "fit_metrics": {
                "r_squared": round(r_squared, 6),
                "rmse": round(rmse, 6),
                "mae": round(mae, 6),
                "mse": round(mse, 6),
            },
            "jq_expression": jq_expr,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(output_dict, f, sort_keys=False, indent=2, allow_unicode=True)

        console.print(f"[green]Results saved to {output_file}[/green]")
