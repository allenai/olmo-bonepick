"""Display utilities for evaluation results."""

from typing import Any


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

    # Per-class table (sorted by label)
    console.print("[bold]Per-Class Statistics:[/bold]")
    class_table = Table(show_header=True)
    class_table.add_column("Label", style="cyan")
    class_table.add_column("Count", justify="right", style="magenta")
    class_table.add_column("Mean Pred", justify="right")
    class_table.add_column("Std Pred", justify="right")
    class_table.add_column("Min", justify="right")
    class_table.add_column("Median", justify="right")
    class_table.add_column("Max", justify="right")

    sorted_per_class = sorted(results["per_class_metrics"], key=lambda x: x["label"])
    for pc in sorted_per_class:
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

        hist_table = Table(show_header=True, box=None)
        hist_table.add_column("Label", justify="right", style="cyan", width=12)
        hist_table.add_column("Mean", justify="right", style="magenta", width=8)
        hist_table.add_column("Distribution", style="blue")

        for pc in sorted_per_class:
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
