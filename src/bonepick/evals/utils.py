import os
import subprocess
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
from model2vec.inference import StaticModelPipeline
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from bonepick.data.expressions import compile_jq
from bonepick.data.utils import FILE_SUFFIXES, FasttextDatasetSplit


def _compute_metrics_from_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    encoded_classes: np.ndarray,
    plain_classes: list[str],
) -> dict:
    y_pred = np.argmax(y_proba, axis=1)

    # Calculate per-class metrics
    precision, recall, f1, support = typing_cast(
        tuple[np.ndarray, ...],
        precision_recall_fscore_support(y_true, y_pred, labels=encoded_classes),
    )

    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Calculate AUC (handle binary and multi-class cases)
    try:
        if len(encoded_classes) == 2:
            # Binary classification: use probabilities for positive class
            auc = roc_auc_score(y_true, y_proba[:, 1])
            per_class_auc = {str(plain_classes[1]): auc}
        else:
            # Multi-class: calculate AUC for each class (one-vs-rest)
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            per_class_auc = {}
            for i, class_label in enumerate(plain_classes):
                try:
                    class_auc = roc_auc_score((y_true == i).astype(int), y_proba[:, i])
                    per_class_auc[str(class_label)] = class_auc
                except ValueError:
                    # Handle case where a class might not be present
                    per_class_auc[str(class_label)] = None
    except ValueError:
        # Handle cases where AUC cannot be calculated
        auc = None
        per_class_auc = {}

    results = {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "macro_auc": float(auc) if auc is not None else None,
        "per_class_metrics": {},
    }

    for i, class_label in enumerate(plain_classes):
        class_name = str(class_label)
        results["per_class_metrics"][class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "auc": per_class_auc.get(class_name),
        }

    return results


def compute_detailed_metrics_model2vec(pipeline: StaticModelPipeline, texts: list[str], labels: list[str]) -> dict:
    """
    Compute detailed classification metrics using predict_proba.

    Returns precision, recall, F1, macro averages, and AUC for each class.
    """
    # Encode labels
    label_encoder = LabelEncoder()
    y_true = typing_cast(np.ndarray, label_encoder.fit_transform(labels))

    plain_classes = typing_cast(list[str], label_encoder.classes_)
    encoded_classes = typing_cast(np.ndarray, label_encoder.transform(plain_classes)).flatten()

    # Get probability predictions
    y_proba = pipeline.predict_proba(texts)

    # Build results dictionary
    return _compute_metrics_from_predictions(y_true, y_proba, encoded_classes, plain_classes)


def compute_detailed_metrics_fasttext(
    model_path: Path,
    dataset_split: FasttextDatasetSplit,
    fasttext_path: Path,
    temp_dir: Path,
) -> dict:
    """
    Compute detailed classification metrics for FastText using predict with probabilities.

    Returns precision, recall, F1, macro averages, and AUC for each class.

    Note: Labels predicted by FastText that are not present in gold labels are ignored
    (their probability mass is discarded). Metrics are computed only on gold label classes.
    """
    # Create temporary input file for predictions
    temp_input = temp_dir / "temp_predict_input.txt"
    gold_labels: list[str] = []
    with open(temp_input, "w", encoding="utf-8") as f:
        for element in dataset_split:
            # Write each text on a line (without label)
            f.write(element.text + "\n")
            gold_labels.append(element.label)

    # Encode labels using only gold labels (we only evaluate on classes present in gold)
    label_encoder = LabelEncoder()
    y_true = typing_cast(np.ndarray, label_encoder.fit_transform(gold_labels))

    # Get names of classes (only those in gold)
    plain_classes = typing_cast(list[str], label_encoder.classes_)
    encoded_classes = typing_cast(np.ndarray, label_encoder.transform(plain_classes)).flatten()
    gold_label_set = set(plain_classes)

    # Run fasttext predict with probabilities (k=-1 means all classes)
    predict_cmd = [
        str(fasttext_path),
        "predict-prob",
        str(model_path),
        str(temp_input),
        "-1",  # Return all class probabilities
    ]

    predict_result = subprocess.run(predict_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if predict_result.returncode != 0:
        raise RuntimeError(
            f"fasttext predict failed with return code {predict_result.returncode}\n"
            f"stderr: {predict_result.stderr}"
        )

    # Parse predictions - each line contains: __label__X prob __label__Y prob ...
    # Filter out any predicted labels not in gold labels (their probability is ignored)
    y_proba = np.zeros((len(dataset_split), len(plain_classes)))
    for i, raw_prediction in enumerate(predict_result.stdout.strip().split("\n")):
        arr = raw_prediction.strip().split()
        pred_labels = arr[::2]
        pred_probas = [float(p) for p in arr[1::2]]

        # Filter to only labels present in gold, then encode
        for label, proba in zip(pred_labels, pred_probas):
            if label in gold_label_set:
                label_enc = label_encoder.transform([label])[0]
                y_proba[i, label_enc] = proba

    return _compute_metrics_from_predictions(y_true, y_proba, encoded_classes, plain_classes)


def result_to_text(
    dataset_dir: tuple[Path, ...],
    model_dir: Path,
    results: dict,
    max_length: int | None = None,
    normalizer: str | None = None,
) -> str:
    per_class_metrics = [
        {
            **{"class_name": class_name},
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        for class_name, metrics in results.pop("per_class_metrics").items()
    ]

    output = {
        "dataset_dir": [str(d) for d in dataset_dir],
        "model_dir": str(model_dir),
        "overall_results": {k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()},
        "per_class_metrics": per_class_metrics,
    }
    return yaml.dump(output, sort_keys=False, indent=2)


def load_predictions_from_single_file(
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
                load_predictions_from_single_file,
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


def load_label_dicts_from_single_file(
    file_path: Path,
    prediction_expression: str,
    label_expression: str,
) -> tuple[list[float], list[dict[str, float]]]:
    """Load predictions and label dicts from a single JSONL file.

    Args:
        file_path: Path to the JSONL file
        prediction_expression: JQ expression to extract prediction (0-1 scalar)
        label_expression: JQ expression to extract label dict {label_name: value}

    Returns:
        Tuple of (predictions, label_dicts)
    """
    decoder = msgspec.json.Decoder()
    prediction_selector = compile_jq(prediction_expression)
    label_selector = compile_jq(label_expression)

    predictions: list[float] = []
    label_dicts: list[dict[str, float]] = []

    with smart_open.open(file_path, "rb") as f:  # pyright: ignore
        for line_num, line in enumerate(f, 1):
            row = decoder.decode(line)

            # Extract prediction
            pred_value = prediction_selector(row)
            if pred_value is None:
                raise ValueError(f"Prediction expression {prediction_expression} returned None for row")
            predictions.append(float(pred_value))

            # Extract label dict
            label_value = label_selector(row)
            if label_value is None:
                raise ValueError(f"Label expression {label_expression} returned None for row")
            if not isinstance(label_value, dict):
                raise ValueError(
                    f"Label expression must return a dict, got {type(label_value).__name__} "
                    f"at {file_path}:{line_num}"
                )
            # Validate all values are numeric
            for k, v in label_value.items():
                if not isinstance(v, (int, float)):
                    raise ValueError(
                        f"Label dict values must be numeric, got {type(v).__name__} for key '{k}' "
                        f"at {file_path}:{line_num}"
                    )
            label_dicts.append({str(k): float(v) for k, v in label_value.items()})

    return predictions, label_dicts


def load_predictions_and_label_dicts(
    dataset_dirs: list[Path],
    prediction_expression: str,
    label_expression: str,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load predictions and label dicts from dataset directories.

    Args:
        dataset_dirs: List of dataset directories
        prediction_expression: JQ expression to extract prediction (0-1 scalar)
        label_expression: JQ expression to extract label dict {label_name: value}
        max_workers: Maximum number of parallel workers

    Returns:
        Tuple of (predictions array, label_matrix array [n_samples, n_labels], label_names)
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
    all_label_dicts: list[dict[str, float]] = []

    with ExitStack() as stack:
        pool_cls = ProcessPoolExecutor if max_workers > 1 else ThreadPoolExecutor
        pool = stack.enter_context(pool_cls(max_workers=max_workers))
        pbar = stack.enter_context(
            tqdm(total=len(all_files), desc="Loading files", unit=" files", unit_scale=True)
        )

        futures = []
        for file_path in all_files:
            future = pool.submit(
                load_label_dicts_from_single_file,
                file_path=file_path,
                prediction_expression=prediction_expression,
                label_expression=label_expression,
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                preds, label_dicts = future.result()
                all_predictions.extend(preds)
                all_label_dicts.extend(label_dicts)
            except Exception as e:
                for f in futures:
                    f.cancel()
                raise e
            pbar.update(1)

    # Get all unique label names
    all_label_names: set[str] = set()
    for label_dict in all_label_dicts:
        all_label_names.update(label_dict.keys())
    label_names = sorted(all_label_names)

    # Build label matrix
    n_samples = len(all_label_dicts)
    n_labels = len(label_names)
    label_name_to_idx = {name: idx for idx, name in enumerate(label_names)}

    label_matrix = np.zeros((n_samples, n_labels), dtype=np.float64)
    for i, label_dict in enumerate(all_label_dicts):
        for name, value in label_dict.items():
            label_matrix[i, label_name_to_idx[name]] = value

    return np.array(all_predictions), label_matrix, label_names


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


def load_prediction_dicts_and_labels_from_single_file(
    file_path: Path,
    prediction_expression: str,
    label_expression: str,
) -> tuple[list[dict[str, float]], list[Any]]:
    """Load prediction dicts and labels from a single JSONL file.

    Args:
        file_path: Path to the JSONL file
        prediction_expression: JQ expression to extract prediction dict {component_name: value}
        label_expression: JQ expression to extract gold label (scalar)

    Returns:
        Tuple of (prediction_dicts, labels)
    """
    decoder = msgspec.json.Decoder()
    prediction_selector = compile_jq(prediction_expression)
    label_selector = compile_jq(label_expression)

    prediction_dicts: list[dict[str, float]] = []
    labels: list[Any] = []

    with smart_open.open(file_path, "rb") as f:  # pyright: ignore
        for line_num, line in enumerate(f, 1):
            row = decoder.decode(line)

            # Extract prediction dict
            pred_value = prediction_selector(row)
            if pred_value is None:
                raise ValueError(f"Prediction expression {prediction_expression} returned None for row")
            if not isinstance(pred_value, dict):
                raise ValueError(
                    f"Prediction expression must return a dict, got {type(pred_value).__name__} "
                    f"at {file_path}:{line_num}"
                )
            # Validate all values are numeric
            for k, v in pred_value.items():
                if not isinstance(v, (int, float)):
                    raise ValueError(
                        f"Prediction dict values must be numeric, got {type(v).__name__} for key '{k}' "
                        f"at {file_path}:{line_num}"
                    )
            prediction_dicts.append({str(k): float(v) for k, v in pred_value.items()})

            # Extract label
            label_value = label_selector(row)
            if label_value is None:
                raise ValueError(f"Label expression {label_expression} returned None for row")
            labels.append(label_value)

    return prediction_dicts, labels


def load_predictions_and_labels_for_calibration(
    dataset_dirs: list[Path],
    prediction_expression: str,
    label_expression: str,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Any], list[str]]:
    """Load prediction dicts and labels from dataset directories for calibration training.

    Args:
        dataset_dirs: List of dataset directories
        prediction_expression: JQ expression to extract prediction dict {component_name: value}
        label_expression: JQ expression to extract gold label (scalar)
        max_workers: Maximum number of parallel workers

    Returns:
        Tuple of (prediction_matrix [n_samples, n_components], encoded_labels, unique_labels, component_names)
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

    all_prediction_dicts: list[dict[str, float]] = []
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
                load_prediction_dicts_and_labels_from_single_file,
                file_path=file_path,
                prediction_expression=prediction_expression,
                label_expression=label_expression,
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                pred_dicts, labs = future.result()
                all_prediction_dicts.extend(pred_dicts)
                all_labels.extend(labs)
            except Exception as e:
                for f in futures:
                    f.cancel()
                raise e
            pbar.update(1)

    # Get all unique component names from prediction dicts
    all_component_names: set[str] = set()
    for pred_dict in all_prediction_dicts:
        all_component_names.update(pred_dict.keys())
    component_names = sorted(all_component_names)

    # Build prediction matrix
    n_samples = len(all_prediction_dicts)
    n_components = len(component_names)
    component_name_to_idx = {name: idx for idx, name in enumerate(component_names)}

    prediction_matrix = np.zeros((n_samples, n_components), dtype=np.float64)
    for i, pred_dict in enumerate(all_prediction_dicts):
        for name, value in pred_dict.items():
            prediction_matrix[i, component_name_to_idx[name]] = value

    # Convert labels to ordinal encoding
    unique_labels = sorted(set(all_labels))
    label_to_rank = {label: rank for rank, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_rank[label] for label in all_labels])

    return prediction_matrix, encoded_labels, unique_labels, component_names


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
