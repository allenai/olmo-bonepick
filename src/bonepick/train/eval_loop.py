import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast as typing_cast

import click
import yaml
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from model2vec.inference import StaticModelPipeline

from bonepick.train.data_utils import (
    load_jsonl_dataset,
    load_fasttext_dataset,
    FasttextDatasetSplit,
)
from bonepick.train.fasttext_utils import fasttext_dataset_signature
from bonepick.cli import PathParamType
from bonepick.train.fasttext_utils import check_fasttext_binary


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


def compute_detailed_metrics(pipeline: StaticModelPipeline, texts: list[str], labels: list[str]) -> dict:
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
    """
    # Create temporary input file for predictions
    temp_input = temp_dir / "temp_predict_input.txt"
    gold_labels: list[str] = []
    with open(temp_input, "w", encoding="utf-8") as f:
        for element in dataset_split:
            # Write each text on a line (without label)
            f.write(element.text + "\n")
            gold_labels.append(element.label)

    # Encode labels
    label_encoder = LabelEncoder()
    y_true = typing_cast(np.ndarray, label_encoder.fit_transform(gold_labels))

    # Get names of classes
    plain_classes = typing_cast(list[str], label_encoder.classes_)
    encoded_classes = typing_cast(np.ndarray, label_encoder.transform(plain_classes)).flatten()

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
    y_proba = np.zeros((len(dataset_split), len(plain_classes)))
    for i, raw_prediction in enumerate(predict_result.stdout.strip().split("\n")):
        labels, probas = (
            (arr := raw_prediction.strip().split())[::2],
            [float(p) for p in arr[1::2]],
        )
        labels_enc = np.array(label_encoder.transform(labels))
        y_proba[i, labels_enc] = np.array(probas)

    return _compute_metrics_from_predictions(y_true, y_proba, encoded_classes, plain_classes)


def result_to_text(dataset_dir: tuple[Path, ...], model_dir: Path, results: dict) -> str:
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
    "-m",
    "--model-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
)
@click.option(
    "-t",
    "--text-field",
    type=str,
    default=None,
    help="field in dataset to use as text",
)
@click.option(
    "-l",
    "--label-field",
    type=str,
    default=None,
    help="field in dataset to use as label",
)
@click.option(
    "-tt",
    "--text-expression",
    type=str,
    default=".text",
    help="expression to extract text from dataset",
)
@click.option(
    "-ll",
    "--label-expression",
    type=str,
    default=".score",
    help="expression to extract label from dataset",
)
def eval_model2vec(
    dataset_dir: tuple[Path, ...],
    model_dir: Path,
    text_field: str | None,
    label_field: str | None,
    text_expression: str,
    label_expression: str,
):
    if text_field is not None:
        msg = (
            "[bold red]WARNING:[/bold red] [red]-t/--text-field[/red] is deprecated, "
            "use [red]-tt/--text-expression[/red] instead."
        )
        text_expression = f".{text_field}"

    if label_field is not None:
        msg = (
            "[bold red]WARNING:[/bold red] [red]-l/--label-field[/red] is deprecated, "
            "use [red]-ll/--label-expression[/red] instead."
        )
        click.echo(msg, err=True, color=True)
        label_expression = f".{label_field}"

    click.echo("Starting model2vec evaluation...")
    click.echo(f"  Dataset directories: {', '.join(str(d) for d in dataset_dir)}")
    click.echo(f"  Model directory: {model_dir}")
    click.echo(f"  Text field: {text_field}")
    click.echo(f"  Label field: {label_field}")

    pipeline_dir = model_dir / "model"
    click.echo(f"\nLoading model from {pipeline_dir}...")
    pipeline = StaticModelPipeline.from_pretrained(pipeline_dir)
    click.echo("Model loaded successfully.")

    click.echo(f"\nLoading dataset from {len(dataset_dir)} director{'y' if len(dataset_dir) == 1 else 'ies'}...")
    dt = load_jsonl_dataset(
        dataset_dirs=list(dataset_dir),
        text_field_expression=text_expression,
        label_field_expression=label_expression,
    )
    click.echo("Dataset loaded successfully.")
    click.echo(f"  Test samples: {len(dt.test.text)}")

    click.echo("\nEvaluating model on test data...")
    assert dt.test.label is not None, "Test labels are required"
    results = compute_detailed_metrics(pipeline, dt.test.text, typing_cast(list[str], dt.test.label))

    results_txt = result_to_text(dataset_dir, model_dir, results)
    click.echo(f"Evaluation results:\n{results_txt}\n")

    results_file = model_dir / f"results_{dt.test.signature[:6]}.yaml"
    click.echo(f"\nSaving results to {results_file}...")
    with open(results_file, "wt", encoding="utf-8") as f:
        f.write(results_txt)
    click.echo(f"Results saved to: {results_file}")
    click.echo("\nEvaluation complete!")


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
    "-m",
    "--model-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    help="Path to the trained fasttext model (.bin file)",
)
@click.option(
    "-t",
    "--text-field",
    type=str,
    default="text",
    help="field in dataset to use as text",
)
@click.option(
    "-l",
    "--label-field",
    type=str,
    default="score",
    help="field in dataset to use as label",
)
def eval_fasttext(
    dataset_dir: tuple[Path, ...],
    model_dir: Path,
    text_field: str,
    label_field: str,
):
    """Evaluate a fasttext classifier on a test set."""
    click.echo("Starting fasttext evaluation...")
    click.echo(f"  Dataset directories: {', '.join(str(d) for d in dataset_dir)}")
    click.echo(f"  Model directory: {model_dir}")
    click.echo(f"  Text field: {text_field}")
    click.echo(f"  Label field: {label_field}")

    fasttext_path = check_fasttext_binary()

    model_path = model_dir / "model.bin"
    assert model_path.exists(), f"Model file {model_path} does not exist"
    assert model_path.is_file(), f"Model file {model_path} is not a file"
    click.echo(f"Model file found: {model_path}")

    with TemporaryDirectory() as _temp_dir:
        # gotta work in from a temporary directory for two reasons:
        # 1. if a user has provided multiple dataset directories, we need
        #    to merge them into a single dataset because fasttext expects a single file
        # 2. we need to output predictions from fasttext into a temporary file
        #    so we can compute the metrics we want on the predictions

        temp_dir = Path(_temp_dir)

        click.echo(
            f"\nLoading dataset from {len(dataset_dir)} director{'y' if len(dataset_dir) == 1 else 'ies'}..."
        )
        dt = load_fasttext_dataset(dataset_dirs=list(dataset_dir), tempdir=temp_dir)
        click.echo("Dataset loaded successfully.")
        click.echo(f"  Test samples: {len(dt.test)}")

        click.echo("\nEvaluating model on test data...")
        results = compute_detailed_metrics_fasttext(
            model_path=model_path,
            dataset_split=dt.test,
            fasttext_path=fasttext_path,
            temp_dir=temp_dir,
        )
        results_txt = result_to_text(dataset_dir, model_dir, results)
        click.echo(f"Evaluation results:\n{results_txt}\n")

    results_file = model_dir / f"results_{fasttext_dataset_signature(dt.test.path)[:6]}.yaml"
    click.echo(f"\nSaving results to {results_file}...")
    with open(results_file, "wt", encoding="utf-8") as f:
        f.write(results_txt)
    click.echo(f"Results saved to: {results_file}")
    click.echo("\nEvaluation complete!")
