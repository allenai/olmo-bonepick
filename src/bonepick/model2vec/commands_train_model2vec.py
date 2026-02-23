from pathlib import Path
from typing import Literal
from typing import cast as typing_cast

import click
import numpy as np
import smart_open
import torch
import yaml
from lazy_imports import try_import
from model2vec.inference import StaticModelPipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from bonepick.cli import PathParamType, PCADimTypeParamType
from bonepick.data.expressions import add_field_or_expression_command_options, field_or_expression
from bonepick.data.normalizers import list_normalizers
from bonepick.data.utils import load_jsonl_dataset
from bonepick.evals import compute_detailed_metrics_model2vec, result_to_text
from bonepick.model2vec.utils import StaticModelForClassification, StaticModelForRegression

with try_import() as extra_dependencies:
    from model2vec.distill import distill


@click.command()
@click.option(
    "-d",
    "--dataset-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    multiple=True,
    help="Dataset directory (can be specified multiple times)",
)
@click.option("-o", "--output-dir", type=PathParamType(mkdir=True, is_dir=True), default=None)
@click.option(
    "-m",
    "--model-name",
    type=str,
    default="minishlab/potion-base-32M",
    help="model name",
)
@click.option("--learning-rate", type=float, default=1e-3, help="learning rate")
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="batch size (if not set, auto-computed)",
)
@click.option("--min-epochs", type=int, default=None, help="minimum number of epochs")
@click.option("--max-epochs", type=int, default=-1, help="max epochs (-1 for unlimited)")
@click.option("--early-stopping-patience", type=int, default=5, help="early stopping patience")
@click.option(
    "--loss-class-weight",
    type=click.Choice(["balanced", "uniform", "sqrt"], case_sensitive=False),
    default="uniform",
    help="Class weighting scheme for loss: 'uniform', 'balanced', 'sqrt' (default: uniform)",
)
@click.option(
    "--regression",
    is_flag=True,
    default=False,
    help="Train a regression model instead of classification",
)
@click.option(
    "--normalizer",
    type=click.Choice(list_normalizers()),
    default=None,
    help="Normalizer to use for text processing",
)
@click.option(
    "--max-length",
    type=int,
    default=None,
    help="Maximum length of text to process",
)
def train_model2vec(
    text_field: str | None,
    label_field: str | None,
    text_expression: str,
    label_expression: str,
    dataset_dir: tuple[Path, ...],
    output_dir: Path | None,
    model_name: str,
    learning_rate: float,
    batch_size: int,
    min_epochs: int,
    max_epochs: int,
    early_stopping_patience: int,
    loss_class_weight: str,
    regression: bool,
    normalizer: str | None,
    max_length: int | None,
):
    """Train a Model2Vec classifier or regressor.

    Uses static embeddings with PyTorch Lightning for efficient CPU-friendly inference.
    """
    task_type = "regression" if regression else "classification"

    text_expression = field_or_expression(text_field, text_expression)
    label_expression = field_or_expression(label_field, label_expression)

    click.echo(f"Starting model2vec {task_type} training...")
    click.echo(f"  Dataset directories: {', '.join(str(d) for d in dataset_dir)}")
    click.echo(f"  Output directory: {output_dir}")
    click.echo(f"  Model name: {model_name}")
    click.echo(f"  Task: {task_type}")
    click.echo(f"  Text expression: {text_expression}")
    click.echo(f"  Label expression: {label_expression}")
    if max_length is not None:
        click.echo(f"  Max length: {max_length}")
    if normalizer is not None:
        click.echo(f"  Normalizer: {normalizer}")

    click.echo(f"\nLoading dataset from {len(dataset_dir)} director{'y' if len(dataset_dir) == 1 else 'ies'}...")

    dataset_tuple = load_jsonl_dataset(
        dataset_dirs=list(dataset_dir),
        text_field_expression=text_expression,
        label_field_expression=label_expression,
        normalizer_name=normalizer,
        text_max_length=max_length,
    )
    click.echo("Dataset loaded successfully.")
    click.echo(f"  Train samples: {len(dataset_tuple.train.text)}")

    click.echo(f"\nLoading pretrained model: {model_name}...")

    if regression:
        model = StaticModelForRegression.from_pretrained(model_name=model_name)
        click.echo("Pretrained regression model loaded.")

        # Convert labels to floats for regression
        train_targets = [float(y) for y in typing_cast(list[str], dataset_tuple.train.label)]
        valid_targets = (
            [float(y) for y in typing_cast(list[str], dataset_tuple.valid.label)]
            if len(dataset_tuple.valid) > 0
            else None
        )

        click.echo("\nFitting regression model on training data...")
        model = model.fit(
            X=dataset_tuple.train.text,
            y=train_targets,
            learning_rate=learning_rate,
            batch_size=batch_size,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            X_val=dataset_tuple.valid.text if len(dataset_tuple.valid) > 0 else None,
            y_val=valid_targets,
            early_stopping_patience=early_stopping_patience,
        )
        click.echo("Model fitting complete.")

        if output_dir is not None:
            click.echo(f"\nSaving model to {output_dir}...")
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / "model"
            # Save the underlying static model for regression
            static_model = model.to_static_model()
            static_model.save_pretrained(str(model_path))
            # Also save the head weights separately
            head_path = output_dir / "regression_head.pt"
            torch.save(model.head.state_dict(), head_path)
            click.echo(f"Model saved to: {model_path}")
            click.echo(f"Regression head saved to: {head_path}")
        else:
            click.echo("\nNo output directory specified, skipping model save.")
    else:
        model = StaticModelForClassification.from_pretrained(model_name=model_name)
        click.echo("Pretrained classification model loaded.")

        if loss_class_weight != "uniform":
            encoded_labels = (label_encoder := LabelEncoder()).fit_transform(dataset_tuple.train.label)
            class_weights = compute_class_weight(
                "balanced",
                classes=label_encoder.transform(label_encoder.classes_),
                y=encoded_labels,
            )

            if loss_class_weight == "sqrt":
                class_weights = np.sqrt(class_weights)

            # renormalize to sum to 1
            class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float)

            click.echo(f"Class weights ({loss_class_weight}):")
            for class_name, class_weight in zip(label_encoder.classes_.tolist(), class_weights.tolist()):  # pyright: ignore
                click.echo(f"  {class_name}: {class_weight:.4f}")
        else:
            class_weights = None

        click.echo("\nFitting model on training data...")

        model = model.fit(
            X=dataset_tuple.train.text,
            y=typing_cast(list[str], dataset_tuple.train.label),
            learning_rate=learning_rate,
            batch_size=batch_size,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            X_val=dataset_tuple.valid.text if len(dataset_tuple.valid) > 0 else None,
            y_val=typing_cast(list[str], dataset_tuple.valid.label) if len(dataset_tuple.valid) > 0 else None,
            early_stopping_patience=early_stopping_patience,
            class_weight=class_weights,
        )
        click.echo("Model fitting complete.")

        if output_dir is None:
            click.echo("\nNo output directory specified, skipping model save.")
            return

        report_dict = {
            "text_expression": str(text_expression),
            "label_expression": str(label_expression),
            "dataset_dir": [str(d) for d in dataset_dir],
            "output_dir": str(output_dir),
            "model_name": str(model_name),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "min_epochs": int(min_epochs),
            "max_epochs": int(max_epochs),
            "early_stopping_patience": int(early_stopping_patience),
            "loss_class_weight": str(loss_class_weight),
            "regression": regression,
            "normalizer": str(normalizer) if normalizer is not None else None,
            "max_length": int(max_length) if max_length is not None else None,
        }

        report_file = output_dir / "report.yaml"
        with open(report_file, "w", encoding="utf-8") as f:
            yaml.dump(report_dict, f, indent=2)
        click.echo(f"Report saved to: {report_file}")

        click.echo(f"\nSaving model to {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline = model.to_pipeline()
        model_path = output_dir / "model"
        pipeline.save_pretrained(str(model_path))
        click.echo(f"Model saved to: {model_path}")


@click.command()
@click.option("-m", "--model-name-or-path", type=str, required=True)
@click.option(
    "-v",
    "--vocabulary-path",
    type=PathParamType(exists=True, is_file=True, optional=True),
    default=None,
)
@click.option("-o", "--output-dir", type=PathParamType(mkdir=True, is_dir=True), required=True)
@click.option("-d", "--pca-dims", type=PCADimTypeParamType(), default=256)
@click.option("-s", "--sif-coefficient", type=float, default=1e-4)
@click.option("-t", "--token-remove-pattern", type=str, default=r"\[unused\d+\]")
@click.option("-r", "--trust-remote-code", is_flag=True, default=False)
@click.option(
    "-q",
    "--quantize-to",
    default="float16",
    type=click.Choice(["float16", "float32", "float64", "int8"]),
)
@click.option("-k", "--vocabulary-quantization", type=int, default=None)
@click.option(
    "-p",
    "--pooling",
    default="mean",
    type=click.Choice(["mean", "last", "first", "pooler"]),
)
def distill_model2vec(
    model_name_or_path: str,
    vocabulary_path: Path | None,
    output_dir: Path,
    pca_dims: int | None | float | Literal["auto"] = 256,
    sif_coefficient: float = 1e-4,
    token_remove_pattern: str = r"\[unused\d+\]",
    trust_remote_code: bool = False,
    quantize_to: str = "float16",
    vocabulary_quantization: int | None = None,
    pooling: str = "mean",
):
    """Distill a transformer to static embeddings.

    Creates fast, lightweight Model2Vec embeddings from larger transformer models.
    """
    # check if the extra dependencies are installed
    extra_dependencies.check()

    click.echo("Starting model distillation...")

    # load vocabulary if provided
    if vocabulary_path is not None:
        click.echo(f"Loading vocabulary from {vocabulary_path}...")
        with smart_open.open(vocabulary_path, "rt", encoding="utf-8") as f:  # pyright: ignore
            vocabulary = [line.strip() for line in f]
        click.echo(f"Vocabulary loaded successfully with {len(vocabulary)} tokens.")
    else:
        vocabulary = None

    # print distillation parameters
    click.echo(f"Distilling model {model_name_or_path}")
    click.echo(f"PCA dimensions: {pca_dims}")
    click.echo(f"SIF coefficient: {sif_coefficient}")
    click.echo(f"Token remove pattern: {token_remove_pattern}")
    click.echo(f"Trust remote code: {trust_remote_code}")
    click.echo(f"Quantize to: {quantize_to}")
    click.echo(f"Vocabulary quantization: {vocabulary_quantization}")
    click.echo(f"Pooling: {pooling}")

    # Distill a Sentence Transformer model
    m2v_model = distill(
        model_name=model_name_or_path,
        vocabulary=vocabulary,
        pca_dims=pca_dims,
        sif_coefficient=sif_coefficient,
        token_remove_pattern=token_remove_pattern,
        trust_remote_code=trust_remote_code,
        quantize_to=quantize_to,
        vocabulary_quantization=vocabulary_quantization,
        pooling=pooling,
    )
    click.echo(f"Model distilled successfully; saving to {output_dir}...")

    # Save the model
    m2v_model.save_pretrained(output_dir)
    click.echo("Distillation complete!")


@click.command()
@add_field_or_expression_command_options
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
    "--max-length",
    type=int,
    default=None,
    help="Maximum length of text to process",
)
@click.option(
    "--normalizer",
    type=click.Choice(list_normalizers()),
    default=None,
    help="Normalizer to use for text processing",
)
def eval_model2vec(
    text_field: str | None,
    label_field: str | None,
    text_expression: str,
    label_expression: str,
    max_length: int | None,
    normalizer: str | None,
    dataset_dir: tuple[Path, ...],
    model_dir: Path,
):
    """Evaluate a Model2Vec classifier on test data.

    Computes precision, recall, F1, and AUC metrics per class and macro averages.
    """
    text_expression = field_or_expression(text_field, text_expression)
    label_expression = field_or_expression(label_field, label_expression)

    click.echo("Starting model2vec evaluation...")
    click.echo(f"  Dataset directories: {', '.join(str(d) for d in dataset_dir)}")
    click.echo(f"  Model directory: {model_dir}")
    click.echo(f"  Text expression: {text_expression}")
    click.echo(f"  Label expression: {label_expression}")
    click.echo(f"  Label field: {label_field}")
    if max_length is not None:
        click.echo(f"  Max length: {max_length}")
    if normalizer is not None:
        click.echo(f"  Normalizer: {normalizer}")

    pipeline_dir = model_dir / "model"
    click.echo(f"\nLoading model from {pipeline_dir}...")
    pipeline = StaticModelPipeline.from_pretrained(pipeline_dir)
    click.echo("Model loaded successfully.")

    click.echo(f"\nLoading dataset from {len(dataset_dir)} director{'y' if len(dataset_dir) == 1 else 'ies'}...")
    dt = load_jsonl_dataset(
        dataset_dirs=list(dataset_dir),
        text_field_expression=text_expression,
        label_field_expression=label_expression,
        normalizer_name=normalizer,
        text_max_length=max_length,
    )
    click.echo("Dataset loaded successfully.")
    click.echo(f"  Test samples: {len(dt.test.text)}")

    click.echo("\nEvaluating model on test data...")
    assert dt.test.label is not None, "Test labels are required"
    results = compute_detailed_metrics_model2vec(pipeline, dt.test.text, typing_cast(list[str], dt.test.label))

    results_txt = result_to_text(
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        results=results,
        max_length=max_length,
        normalizer=normalizer,
    )
    click.echo(f"Evaluation results:\n{results_txt}\n")

    results_file = model_dir / f"results_{dt.test.signature[:6]}.yaml"
    click.echo(f"\nSaving results to {results_file}...")
    with open(results_file, "wt", encoding="utf-8") as f:
        f.write(results_txt)
    click.echo(f"Results saved to: {results_file}")
    click.echo("\nEvaluation complete!")
