import json
from multiprocessing import cpu_count
import subprocess
from pathlib import Path
from typing import cast as typing_cast

import click
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

from bonepick.train.data_utils import load_jsonl_dataset
from bonepick.cli import PathParamType
from bonepick.train.fasttext_utils import check_fasttext_binary
from bonepick.train.model2vec_utils import StaticModelForClassification, StaticModelForRegression
from bonepick.train.jq_utils import add_field_or_expression_command_options, field_or_expression


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
):
    task_type = "regression" if regression else "classification"
    click.echo(f"Starting model2vec {task_type} training...")
    click.echo(f"  Dataset directories: {', '.join(str(d) for d in dataset_dir)}")
    click.echo(f"  Output directory: {output_dir}")
    click.echo(f"  Text field: {text_field}")
    click.echo(f"  Label field: {label_field}")
    click.echo(f"  Model name: {model_name}")
    click.echo(f"  Task: {task_type}")

    click.echo(f"\nLoading dataset from {len(dataset_dir)} director{'y' if len(dataset_dir) == 1 else 'ies'}...")

    text_expression = field_or_expression(text_field, text_expression)
    label_expression = field_or_expression(label_field, label_expression)

    dataset_tuple = load_jsonl_dataset(
        dataset_dirs=list(dataset_dir),
        text_field_expression=text_expression,
        label_field_expression=label_expression,
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

        if output_dir is not None:
            click.echo(f"\nSaving model to {output_dir}...")
            output_dir.mkdir(parents=True, exist_ok=True)
            pipeline = model.to_pipeline()
            model_path = output_dir / "model"
            pipeline.save_pretrained(str(model_path))
            click.echo(f"Model saved to: {model_path}")
        else:
            click.echo("\nNo output directory specified, skipping model save.")

    click.echo("\nTraining complete!")


@click.command()
@click.option(
    "-d",
    "--dataset-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    multiple=True,
    help="Directory containing the dataset (can be specified multiple times)",
)
@click.option(
    "-o",
    "--output-dir",
    type=PathParamType(mkdir=True, is_dir=True),
    required=True,
    help="Directory to save the trained model",
)
@click.option("--learning-rate", type=float, default=0.1, help="Learning rate")
@click.option("--word-ngrams", type=int, default=3, help="Max length of word n-gram")
@click.option("--min-count", type=int, default=5, help="Minimal number of word occurrences")
@click.option("--epoch", type=int, default=3, help="Number of training epochs")
@click.option(
    "--bucket",
    type=int,
    default=2_000_000,
    help="Number of buckets for hashing n-grams",
)
@click.option("--min-char-ngram", type=int, default=0, help="Min length of char n-gram")
@click.option("--max-char-ngram", type=int, default=0, help="Max length of char n-gram")
@click.option("--window-size", type=int, default=5, help="Window size for word n-gram")
@click.option("--dimension", type=int, default=256, help="Size of word vectors")
@click.option(
    "--loss",
    type=click.Choice(["softmax", "hs", "ova"]),
    default="softmax",
    help="Loss function",
)
@click.option("--num-negatives", type=int, default=5, help="Number of negative samples")
@click.option(
    "--thread",
    type=int,
    default=cpu_count(),
    help="Number of threads (default: number of CPUs)",
)
@click.option(
    "--pretrained-vectors",
    type=PathParamType(exists=True, is_file=True),
    help="Path to pretrained vectors",
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--verbose", type=int, default=2, help="Verbosity level (0-2)")
def train_fasttext(
    dataset_dir: tuple[Path, ...],
    output_dir: Path,
    learning_rate: float,
    word_ngrams: int,
    min_count: int,
    epoch: int,
    bucket: int,
    min_char_ngram: int,
    max_char_ngram: int,
    window_size: int,
    dimension: int,
    num_negatives: int,
    loss: str,
    thread: int,
    pretrained_vectors: Path | None,
    seed: int,
    verbose: int,
):
    """Train a fasttext classifier by shelling out to the fasttext binary."""
    click.echo("Starting fasttext training...")
    click.echo(f"  Dataset directories: {', '.join(str(d) for d in dataset_dir)}")
    click.echo(f"  Output directory: {output_dir}")
    click.echo(f"  Learning rate: {learning_rate}")
    click.echo(f"  Word n-grams: {word_ngrams}")
    click.echo(f"  Min count: {min_count}")
    click.echo(f"  Epochs: {epoch}")
    click.echo(f"  Bucket: {bucket}")
    click.echo(f"  Char n-gram range: {min_char_ngram}-{max_char_ngram}")
    click.echo(f"  Window size: {window_size}")
    click.echo(f"  Dimension: {dimension}")
    click.echo(f"  Num negatives: {num_negatives}")
    click.echo(f"  Loss: {loss}")
    click.echo(f"  Threads: {thread}")
    click.echo(f"  Seed: {seed}")
    click.echo(f"  Verbose: {verbose}")
    if pretrained_vectors:
        click.echo(f"  Pretrained vectors: {pretrained_vectors}")

    fasttext_path = check_fasttext_binary()

    click.echo(f"\nCreating output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "model"

    # Collect all train.txt files from all directories
    train_files: list[Path] = []
    for d in dataset_dir:
        train_file = d / "train.txt"
        click.echo(f"Checking for training file: {train_file}")
        assert train_file.exists(), f"Train file {train_file} does not exist"
        assert train_file.is_file(), f"Train file {train_file} is not a file"
        train_files.append(train_file)
    click.echo(f"Found {len(train_files)} training file(s)")

    # If multiple directories, concatenate into a temporary file
    if len(train_files) == 1:
        train_file = train_files[0]
    else:
        combined_train = output_dir / "combined_train.txt"
        click.echo(f"Concatenating training files to: {combined_train}")
        with open(combined_train, "w", encoding="utf-8") as out_f:
            for tf in train_files:
                with open(tf, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
        train_file = combined_train

    # Build the training command
    click.echo("\nBuilding training command...")
    train_cmd = [
        str(fasttext_path),
        "supervised",
        "-input",
        str(train_file),
        "-output",
        str(model_prefix),
        "-dim",
        str(dimension),
        "-lr",
        str(learning_rate),
        "-wordNgrams",
        str(word_ngrams),
        "-minCount",
        str(min_count),
        "-epoch",
        str(epoch),
        "-bucket",
        str(bucket),
        "-minn",
        str(min_char_ngram),
        "-maxn",
        str(max_char_ngram),
        "-ws",
        str(window_size),
        "-neg",
        str(num_negatives),
        "-seed",
        str(seed),
        "-thread",
        str(thread),
        "-loss",
        loss,
        "-verbose",
        str(verbose),
        *(["-pretrainedVectors", str(pretrained_vectors)] if pretrained_vectors is not None else []),
    ]

    click.echo("\nTraining fasttext model...")
    click.echo(f"Command: {' '.join(train_cmd)}")

    train_result = subprocess.run(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if train_result.returncode != 0:
        raise click.ClickException(f"fasttext training failed with return code {train_result.returncode}")

    click.echo("Training subprocess completed successfully.")

    model_bin = model_prefix.with_suffix(".bin")
    click.echo(f"\nVerifying model file exists: {model_bin}")
    if not model_bin.exists():
        raise click.ClickException(f"Expected model file not found: {model_bin}")
    click.echo(f"Model file verified: {model_bin}")

    # Save training parameters to file
    click.echo("\nSaving training parameters...")
    params = {
        "train_command": " ".join(train_cmd),
        "parameters": {
            "dimension": dimension,
            "learning_rate": learning_rate,
            "word_ngrams": word_ngrams,
            "min_count": min_count,
            "epoch": epoch,
            "bucket": bucket,
            "min_char_ngram": min_char_ngram,
            "max_char_ngram": max_char_ngram,
            "window_size": window_size,
            "num_negatives": num_negatives,
            "loss": loss,
            "seed": seed,
        },
    }

    params_file = output_dir / "train_params.json"
    with open(params_file, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    click.echo(f"\nModel saved to: {model_bin}")
    click.echo(f"Training params saved to: {params_file}")
    click.echo("\nFasttext training complete!")
