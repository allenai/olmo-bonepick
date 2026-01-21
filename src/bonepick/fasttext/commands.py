import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from multiprocessing import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory

import click
from tqdm import tqdm

from bonepick.cli import PathParamType
from bonepick.data.expressions import field_or_expression
from bonepick.data.normalizers import list_normalizers
from bonepick.data.utils import FILE_SUFFIXES, load_fasttext_dataset
from bonepick.evals import compute_detailed_metrics_fasttext, result_to_text
from bonepick.fasttext.utils import check_fasttext_binary, fasttext_dataset_signature, infer_single_file


@click.command()
@click.option(
    "-d",
    "--dataset-dir",
    required=True,
    multiple=True,
    type=PathParamType(mkdir=False, is_dir=True),
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
    """Train a FastText classifier.

    Shells out to the fasttext binary for fast n-gram based text classification.
    """
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

    train_result = subprocess.run(train_cmd)

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


@click.command()
@click.option(
    "-m",
    "--model-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    help="Path to the fasttext model directory (must contain model.bin)",
)
@click.option(
    "-i",
    "--input-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    help="Input directory containing JSONL files",
)
@click.option(
    "-o",
    "--output-dir",
    type=PathParamType(mkdir=True),
    required=True,
    help="Output directory for annotated JSONL files",
)
@click.option(
    "-t",
    "--text-field",
    type=str,
    default=None,
    help="Field in dataset to use as text",
)
@click.option(
    "-tt",
    "--text-expression",
    type=str,
    default=".text",
    help="JQ expression to extract text from dataset",
)
@click.option(
    "-c",
    "--classifier-name",
    type=str,
    required=True,
    help="Name for the classifier (results stored in .metadata.{classifier_name})",
)
@click.option(
    "--normalizer",
    type=click.Choice(list_normalizers()),
    default=None,
    help="Normalizer to apply to text before inference",
)
@click.option(
    "--num-proc",
    type=int,
    default=os.cpu_count() or 1,
    help="Maximum number of parallel workers (default: number of CPUs)",
)
@click.option("--max-length", type=int, default=None, help="Maximum length of text to process")
def infer_fasttext(
    model_dir: Path,
    input_dir: Path,
    output_dir: Path,
    text_field: str | None,
    text_expression: str,
    classifier_name: str,
    normalizer: str | None,
    num_proc: int,
    max_length: int | None,
):
    """Run FastText inference on JSONL files.

    Adds predictions to .metadata.{classifier_name} as label-probability dicts.
    """

    text_expression = field_or_expression(text_field, text_expression)

    click.echo("Starting fasttext inference...")
    click.echo(f"  Model directory: {model_dir}")
    click.echo(f"  Input directory: {input_dir}")
    click.echo(f"  Output directory: {output_dir}")
    click.echo(f"  Text expression: {text_expression}")
    click.echo(f"  Classifier name: {classifier_name}")
    if normalizer:
        click.echo(f"  Normalizer: {normalizer}")

    if max_length is not None:
        click.echo(f"  Maximum text length: {max_length}")

    # Check fasttext binary
    fasttext_path = check_fasttext_binary()

    # Check model file
    model_path = model_dir / "model.bin"
    if not model_path.exists():
        raise click.ClickException(f"Model file {model_path} does not exist")
    click.echo(f"Model file found: {model_path}")

    # Collect input files
    input_files: list[Path] = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = Path(root) / file
            if "".join(file_path.suffixes) in FILE_SUFFIXES:
                input_files.append(file_path)

    if not input_files:
        raise click.ClickException(f"No JSONL files found in {input_dir}")

    click.echo(f"Found {len(input_files)} input files")

    # Process files
    executor_cls = ProcessPoolExecutor if num_proc > 1 else ThreadPoolExecutor

    total_rows = 0

    with ExitStack() as stack:
        pbar = stack.enter_context(tqdm(total=len(input_files), desc="Processing files", unit=" files"))
        executor = stack.enter_context(executor_cls(max_workers=num_proc))
        futures = []

        for source_path in input_files:
            # Compute relative path to preserve directory structure
            relative_path = source_path.relative_to(input_dir)
            destination_path = output_dir / relative_path

            future = executor.submit(
                infer_single_file,
                source_path=source_path,
                destination_path=destination_path,
                fasttext_path=fasttext_path,
                model_path=model_path,
                text_expression=text_expression,
                classifier_name=classifier_name,
                normalizer_name=normalizer,
                max_length=max_length,
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                rows_processed = future.result()
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e

            total_rows += rows_processed
            pbar.update(1)

    click.echo("\nInference complete!")
    click.echo(f"  Total files processed: {len(input_files)}")
    click.echo(f"  Total rows processed: {total_rows:,}")
    click.echo(f"  Output directory: {output_dir}")


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
