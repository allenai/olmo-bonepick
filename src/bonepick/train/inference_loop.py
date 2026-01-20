import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile

import click
import msgspec
import smart_open
from tqdm import tqdm

from bonepick.cli import PathParamType
from bonepick.train.data_utils import FILE_SUFFIXES
from bonepick.train.fasttext_utils import check_fasttext_binary
from bonepick.train.jq_utils import compile_jq, field_or_expression
from bonepick.train.normalizers import get_normalizer, list_normalizers


def infer_single_file(
    source_path: Path,
    destination_path: Path,
    fasttext_path: Path,
    model_path: Path,
    text_expression: str,
    classifier_name: str,
    normalizer_name: str | None = None,
) -> int:
    """Run fasttext inference on a single file and add predictions to metadata.

    Args:
        source_path: Path to source JSONL file
        destination_path: Path to destination JSONL file
        fasttext_path: Path to fasttext binary
        model_path: Path to fasttext model file
        text_expression: JQ expression to extract text
        classifier_name: Name to use in .metadata.{classifier_name}
        normalizer_name: Optional normalizer to apply to text

    Returns:
        Number of rows processed
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    decoder = msgspec.json.Decoder()
    encoder = msgspec.json.Encoder()
    text_selector = compile_jq(text_expression)
    normalizer = get_normalizer(normalizer_name) if normalizer_name else None

    # Step 1: Read source file and prepare texts for fasttext
    rows: list[dict] = []
    texts: list[str] = []

    with smart_open.open(source_path, "rb") as f:  # pyright: ignore
        for line in f:
            row = decoder.decode(line)
            rows.append(row)

            text = str(text_selector(row))
            if normalizer:
                text = normalizer.normalize(text)
            # FastText expects single-line input
            text = text.replace("\n", " ").replace("\r", " ")
            texts.append(text)

    if not rows:
        return 0

    # Step 2: Write texts to temp file and run fasttext
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as temp_input:
        for text in texts:
            temp_input.write(text + "\n")
        temp_input_path = temp_input.name

    try:
        # Run fasttext predict-prob with all classes (-1)
        predict_cmd = [
            str(fasttext_path),
            "predict-prob",
            str(model_path),
            temp_input_path,
            "-1",
        ]

        result = subprocess.run(
            predict_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"fasttext predict failed with return code {result.returncode}\nstderr: {result.stderr}"
            )

        # Step 3: Parse predictions and add to rows
        predictions = result.stdout.strip().split("\n")

        if len(predictions) != len(rows):
            raise RuntimeError(
                f"Number of predictions ({len(predictions)}) does not match number of rows ({len(rows)})"
            )

        for row, prediction in zip(rows, predictions):
            # Parse prediction: __label__X prob __label__Y prob ...
            parts = prediction.strip().split()
            labels = parts[::2]
            probas = [float(p) for p in parts[1::2]]

            # Build probability dict for all classes (keep __label__ prefix); sort ensures order by label name
            proba_dict = {label: proba for label, proba in sorted(zip(labels, probas))}

            # Add to metadata
            if "metadata" not in row:
                row["metadata"] = {}

            row["metadata"][classifier_name] = proba_dict

    finally:
        os.unlink(temp_input_path)

    # Step 4: Write output file
    with smart_open.open(destination_path, "wb") as f:  # pyright: ignore
        for row in rows:
            f.write(encoder.encode(row) + b"\n")

    return len(rows)


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
    default="whitespace",
    help="Normalizer to apply to text before inference",
)
@click.option(
    "--num-proc",
    type=int,
    default=os.cpu_count() or 1,
    help="Maximum number of parallel workers (default: number of CPUs)",
)
def infer_fasttext(
    model_dir: Path,
    input_dir: Path,
    output_dir: Path,
    text_field: str | None,
    text_expression: str,
    classifier_name: str,
    normalizer: str,
    num_proc: int,
):
    """Run fasttext inference on JSONL files and add predictions to metadata.

    For each row in the input files, extracts text using the text-expression,
    runs fasttext classification, and stores the result in .metadata.{classifier_name}
    as a dict mapping labels to probabilities (e.g., {"__label__pos": 0.52, "__label__neg": 0.48}).
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
