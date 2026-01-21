import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import click
import msgspec
import smart_open
import yaml

from bonepick.data.expressions import compile_jq
from bonepick.data.normalizers import get_normalizer


def check_fasttext_binary() -> Path:
    """Check if fasttext binary is available and return its path."""
    click.echo("Checking for fasttext binary in PATH...")
    fasttext_path = shutil.which("fasttext")
    if fasttext_path is None:
        raise click.ClickException(
            "fasttext binary not found in PATH. Please install fasttext: https://fasttext.cc/docs/en/support.html"
        )
    click.echo(f"Found fasttext binary at: {fasttext_path}")
    return Path(fasttext_path)


def fasttext_dataset_signature(fasttext_file: Path) -> str:
    assert fasttext_file.exists(), f"Fasttext file {fasttext_file} does not exist"
    assert fasttext_file.is_file(), f"Fasttext file {fasttext_file} is not a file"

    h = hashlib.sha256()
    with smart_open.open(fasttext_file, "rb") as f:  # pyright: ignore
        for line in f:
            h.update(line)
    return h.hexdigest()


def infer_single_file(
    source_path: Path,
    destination_path: Path,
    fasttext_path: Path,
    model_path: Path,
    text_expression: str,
    classifier_name: str,
    normalizer_name: str | None = None,
    max_length: int | None = None,
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

            if max_length is not None and len(text) > max_length:
                text = text[:max_length]

            if normalizer:
                text = normalizer.normalize(text)
            else:
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

            # add to metadata
            row.setdefault("metadata", {})[classifier_name] = proba_dict

    finally:
        os.unlink(temp_input_path)

    # Step 4: Write output file
    with smart_open.open(destination_path, "wb") as f:  # pyright: ignore
        for row in rows:
            f.write(encoder.encode(row) + b"\n")

    return len(rows)


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
