import json
import subprocess
from pathlib import Path

import click
from model2vec.inference import StaticModelPipeline

from better_cls.data_utils import load_dataset
from better_cls.cli import PathParamType
from better_cls.fasttext_utils import check_fasttext_binary, fasttext_dataset_signature


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
@click.option("-t", "--text-field", type=str, default="text", help="field in dataset to use as text")
@click.option("-l", "--label-field", type=str, default="score", help="field in dataset to use as label")
def eval_model2vec(
    dataset_dir: tuple[Path, ...],
    model_dir: Path,
    text_field: str,
    label_field: str,
):
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
    dt = load_dataset(
        dataset_dirs=list(dataset_dir),
        text_field_name=text_field,
        label_field_name=label_field,
    )
    click.echo("Dataset loaded successfully.")
    click.echo(f"  Test samples: {len(dt.test.text)}")

    click.echo("\nEvaluating model on test data...")
    results = pipeline.evaluate(dt.test.text, dt.test.label)
    click.echo("\nEvaluation results:")
    click.echo(results)

    results_txt = json.dumps(results, indent=4) if isinstance(results, dict) else str(results)
    results_file = model_dir / f"results_{dt.test.signature}.txt"
    click.echo(f"\nSaving results to {results_file}...")
    with open(results_file, "wt", encoding="utf-8") as f:
        f.write(results_txt)
    click.echo(f"Results saved to: {results_file}")

    click.echo("\nEvaluation complete!")


@click.command()
@click.option(
    "-m",
    "--model-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    help="Path to the trained fasttext model (.bin file)",
)
@click.option(
    "-d",
    "--dataset-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    multiple=True,
    help="Directory containing the dataset (can be specified multiple times)",
)
def eval_fasttext(
    model_dir: Path,
    dataset_dir: tuple[Path, ...],
):
    """Evaluate a fasttext classifier on a test set."""
    click.echo("Starting fasttext evaluation...")
    click.echo(f"  Model directory: {model_dir}")
    click.echo(f"  Dataset directories: {', '.join(str(d) for d in dataset_dir)}")

    fasttext_path = check_fasttext_binary()

    # Collect all test.txt files from all directories
    test_files: list[Path] = []
    for d in dataset_dir:
        test_file = d / "test.txt"
        assert test_file.exists(), f"Test file {test_file} does not exist"
        assert test_file.is_file(), f"Test file {test_file} is not a file"
        test_files.append(test_file)
    click.echo(f"Found {len(test_files)} test file(s)")

    # If multiple directories, concatenate into a temporary file
    if len(test_files) == 1:
        test_file = test_files[0]
    else:
        combined_test = model_dir / "combined_test.txt"
        click.echo(f"Concatenating test files to: {combined_test}")
        with open(combined_test, "w", encoding="utf-8") as out_f:
            for tf in test_files:
                with open(tf, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
        test_file = combined_test
    click.echo(f"Test file: {test_file}")

    model_path = model_dir / "model.bin"
    assert model_path.exists(), f"Model file {model_path} does not exist"
    assert model_path.is_file(), f"Model file {model_path} is not a file"
    click.echo(f"Model file found: {model_path}")

    click.echo("\nBuilding evaluation command...")
    test_cmd = [
        str(fasttext_path),
        "test-label",
        str(model_path),
        str(test_file),
    ]

    click.echo("\nEvaluating fasttext model on test set...")
    click.echo(f"Command: {' '.join(test_cmd)}")

    test_result = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if test_result.returncode != 0:
        click.echo(f"Evaluation failed with return code: {test_result.returncode}", err=True)
        raise click.ClickException(
            f"fasttext test failed with return code {test_result.returncode}\nstderr: {test_result.stderr}"
        )

    click.echo("Evaluation subprocess completed successfully.")

    # Parse and display results
    test_output = test_result.stdout.strip().decode("utf-8")
    click.echo(f"\nTest results:\n{test_output}")

    # Save results to file
    click.echo("\nPreparing results for saving...")
    results = {
        "test_command": " ".join(test_cmd),
        "model_path": str(model_path),
        "test_file": str(test_file),
        "test_output": test_output,
    }

    click.echo("Computing dataset signature...")
    signature = fasttext_dataset_signature(test_file)
    click.echo(f"Dataset signature: {signature}")

    results_file = model_path.parent / f"results_{signature}.json"
    click.echo(f"Saving results to: {results_file}")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    click.echo(f"\nResults saved to: {results_file}")
    click.echo("\nFasttext evaluation complete!")
