import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from functools import partial
from math import log10
from pathlib import Path

import click
import datasets
import smart_open
import yaml
from tqdm import tqdm

from bonepick.cli import ByteSizeParamType, FloatOrIntParamType, PathParamType
from bonepick.data.expressions import add_field_or_expression_command_options, field_or_expression
from bonepick.data.normalizers import list_normalizers
from bonepick.data.utils import (
    FILE_SUFFIXES,
    DatasetSplit,
    DatasetTuple,
    batch_save_hf_dataset,
    convert_single_file_to_fasttext,
    count_tokens_in_file,
    extract_numeric_labels_from_file,
    load_jsonl_dataset,
    normalize_single_file,
    pretty_size,
    sample_single_file,
    transform_single_file,
    write_dataset,
)

__all__ = [
    "balance_dataset",
    "convert_to_fasttext",
    "count_tokens",
    "import_hf_dataset",
    "normalize_dataset",
    "reshard_dataset",
    "sample_dataset",
    "transform_dataset",
]


@click.command()
@click.option("-n", "--name", type=str, required=True)
@click.option("-s", "--subset", type=str, default=None)
@click.option("-o", "--output-dir", type=PathParamType(mkdir=True, is_dir=True), required=True)
@click.option("-t", "--test-split", type=FloatOrIntParamType(), default=None)
@click.option("-b", "--batch-size", type=int, default=100_000)
@click.option("-p", "--num-proc", type=int, default=os.cpu_count())
@click.option("-S", "--seed", type=int, default=333)
def import_hf_dataset(
    name: str,
    output_dir: Path,
    subset: str | None,
    test_split: float | int | None,
    seed: int,
    batch_size: int,
    num_proc: int,
):
    """Import a HuggingFace dataset to local JSONL files.

    Downloads and saves to train/ and test/ subdirectories with optional train/test splitting.
    """
    dataset = datasets.load_dataset(name, name=subset)
    assert isinstance(dataset, datasets.DatasetDict), "Dataset is not a DatasetDict"

    if "test" not in dataset and test_split is None:
        raise ValueError("Test split is required if test split is not in dataset")
    elif "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=test_split)

    for split in ("train", "test"):
        dataset_split: datasets.Dataset = dataset[split]
        dataset_split = dataset_split.shuffle(seed=seed)

        (split_dest := output_dir / split).mkdir(parents=True, exist_ok=True)
        fn = partial(batch_save_hf_dataset, destination_dir=split_dest)

        dataset_split.map(
            fn,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            with_indices=True,
        )


@click.command()
@click.option("-i", "--input-dir", type=PathParamType(exists=True, is_dir=True), required=True)
@click.option("-o", "--output-dir", type=PathParamType(mkdir=True, is_dir=True), required=True)
@click.option("-t", "--text-transform", type=str, default="{text: .text}")
@click.option("-l", "--label-transform", type=str, default="{score: .score}")
@click.option("-p", "--num-proc", type=int, default=os.cpu_count())
def transform_dataset(
    input_dir: Path,
    output_dir: Path,
    text_transform: str,
    label_transform: str,
    num_proc: int,
):
    """Transform dataset fields using jq expressions.

    Applies jq transformations to reshape text and label fields in JSONL files.
    """
    input_files: list[Path] = []
    output_files: list[Path] = []
    for root, _, files in os.walk(input_dir):
        for _fn in files:
            fn = Path(root) / _fn
            if "".join(fn.suffixes) not in FILE_SUFFIXES:
                continue

            input_files.append(fn)
            output_files.append(output_dir / fn.relative_to(input_dir))

    executor_cls = ProcessPoolExecutor if num_proc > 1 else ThreadPoolExecutor

    with executor_cls(max_workers=num_proc) as pool:
        futures = []
        for input_file, output_file in zip(input_files, output_files):
            future = pool.submit(
                transform_single_file,
                source_path=input_file,
                destination_path=output_file,
                text_transform=text_transform,
                label_transform=label_transform,
            )
            futures.append(future)

        pbar = tqdm(total=len(futures), desc="Processing files", unit="file")
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e
            pbar.update(1)

        pbar.close()


@click.command()
@click.option("-i", "--input-dir", type=PathParamType(exists=True, is_dir=True), required=True)
@click.option("-o", "--output-dir", type=PathParamType(mkdir=True, is_dir=True), required=True)
@click.option("-n", "--normalization", type=click.Choice(list_normalizers()), default="plsfix")
@click.option("-t", "--text-field", type=str, default="text")
@click.option("-l", "--label-field", type=str, default="score")
@click.option("-p", "--num-proc", type=int, default=os.cpu_count())
def normalize_dataset(
    input_dir: Path,
    output_dir: Path,
    normalization: str,
    text_field: str,
    label_field: str,
    num_proc: int,
):
    """Apply text normalization to a dataset.

    Supports whitespace, plsfix, tokenizer, ultrafine, hyperfine, and potion normalizers.
    """
    input_files: list[Path] = []
    output_files: list[Path] = []
    for root, _, files in os.walk(input_dir):
        for _fn in files:
            fn = Path(root) / _fn
            if "".join(fn.suffixes) not in FILE_SUFFIXES:
                continue

            input_files.append(fn)
            output_files.append(output_dir / fn.relative_to(input_dir))

    executor_cls = ProcessPoolExecutor if num_proc > 1 else ThreadPoolExecutor

    with executor_cls(max_workers=num_proc) as pool:
        futures = []
        for input_file, output_file in zip(input_files, output_files):
            future = pool.submit(
                normalize_single_file,
                source_path=input_file,
                destination_path=output_file,
                text_field=text_field,
                label_field=label_field,
                normalization=normalization,
            )
            futures.append(future)

        pbar = tqdm(total=len(futures), desc="Processing files", unit="file")
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e
            pbar.update(1)

        pbar.close()


@click.command()
@add_field_or_expression_command_options
@click.option("-i", "--input-dir", type=PathParamType(exists=True, is_dir=True), required=True)
@click.option("-o", "--output-dir", type=PathParamType(mkdir=True, is_dir=True), required=True)
@click.option("-p", "--num-proc", type=int, default=os.cpu_count())
@click.option("-n", "--normalization", type=click.Choice(list_normalizers()), default="whitespace")
@click.option("--max-length", type=int, default=None, help="Maximum length of text to process")
@click.option(
    "--auto",
    type=int,
    default=None,
    help="Auto-bin numeric labels into N equal-count (quantile) bins (requires 2 <= N <= unique labels)",
)
@click.option(
    "--multi-label",
    is_flag=True,
    default=False,
    help="Multi-label mode: label expression should return a dict {criterion_name: 0_or_1}. "
    "Generates __label__<criterion> for each criterion with value 1.",
)
def convert_to_fasttext(
    text_field: str | None,
    label_field: str | None,
    text_expression: str,
    label_expression: str,
    input_dir: Path,
    output_dir: Path,
    num_proc: int,
    normalization: str,
    max_length: int | None,
    auto: int | None,
    multi_label: bool,
):
    """Convert JSONL dataset to FastText format.

    Outputs __label__<label> <text> format for FastText training.
    """
    text_expression = field_or_expression(text_field, text_expression)
    label_expression = field_or_expression(label_field, label_expression)
    row_count: dict[str, int] = {}

    # Validate mutually exclusive options
    if multi_label and auto is not None:
        raise click.BadParameter("--multi-label and --auto are mutually exclusive", param_hint="--multi-label")

    # Label mapper for auto-binning (computed from training data if --auto is specified)
    label_mapper: tuple[list[float], list[str]] | None = None

    if auto is not None:
        if auto < 2:
            raise click.BadParameter("--auto must be at least 2", param_hint="--auto")

        click.echo(f"Auto-binning enabled with {auto} bins")
        click.echo("Pass 1: Extracting labels from training data...")

        # Collect all training files
        train_dir = input_dir / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory {train_dir} does not exist")

        train_files: list[Path] = []
        for root, _, files in os.walk(train_dir):
            for _fn in files:
                fn = Path(root) / _fn
                if "".join(fn.suffixes) not in FILE_SUFFIXES:
                    continue
                train_files.append(fn)

        if not train_files:
            raise click.ClickException(f"No JSONL files found in {train_dir}")

        click.echo(f"  Found {len(train_files)} training files")

        # Extract labels in parallel
        all_labels: list[float] = []
        executor_cls = ProcessPoolExecutor if num_proc > 1 else ThreadPoolExecutor

        with executor_cls(max_workers=num_proc) as pool:
            futures = [pool.submit(extract_numeric_labels_from_file, fn, label_expression) for fn in train_files]

            with tqdm(total=len(futures), desc="Extracting labels", unit="file") as pbar:
                for future in as_completed(futures):
                    try:
                        labels = future.result()
                        all_labels.extend(labels)
                        pbar.set_postfix(total_labels=f"{len(all_labels):,}")
                    except Exception as e:
                        for f in futures:
                            f.cancel()
                        raise e
                    pbar.update(1)

        if not all_labels:
            raise click.ClickException("No labels extracted from training data")

        # Compute statistics
        min_label = min(all_labels)
        max_label = max(all_labels)
        unique_labels = len(set(all_labels))

        click.echo(f"  Total labels: {len(all_labels):,}")
        click.echo(f"  Unique labels: {unique_labels:,}")
        click.echo(f"  Min label: {min_label}")
        click.echo(f"  Max label: {max_label}")

        if auto > unique_labels:
            raise click.BadParameter(
                f"--auto ({auto}) cannot exceed unique labels ({unique_labels})", param_hint="--auto"
            )

        # Compute equal-count (quantile-based) bins
        # Sort labels and find quantile boundaries
        sorted_labels = sorted(all_labels)
        n = len(sorted_labels)
        bin_edges: list[float] = [sorted_labels[0]]  # Start with min

        for i in range(1, auto):
            # Find the index for the i-th quantile boundary
            idx = int(i * n / auto)
            bin_edges.append(sorted_labels[idx])

        bin_edges.append(sorted_labels[-1])  # End with max

        # Create bin labels (e.g., "bin_0", "bin_1", ...)
        bin_labels = [f"bin_{i}" for i in range(auto)]

        click.echo("  Bin edges and labels (equal-count/quantile bins):")
        for i in range(auto):
            # Use '(' instead of '[' if previous bin was single-value (to avoid overlap)
            left_bracket = "(" if i > 0 and bin_edges[i - 1] == bin_edges[i] else "["
            if bin_edges[i] == bin_edges[i + 1]:
                click.echo(f"    {bin_labels[i]}: [{bin_edges[i]:.4f}] (single-value bin)")
            else:
                click.echo(f"    {bin_labels[i]}: {left_bracket}{bin_edges[i]:.4f}, {bin_edges[i + 1]:.4f})")

        label_mapper = (bin_edges, bin_labels)

        # Count samples per bin for reporting
        bin_counts: dict[str, int] = {label: 0 for label in bin_labels}
        for label_value in all_labels:
            for i in range(len(bin_edges) - 1):
                lower, upper = bin_edges[i], bin_edges[i + 1]
                # Handle single-value bins (lower == upper) with exact match
                if lower == upper:
                    if label_value == lower:
                        bin_counts[bin_labels[i]] += 1
                        break
                elif lower <= label_value < upper:
                    bin_counts[bin_labels[i]] += 1
                    break
            else:
                # Handle edge case: value equals max edge
                if label_value == bin_edges[-1]:
                    bin_counts[bin_labels[-1]] += 1

        click.echo("  Samples per bin:")
        for label in bin_labels:
            pct = 100 * bin_counts[label] / len(all_labels)
            click.echo(f"    {label}: {bin_counts[label]:,} ({pct:.1f}%)")

        click.echo("\nPass 2: Converting to FastText format...")

    for split, must_exist in (("train", True), ("valid", False), ("test", True)):
        split_dir = input_dir / split

        if not split_dir.exists():
            if must_exist:
                raise FileNotFoundError(f"Split directory {split_dir} does not exist")
            continue

        with ExitStack() as stack:
            # this will handle executing the conversion in parallel
            pool_cls = ProcessPoolExecutor if num_proc > 1 else ThreadPoolExecutor
            pool = stack.enter_context(pool_cls(max_workers=num_proc))

            # output to a single text file for each split
            output_file = output_dir / f"{split}.txt"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file = stack.enter_context(smart_open.open(output_file, "wt", encoding="utf-8"))  # pyright: ignore

            futures = []
            for root, _, files in os.walk(split_dir):
                for _fn in files:
                    fn = Path(root) / _fn
                    if "".join(fn.suffixes) not in FILE_SUFFIXES:
                        continue

                    future = pool.submit(
                        convert_single_file_to_fasttext,
                        source_path=fn,
                        text_expression=text_expression,
                        label_expression=label_expression,
                        normalization=normalization,
                        max_length=max_length,
                        label_mapper=label_mapper,
                        multi_label=multi_label,
                    )
                    futures.append(future)

            files_pbar = stack.enter_context(
                tqdm(total=len(futures), desc=f"Converting {split} files", unit="file")
            )
            rows_pbar = stack.enter_context(tqdm(desc=f"Writing {split} rows", unit=" rows", unit_scale=True))

            for future in as_completed(futures):
                try:
                    future.result()
                    for row in future.result():
                        output_file.write(row + "\n")
                        rows_pbar.update(1)
                    files_pbar.update(1)
                except Exception as e:
                    for future in futures:
                        future.cancel()
                    raise e

            row_count[split] = rows_pbar.n

    report_dict: dict = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "text_expression": text_expression,
        "label_expression": label_expression,
        "normalization": normalization,
        "max_length": max_length,
        "multi_label": multi_label,
        "num_rows": row_count,
    }

    if auto is not None and label_mapper is not None:
        bin_edges, bin_labels = label_mapper
        report_dict["auto_bins"] = {
            "num_bins": auto,
            "bin_edges": bin_edges,
            "bin_labels": bin_labels,
        }

    report_file = output_dir / "report.yaml"
    with open(report_file, "w", encoding="utf-8") as f:
        yaml.dump(report_dict, f, indent=2)
    click.echo(f"Report saved to: {report_file}")


@click.command()
@add_field_or_expression_command_options
@click.option(
    "-i",
    "--input-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    multiple=True,
    help="Input directory (can be specified multiple times)",
)
@click.option(
    "-o",
    "--output-dir",
    type=PathParamType(mkdir=True, is_dir=True),
    required=True,
    help="Output directory for balanced dataset",
)
@click.option("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option(
    "-p",
    "--num-proc",
    type=int,
    default=os.cpu_count(),
    help="Number of processes for parallel processing",
)
def balance_dataset(
    text_field: str | None,
    label_field: str | None,
    text_expression: str,
    label_expression: str,
    input_dir: tuple[Path, ...],
    output_dir: Path,
    seed: int,
    num_proc: int,
):
    """Balance dataset labels via downsampling.

    Ensures each label has equal representation across train/test splits.
    """
    click.echo("Starting dataset balancing...")
    click.echo(f"  Input directories: {', '.join(str(d) for d in input_dir)}")
    click.echo(f"  Output directory: {output_dir}")
    click.echo(f"  Text field: {text_field}")
    click.echo(f"  Label field: {label_field}")
    click.echo(f"  Seed: {seed}")
    click.echo(f"  Num processes: {num_proc}")

    rng = random.Random(seed)

    text_expression = field_or_expression(text_field, text_expression)
    label_expression = field_or_expression(label_field, label_expression)

    dataset_tuple = load_jsonl_dataset(
        dataset_dirs=list(input_dir),
        text_field_expression=text_expression,
        label_field_expression=label_expression,
    )
    sampled_dataset_splits: dict[str, DatasetSplit] = {}

    for split_name, split_data in dataset_tuple:
        if len(split_data) == 0:
            click.echo(f"  {split_name} split is empty, skipping...")
            sampled_dataset_splits[split_name] = DatasetSplit.new()
            continue

        click.echo(f"\nProcessing {split_name} split...")
        click.echo(f"  Samples: {len(split_data.text)}")

        label_counts = Counter(split_data.label)
        click.echo("  Label counts:")
        for label, count in label_counts.most_common(len(label_counts)):
            click.echo(f"    {label}: {count}")

        target_count = min(label_counts.values())
        click.echo(f"  Target count per label: {target_count}")

        sampling_ratio = {k: target_count / v for k, v in label_counts.items()}
        click.echo("  Sampling ratio:")
        for label, ratio in sampling_ratio.items():
            click.echo(f"    {label}: {ratio:.4f}")

        click.echo(f"  Creating sampled split for {split_name}...")
        sampled_split = DatasetSplit.new()
        for text, label in split_data:
            if rng.random() >= sampling_ratio[label]:
                continue
            sampled_split.text.append(text)
            sampled_split.label.append(label)

        # give it a shuffle and append to the sampled dataset splits
        sampled_split = sampled_split.shuffle(rng=rng)
        sampled_dataset_splits[split_name] = sampled_split

    sampled_dataset_tuple = DatasetTuple(**sampled_dataset_splits)
    click.echo("\nBalancing complete!")

    click.echo(f"Writing sampled dataset to {output_dir}...")
    write_dataset(
        dataset=sampled_dataset_tuple,
        destination_dir=output_dir,
        text_field_name=text_field or "text",
        label_field_name=label_field or "score",
    )
    click.echo(f"  Written to: {output_dir}")


@click.command()
@click.option(
    "-i",
    "--dataset-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    multiple=True,
    help="Input dataset directory (can be specified multiple times)",
)
@click.option(
    "-o",
    "--output-dir",
    type=PathParamType(mkdir=True, is_dir=True),
    required=True,
    help="Output directory for sampled dataset",
)
@click.option(
    "-r",
    "--sampling-rate",
    type=float,
    default=None,
    help="Sampling rate (0.0-1.0). Mutually exclusive with --target-size",
)
@click.option(
    "-t",
    "--target-size",
    type=ByteSizeParamType(),
    default=None,
    help="Target total size (e.g., '1GB', '500MB'). Mutually exclusive with --sampling-rate",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.option(
    "-p",
    "--num-proc",
    type=int,
    default=os.cpu_count(),
    help="Number of processes for parallel processing",
)
def sample_dataset(
    dataset_dir: tuple[Path, ...],
    output_dir: Path,
    sampling_rate: float | None,
    target_size: int | None,
    seed: int,
    num_proc: int,
):
    """Create a random sample of a dataset.

    Use --sampling-rate (0-1) or --target-size (e.g., '1GB') to control output size.
    """
    # Validate mutually exclusive options
    if sampling_rate is None and target_size is None:
        raise click.BadParameter("Either --sampling-rate or --target-size must be specified")
    if sampling_rate is not None and target_size is not None:
        raise click.BadParameter("--sampling-rate and --target-size are mutually exclusive")
    if sampling_rate is not None and (sampling_rate <= 0 or sampling_rate > 1.0):
        raise click.BadParameter("--sampling-rate must be between 0 and 1.0")

    click.echo("Starting dataset sampling...")
    click.echo(f"  Input directories: {', '.join(str(d) for d in dataset_dir)}")
    click.echo(f"  Output directory: {output_dir}")
    click.echo(f"  Seed: {seed}")
    click.echo(f"  Num processes: {num_proc}")

    # Step 1: Collect all files and their sizes
    click.echo("\nCollecting files...")
    file_info: list[tuple[Path, Path, int]] = []  # (source_path, relative_path, size)
    total_size = 0

    for input_dir in dataset_dir:
        for root, _, files in os.walk(input_dir):
            for _fn in files:
                fn = Path(root) / _fn
                if "".join(fn.suffixes) not in FILE_SUFFIXES:
                    continue

                file_size = fn.stat().st_size
                relative_path = fn.relative_to(input_dir)
                file_info.append((fn, relative_path, file_size))
                total_size += file_size

    if not file_info:
        click.echo("No files found to sample. Exiting.")
        return

    click.echo(f"  Found {len(file_info)} files")
    click.echo(f"  Total size: {total_size:,} bytes ({total_size / (1024**3):.2f} GB)")

    # Step 2: Calculate target size per file
    if sampling_rate is not None:
        total_target_size = int(total_size * sampling_rate)
        click.echo(f"  Sampling rate: {sampling_rate:.2%}")
    else:
        # this assert is just to make mypy happy
        assert target_size is not None, "This should be impossible"

        total_target_size = target_size
        click.echo(f"  Target size: {target_size:,} bytes ({target_size / (1024**3):.2f} GB)")

    effective_sampling_rate = total_target_size / total_size
    click.echo(f"  Effective sampling rate: {effective_sampling_rate:.2%}")

    # Optimization: If sampling rate is low, select a subset of files instead of sampling all
    # This is more efficient when target size << total size
    # Use threshold of 5% - if we're sampling less than 5%, select subset of files
    file_tasks: list[tuple[Path, Path, int]] = []  # (source, dest, target_size)

    if effective_sampling_rate < 0.05:
        click.echo("\n  Optimization: Selecting subset of files due to low sampling rate...")

        # Shuffle files to get random selection
        rng = random.Random(seed)
        shuffled_file_info = list(file_info)
        rng.shuffle(shuffled_file_info)

        # Select files until we reach approximately the target size
        selected_files: list[tuple[Path, Path, int]] = []
        accumulated_size = 0
        for source_path, relative_path, file_size in shuffled_file_info:
            if accumulated_size >= total_target_size:
                break
            selected_files.append((source_path, relative_path, file_size))
            accumulated_size += file_size

        # Now calculate the sampling rate for the selected files
        if accumulated_size > 0:
            subset_sampling_rate = total_target_size / accumulated_size
        else:
            subset_sampling_rate = 1.0

        click.echo(f"  Selected {len(selected_files)} files (out of {len(file_info)})")
        click.echo(
            f"  Selected files total size: {accumulated_size:,} bytes ({accumulated_size / (1024**3):.2f} GB)"
        )
        click.echo(f"  Per-file sampling rate: {subset_sampling_rate:.2%}")

        # Create tasks with the subset sampling rate
        for source_path, relative_path, file_size in selected_files:
            file_target_size = int(file_size * subset_sampling_rate)
            dest_path = output_dir / relative_path
            file_tasks.append((source_path, dest_path, file_target_size))
    else:
        # Normal case: sample all files proportionally
        click.echo("\n  Using proportional sampling across all files...")
        for source_path, relative_path, file_size in file_info:
            file_target_size = int((file_size / total_size) * total_target_size)
            dest_path = output_dir / relative_path
            file_tasks.append((source_path, dest_path, file_target_size))

    # Step 3: Process files in parallel using multiprocessing
    click.echo(f"\nSampling {len(file_tasks)} files using {num_proc} processes...")

    executor_cls = ProcessPoolExecutor if num_proc > 1 else ThreadPoolExecutor

    with executor_cls(max_workers=num_proc) as pool:
        futures = []
        for source_path, dest_path, file_target_size in file_tasks:
            future = pool.submit(
                sample_single_file,
                source_path=source_path,
                destination_path=dest_path,
                target_size=file_target_size,
                seed=seed,
            )
            futures.append(future)

        pbar = tqdm(total=len(futures), desc="Sampling files", unit="file")
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e
            pbar.update(1)

        pbar.close()

    # Report final size
    final_size = 0
    for root, _, files in os.walk(output_dir):
        for _fn in files:
            fn = Path(root) / _fn
            final_size += fn.stat().st_size

    click.echo("\nSampling complete!")
    click.echo(f"  Output size: {final_size:,} bytes ({final_size / (1024**3):.2f} GB)")
    click.echo(f"  Actual sampling rate: {final_size / total_size:.2%}")
    click.echo(f"  Written to: {output_dir}")


@click.command()
@click.option(
    "-d",
    "--dataset-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    multiple=True,
    help="Input dataset directory (can be specified multiple times)",
)
@click.option(
    "-t",
    "--tokenizer-name-or-path",
    type=str,
    default="allenai/dolma2-tokenizer",
    help="Tokenizer name or path (HuggingFace tokenizer identifier or local path)",
)
@click.option(
    "-i",
    "--input-field-expression",
    type=str,
    default=".text",
    help="JQ expression to extract text field (default: '.text')",
)
@click.option(
    "-p",
    "--num-proc",
    type=int,
    default=os.cpu_count(),
    help="Number of processes for parallel processing",
)
def count_tokens(
    dataset_dir: tuple[Path, ...],
    tokenizer_name_or_path: str,
    input_field_expression: str,
    num_proc: int,
):
    """Count tokens in dataset directories.

    Uses a tokenizer to count total tokens across all JSONL files in parallel.
    """
    from tokenizers import Tokenizer

    click.echo("Starting token counting...")
    click.echo(f"  Dataset directories: {', '.join(str(d) for d in dataset_dir)}")
    click.echo(f"  Tokenizer: {tokenizer_name_or_path}")
    click.echo(f"  Input field expression: {input_field_expression}")
    click.echo(f"  Num processes: {num_proc}")

    # Load and serialize tokenizer once
    click.echo("\nLoading tokenizer...")
    try:
        tokenizer_obj = Tokenizer.from_pretrained(tokenizer_name_or_path)
    except Exception:
        # Try loading from local path
        tokenizer_obj = Tokenizer.from_file(tokenizer_name_or_path)
    tokenizer_json = tokenizer_obj.to_str()
    click.echo("  Tokenizer loaded successfully")

    # Collect all files from all dataset directories
    click.echo("\nCollecting files...")
    all_files: list[Path] = []
    file_sizes: list[int] = []
    for input_dir in dataset_dir:
        for root, _, files in os.walk(input_dir):
            for _fn in files:
                fn = Path(root) / _fn
                if "".join(fn.suffixes) not in FILE_SUFFIXES:
                    continue
                all_files.append(fn)
                file_sizes.append(fn.stat().st_size)

    if not all_files:
        click.echo("No files found to process. Exiting.")
        return

    click.echo(f"  Found {len(all_files):,} files")
    click.echo(f"  Total size: {pretty_size(sum(file_sizes))}")
    # Process files in parallel
    click.echo(f"\nCounting tokens using {num_proc} processes...")

    executor_cls = ProcessPoolExecutor if num_proc > 1 else ThreadPoolExecutor

    with executor_cls(max_workers=num_proc) as pool:
        futures = []
        for file_path in all_files:
            future = pool.submit(
                count_tokens_in_file,
                source_path=file_path,
                tokenizer_json=tokenizer_json,
                input_field_expression=input_field_expression,
            )
            futures.append(future)

        total_tokens = 0
        pbar = tqdm(total=len(futures), desc="Processing files", unit="file")
        for future in as_completed(futures):
            try:
                file_token_count = future.result()
                total_tokens += file_token_count
                pbar.set_postfix(total_tokens=pretty_size(total_tokens, unit="T", precision=1))
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e
            pbar.update(1)

        pbar.close()

    click.echo("\nToken counting complete!")
    click.echo(f"  Total files processed: {len(all_files):,}")
    click.echo(f"  Total tokens: {total_tokens:,}")
    click.echo(f"  Total size: {pretty_size(sum(file_sizes))}")
    click.echo(f"  Average tokens per file: {total_tokens / len(all_files):.2f}")
    click.echo(f"  Average tokens per byte: {total_tokens / sum(file_sizes):.2f}")


@click.command()
@click.option(
    "-i",
    "--dataset-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    help="Input directory containing dataset files (all files in directory and subdirectories will be resharded)",
)
@click.option(
    "-o",
    "--output-dir",
    type=PathParamType(mkdir=True, is_dir=True),
    required=True,
    help="Output directory for resharded dataset",
)
@click.option(
    "-n",
    "--num-files",
    type=int,
    required=True,
    help="Target number of output files (total across train and test if --test-split-frac is specified)",
)
@click.option(
    "-t",
    "--test-split-frac",
    type=FloatOrIntParamType(),
    default=None,
    help="Test split fraction (float 0.0-1.0 for percentage, or int for number of instances)",
)
@click.option(
    "-v",
    "--valid-split-frac",
    type=FloatOrIntParamType(),
    default=None,
    help="Validation split fraction (float 0.0-1.0 for percentage, or int for number of instances)",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility when splitting into train/test/valid",
)
@click.option(
    "-p",
    "--num-proc",
    type=int,
    default=os.cpu_count(),
    help="Number of processes for parallel processing",
)
def reshard_dataset(
    dataset_dir: Path,
    output_dir: Path,
    num_files: int,
    test_split_frac: float | int | None,
    valid_split_frac: float | int | None,
    seed: int,
    num_proc: int,
):
    """Combine files into N equal-sized shards.

    Redistributes data from multiple small files into a specified number of larger files,
    with output files being roughly equal in size. All files in the input directory
    and its subdirectories will be combined.

    If --test-split-frac and/or --valid-split-frac are specified, the data will be split
    into train/, test/, and/or valid/ subdirectories, with num-files distributed
    proportionally between the splits.

    Useful for:
    - Reducing the number of small files for more efficient I/O
    - Creating evenly-sized shards for distributed processing
    - Preparing data for systems that work better with fewer, larger files
    - Creating train/test/valid splits during resharding

    Note: Call this command separately for train/, test/, valid/ directories if you need to
    maintain split separation without creating a new split.
    """
    if num_files <= 0:
        raise click.BadParameter("--num-files must be greater than 0")

    # Step 0: Print configuration
    click.echo("Starting dataset resharding...")
    click.echo(f"  Input directory: {dataset_dir}")
    click.echo(f"  Output directory: {output_dir}")
    click.echo(f"  Target output files: {num_files}")
    if test_split_frac is not None:
        click.echo(f"  Test split: {test_split_frac}")
    if valid_split_frac is not None:
        click.echo(f"  Valid split: {valid_split_frac}")
    if test_split_frac is not None or valid_split_frac is not None:
        click.echo(f"  Random seed: {seed}")
    click.echo(f"  Num processes: {num_proc}")

    # Step 1: Collect all input files and their sizes
    click.echo("\nCollecting files...")
    input_files: list[tuple[Path, int]] = []
    for root, _, files in os.walk(dataset_dir):
        for _fn in files:
            fn = Path(root) / _fn
            if "".join(fn.suffixes) not in FILE_SUFFIXES:
                continue
            input_files.append((fn, fn.stat().st_size))

    if not input_files:
        click.echo("  No files found, exiting...")
        return

    total_size = sum(size for _, size in input_files)
    click.echo(f"  Input files: {len(input_files):,}")
    click.echo(f"  Total size: {pretty_size(total_size)}")

    # Step 2: simply read all rows into memory
    all_rows: list[bytes] = []
    with tqdm(total=len(input_files), desc="Reading files", unit="file") as pbar:
        for file_path, _ in input_files:
            with smart_open.open(file_path, "rb") as f:  # pyright: ignore
                for line in f:
                    all_rows.append(line)
                    pbar.set_postfix(rows=f"{len(all_rows):,}")
                pbar.update(1)
                pbar.refresh()

    total_rows = len(all_rows)
    click.echo(f"  Total rows: {total_rows:,}")

    # Step 3: shuffle rows
    click.echo(f"  Shuffling rows with seed {seed}...")
    rng = random.Random(seed)
    rng.shuffle(all_rows)

    # Step 4: split rows into train, test, and valid
    train_size, valid_size, test_size = len(all_rows), 0, 0
    if valid_split_frac is not None:
        valid_size = (
            round(total_rows * valid_split_frac) if isinstance(valid_split_frac, float) else valid_split_frac
        )
        if valid_size > total_rows:
            raise click.BadParameter(
                f"Requested more validation rows ({valid_size:,}) than total rows ({total_rows:,})"
            )
        train_size -= valid_size
    if test_split_frac is not None:
        test_size = round(total_rows * test_split_frac) if isinstance(test_split_frac, float) else test_split_frac
        if test_size > total_rows:
            raise click.BadParameter(f"Requested more test rows ({test_size:,}) than total rows ({total_rows:,})")
        train_size -= test_size

    # Step 5: calculate number of files for each split
    num_valid_files = round(max(1.0, num_files * valid_size / total_rows))
    num_test_files = round(max(1.0, num_files * test_size / total_rows))
    num_train_files = num_files - num_valid_files - num_test_files
    if num_train_files <= 0:
        raise click.BadParameter(f"Not enough files ({num_files:,}) for the split requested")
    suffix_digits_name = int(log10(num_files)) + 1

    click.echo(f"  Train files: {num_train_files:,}")
    click.echo(f"  Valid files: {num_valid_files:,}")
    click.echo(f"  Test files: {num_test_files:,}")

    with tqdm(total=len(all_rows), desc="Writing files", unit="row") as pbar:
        total_files_written = 0

        for split_name, split_start, split_end, num_files in [
            ("train", 0, train_size, num_train_files),
            ("valid", train_size, train_size + valid_size, num_valid_files),
            ("test", train_size + valid_size, train_size + valid_size + test_size, num_test_files),
        ]:
            num_rows_per_file = (split_end - split_start) / num_files
            row_ranges = [
                (round(split_start + i * num_rows_per_file), round(split_start + (i + 1) * num_rows_per_file))
                for i in range(num_files)
            ]

            # if we have valid and/or test split, we include the split name in the path
            split_output_dir = (output_dir / split_name) if valid_size != 0 or test_size != 0 else output_dir
            split_output_dir.mkdir(parents=True, exist_ok=True)

            for range_start, range_end in row_ranges:
                dest_path = split_output_dir / f"shard_{total_files_written:0{suffix_digits_name}d}.jsonl.zst"

                with smart_open.open(dest_path, "wb") as f:  # pyright: ignore
                    for row in all_rows[range_start:range_end]:
                        f.write(row)
                        pbar.update(1)

                total_files_written += 1
                pbar.set_postfix(split=split_name, files_written=total_files_written)
                pbar.refresh()
