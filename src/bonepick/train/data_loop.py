import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import click
import datasets
import smart_open
from tqdm import tqdm

from bonepick.cli import ByteSizeParamType, FloatOrIntParamType, PathParamType
from bonepick.train.data_utils import (
    load_jsonl_dataset,
    DatasetSplit,
    DatasetTuple,
    batch_save_hf_dataset,
    convert_single_file_to_fasttext,
    normalize_single_file,
    sample_single_file,
    write_dataset,
    FILE_SUFFIXES,
    transform_single_file,
)
from bonepick.train.normalizers import list_normalizers


__all__ = [
    "balance_dataset",
    "import_hf_dataset",
    "sample_dataset",
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
@click.option("-i", "--input-dir", type=PathParamType(exists=True, is_dir=True), required=True)
@click.option("-o", "--output-dir", type=PathParamType(mkdir=True, is_dir=True), required=True)
@click.option("-t", "--text-field", type=str, default="text")
@click.option("-l", "--label-field", type=str, default="score")
@click.option("-p", "--num-proc", type=int, default=os.cpu_count())
@click.option("-n", "--normalization", type=click.Choice(list_normalizers()), default="whitespace")
def convert_to_fasttext(
    input_dir: Path,
    output_dir: Path,
    text_field: str,
    label_field: str,
    num_proc: int,
    normalization: str,
):
    for split in ("train", "test"):
        split_dir = input_dir / split
        assert split_dir.exists(), f"Split directory {split_dir} does not exist"
        assert split_dir.is_dir(), f"Split directory {split_dir} is not a directory"

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
                        text_field=text_field,
                        label_field=label_field,
                        normalization=normalization,
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


@click.command()
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
@click.option(
    "-t",
    "--text-field",
    type=str,
    default="text",
    help="Field in dataset to use as text",
)
@click.option(
    "-l",
    "--label-field",
    type=str,
    default="score",
    help="Field in dataset to use as label",
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
    input_dir: tuple[Path, ...],
    output_dir: Path,
    text_field: str,
    label_field: str,
    seed: int,
    num_proc: int,
):
    """Balance dataset so each label has equal representation in all splits."""
    import random

    click.echo("Starting dataset balancing...")
    click.echo(f"  Input directories: {', '.join(str(d) for d in input_dir)}")
    click.echo(f"  Output directory: {output_dir}")
    click.echo(f"  Text field: {text_field}")
    click.echo(f"  Label field: {label_field}")
    click.echo(f"  Seed: {seed}")
    click.echo(f"  Num processes: {num_proc}")

    rng = random.Random(seed)

    dataset_tuple = load_jsonl_dataset(
        dataset_dirs=list(input_dir),
        text_field_name=text_field,
        label_field_name=label_field,
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
        text_field_name=text_field,
        label_field_name=label_field,
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
    """Sample a dataset to a smaller size using random sampling.

    Either --sampling-rate or --target-size must be specified (but not both).
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
        import random

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
