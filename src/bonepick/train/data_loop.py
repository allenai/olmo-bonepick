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
    count_tokens_in_file,
    normalize_single_file,
    sample_single_file,
    write_dataset,
    FILE_SUFFIXES,
    transform_single_file,
    pretty_size,
    reshard_single_output,
)
from bonepick.train.normalizers import list_normalizers


__all__ = [
    "balance_dataset",
    "count_tokens",
    "import_hf_dataset",
    "sample_dataset",
    "reshard_dataset",
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
    """Count tokens in one or more dataset directories using parallel processing.

    This command counts the total number of tokens in the specified dataset directories
    using a specified tokenizer. It processes files in parallel for efficiency.
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


def _write_rows_to_file(
    rows: list[bytes],
    destination_path: Path,
) -> tuple[int, int]:
    """Write a list of rows to a destination file.

    Args:
        rows: List of byte strings (lines) to write
        destination_path: Path to destination file

    Returns:
        Tuple of (total_rows, total_bytes) written
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_bytes = 0

    with smart_open.open(destination_path, "wb") as dest_file:  # pyright: ignore
        for line in rows:
            dest_file.write(line)
            total_rows += 1
            total_bytes += len(line)

    return total_rows, total_bytes


def _reshard_rows_to_shards(
    rows: list[bytes],
    output_dir: Path,
    num_files: int,
    num_proc: int,
    split_name: str | None = None,
) -> tuple[int, int, int]:
    """Reshard a list of rows into multiple shard files.

    Args:
        rows: List of byte strings (lines) to write
        output_dir: Output directory for shards
        num_files: Number of output files to create
        num_proc: Number of processes for parallel processing
        split_name: Optional split name (e.g., "train", "test") for logging

    Returns:
        Tuple of (num_files_written, total_rows, total_bytes)
    """
    if len(rows) == 0:
        if split_name:
            click.echo(f"\nSkipping {split_name} split (no rows)")
        return 0, 0, 0

    if split_name:
        click.echo(f"\nProcessing {split_name} split...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Distribute rows into shards
    rows_per_shard = len(rows) // num_files
    remainder = len(rows) % num_files

    executor_cls = ProcessPoolExecutor if num_proc > 1 else ThreadPoolExecutor

    with executor_cls(max_workers=num_proc) as pool:
        futures = []
        start_idx = 0

        for shard_idx in range(num_files):
            # Calculate rows for this shard (distribute remainder evenly)
            shard_rows_count = rows_per_shard + (1 if shard_idx < remainder else 0)
            end_idx = start_idx + shard_rows_count
            shard_rows = rows[start_idx:end_idx]
            start_idx = end_idx

            dest_path = output_dir / f"shard_{shard_idx:05d}.jsonl.zst"
            future = pool.submit(
                _write_rows_to_file,
                rows=shard_rows,
                destination_path=dest_path,
            )
            futures.append(future)

        total_rows = 0
        total_bytes = 0
        desc = f"Writing {split_name} shards" if split_name else "Writing shards"
        pbar = tqdm(total=len(futures), desc=desc, unit="file")
        for future in as_completed(futures):
            try:
                rows, bytes_written = future.result()
                total_rows += len(rows)
                total_bytes += bytes_written
                pbar.set_postfix(rows=f"{total_rows:,}", size=pretty_size(total_bytes))
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e
            pbar.update(1)

        pbar.close()

    if split_name:
        click.echo(f"  {split_name.capitalize()} output files: {len(futures)}")
        click.echo(f"  {split_name.capitalize()} total rows: {total_rows:,}")
        click.echo(f"  {split_name.capitalize()} total size: {pretty_size(total_bytes)}")

    return len(futures), total_rows, total_bytes


def _reshard_files_to_shards(
    input_files: list[tuple[Path, int]],
    output_dir: Path,
    num_files: int,
    num_proc: int,
) -> tuple[int, int, int]:
    """Reshard multiple input files into shards using greedy bin packing.

    Args:
        input_files: List of (file_path, file_size) tuples
        output_dir: Output directory for shards
        num_files: Number of output files to create
        num_proc: Number of processes for parallel processing

    Returns:
        Tuple of (num_files_written, total_rows, total_bytes)
    """
    # Sort files by size (largest first) for better distribution
    input_files.sort(key=lambda x: x[1], reverse=True)

    total_size = sum(size for _, size in input_files)

    # Distribute files into output shards using greedy bin packing
    # This ensures output files are roughly equal in size
    target_size_per_shard = total_size / num_files
    click.echo(f"  Target size per output file: {pretty_size(target_size_per_shard)}")

    # Initialize shards with empty lists and size tracking
    shards: list[tuple[list[Path], int]] = [([], 0) for _ in range(num_files)]

    # Greedy bin packing: assign each file to the shard with smallest current size
    for file_path, file_size in input_files:
        # Find shard with smallest current size
        min_shard_idx = min(range(num_files), key=lambda i: shards[i][1])
        shards[min_shard_idx][0].append(file_path)
        shards[min_shard_idx] = (shards[min_shard_idx][0], shards[min_shard_idx][1] + file_size)

    # Display distribution statistics
    shard_sizes = [size for _, size in shards]
    click.echo(f"  Output file size range: {pretty_size(min(shard_sizes))} - {pretty_size(max(shard_sizes))}")
    click.echo(
        f"  Size standard deviation: {pretty_size(sum((s - target_size_per_shard) ** 2 for s in shard_sizes) ** 0.5 / num_files)}"
    )

    # Process shards in parallel
    click.echo(f"\nResharding using {num_proc} processes...")
    output_dir.mkdir(parents=True, exist_ok=True)

    executor_cls = ProcessPoolExecutor if num_proc > 1 else ThreadPoolExecutor

    with executor_cls(max_workers=num_proc) as pool:
        futures = []
        for shard_idx, (shard_files, _) in enumerate(shards):
            if not shard_files:
                continue

            dest_path = output_dir / f"shard_{shard_idx:05d}{'.jsonl.zst'}"
            future = pool.submit(
                reshard_single_output,
                source_files=shard_files,
                destination_path=dest_path,
                shard_index=shard_idx,
            )
            futures.append(future)

        total_rows = 0
        total_bytes = 0
        pbar = tqdm(total=len(futures), desc="Resharding files", unit="file")
        for future in as_completed(futures):
            try:
                rows, bytes_written = future.result()
                total_rows += rows
                total_bytes += bytes_written
                pbar.set_postfix(rows=f"{total_rows:,}", size=pretty_size(total_bytes))
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e
            pbar.update(1)

        pbar.close()

    return len(futures), total_rows, total_bytes


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
    """Reshard a dataset by combining multiple files into exactly num-files output files.

    This command redistributes the data from multiple small files into a specified number
    of larger files, with output files being roughly equal in size. All files in the input
    directory and its subdirectories will be combined.

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

    # Step 2: Handle train/test/valid split if requested
    if test_split_frac is not None or valid_split_frac is not None:
        import random
        import smart_open

        # Read all rows from all files
        click.echo("\nReading all rows for splitting...")
        all_rows: list[bytes] = []
        for file_path, _ in tqdm(input_files, desc="Reading files", unit="file"):
            with smart_open.open(file_path, "rb") as f:  # pyright: ignore
                for line in f:
                    all_rows.append(line)

        total_rows = len(all_rows)
        click.echo(f"  Total rows: {total_rows:,}")

        # Shuffle rows
        click.echo(f"  Shuffling rows with seed {seed}...")
        rng = random.Random(seed)
        rng.shuffle(all_rows)

        # Calculate test split size
        test_size = 0
        if test_split_frac is not None:
            if isinstance(test_split_frac, float):
                if test_split_frac <= 0 or test_split_frac >= 1.0:
                    raise click.BadParameter("--test-split-frac must be between 0 and 1.0 when using float")
                test_size = int(total_rows * test_split_frac)
            else:
                if test_split_frac <= 0 or test_split_frac >= total_rows:
                    raise click.BadParameter(f"--test-split-frac must be between 0 and {total_rows} when using int")
                test_size = test_split_frac

        # Calculate valid split size
        valid_size = 0
        if valid_split_frac is not None:
            if isinstance(valid_split_frac, float):
                if valid_split_frac <= 0 or valid_split_frac >= 1.0:
                    raise click.BadParameter("--valid-split-frac must be between 0 and 1.0 when using float")
                valid_size = int(total_rows * valid_split_frac)
            else:
                if valid_split_frac <= 0 or valid_split_frac >= total_rows:
                    raise click.BadParameter(f"--valid-split-frac must be between 0 and {total_rows} when using int")
                valid_size = valid_split_frac

        # Validate total split sizes
        if test_size + valid_size >= total_rows:
            raise click.BadParameter(
                f"Combined test ({test_size}) and valid ({valid_size}) sizes exceed total rows ({total_rows})"
            )

        train_size = total_rows - test_size - valid_size
        click.echo(f"  Train rows: {train_size:,}")
        if test_size > 0:
            click.echo(f"  Test rows: {test_size:,}")
        if valid_size > 0:
            click.echo(f"  Valid rows: {valid_size:,}")

        # Split rows: train first, then test, then valid
        train_rows = all_rows[:train_size]
        test_rows = all_rows[train_size : train_size + test_size] if test_size > 0 else []
        valid_rows = all_rows[train_size + test_size :] if valid_size > 0 else []

        # Calculate number of files for each split (proportional to data size)
        # Only count splits that have data
        active_splits: list[tuple[str, list[bytes], int]] = []

        # Always have train
        train_num_files = max(1, int(num_files * train_size / total_rows))
        remaining_files = num_files - train_num_files
        active_splits.append(("train", train_rows, train_num_files))

        if test_size > 0 and valid_size > 0:
            # Both test and valid - split remaining files proportionally
            test_num_files = max(1, int(remaining_files * test_size / (test_size + valid_size)))
            valid_num_files = max(1, remaining_files - test_num_files)
            active_splits.append(("test", test_rows, test_num_files))
            active_splits.append(("valid", valid_rows, valid_num_files))
        elif test_size > 0:
            test_num_files = max(1, remaining_files)
            active_splits.append(("test", test_rows, test_num_files))
        elif valid_size > 0:
            valid_num_files = max(1, remaining_files)
            active_splits.append(("valid", valid_rows, valid_num_files))

        for split_name, _, split_num_files in active_splits:
            click.echo(f"  {split_name.capitalize()} files: {split_num_files}")

        # Process each split using helper function
        for split_name, split_rows, split_num_files in active_splits:
            _reshard_rows_to_shards(
                rows=split_rows,
                output_dir=output_dir / split_name,
                num_files=split_num_files,
                num_proc=num_proc,
                split_name=split_name,
            )

        click.echo("\nResharding complete!")
        click.echo(f"  Output written to: {output_dir}")
        return

    # Original code path (no train/test split) - use helper function
    num_output_files, total_rows, total_bytes = _reshard_files_to_shards(
        input_files=input_files,
        output_dir=output_dir,
        num_files=num_files,
        num_proc=num_proc,
    )

    click.echo("\nResharding complete!")
    click.echo(f"  Output files: {num_output_files}")
    click.echo(f"  Total rows: {total_rows:,}")
    click.echo(f"  Total size: {pretty_size(total_bytes)}")
    click.echo(f"  Output written to: {output_dir}")
