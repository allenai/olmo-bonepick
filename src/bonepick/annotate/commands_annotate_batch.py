"""Batch annotation CLI commands.

This module focuses on command orchestration and local file transforms:
- ``batch_annotate_submit`` prepares shards and submits provider batches;
- ``batch_annotate_retrieve`` merges cached batch outputs back into rows.

Provider polling and result downloading live in ``retrieve_utils.py``.
"""

import asyncio
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from functools import reduce
from pathlib import Path

import click
import msgspec
import smart_open
from lazy_imports import try_import
from tqdm import tqdm

from bonepick.cli import PathParamType
from bonepick.data.file_estimates import group_files_by_min_rows
from bonepick.data.utils import common_path_prefix, is_valid_suffix

with try_import() as extra_dependencies:
    from lm_deluge import Conversation, LLMClient, Message
    from lm_deluge.models import registry

    # import here to register all the prompts
    from bonepick.annotate import prompt_collections  # noqa: F401


from bonepick.annotate.commands_annotate_stream import ReasoningEffort
from bonepick.annotate.retrieve_utils import (
    collect_unique_batch_ids,
    result_file_uses_zstd,
    result_path_for_batch,
    wait_for_batch_downloads,
)

# paths to be used within batch directory for different stages of the annotation process.
ROWS_ALREADY_ANNOTATED_SUBDIR = "rows_already_annotated"
ROWS_TO_ANNOTATE_SUBDIR = "rows_to_annotate"
BATCHES_TO_ANNOTATE_SUBDIR = "batches_to_annotate"
ANNOTATION_BATCH_SUBDIR = "annotation_batch"
RESULTS_SUBDIR = "results"


def _extract_batch_completion(result: dict, provider: str) -> str | None:
    """Extract the completion text from a batch API result dict."""
    try:
        if provider == "openai":
            return result["response"]["body"]["choices"][0]["message"]["content"]
        elif provider == "anthropic":
            if result["result"]["type"] != "succeeded":
                return None
            return result["result"]["message"]["content"][0]["text"]
    except (KeyError, IndexError):
        return None
    return None


async def _read_and_submit_all_batches(
    annotation_paths: list[Path],
    client: "LLMClient",  # pyright: ignore
    batch_size: int,
    limit_rows: int | None,
    max_concurrent_submissions: int = 4,
) -> tuple[list[tuple[Path, int, list[str]]], int]:
    """Read prompt shards and submit them with bounded concurrency.

    Returns ``(batch_results, total_count)`` where each batch result is
    ``(source_path, num_prompts, batch_ids)``.
    """
    decoder = msgspec.json.Decoder()
    semaphore = asyncio.Semaphore(max_concurrent_submissions)
    tasks: list[asyncio.Task] = []
    batch_meta: list[tuple[Path, int]] = []
    total_count = 0

    async def _submit_with_semaphore(prompts, batch_sz, max_retries=5, base_delay=5.0):
        async with semaphore:
            for attempt in range(max_retries + 1):
                try:
                    return await client.submit_batch_job(prompts, batch_size=batch_sz)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    delay = base_delay * (2**attempt)
                    click.echo(f"  Batch submission failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    click.echo(f"  Retrying in {delay:.0f}s...")
                    await asyncio.sleep(delay)

    for path in annotation_paths:
        with smart_open.open(path, "rb") as f:  # pyright: ignore
            prompts = []
            for line in f:
                try:
                    prompt = decoder.decode(line)
                except msgspec.DecodeError:
                    continue

                hydrated_prompt = Conversation.from_log(prompt)
                prompts.append(hydrated_prompt)

                if limit_rows is not None and total_count + len(prompts) > limit_rows:
                    break

        # Fire off submission with concurrency limit
        task = asyncio.create_task(_submit_with_semaphore(prompts, batch_size))
        tasks.append(task)
        batch_meta.append((path, len(prompts)))
        total_count += len(prompts)

        click.echo(f"  Queued batch {len(tasks):,} ({len(prompts):,} prompts) from {path.name}")

        # Yield to event loop so in-flight submissions can make progress
        await asyncio.sleep(0)

        if limit_rows is not None and total_count >= limit_rows:
            click.echo(f"Reached row limit of {limit_rows:,}.")
            break

    click.echo(f"Awaiting {len(tasks):,} batch submissions ({total_count:,} prompts)...")
    all_batch_ids = await asyncio.gather(*tasks)

    return [
        (path, num_prompts, batch_ids) for (path, num_prompts), batch_ids in zip(batch_meta, all_batch_ids)
    ], total_count


class BatchedFileCounter:
    """Context manager that writes rows across sequentially-numbered output files.

    Automatically rolls over to a new file when ``max_rows`` is reached.
    """

    def __init__(
        self,
        dest_filename: str | Path,
        max_rows: int | None = None,
    ):
        self.destination_dir = Path(dest_filename).parent
        self.destination_dir.mkdir(parents=True, exist_ok=True)
        self.suffix = "".join(Path(dest_filename).suffixes)
        self.prefix = Path(dest_filename).name[: -len(self.suffix)]
        self.max_rows = max_rows
        self.current_count = 0
        self.current_file = None
        self.paths: list[str] = []

    def __enter__(self):
        return self

    def _open_next_file(self) -> None:
        dest_path = self.destination_dir / f"{self.prefix}_{len(self.paths):08d}{self.suffix}"
        self.paths.append(str(dest_path))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_file = smart_open.open(dest_path, "wb")  # pyright: ignore

    def write_row(self, row: bytes):
        """Write a single row, rolling over to a new file if max_rows is reached."""
        if self.max_rows is not None and self.current_count >= self.max_rows:
            if self.current_file is not None:
                self.current_file.close()  # pyright: ignore
            self.current_count = 0
            self.current_file = None

        if self.current_file is None:
            self._open_next_file()

        self.current_file.write(row.rstrip(b"\n") + b"\n")  # pyright: ignore
        self.current_count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_file is not None:
            self.current_file.close()  # pyright: ignore
            self.current_file = None


def _make_annotation_batches(
    file_group: list[str | Path],
    common_prefix: Path,
    base_destination_dir: str | Path,
    task_prompt_name: str,
    input_field_expression: str,
    reprocess_all_rows: bool,
    max_request_count: int,
    system_prompt_name: str | None = None,
    max_text_length: int | None = None,
) -> list[str]:
    """Prepare sharded inputs for batch APIs.

    For each source file, rows are split into:
    1. already-annotated pass-through rows;
    2. rows that still need annotation;
    3. serialized prompt payloads for batch submission.
    """

    from bonepick.annotate.prompts import BaseAnnotationPrompt, BaseSystemPrompt
    from bonepick.data.expressions import compile_jq

    # Prompt loaders and input selector for the source row.
    system_prompt = BaseSystemPrompt.get(system_prompt_name) if system_prompt_name else None
    task_prompt = BaseAnnotationPrompt.get(task_prompt_name)
    input_field_selector = compile_jq(input_field_expression)

    # Fast JSON codec for row and prompt serialization.
    decoder = msgspec.json.Decoder()
    encoder = msgspec.json.Encoder()

    base_destination_dir = Path(base_destination_dir)

    all_paths_to_annotate: list[str] = []

    for source_path in file_group:
        source_path = Path(source_path)
        with ExitStack() as stack:
            input_file = stack.enter_context(smart_open.open(source_path, "rb"))  # pyright: ignore

            annotated_rows_subpath = (
                base_destination_dir / ROWS_ALREADY_ANNOTATED_SUBDIR / source_path.relative_to(common_prefix)
            )
            rows_already_annotated = stack.enter_context(
                BatchedFileCounter(dest_filename=annotated_rows_subpath, max_rows=max_request_count)
            )
            to_annotate_rows_subpath = (
                base_destination_dir / ROWS_TO_ANNOTATE_SUBDIR / source_path.relative_to(common_prefix)
            )
            rows_to_annotate = stack.enter_context(
                BatchedFileCounter(dest_filename=to_annotate_rows_subpath, max_rows=max_request_count)
            )
            batches_to_annotate_subpath = (
                base_destination_dir / BATCHES_TO_ANNOTATE_SUBDIR / source_path.relative_to(common_prefix)
            )
            requests_to_annotate = stack.enter_context(
                BatchedFileCounter(dest_filename=batches_to_annotate_subpath, max_rows=max_request_count)
            )

            for line in input_file:
                row = decoder.decode(line)

                if not reprocess_all_rows and task_prompt.name in row:
                    rows_already_annotated.write_row(line)
                    continue

                conversation = Conversation()
                if system_prompt:
                    conversation.add(Message.system(system_prompt.apply()))
                content = input_field_selector(row)
                conversation.add(Message.user(task_prompt.apply(content, max_text_length)))

                rows_to_annotate.write_row(line)
                requests_to_annotate.write_row(encoder.encode(conversation.to_log()))

            all_paths_to_annotate.extend(requests_to_annotate.paths)

    return all_paths_to_annotate


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
    "-b",
    "--batch-dir",
    type=PathParamType(mkdir=True, is_dir=True),
    required=True,
    help="Batch output directory",
)
@click.option(
    "-m",
    "--model-name",
    default="gpt-5.2",
    help="Name of the model to use for annotation",
)
@click.option(
    "-T",
    "--annotation-task-prompt",
    required=True,
    type=str,
    help="Name of the annotation task prompt to use",
)
@click.option(
    "-S",
    "--annotation-system-prompt",
    default=None,
    type=str,
    help="Name of the annotation system prompt to use",
)
@click.option(
    "-i",
    "--input-field-expression",
    type=str,
    default=".text",
    help="Expression to extract the input text from the row",
)
@click.option(
    "-f",
    "--input-field-format",
    type=click.Choice(["text", "conversation"]),
    default="text",
    help="Format of the input: `text` is a string, `conversation` is a list of messages.",
)
@click.option(
    "-r",
    "--reasoning-effort",
    type=click.Choice([effort.value for effort in ReasoningEffort]),
    default=None,
    help="Reasoning effort to use for annotation",
)
@click.option(
    "--reprocess-all-rows/--process-missing-rows",
    is_flag=True,
    default=False,
    help="Reprocess all rows or only missing rows",
)
@click.option(
    "--max-text-length",
    default=100_000,
    type=int,
    help="Maximum text length",
)
@click.option(
    "--max-new-tokens",
    default=16_384,
    type=int,
    help="Maximum new tokens",
)
@click.option(
    "--limit-rows",
    default=None,
    type=int,
    help="Maximum number of rows to annotate",
)
@click.option(
    "--annotation-batch-size",
    default=50_000,
    type=int,
    help="Max items per API batch",
)
@click.option(
    "-p",
    "--num-proc",
    default=None,
    type=int,
    help="Number of parallel workers for file processing (default: cpu_count)",
)
@click.option(
    "--seed",
    default=0,
    type=int,
    help="Random seed for shuffling source files",
)
def batch_annotate_submit(
    dataset_dir: tuple[Path, ...],
    batch_dir: Path,
    model_name: str,
    annotation_task_prompt: str,
    annotation_system_prompt: str | None,
    input_field_expression: str,
    input_field_format: str,
    reasoning_effort: str | None,
    reprocess_all_rows: bool,
    max_text_length: int | None,
    max_new_tokens: int,
    limit_rows: int | None,
    annotation_batch_size: int,
    num_proc: int | None,
    seed: int,
):
    """Build batch inputs and submit batch annotation jobs.

    Submit flow:
    1. discover source dataset files and group them by estimated rows;
    2. split rows into pass-through/to-annotate/request shards in parallel;
    3. submit request shards to the provider batch API;
    4. persist batch metadata under ``annotation_batch/`` for retrieval.
    """
    extra_dependencies.check()

    from bonepick.annotate.deluge_utils import _batch_output_schema, lm_deluge_monkey_patch
    from bonepick.annotate.prompts import BaseAnnotationPrompt

    lm_deluge_monkey_patch()

    num_proc = num_proc or os.cpu_count() or 1

    click.echo("Locations:")
    click.echo("  Dataset directories:")
    for d in dataset_dir:
        click.echo(f"    - {str(d)}")
    click.echo(f"  Batch directory: {str(batch_dir)}")
    click.echo()

    click.echo("Annotation task")
    click.echo(f"  Prompt: {annotation_task_prompt}")
    click.echo(f"  System prompt: {annotation_system_prompt}")
    click.echo(f"  Input field expression: {input_field_expression}")
    click.echo(f"  Input field format: {input_field_format}")
    click.echo(f"  Max text length: {max_text_length:,}")
    click.echo(f"  Num workers: {num_proc}")
    click.echo()

    task_prompt = BaseAnnotationPrompt.get(annotation_task_prompt)

    if input_field_format != "text":
        raise NotImplementedError("Only text format is supported for now")

    # Step 1: Collect all files
    click.echo("Collecting files...")
    source_files: list[Path] = []

    for input_dir in dataset_dir:

        # is a file, collect it as-is
        if Path(input_dir).is_file():
            source_files.append(Path(input_dir))
            continue

        # is a directory, walk it and collect all files
        for root, _, files in os.walk(input_dir):
            for _fn in files:
                fn = Path(root) / _fn
                if not is_valid_suffix(fn):
                    continue
                source_files.append(fn)

    import random

    random.Random(seed).shuffle(source_files)

    if not source_files:
        click.echo("No files found to annotate. Exiting.")
        return

    click.echo(f"Found {len(source_files):,} files")

    # Step 2: Group files by min rows (using annotation_batch_size as threshold)
    file_groups = group_files_by_min_rows(
        files=source_files, min_rows_per_group=annotation_batch_size, num_proc=num_proc
    )
    click.echo(f"Created {len(file_groups):,} file groups for parallel processing")

    common_prefix = common_path_prefix([p.path for fp in file_groups for p in fp] + [Path(p) for p in dataset_dir])

    if limit_rows is not None:
        # cheecky cumsum using reduce to accummulate row counts.
        cumulative_rows = reduce(
            lambda acc, fg: acc + [sum(f.est for f in fg) + (acc[-1] if acc else 0)], file_groups, []
        )
        file_groups = [
            fg
            # the shift by one is because we want to stop if next cumulative sum is larger than
            # limit, not the current one.
            for fg, cum_rows in zip(file_groups, [0] + cumulative_rows)
            if cum_rows <= limit_rows
        ]
        click.echo(f"Limited to {limit_rows:,} rows across {len(file_groups):,} file groups")

    # Step 3: Process file groups in parallel
    click.echo("Building prompts and processing rows...")
    pool_cls = ProcessPoolExecutor if num_proc > 1 else ThreadPoolExecutor

    # Submit all groups and iterate in submission order to preserve file ordering
    annotation_paths_to_submit: list[Path] = []

    with pool_cls(max_workers=num_proc) as pool:
        futures = []
        for fg in file_groups:
            future = pool.submit(
                _make_annotation_batches,
                file_group=[f.path for f in fg],  # type: ignore
                common_prefix=common_prefix,
                task_prompt_name=annotation_task_prompt,
                system_prompt_name=annotation_system_prompt,
                base_destination_dir=batch_dir,
                input_field_expression=input_field_expression,
                reprocess_all_rows=reprocess_all_rows,
                max_request_count=annotation_batch_size,
                max_text_length=max_text_length,
            )
            futures.append(future)

        for future in tqdm(futures, total=len(futures), desc="Processing groups", unit="group"):
            try:
                group_result = future.result()
            except Exception as e:
                # cancel remaining futures to avoid unnecessary work
                click.echo(f"Error processing group: {e}")
                for f in futures:
                    f.cancel()
                raise click.ClickException(f"Error processing group: {e}") from e

            annotation_paths_to_submit.extend([Path(p) for p in group_result])

    # Step 6: Submit batch job
    click.echo("\nInitializing LLM client...")
    click.echo(f"  Model name:       {model_name}")
    click.echo(f"  Reasoning effort: {reasoning_effort}")
    click.echo(f"  Max new tokens:   {max_new_tokens:,}")
    click.echo()

    client = LLMClient(
        model_name,
        reasoning_effort=ReasoningEffort(reasoning_effort).value if reasoning_effort else None,
        max_new_tokens=max_new_tokens,
    )

    click.echo(f"Submitting up to {len(annotation_paths_to_submit):,} batches...")

    token = _batch_output_schema.set(task_prompt.schema)  # pyright: ignore
    try:
        batch_results, total_count = asyncio.run(
            _read_and_submit_all_batches(
                annotation_paths=annotation_paths_to_submit,
                client=client,
                batch_size=annotation_batch_size,
                limit_rows=limit_rows,
            )
        )
    finally:
        _batch_output_schema.reset(token)

    # Write batch info files
    provider = registry[model_name].api_spec
    batches_base = batch_dir / BATCHES_TO_ANNOTATE_SUBDIR
    for path, num_prompts, batch_ids in batch_results:
        batch_info_path = batch_dir / ANNOTATION_BATCH_SUBDIR / path.relative_to(batches_base)
        batch_info_path.parent.mkdir(parents=True, exist_ok=True)
        batch_info = {
            "batch_ids": batch_ids,
            "model": model_name,
            "task_prompt_name": annotation_task_prompt,
            "reasoning_effort": reasoning_effort,
            "max_new_tokens": max_new_tokens,
            "provider": provider,
            "num_prompts": num_prompts,
            "prompts_path": str(path),
        }
        with smart_open.open(batch_info_path, "w") as f:  # pyright: ignore
            json.dump(batch_info, f, indent=2)

    click.echo("\nBatch submitted successfully!")
    click.echo(f"  Total prompts submitted: {total_count:,}")


def _process_single_batch_result(
    info: dict,
    batch_dir: Path,
    batches_base: Path,
    results_base: Path,
    output_dir: Path,
    provider: str,
    task_prompt_name: str,
    skip_missing_batch_results: bool = False,
) -> tuple[int, int]:
    """Merge one batch-info file into an output shard.

    Returns ``(success_count, fail_count)``.
    """
    from bonepick.annotate.prompts import BaseAnnotationPrompt

    decoder = msgspec.json.Decoder()
    encoder = msgspec.json.Encoder()
    task_prompt = BaseAnnotationPrompt.get(task_prompt_name)

    prompts_path = Path(info["prompts_path"])
    relative_subpath = str(prompts_path.relative_to(batches_base))
    rows_path = batch_dir / ROWS_TO_ANNOTATE_SUBDIR / relative_subpath
    output_path = output_dir / relative_subpath
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completions: dict[int, str | None] = {}
    for batch_id in info["batch_ids"]:
        result_path = result_path_for_batch(results_dir=results_base, batch_id=str(batch_id))
        if not result_path.exists():
            if skip_missing_batch_results:
                continue
            raise FileNotFoundError(f"Missing downloaded batch results: {result_path}")

        # Support legacy cache files that were plain JSONL with a .zst suffix.
        result_file_opener = smart_open.open if result_file_uses_zstd(result_path) else open
        with result_file_opener(result_path, "rb") as result_file:  # pyright: ignore
            for line in result_file:
                try:
                    result = decoder.decode(line)
                except msgspec.DecodeError:
                    continue
                try:
                    cid = int(result["custom_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                completions[cid] = _extract_batch_completion(result=result, provider=provider)

    successful = 0
    failed = 0

    with (
        smart_open.open(rows_path, "rb") as rows_file,  # pyright: ignore
        smart_open.open(output_path, "wb") as output_file,  # pyright: ignore
    ):
        for idx, line in enumerate(rows_file):
            completion = completions.get(idx)
            if completion is None:
                failed += 1
                continue
            try:
                parsed = task_prompt.parse(completion)
            except Exception:
                failed += 1
                continue
            row = decoder.decode(line)
            row[task_prompt.name] = parsed
            output_file.write(encoder.encode(row) + b"\n")  # pyright: ignore
            successful += 1

    return successful, failed


def _copy_single_passthrough(
    src: Path,
    already_annotated_base: Path,
    output_dir: Path,
) -> int:
    """Copy one pass-through file into output, appending if target exists."""
    relative_subpath = str(src.relative_to(already_annotated_base))
    output_path = output_dir / relative_subpath
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "ab" if output_path.exists() else "wb"
    row_count = 0
    with (
        smart_open.open(src, "rb") as read_file,  # pyright: ignore
        smart_open.open(output_path, mode) as write_file,  # pyright: ignore
    ):
        for line in read_file:
            write_file.write(line)  # pyright: ignore
            row_count += 1

    return row_count


def _discover_batch_info_files(batch_dir: Path) -> list[Path]:
    """Return sorted batch-info files under ``annotation_batch``."""
    annotation_batch_base = batch_dir / ANNOTATION_BATCH_SUBDIR
    if not annotation_batch_base.exists():
        return []
    return sorted(path for path in annotation_batch_base.rglob("*") if path.is_file())


def _load_batch_infos(batch_info_files: list[Path]) -> list[dict]:
    """Load JSON metadata for each batch-info file."""
    batch_infos: list[dict] = []
    for batch_info_file in batch_info_files:
        with smart_open.open(batch_info_file, "r") as f:  # pyright: ignore
            batch_infos.append(json.load(f))
    return batch_infos


@click.command()
@click.option(
    "-b",
    "--batch-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    help="Batch directory from submit step",
)
@click.option(
    "-o",
    "--output-dir",
    type=PathParamType(mkdir=True, is_dir=True),
    required=True,
    help="Final output directory",
)
@click.option(
    "--timeout",
    default=None,
    type=int,
    help="Timeout in seconds for waiting for batch completion",
)
@click.option(
    "--skip-failed-batches",
    is_flag=True,
    default=False,
    help="Skip failed/cancelled/expired batch downloads and continue with available results",
)
@click.option(
    "--skip-in-progress",
    is_flag=True,
    default=False,
    help="Skip batches that are still processing and continue with available completed results",
)
@click.option(
    "-p",
    "--num-proc",
    default=None,
    type=int,
    help="Number of parallel workers for result processing (default: cpu_count)",
)
def batch_annotate_retrieve(
    batch_dir: Path,
    output_dir: Path,
    timeout: int | None,
    skip_failed_batches: bool,
    skip_in_progress: bool,
    num_proc: int | None,
):
    """Retrieve batch annotation results.

    Retrieval flow:
    1. load per-shard batch metadata from ``annotation_batch/``;
    2. wait for pending provider batches and cache outputs in ``results/``;
    3. merge newly annotated rows into output shards in parallel;
    4. copy pre-annotated pass-through rows into those same output shards.
    """
    extra_dependencies.check()

    from bonepick.annotate.deluge_utils import lm_deluge_monkey_patch
    from bonepick.annotate.prompts import BaseAnnotationPrompt

    lm_deluge_monkey_patch()

    # Step 1: read batch metadata produced by batch_annotate_submit.
    batch_info_files = _discover_batch_info_files(batch_dir)
    if not batch_info_files:
        raise click.ClickException(f"No batch info files found under {batch_dir}")

    batch_infos = _load_batch_infos(batch_info_files)

    # Extract common config from first batch info
    task_prompt_name = batch_infos[0]["task_prompt_name"]
    provider = batch_infos[0]["provider"]
    model = batch_infos[0]["model"]
    BaseAnnotationPrompt.get(task_prompt_name)  # validate prompt exists early

    batches_base = batch_dir / BATCHES_TO_ANNOTATE_SUBDIR
    already_annotated_base = batch_dir / ROWS_ALREADY_ANNOTATED_SUBDIR
    total_prompts = sum(info["num_prompts"] for info in batch_infos)

    click.echo("Batch retrieval")
    click.echo(f"  Batch dir:     {batch_dir}")
    click.echo(f"  Output dir:    {output_dir}")
    click.echo(f"  Model:         {model}")
    click.echo(f"  Provider:      {provider}")
    click.echo(f"  Task prompt:   {task_prompt_name}")
    click.echo(f"  Batch infos:   {len(batch_infos):,}")
    click.echo(f"  Total prompts: {total_prompts:,}")
    click.echo()

    # Step 2: Wait for all batches concurrently and download each output file.
    successful_docs = 0
    failed_docs = 0
    results_base = batch_dir / RESULTS_SUBDIR
    results_base.mkdir(parents=True, exist_ok=True)

    all_batch_ids = collect_unique_batch_ids(batch_infos)

    timeout_msg = f" (timeout: {timeout:,}s)" if timeout else ""
    click.echo(f"Waiting for batch completion and downloading results...{timeout_msg}")

    result_files = wait_for_batch_downloads(
        batch_ids=all_batch_ids,
        provider=provider,
        results_dir=results_base,
        timeout=timeout,
        reporter=click.echo,
        skip_failed_batches=skip_failed_batches,
        skip_in_progress=skip_in_progress,
    )
    click.echo(f"Result files ready: {len(result_files):,}")
    missing_batch_ids = [
        batch_id
        for batch_id in all_batch_ids
        if not result_path_for_batch(results_dir=results_base, batch_id=batch_id).exists()
    ]
    allow_partial_batch_results = skip_failed_batches or skip_in_progress
    if missing_batch_ids:
        if not allow_partial_batch_results:
            raise click.ClickException(
                f"{len(missing_batch_ids):,} batch results are missing. "
                "Re-run with --skip-failed-batches or --skip-in-progress to continue with partial results."
            )
        click.echo(f"Skipping {len(missing_batch_ids):,} batches without result files.")
    click.echo()

    # Step 3: Process each annotation batch info in parallel. Each worker reads
    # rows + downloaded result files and writes annotated output directly.
    num_proc = num_proc or os.cpu_count() or 1
    pool_cls = ProcessPoolExecutor if num_proc > 1 else ThreadPoolExecutor

    click.echo(f"Writing output files (workers: {num_proc})...")
    passthrough_docs = 0

    with pool_cls(max_workers=num_proc) as pool:
        batch_futures = [
            pool.submit(
                _process_single_batch_result,
                info=info,
                batch_dir=batch_dir,
                batches_base=batches_base,
                results_base=results_base,
                output_dir=output_dir,
                provider=provider,
                task_prompt_name=task_prompt_name,
                skip_missing_batch_results=allow_partial_batch_results,
            )
            for info in batch_infos
        ]

        for future in tqdm(batch_futures, desc="Processing batches", unit="batch"):
            successful, failed = future.result()
            successful_docs += successful
            failed_docs += failed

    # Step 4: Copy pass-through rows (already annotated) from disk.
    if already_annotated_base.exists():
        passthrough_files = sorted(f for f in already_annotated_base.rglob("*") if f.is_file())
        with ThreadPoolExecutor(max_workers=num_proc) as copy_pool:
            pt_futures = [
                copy_pool.submit(
                    _copy_single_passthrough,
                    src=src,
                    already_annotated_base=already_annotated_base,
                    output_dir=output_dir,
                )
                for src in passthrough_files
            ]

            for future in tqdm(pt_futures, desc="Copying pass-through rows", unit="file"):
                passthrough_docs += future.result()

    click.echo("\nSummary:")
    click.echo(f"  Pass-through rows: {passthrough_docs:,}")
    click.echo(f"  Annotated rows:    {successful_docs:,}")
    click.echo(f"  Failed rows:       {failed_docs:,}")
    click.echo(f"  Skipped batches:   {len(missing_batch_ids):,}")
    click.echo(f"  Output directory:  {output_dir}")
