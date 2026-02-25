"""Batch annotation commands for submitting and retrieving LLM batch API jobs."""

import asyncio
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from functools import reduce
from hashlib import sha256
from pathlib import Path
from typing import Self

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

# paths to be used within batch directory for different stages of the annotation process.
ROWS_ALREADY_ANNOTATED_SUBDIR = "rows_already_annotated"
ROWS_TO_ANNOTATE_SUBDIR = "rows_to_annotate"
BATCHES_TO_ANNOTATE_SUBDIR = "batches_to_annotate"
ANNOTATION_BATCH_SUBDIR = "annotation_batch"
RESULTS_SUBDIR = "results"


async def _download_response_to_file(
    session,
    url: str,
    headers: dict[str, str],
    destination_path: Path,
) -> None:
    """Stream an HTTP response body into a local compressed result file."""
    # Keep the same compression suffix on temp files so smart_open applies
    # the expected compressor instead of writing plain JSONL bytes.
    tmp_path = destination_path.with_name(f"{destination_path.stem}.tmp{destination_path.suffix}")
    try:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                raise ValueError(f"Error downloading batch output from {url}: {text}")

            with smart_open.open(tmp_path, "wb") as wf:  # pyright: ignore
                async for chunk in response.content.iter_chunked(1024 * 1024):
                    if chunk:
                        wf.write(chunk)

        tmp_path.replace(destination_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


async def _wait_and_download_openai_batch_async(
    session,
    batch_id: str,
    destination_path: Path,
    poll_interval: int = 30,
) -> str:
    """Poll OpenAI batch until completion, then download JSONL output to disk."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable must be set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    status_url = f"https://api.openai.com/v1/batches/{batch_id}"

    while True:
        async with session.get(status_url, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                raise ValueError(f"Error checking OpenAI batch status for {batch_id}: {text}")

            batch_data = await response.json()
            status = batch_data["status"]

            if status == "completed":
                output_file_id = batch_data.get("output_file_id")
                if not output_file_id:
                    raise ValueError(f"No output file available for OpenAI batch {batch_id}")

                output_url = f"https://api.openai.com/v1/files/{output_file_id}/content"
                await _download_response_to_file(
                    session=session,
                    url=output_url,
                    headers=headers,
                    destination_path=destination_path,
                )
                return batch_id

            if status in ("failed", "expired", "cancelled"):
                raise ValueError(f"OpenAI batch {batch_id} failed with status: {status}")

        await asyncio.sleep(poll_interval)


async def _wait_and_download_anthropic_batch_async(
    session,
    batch_id: str,
    destination_path: Path,
    poll_interval: int = 30,
) -> str:
    """Poll Anthropic batch until completion, then download JSONL output to disk."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key is None:
        raise ValueError("ANTHROPIC_API_KEY environment variable must be set.")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    status_url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}"

    while True:
        async with session.get(status_url, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                raise ValueError(f"Error checking Anthropic batch status for {batch_id}: {text}")

            batch_data = await response.json()
            status = batch_data["processing_status"]

            if status == "ended":
                output_url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
                await _download_response_to_file(
                    session=session,
                    url=output_url,
                    headers=headers,
                    destination_path=destination_path,
                )
                return batch_id

            if status in ("canceled", "expired"):
                raise ValueError(f"Anthropic batch {batch_id} failed with status: {status}")

        await asyncio.sleep(poll_interval)


async def _wait_for_batch_groups_async(
    batch_ids: list[str],
    provider: str,
    results_dir: Path,
    timeout: int | None,
    poll_interval: int = 30,
) -> list[Path]:
    """Wait for batches concurrently and persist each result as ``<batch_id>.jsonl.zst``."""
    import aiohttp

    assert provider in ("openai", "anthropic"), "Only openai or anthropic support batch mode"

    results_dir.mkdir(parents=True, exist_ok=True)

    normalized_batch_ids = [str(bid) for bid in batch_ids]
    existing_batch_ids = [bid for bid in normalized_batch_ids if (results_dir / f"{bid}.jsonl.zst").exists()]
    existing_batch_id_set = set(existing_batch_ids)
    if existing_batch_ids:
        click.echo(f"Found {len(existing_batch_ids):,} existing result files; skipping those downloads.")

    remaining_batch_ids = [bid for bid in normalized_batch_ids if bid not in existing_batch_id_set]
    if not remaining_batch_ids:
        click.echo("No remaining batch downloads needed.")
        return [results_dir / f"{bid}.jsonl.zst" for bid in normalized_batch_ids]

    click.echo(f"Waiting for and downloading {len(remaining_batch_ids):,} remaining batches...")

    async with aiohttp.ClientSession() as session:
        tasks: list[asyncio.Task[str]] = []
        for batch_id in remaining_batch_ids:
            destination = results_dir / f"{batch_id}.jsonl.zst"
            if provider == "openai":
                task = asyncio.create_task(
                    _wait_and_download_openai_batch_async(
                        session=session,
                        batch_id=batch_id,
                        destination_path=destination,
                        poll_interval=poll_interval,
                    )
                )
            else:
                task = asyncio.create_task(
                    _wait_and_download_anthropic_batch_async(
                        session=session,
                        batch_id=batch_id,
                        destination_path=destination,
                        poll_interval=poll_interval,
                    )
                )
            tasks.append(task)

        async def _wait_and_report() -> None:
            completed = 0
            for finished in asyncio.as_completed(tasks):
                downloaded_batch_id = await finished
                completed += 1
                click.echo(f"  Downloaded {completed:,}/{len(tasks):,}: {downloaded_batch_id}.jsonl.zst")

        if timeout is not None:
            await asyncio.wait_for(_wait_and_report(), timeout=timeout)
        else:
            await _wait_and_report()

    return [results_dir / f"{bid}.jsonl.zst" for bid in normalized_batch_ids]


def _wait_for_batch_groups(
    batch_ids: list[str],
    provider: str,
    results_dir: Path,
    timeout: int | None,
    poll_interval: int = 30,
) -> list[Path]:
    """Wait for batches concurrently and persist outputs to local result files."""
    try:
        return asyncio.run(
            _wait_for_batch_groups_async(
                batch_ids=batch_ids,
                provider=provider,
                results_dir=results_dir,
                timeout=timeout,
                poll_interval=poll_interval,
            )
        )
    except asyncio.TimeoutError:
        raise click.ClickException(f"Batch wait timed out after {timeout:,}s")


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
    """Read prompts from each file and submit with limited concurrency.

    Returns (batch_results, total_count) where each batch_result is
    (path, num_prompts, batch_ids).
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
        self.stack: ExitStack | None = None
        self.current_file = None
        self.paths: list[str] = []

    @classmethod
    def from_file_group(
        cls: type[Self],
        file_group: list[str | Path],
        base_dest_dir: str | Path,
        max_rows: int | None = None,
    ) -> Self:
        """Create a counter whose filename is derived from a hash of the file group."""
        file_group_hash = sha256()
        for file_path in sorted(file_group):
            file_group_hash.update(str(file_path).encode())
        hash_prefix = file_group_hash.hexdigest()[:8]
        dest_filename = Path(base_dest_dir) / f"{hash_prefix}.jsonl.zst"
        return cls(dest_filename=dest_filename, max_rows=max_rows)

    def __enter__(self):
        if self.stack is None:
            self.stack = ExitStack()
        return self

    def write_row(self, row: bytes):
        """Write a single row, rolling over to a new file if max_rows is reached."""
        if self.stack is None:
            raise ValueError(f"{self.__class__.__name__} must be used as a context manager")

        if self.current_file is None:
            dest_path = self.destination_dir / f"{self.prefix}_{len(self.paths):08d}{self.suffix}"
            self.paths.append(str(dest_path))
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            self.current_file = self.stack.enter_context(smart_open.open(dest_path, "wb"))  # pyright: ignore

        if self.max_rows is not None and self.current_count >= self.max_rows:
            self.stack.pop_all()  # close current file
            self.current_count = 0
            self.current_file = None
            return self.write_row(row)  # try writing again, will open new file

        self.current_file.write(row.rstrip(b"\n") + b"\n")  # pyright: ignore
        self.current_count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stack:
            self.stack.close()


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
    """Process file groups in parallel and create annotation batches."""

    from bonepick.annotate.prompts import BaseAnnotationPrompt, BaseSystemPrompt
    from bonepick.data.expressions import compile_jq

    # get prompts and tool to extract input field.
    system_prompt = BaseSystemPrompt.get(system_prompt_name) if system_prompt_name else None
    task_prompt = BaseAnnotationPrompt.get(task_prompt_name)
    input_field_selector = compile_jq(input_field_expression)

    # encoder/decoder for writing files
    decoder = msgspec.json.Decoder()
    encoder = msgspec.json.Encoder()

    base_destination_dir = Path(base_destination_dir)

    all_paths_to_annotate: list[str] = []

    with ExitStack() as stack:
        for source_path in file_group:
            input_file = stack.enter_context(smart_open.open(source_path, "rb"))  # pyright: ignore

            source_path = Path(source_path)
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

                # Already annotated — pass through
                if not reprocess_all_rows and task_prompt.name in row:
                    rows_already_annotated.write_row(line)
                    continue

                # build conversation
                conversation = Conversation()
                if system_prompt:
                    conversation.add(Message.system(system_prompt.apply()))
                content = input_field_selector(row)
                conversation.add(Message.user(task_prompt.apply(content, max_text_length)))

                # build row record
                rows_to_annotate.write_row(line)

                # build request record
                requests_to_annotate.write_row(encoder.encode(conversation.to_log()))

            all_paths_to_annotate.extend(requests_to_annotate.paths)
            stack.pop_all()  # ensure all files are closed before returning paths

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
    """Submit batch annotation job to LLM batch API.

    Creates a batch directory with a manifest and compressed rows file,
    then submits prompts via the provider's batch API (OpenAI or Anthropic).
    Files are processed in parallel using ProcessPoolExecutor.
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
) -> tuple[str, int, int]:
    """Process one batch and write annotated rows directly to output.

    Returns (relative_subpath, success_count, fail_count).
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
        result_path = results_base / f"{batch_id}.jsonl.zst"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing downloaded batch results: {result_path}")

        with open(result_path, "rb") as probe:
            prefix = probe.read(4)

        # Support legacy cache files that were plain JSONL with a .zst suffix.
        result_file_opener = smart_open.open if prefix == b"\x28\xb5\x2f\xfd" else open
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

    return relative_subpath, successful, failed


def _copy_single_passthrough(
    src: Path,
    already_annotated_base: Path,
    output_dir: Path,
) -> tuple[str, int]:
    """Copy one passthrough file into output, appending if target exists."""
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

    return relative_subpath, row_count


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
    num_proc: int | None,
):
    """Retrieve batch annotation results.

    Reads batch info files from the batch directory, waits for batch
    completion, then writes annotated output combining newly annotated
    rows with already-annotated pass-through rows. Preserves the
    subdirectory structure from the original dataset.
    """
    extra_dependencies.check()

    from bonepick.annotate.deluge_utils import lm_deluge_monkey_patch
    from bonepick.annotate.prompts import BaseAnnotationPrompt

    lm_deluge_monkey_patch()

    # Step 1: Find and read all batch info files (may be nested)
    annotation_batch_base = batch_dir / ANNOTATION_BATCH_SUBDIR
    batch_info_files: list[Path] = []
    if annotation_batch_base.exists():
        for root, _, files in os.walk(annotation_batch_base):
            for fn in files:
                batch_info_files.append(Path(root) / fn)
    batch_info_files.sort()

    if not batch_info_files:
        raise click.ClickException(f"No batch info files found under {batch_dir}")

    batch_infos: list[dict] = []
    for bif in batch_info_files:
        with smart_open.open(bif, "r") as f:  # pyright: ignore
            batch_infos.append(json.load(f))

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

    all_batch_ids: list[str] = []
    seen_batch_ids: set[str] = set()
    for info in batch_infos:
        for batch_id in info["batch_ids"]:
            batch_id_str = str(batch_id)
            if batch_id_str in seen_batch_ids:
                continue
            seen_batch_ids.add(batch_id_str)
            all_batch_ids.append(batch_id_str)

    timeout_msg = f" (timeout: {timeout:,}s)" if timeout else ""
    click.echo(f"Waiting for batch completion and downloading results...{timeout_msg}")

    result_files = _wait_for_batch_groups(
        batch_ids=all_batch_ids,
        provider=provider,
        results_dir=results_base,
        timeout=timeout,
    )
    click.echo(f"Result files ready: {len(result_files):,}")
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
            )
            for info in batch_infos
        ]

        for future in tqdm(batch_futures, desc="Processing batches", unit="batch"):
            _, successful, failed = future.result()
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
                _, copied_rows = future.result()
                passthrough_docs += copied_rows

    click.echo("\nSummary:")
    click.echo(f"  Pass-through rows: {passthrough_docs:,}")
    click.echo(f"  Annotated rows:    {successful_docs:,}")
    click.echo(f"  Failed rows:       {failed_docs:,}")
    click.echo(f"  Output directory:  {output_dir}")
