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


def _wait_for_batch_groups(
    batch_id_groups: list[list[str]],
    provider: str,
    timeout: int | None,
    poll_interval: int = 30,
) -> list[list[dict]]:
    """Wait for multiple batch groups concurrently, returning results per group.

    Each group's batch_ids are waited on via ``wait_for_batch_completion_async``
    (which itself uses ``asyncio.gather``), and all groups are awaited
    concurrently so polling happens in parallel.
    """
    from lm_deluge.batches import wait_for_batch_completion_async

    assert provider in ("openai", "anthropic"), "Only openai or anthropic support batch mode"

    async def _run() -> list[list[dict]]:
        tasks = [
            wait_for_batch_completion_async(group, provider, poll_interval=poll_interval)  # pyright: ignore
            for group in batch_id_groups
        ]
        coro = asyncio.gather(*tasks)
        if timeout is not None:
            coro = asyncio.wait_for(coro, timeout=timeout)
        return await coro  # pyright: ignore

    try:
        return asyncio.run(_run())
    except asyncio.TimeoutError:
        raise click.ClickException(f"Batch wait timed out after {timeout:,}s")


def _extract_batch_completion(result: dict, provider: str) -> str | None:
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


class BatchedFileCounter:
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

    if not source_files:
        click.echo("No files found to annotate. Exiting.")
        return

    click.echo(f"Found {len(source_files):,} files")

    # Step 2: Group files by min rows (using annotation_batch_size as threshold)
    file_groups = group_files_by_min_rows(
        files=source_files, min_rows_per_group=annotation_batch_size, num_proc=num_proc
    )
    click.echo(f"Created {len(file_groups):,} file groups for parallel processing")

    common_prefix = common_path_prefix([p.path for fp in file_groups for p in fp])

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
    num_proc = 1
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

    decoder = msgspec.json.Decoder()
    total_count = 0

    with tqdm(total=len(annotation_paths_to_submit), desc="Submitting batches", unit="batch") as pbar:
        for path in annotation_paths_to_submit:
            with smart_open.open(path, "rb") as f:  # pyright: ignore
                prompts = [Conversation.from_log(decoder.decode(line)) for line in f]

            if limit_rows is not None and total_count + len(prompts) > limit_rows:
                prompts = prompts[: limit_rows - total_count]

            # Set output_schema via contextvar for the batch submission
            token = _batch_output_schema.set(task_prompt.schema)  # pyright: ignore
            try:
                batch_ids = asyncio.run(client.submit_batch_job(prompts, batch_size=annotation_batch_size))
            finally:
                _batch_output_schema.reset(token)

            batch_info_path = path.parent.parent / ANNOTATION_BATCH_SUBDIR / path.name
            batch_info_path.parent.mkdir(parents=True, exist_ok=True)
            provider = registry[model_name].api_spec
            batch_info = {
                "batch_ids": batch_ids,
                "model": model_name,
                "task_prompt_name": annotation_task_prompt,
                "reasoning_effort": reasoning_effort,
                "max_new_tokens": max_new_tokens,
                "provider": provider,
                "num_prompts": len(prompts),
                "prompts_path": str(path),
            }
            total_count += len(prompts)
            with smart_open.open(batch_info_path, "w") as f:  # pyright: ignore
                json.dump(batch_info, f, indent=2)

            pbar.update(1)
            pbar.set_postfix(dict(total_prompts=total_count))

            if limit_rows is not None and total_count >= limit_rows:
                click.echo(f"Reached row limit of {limit_rows:,}. Stopping submission.")
                break

    click.echo("\nBatch submitted successfully!")
    click.echo(f"  Total prompts submitted: {total_count:,}")


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
def batch_annotate_retrieve(
    batch_dir: Path,
    output_dir: Path,
    timeout: int | None,
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

    encoder = msgspec.json.Encoder()
    decoder = msgspec.json.Decoder()

    # Step 1: Find and read all batch info files (may be nested)
    batch_info_files = sorted(batch_dir.rglob(f"{ANNOTATION_BATCH_SUBDIR}/*"))
    batch_info_files = [f for f in batch_info_files if f.is_file()]

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
    task_prompt = BaseAnnotationPrompt.get(task_prompt_name)

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

    # Step 2: Wait for all batches concurrently. Results are returned per group
    # (one group per batch_info) so custom_ids (0..N-1) don't collide.
    successful_docs = 0
    failed_docs = 0
    output_handles: dict[str, object] = {}

    timeout_msg = f" (timeout: {timeout:,}s)" if timeout else ""
    click.echo(f"Waiting for batch completion...{timeout_msg}")

    all_results = _wait_for_batch_groups(
        batch_id_groups=[[str(bid) for bid in info["batch_ids"]] for info in batch_infos],
        provider=provider,
        timeout=timeout,
    )

    total_results = sum(len(r) for r in all_results)
    click.echo(f"Retrieved {total_results:,} results")
    click.echo()

    # Step 3: Write annotated rows. Output files are keyed by their relative
    # subpath so that annotated and pass-through rows from the same source merge
    # into the same output file.
    click.echo("Writing output files...")

    def _get_output_handle(relative_subpath: str) -> object:
        if relative_subpath not in output_handles:
            output_path = output_dir / relative_subpath
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handles[relative_subpath] = smart_open.open(output_path, "wb")  # pyright: ignore
        return output_handles[relative_subpath]

    try:
        for info, results in tqdm(
            zip(batch_infos, all_results), total=len(batch_infos), desc="Processing batches", unit="batch"
        ):
            prompts_path = Path(info["prompts_path"])

            # Derive relative subpath and corresponding rows file
            relative_subpath = str(prompts_path.relative_to(batches_base))
            rows_path = batch_dir / ROWS_TO_ANNOTATE_SUBDIR / relative_subpath

            # Build completions dict {custom_id: completion_text}
            completions: dict[int, str | None] = {}
            for result in results:
                cid = int(result["custom_id"])
                completions[cid] = _extract_batch_completion(result, provider)

            # Read rows, pair with completions, write annotated output
            out_file = _get_output_handle(relative_subpath)

            with smart_open.open(rows_path, "rb") as rows_file:  # pyright: ignore
                for idx, line in enumerate(rows_file):
                    completion = completions.get(idx)
                    if completion is None:
                        failed_docs += 1
                        continue

                    try:
                        parsed = task_prompt.parse(completion)
                    except Exception:
                        failed_docs += 1
                        continue

                    row = decoder.decode(line)
                    row[task_prompt.name] = parsed
                    out_file.write(encoder.encode(row) + b"\n")  # pyright: ignore
                    successful_docs += 1

        # Step 3: Copy pass-through rows (already annotated), preserving subdirectory structure
        passthrough_docs = 0

        if already_annotated_base.exists():
            passthrough_files = sorted(f for f in already_annotated_base.rglob("*") if f.is_file())
            for src in tqdm(passthrough_files, desc="Copying pass-through rows", unit="file"):
                relative_subpath = str(src.relative_to(already_annotated_base))
                out_file = _get_output_handle(relative_subpath)

                with smart_open.open(src, "rb") as rf:  # pyright: ignore
                    for line in rf:
                        out_file.write(line)  # pyright: ignore
                        passthrough_docs += 1
    finally:
        for handle in output_handles.values():
            handle.close()  # pyright: ignore

    click.echo("\nSummary:")
    click.echo(f"  Pass-through rows: {passthrough_docs:,}")
    click.echo(f"  Annotated rows:    {successful_docs:,}")
    click.echo(f"  Failed rows:       {failed_docs:,}")
    click.echo(f"  Output directory:  {output_dir}")
