import asyncio
import os
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from typing import Literal, TypedDict

import click
import msgspec
import smart_open
from lazy_imports import try_import
from tqdm import tqdm

from bonepick.annotate.prompts import BaseAnnotationPrompt, BaseSystemPrompt
from bonepick.cli import PathParamType
from bonepick.data.expressions import compile_jq
from bonepick.data.utils import FILE_SUFFIXES

with try_import() as extra_dependencies:
    # extra imports; they won't fail here, but later when running the command they will
    from lm_deluge import Conversation, LLMClient, Message
    from platformdirs import user_cache_dir

    # import here to register all the prompts
    from bonepick.annotate import prompt_collections  # noqa: F401


class DatasetRow(TypedDict):
    text: str
    label: str | None


class ServiceTier(Enum):
    AUTO = "auto"
    DEFAULT = "default"
    FLEX = "flex"
    PRIORITY = "priority"
    NONE = None


class ReasoningEffort(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    NONE = "none"


@click.command()
@click.option(
    "-d",
    "--dataset-dir",
    type=PathParamType(exists=True, is_dir=True),
    required=True,
    multiple=True,
    help="Dataset directory (can be specified multiple times)",
)
@click.option("-o", "--output-dir", type=PathParamType(mkdir=True, is_dir=True), default=None)
@click.option(
    "-m",
    "--model-name",
    default="gpt-5.2",
    help="Name of the model to use for annotation",
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
    help="Format of the input: `text` is a string, `conversation` is a list of messages in OpenAI chat format.",
)
@click.option(
    "-r",
    "--reasoning-effort",
    type=click.Choice([effort.value for effort in ReasoningEffort]),
    default=None,
    help="Reasoning effort to use for annotation",
)
@click.option(
    "-c",
    "--cache-location",
    default=None,
    type=PathParamType(is_dir=True, mkdir=True, optional=True),
    help="location to cache data (if not set, will use default cache location)",
)
@click.option(
    "-T",
    "--annotation-task-prompt",
    required=True,
    type=str,
    help="Name of the annotation task prompt to use; use `bonepick annotation-prompts` to list available prompts",
)
@click.option(
    "-S",
    "--annotation-system-prompt",
    default=None,
    type=str,
    help="Name of the annotation system prompt to use; use `bonepick annotation-prompts` to list available prompts",
)
@click.option(
    "-e",
    "--service-tier",
    type=click.Choice([tier.value for tier in ServiceTier]),
    default=ServiceTier.NONE.value,
    help="service tier to use for openai",
)
@click.option(
    "--reprocess-all-rows/--process-missing-rows",
    is_flag=True,
    default=False,
    help="Reprocess all rows or only missing rows",
)
@click.option(
    "--max-requests-per-minute",
    default=10_000,
    help="Maximum requests per minute",
)
@click.option(
    "--max-tokens-per-minute",
    default=100_000_000,
    help="Maximum tokens per minute",
)
@click.option(
    "--max-concurrent-requests",
    default=1_000,
    help="Maximum concurrent requests",
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
    default=2_000,
    type=int,
    help="Batch size to use for annotation",
)
@click.option(
    "--show-progress/--show-no-progress",
    is_flag=True,
    default=True,
    help="Show progress bar",
)
@click.option(
    "--disable-cache/--enable-cache",
    is_flag=True,
    default=False,
    help="Disable cache",
)
def annotate_dataset(
    dataset_dir: tuple[Path, ...],
    output_dir: Path,
    model_name: str,
    service_tier: str | None,
    max_requests_per_minute: int,
    max_tokens_per_minute: int,
    max_concurrent_requests: int,
    annotation_task_prompt: str,
    annotation_system_prompt: str | None,
    input_field_expression: str,
    reprocess_all_rows: bool,
    input_field_format: str,
    cache_location: Path | str | None,
    reasoning_effort: str | None,
    max_text_length: int | None,
    max_new_tokens: int,
    limit_rows: int | None,
    annotation_batch_size: int,
    show_progress: bool,
    disable_cache: bool,
):
    """Annotate dataset rows using LLM APIs.

    Supports rate limiting, caching, and batch processing for efficient annotation.
    """
    # check if the extra dependencies are installed
    extra_dependencies.check()

    # import these here to avoid annoying warning about plyvel not being installed
    from bonepick.annotate.deluge_utils import SqliteInvalidableCache, lm_deluge_monkey_patch

    # make sure the models available in the registry are updated
    lm_deluge_monkey_patch()

    click.echo("Locations:")
    click.echo(f"  Cache location: {cache_location}")
    click.echo("  Dataset directories:")
    for d in dataset_dir:
        click.echo(f"    - {str(d)}")
    click.echo(f"  Output directory: {str(output_dir)}")
    click.echo()

    click.echo("Annotation task")
    click.echo(f"  Prompt: {annotation_task_prompt}")
    click.echo(f"  System prompt: {annotation_system_prompt}")
    click.echo(f"  Input field expression: {input_field_expression}")
    click.echo(f"  Input field format: {input_field_format}")
    click.echo(f"  Max text length: {max_text_length:,}")
    click.echo()

    # these are the prompts to use for annotation
    task_prompt = BaseAnnotationPrompt.get(annotation_task_prompt)
    system_prompt = BaseSystemPrompt.get(annotation_system_prompt) if annotation_system_prompt else None

    # only supporting text format for now
    if input_field_format != "text":
        raise NotImplementedError("Only text format is supported for now")

    # setup cache location
    cache_location = Path(cache_location or user_cache_dir(__package__))
    cache_location.mkdir(parents=True, exist_ok=True)

    # # we need to disable the cache if we to reprocess rows that do not meet validation
    if disable_cache:
        cache = None
    else:
        cache = SqliteInvalidableCache(
            path=str(cache_location / f"{model_name}.db"), invalidate=reprocess_all_rows
        )

    click.echo("Initializing LLM client...")
    click.echo(f"  Model name:              {model_name}")
    click.echo(f"  Reasoning effort:        {reasoning_effort}")
    click.echo(f"  Service tier:            {service_tier}")
    click.echo(f"  Max requests per minute: {max_requests_per_minute:,}")
    click.echo(f"  Max tokens per minute:   {max_tokens_per_minute:,}")
    click.echo(f"  Max concurrent requests: {max_concurrent_requests:,}")
    click.echo(f"  Max new tokens:          {max_new_tokens:,}")
    click.echo()

    client = LLMClient(
        model_name,
        cache=cache,
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        max_concurrent_requests=max_concurrent_requests,
        reasoning_effort=ReasoningEffort(reasoning_effort).value if reasoning_effort else None,
        max_new_tokens=max_new_tokens,
    )

    click.echo("LLM client initialized")

    # Step 1: Collect all files and their sizes
    click.echo("\nCollecting files...")
    source_files: list[Path] = []
    destination_files: list[Path] = []

    for input_dir in dataset_dir:
        for root, _, files in os.walk(input_dir):
            for _fn in files:
                fn = Path(root) / _fn
                if "".join(fn.suffixes) not in FILE_SUFFIXES:
                    continue
                relative_path = fn.relative_to(input_dir)
                source_files.append(fn)
                destination_files.append(output_dir / relative_path)

    if not source_files:
        click.echo("No files found to annotate. Exiting.")
        return
    else:
        click.echo(f"Found {len(source_files):,} files")

    # these are used to read from and to jsonl files
    encoder, decoder = msgspec.json.Encoder(), msgspec.json.Decoder()
    input_field_selector = compile_jq(input_field_expression)

    with tqdm(total=len(source_files), desc="Processing files", unit="file") as pbar, ExitStack() as file_stack:
        failed_docs_cnt = successful_docs_cnt = 0
        to_annotate_docs_cnt = 0

        if not show_progress:
            # disable the progress bars if the user requested it
            pbar.disable = True

        for source_file, destination_file in zip(source_files, destination_files):
            if limit_rows is not None and to_annotate_docs_cnt >= limit_rows:
                click.echo(f"\nReached limit of {limit_rows:,} rows to annotate\n")
                break

            click.echo(f"\nProcessing {source_file.name}\n")
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            input_file = file_stack.enter_context(smart_open.open(source_file, "rb"))  # pyright: ignore
            output_file = file_stack.enter_context(smart_open.open(destination_file, "wb"))  # pyright: ignore

            batch_rows: list[dict] = []
            for line in input_file:
                row = decoder.decode(line)

                # already annotated; we skip and write to output file immediately
                if not reprocess_all_rows and task_prompt.name in row:
                    output_file.write(line)
                    successful_docs_cnt += 1
                    continue

                # to annotate; add to batch
                batch_rows.append(row)
                to_annotate_docs_cnt += 1

                if limit_rows is not None and to_annotate_docs_cnt >= limit_rows:
                    click.echo(f"\nReached limit of {limit_rows:,} rows to annotate\n")
                    break

            # keep track of the progress
            pbar.set_postfix(successful=successful_docs_cnt, failed=failed_docs_cnt)

            if not batch_rows:
                # nothing to annotate; move onto next file
                pbar.update(1)
                file_stack.pop_all()
                click.echo(f"\nSkipping {source_file.name} because it has no rows to annotate\n")
                continue

            click.echo(f"\nAnnotating {len(batch_rows):,} rows from {source_file.name}\n")
            batch_prompts: list[Conversation] = []
            for row in batch_rows:
                # use JQ expression to extract the input value from the row
                # (e.g., ".text" -> will extract the value of the "text" field)
                content = input_field_selector(row)
                assert isinstance(content, (str, list)), f"Expected str or list[TurnDict], got {type(content)}"

                # build conversation
                conversation = Conversation()
                if system_prompt:
                    conversation.add(Message.system(system_prompt.apply()))
                conversation.add(Message.user(task_prompt.apply(content, max_text_length)))
                batch_prompts.append(conversation)

            # annotate batches in chunk on `annotation_batch_size`` rows at the time
            for i in range(0, len(batch_prompts), annotation_batch_size):
                batch_prompts_chunk = batch_prompts[i : i + annotation_batch_size]
                batch_rows_chunk = batch_rows[i : i + annotation_batch_size]

                # we have to use the async cuz the sync APIs don't support service tier
                batch_responses = asyncio.run(
                    client.process_prompts_async(
                        batch_prompts_chunk,
                        service_tier=ServiceTier(service_tier).value if service_tier else None,
                        output_schema=task_prompt.schema,
                        show_progress=show_progress,
                    )
                )

                # write responses to output file
                for response, row in zip(batch_responses, batch_rows_chunk):
                    if response is None or response.completion is None:
                        failed_docs_cnt += 1
                        continue
                    try:
                        parsed_response = task_prompt.parse(response.completion)
                    except Exception:
                        failed_docs_cnt += 1
                        continue

                    output_file.write(encoder.encode({**row, task_prompt.name: parsed_response}) + b"\n")
                    successful_docs_cnt += 1

            click.echo(f"  Wrote {len(batch_rows):,} rows to {destination_file.name}")
            pbar.update(1)

        pbar.close()
        file_stack.close()

    click.echo("\nSummary:")
    click.echo(f"  Processed {len(source_files):,} files")
    click.echo(f"  Annotated {to_annotate_docs_cnt:,} rows")
    click.echo(f"  Failed {failed_docs_cnt:,} rows")


@click.command()
@click.argument("prompt-type", type=click.Choice(["task", "system"]))
def list_prompts(prompt_type: Literal["task", "system"]):
    """List available annotation prompts"""

    click.echo(f"Available {prompt_type} prompts:")
    if prompt_type == "task":
        for prompt_name in BaseAnnotationPrompt.prompts():
            click.echo(f"- {prompt_name}")

    if prompt_type == "system":
        for prompt_name in BaseSystemPrompt.prompts():
            click.echo(f"- {prompt_name}")
