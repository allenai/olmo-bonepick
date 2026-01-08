import asyncio
import os
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from functools import partial
from typing import Literal, TypedDict, cast as typing_cast

import click
from lazy_imports import try_import

from bonepick.train.data_utils import (
    load_jsonl_dataset,
    DatasetSplit,
    write_dataset,
    DatasetTuple,
)
from bonepick.cli import PathParamType
from bonepick.annotate.prompts import BaseAnnotationPrompt, BaseSystemPrompt
from bonepick.train.data_utils import ChunkedDatasetPath, ChunkedDataset


with try_import() as extra_dependencies:
    # extra imports; they won't fail here, but later when running the command they will
    from lm_deluge import LLMClient, Conversation, Message
    from lm_deluge.client import _LLMClient as LLMClientType

    # from lm_deluge.cache import SqliteCache
    # from lm_deluge.models import registry as lm_deluge_registry
    # from lm_deluge.api_requests.base import APIResponse
    from platformdirs import user_cache_dir
    from bonepick.annotate.deluge_utils import SqliteInvalidableCache

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


def annotate_batch(
    batch_input: ChunkedDatasetPath[DatasetRow],
    batch_output: ChunkedDataset[DatasetRow],
    client: LLMClientType,
    annotation_task_prompt: str,
    annotation_system_prompt: str | None,
    service_tier: ServiceTier,
):
    task_prompt = BaseAnnotationPrompt.get(annotation_task_prompt)
    system_prompt = (
        BaseSystemPrompt.get(annotation_system_prompt)
        if annotation_system_prompt
        else None
    )
    batch_prompts: list[Conversation] = []

    for row in batch_input:
        # build conversation
        conversation = Conversation()
        if system_prompt:
            conversation.add(Message.system(system_prompt.apply()))
        conversation.add(Message.user(task_prompt.apply(row["text"])))
        batch_prompts.append(conversation)

    responses = asyncio.run(
        client.process_prompts_async(
            batch_prompts,
            service_tier=service_tier.value,
            output_schema=task_prompt.schema,
        )
    )  # pyright: ignore
    failed_cnt = 0

    output: list[DatasetRow] = []
    for response, row in zip(responses, batch_input):
        if response is None or response.completion is None:
            failed_cnt += 1
            continue

        # TODO: properly support non-string outputs
        result = typing_cast(str, task_prompt.parse(response.completion))
        output.append(DatasetRow(text=row["text"], label=result))

    click.echo(
        f"batch {batch_input.chunk_path.name}: failed to annotate {failed_cnt:,} rows"
    )

    batch_output.add_chunk(output)

    return len(output)


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
    "-o", "--output-dir", type=PathParamType(mkdir=True, is_dir=True), default=None
)
@click.option(
    "-m",
    "--model-name",
    default="gpt-5.2",
    help="Name of the model to use for annotation",
)
@click.option(
    "-i",
    "--input-field",
    type=str,
    default="text",
    help="Field in dataset to use as input for annotation",
)
@click.option(
    "-f",
    "--input-field-format",
    type=click.Choice(["text", "conversation"]),
    default="text",
    help="Format of the input: `text` is a string, `conversation` is a list of messages in OpenAI chat format.",
)
@click.option(
    "-c",
    "--cache-location",
    default=None,
    type=PathParamType(is_dir=True, optional=True),
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
    default=None,
    help="service tier to use for openai",
    type=str,
)
@click.option(
    "--num-proc",
    default=os.cpu_count(),
    help="number of processes to use for processing",
)
@click.option(
    "--reprocess-missing/--process-all",
    default=False,
    help="Whether to reprocess missing/invalid rows",
)
@click.option(
    "--max-requests-per-minute",
    default=10_000,
    help="Maximum requests per minute",
)
@click.option(
    "--max-tokens-per-minute",
    default=10_000_000,
    help="Maximum tokens per minute",
)
@click.option(
    "--max-concurrent-requests",
    default=5_000,
    help="Maximum concurrent requests",
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
    num_proc: int,
    input_field: str,
    input_field_format: str,
    cache_location: Path | str | None,
    reprocess_missing: bool,
):
    # check if the extra dependencies are installed
    extra_dependencies.check()

    # only supporting text format for now
    if input_field_format != "text":
        raise NotImplementedError("Only text format is supported for now")

    # setup cache location
    cache_location = Path(cache_location or user_cache_dir(__package__))
    cache_location.mkdir(parents=True, exist_ok=True)

    # we need to disable the cache if we to reprocess rows that do not meet validation
    cache = SqliteInvalidableCache(
        path=str(cache_location / f"{model_name}.db"), invalidate=reprocess_missing
    )

    client = LLMClient(
        model_name,
        cache=cache,
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        max_concurrent_requests=max_concurrent_requests,
    )

    dataset = load_jsonl_dataset(
        list(dataset_dir),
        text_field_name=input_field,
        label_field_name=annotation_task_prompt,
        valid_split_required=False,
        test_split_required=False,
        allow_missing_label=True,
    )

    click.echo(f"Loaded {len(dataset.train):,} rows from {dataset_dir}")

    existing_text, existing_label, to_annotate_text = [], [], []
    for text, label in dataset.train:
        if label is not None:
            existing_text.append(text)
            existing_label.append(label)
        else:
            to_annotate_text.append(text)
    del dataset

    click.echo(
        f"Found {len(existing_text):,} existing rows and {len(to_annotate_text):,} rows to annotate"
    )

    with ExitStack() as stack:
        batch_input = stack.enter_context(ChunkedDataset())
        batch_input.add_dataset(
            [DatasetRow(text=text, label=None) for text in to_annotate_text]
        )
        batch_output = stack.enter_context(ChunkedDataset())

        pool_cls = ProcessPoolExecutor if num_proc > 1 else ThreadPoolExecutor
        pool = stack.enter_context(pool_cls(max_workers=num_proc))

        process_fn = partial(
            annotate_batch,
            client=client,
            annotation_task_prompt=annotation_task_prompt,
            annotation_system_prompt=annotation_system_prompt,
            service_tier=ServiceTier(service_tier)
            if service_tier
            else ServiceTier.NONE,
            batch_output=batch_output,
        )

        # we don't pass the number of processes to make sure we run this in single thread
        annotate_pbar = stack.enter_context(
            tqdm(
                total=len(to_annotate_text), desc="Annotating dataset", unit_scale=True
            )
        )
        futures = []
        for batch_input_path in batch_input:
            future = pool.submit(process_fn, batch_input=batch_input_path)
            futures.append(future)

        for future in as_completed(futures):
            try:
                successful_cnt = future.result()
                annotate_pbar.update(successful_cnt)
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e
        annotate_pbar.close()

        retrieve_pbar = stack.enter_context(
            tqdm(
                total=len(existing_text),
                desc="Retrieving annotated dataset",
                unit_scale=True,
            )
        )
        for batch_output_path in batch_output:
            for row in batch_output_path:
                existing_text.append(row["text"])
                existing_label.append(row["label"])
                retrieve_pbar.update(1)
        retrieve_pbar.close()

    dataset_split = DatasetSplit(text=existing_text, label=existing_label)
    dataset = DatasetTuple(
        train=dataset_split, valid=DatasetSplit.new(), test=DatasetSplit.new()
    )
    write_dataset(
        dataset,
        output_dir,
        text_field_name=input_field,
        label_field_name=annotation_task_prompt,
    )

    # processed_dataset.push_to_hub(destination_path, private=keep_private)
    # print("Dataset pushed to https://huggingface.co/datasets/" + destination_path)


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
