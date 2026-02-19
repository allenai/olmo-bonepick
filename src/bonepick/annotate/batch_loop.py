import asyncio
import json
import os
from pathlib import Path

import click
import msgspec
import smart_open
from lazy_imports import try_import
from tqdm import tqdm

from bonepick.annotate.prompts import BaseAnnotationPrompt, BaseSystemPrompt
from bonepick.cli import PathParamType
from bonepick.data.expressions import compile_jq
from bonepick.data.utils import is_valid_suffix

with try_import() as extra_dependencies:
    from lm_deluge import Conversation, LLMClient, Message
    from lm_deluge.models import registry

    # import here to register all the prompts
    from bonepick.annotate import prompt_collections  # noqa: F401


from bonepick.annotate.annotate_loop import ReasoningEffort


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
):
    """Submit batch annotation job to LLM batch API.

    Creates a batch directory with a manifest and compressed rows file,
    then submits prompts via the provider's batch API (OpenAI or Anthropic).
    """
    extra_dependencies.check()

    from bonepick.annotate.deluge_utils import _batch_output_schema, lm_deluge_monkey_patch

    lm_deluge_monkey_patch()

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
    click.echo()

    task_prompt = BaseAnnotationPrompt.get(annotation_task_prompt)
    system_prompt = BaseSystemPrompt.get(annotation_system_prompt) if annotation_system_prompt else None

    if input_field_format != "text":
        raise NotImplementedError("Only text format is supported for now")

    # Step 1: Collect all files
    click.echo("Collecting files...")
    source_files: list[Path] = []
    relative_paths: list[str] = []

    for input_dir in dataset_dir:
        for root, _, files in os.walk(input_dir):
            for _fn in files:
                fn = Path(root) / _fn
                if not is_valid_suffix(fn):
                    continue
                relative_path = str(fn.relative_to(input_dir))
                source_files.append(fn)
                relative_paths.append(relative_path)

    if not source_files:
        click.echo("No files found to annotate. Exiting.")
        return

    click.echo(f"Found {len(source_files):,} files")

    encoder, decoder = msgspec.json.Encoder(), msgspec.json.Decoder()
    input_field_selector = compile_jq(input_field_expression)

    # Step 2: Stream-write rows.jsonl.zst and build prompts
    rows_path = batch_dir / "rows.jsonl.zst"
    prompts: list[Conversation] = []
    custom_id_counter = 0
    total_rows = 0
    skipped_rows = 0

    click.echo("Building prompts and writing rows...")
    limit_reached = False
    with smart_open.open(rows_path, "wb") as rows_file:  # pyright: ignore
        for source_file, rel_path in tqdm(
            zip(source_files, relative_paths), total=len(source_files), desc="Processing files", unit="file"
        ):
            if limit_reached:
                break

            with smart_open.open(source_file, "rb") as input_file:  # pyright: ignore
                for line in input_file:
                    row = decoder.decode(line)
                    total_rows += 1

                    # Already annotated — pass through
                    if not reprocess_all_rows and task_prompt.name in row:
                        row_entry = {"custom_id": None, "dest_file": rel_path, "row": row}
                        rows_file.write(encoder.encode(row_entry) + b"\n")
                        skipped_rows += 1
                        continue

                    if limit_rows is not None and custom_id_counter >= limit_rows:
                        click.echo(f"\nReached limit of {limit_rows:,} rows to annotate")
                        limit_reached = True
                        break

                    # Needs annotation
                    row_entry = {"custom_id": custom_id_counter, "dest_file": rel_path, "row": row}
                    rows_file.write(encoder.encode(row_entry) + b"\n")

                    # Build conversation prompt
                    content = input_field_selector(row)
                    assert isinstance(content, (str, list)), f"Expected str or list, got {type(content)}"

                    conversation = Conversation()
                    if system_prompt:
                        conversation.add(Message.system(system_prompt.apply()))
                    conversation.add(Message.user(task_prompt.apply(content, max_text_length)))
                    prompts.append(conversation)

                    custom_id_counter += 1

    click.echo(f"\nTotal rows: {total_rows:,}")
    click.echo(f"Rows to annotate: {custom_id_counter:,}")
    click.echo(f"Rows skipped (already annotated): {skipped_rows:,}")

    if custom_id_counter == 0:
        click.echo("No rows need annotation. Exiting.")
        return

    # Step 3: Submit batch job
    click.echo(f"\nInitializing LLM client...")
    click.echo(f"  Model name:       {model_name}")
    click.echo(f"  Reasoning effort: {reasoning_effort}")
    click.echo(f"  Max new tokens:   {max_new_tokens:,}")
    click.echo()

    client = LLMClient(
        model_name,
        reasoning_effort=ReasoningEffort(reasoning_effort).value if reasoning_effort else None,
        max_new_tokens=max_new_tokens,
    )

    click.echo(f"Submitting {custom_id_counter:,} prompts to batch API...")

    # Set output_schema via contextvar for the batch submission
    token = _batch_output_schema.set(task_prompt.schema)
    try:
        batch_ids = asyncio.run(client.submit_batch_job(prompts, batch_size=annotation_batch_size))
    finally:
        _batch_output_schema.reset(token)

    # Convert to list of strings for JSON serialization
    batch_ids = [str(bid) for bid in batch_ids]

    # Step 4: Write manifest
    provider = registry[model_name].api_spec
    manifest = {
        "batch_ids": batch_ids,
        "provider": provider,
        "model": model_name,
        "task_prompt_name": annotation_task_prompt,
        "system_prompt_name": annotation_system_prompt,
        "num_prompts": custom_id_counter,
        "num_total_rows": total_rows,
    }

    manifest_path = batch_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    click.echo(f"\nBatch submitted successfully!")
    click.echo(f"  Batch IDs: {batch_ids}")
    click.echo(f"  Manifest: {manifest_path}")
    click.echo(f"  Rows file: {rows_path}")


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
def batch_annotate_retrieve(
    batch_dir: Path,
    output_dir: Path,
):
    """Retrieve batch annotation results and merge with original data.

    Reads the manifest and rows from a batch directory, waits for batch
    completion, then writes annotated output preserving directory structure.
    """
    extra_dependencies.check()

    from bonepick.annotate.deluge_utils import lm_deluge_monkey_patch

    lm_deluge_monkey_patch()

    # Step 1: Read manifest
    manifest_path = batch_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    batch_ids = manifest["batch_ids"]
    model = manifest["model"]
    task_prompt_name = manifest["task_prompt_name"]
    num_prompts = manifest["num_prompts"]

    provider = registry[model].api_spec

    click.echo("Batch retrieval")
    click.echo(f"  Batch dir:    {batch_dir}")
    click.echo(f"  Output dir:   {output_dir}")
    click.echo(f"  Model:        {model}")
    click.echo(f"  Provider:     {provider}")
    click.echo(f"  Task prompt:  {task_prompt_name}")
    click.echo(f"  Batch IDs:    {batch_ids}")
    click.echo(f"  Num prompts:  {num_prompts:,}")
    click.echo()

    task_prompt = BaseAnnotationPrompt.get(task_prompt_name)

    # Step 2: Wait for batch completion
    click.echo("Waiting for batch completion...")
    from lm_deluge.batches import wait_for_batch_completion_async

    results = asyncio.run(wait_for_batch_completion_async(batch_ids, provider))

    click.echo(f"Retrieved {len(results):,} results")

    # Step 3: Build lookup dict {custom_id: completion_text}
    completions: dict[int, str | None] = {}
    for result in results:
        cid = int(result["custom_id"])
        completions[cid] = _extract_batch_completion(result, provider)

    succeeded = sum(1 for v in completions.values() if v is not None)
    failed = sum(1 for v in completions.values() if v is None)
    click.echo(f"  Succeeded: {succeeded:,}")
    click.echo(f"  Failed:    {failed:,}")

    # Step 4: Stream through rows and write output
    rows_path = batch_dir / "rows.jsonl.zst"
    encoder, decoder = msgspec.json.Encoder(), msgspec.json.Decoder()

    # Open output files as needed
    output_handles: dict[str, object] = {}
    successful_docs = 0
    failed_docs = 0
    passthrough_docs = 0

    click.echo("\nWriting output files...")
    try:
        with smart_open.open(rows_path, "rb") as rows_file:  # pyright: ignore
            for line in tqdm(rows_file, desc="Processing rows", unit="row"):
                row_entry = decoder.decode(line)
                custom_id = row_entry["custom_id"]
                dest_file = row_entry["dest_file"]
                row = row_entry["row"]

                dest_path = output_dir / dest_file

                # Open output file if not already open
                if dest_file not in output_handles:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    output_handles[dest_file] = smart_open.open(dest_path, "wb")  # pyright: ignore

                out_f = output_handles[dest_file]

                if custom_id is None:
                    # Pass-through row (already annotated)
                    out_f.write(encoder.encode(row) + b"\n")  # pyright: ignore
                    passthrough_docs += 1
                    continue

                # Look up completion
                completion = completions.get(custom_id)
                if completion is None:
                    failed_docs += 1
                    continue

                try:
                    parsed_response = task_prompt.parse(completion)
                except Exception:
                    failed_docs += 1
                    continue

                out_f.write(encoder.encode({**row, task_prompt.name: parsed_response}) + b"\n")  # pyright: ignore
                successful_docs += 1
    finally:
        for handle in output_handles.values():
            handle.close()  # pyright: ignore

    click.echo(f"\nSummary:")
    click.echo(f"  Pass-through rows: {passthrough_docs:,}")
    click.echo(f"  Annotated rows:    {successful_docs:,}")
    click.echo(f"  Failed rows:       {failed_docs:,}")
    click.echo(f"  Output directory:  {output_dir}")
