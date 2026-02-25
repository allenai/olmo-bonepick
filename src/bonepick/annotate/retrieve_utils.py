"""Utilities for waiting on and downloading batch retrieval outputs.

This module keeps retrieval I/O concerns isolated from CLI command wiring:
1. figure out which batch ids still need downloading;
2. wait for completion for each pending id (provider-specific polling);
3. stream each result payload into ``results/<batch_id>.jsonl.zst``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable

import click
import smart_open

RESULT_SUFFIX = ".jsonl.zst"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def collect_unique_batch_ids(batch_infos: list[dict]) -> list[str]:
    """Collect unique batch ids while preserving first-seen order."""
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for info in batch_infos:
        for batch_id in info["batch_ids"]:
            batch_id_str = str(batch_id)
            if batch_id_str in seen:
                continue
            seen.add(batch_id_str)
            ordered_ids.append(batch_id_str)
    return ordered_ids


def result_path_for_batch(results_dir: Path, batch_id: str) -> Path:
    """Build local cache path for one batch id."""
    return results_dir / f"{batch_id}{RESULT_SUFFIX}"


def result_file_uses_zstd(result_path: Path) -> bool:
    """Return whether a cached result file is zstd-compressed."""
    with open(result_path, "rb") as probe:
        return probe.read(4) == _ZSTD_MAGIC


async def _download_response_to_file(
    session,
    url: str,
    headers: dict[str, str],
    destination_path: Path,
) -> None:
    """Stream an HTTP response body into a local compressed result file."""
    # Keep the compression suffix on temp files so smart_open compresses output.
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


async def wait_and_download_openai_batch_async(
    session,
    batch_id: str,
    destination_path: Path,
    poll_interval: int,
) -> str:
    """Poll OpenAI batch until completion, then download output to disk."""
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


async def wait_and_download_anthropic_batch_async(
    session,
    batch_id: str,
    destination_path: Path,
    poll_interval: int,
) -> str:
    """Poll Anthropic batch until completion, then download output to disk."""
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


async def wait_for_batch_downloads_async(
    batch_ids: list[str],
    provider: str,
    results_dir: Path,
    timeout: int | None,
    poll_interval: int,
    reporter: Callable[[str], None],
) -> list[Path]:
    """Wait for batch completion concurrently and persist outputs on disk."""
    import aiohttp

    if provider not in ("openai", "anthropic"):
        raise ValueError("Only openai or anthropic support batch mode")

    results_dir.mkdir(parents=True, exist_ok=True)

    normalized_batch_ids = [str(batch_id) for batch_id in batch_ids]
    existing_batch_ids = [
        batch_id
        for batch_id in normalized_batch_ids
        if result_path_for_batch(results_dir=results_dir, batch_id=batch_id).exists()
    ]
    existing_batch_id_set = set(existing_batch_ids)

    if existing_batch_ids:
        reporter(f"Found {len(existing_batch_ids):,} existing result files; skipping those downloads.")

    pending_batch_ids = [batch_id for batch_id in normalized_batch_ids if batch_id not in existing_batch_id_set]
    if not pending_batch_ids:
        reporter("No remaining batch downloads needed.")
        return [
            result_path_for_batch(results_dir=results_dir, batch_id=batch_id) for batch_id in normalized_batch_ids
        ]

    reporter(f"Waiting for and downloading {len(pending_batch_ids):,} remaining batches...")

    async with aiohttp.ClientSession() as session:
        tasks: list[asyncio.Task[str]] = []
        for batch_id in pending_batch_ids:
            destination_path = result_path_for_batch(results_dir=results_dir, batch_id=batch_id)
            if provider == "openai":
                task = asyncio.create_task(
                    wait_and_download_openai_batch_async(
                        session=session,
                        batch_id=batch_id,
                        destination_path=destination_path,
                        poll_interval=poll_interval,
                    )
                )
            else:
                task = asyncio.create_task(
                    wait_and_download_anthropic_batch_async(
                        session=session,
                        batch_id=batch_id,
                        destination_path=destination_path,
                        poll_interval=poll_interval,
                    )
                )
            tasks.append(task)

        async def _wait_and_report() -> None:
            completed = 0
            for finished_task in asyncio.as_completed(tasks):
                downloaded_batch_id = await finished_task
                completed += 1
                reporter(f"  Downloaded {completed:,}/{len(tasks):,}: {downloaded_batch_id}{RESULT_SUFFIX}")

        if timeout is not None:
            await asyncio.wait_for(_wait_and_report(), timeout=timeout)
        else:
            await _wait_and_report()

    return [result_path_for_batch(results_dir=results_dir, batch_id=batch_id) for batch_id in normalized_batch_ids]


def wait_for_batch_downloads(
    batch_ids: list[str],
    provider: str,
    results_dir: Path,
    timeout: int | None,
    poll_interval: int = 30,
    reporter: Callable[[str], None] = click.echo,
) -> list[Path]:
    """Synchronous wrapper for concurrent retrieval downloads."""
    try:
        return asyncio.run(
            wait_for_batch_downloads_async(
                batch_ids=batch_ids,
                provider=provider,
                results_dir=results_dir,
                timeout=timeout,
                poll_interval=poll_interval,
                reporter=reporter,
            )
        )
    except asyncio.TimeoutError as exc:
        raise click.ClickException(f"Batch wait timed out after {timeout:,}s") from exc
