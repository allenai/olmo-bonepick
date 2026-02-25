"""Utilities for waiting on and downloading batch retrieval outputs.

This module keeps retrieval I/O concerns isolated from CLI command wiring:
1. figure out which batch ids still need downloading;
2. wait for completion for each pending id (provider-specific polling);
3. stream each result payload into ``results/<batch_id>.jsonl.zst``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Callable, Literal

import click
import smart_open

RESULT_SUFFIX = ".jsonl.zst"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
DEFAULT_MAX_CONCURRENT_DOWNLOADS = 8

BatchDownloadStatus = Literal["ready", "downloading", "in_progress", "other", "completed"]
BatchStatusReporter = Callable[[str, BatchDownloadStatus], None]


def _format_batch_status_line(
    *,
    completed: int,
    ready: int,
    downloading: int,
    in_progress: int,
    other: int,
    total: int,
) -> str:
    """Build a colorized one-line status summary for batch retrieval progress."""
    return (
        f"  {click.style('Status:', fg='cyan', bold=True)} "
        f"{click.style('completed', fg='green')}={click.style(f'{completed:,}', fg='green', bold=True)}/"
        f"{click.style(f'{total:,}', fg='green')} "
        f"{click.style('ready', fg='yellow')}={click.style(f'{ready:,}', fg='yellow', bold=True)} "
        f"{click.style('downloading', fg='blue')}={click.style(f'{downloading:,}', fg='blue', bold=True)} "
        f"{click.style('in-progress', fg='magenta')}={click.style(f'{in_progress:,}', fg='magenta', bold=True)} "
        f"{click.style('other', fg='red')}={click.style(f'{other:,}', fg='red', bold=True)}"
    )


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
        return probe.read(4) == ZSTD_MAGIC


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
    on_status_change: BatchStatusReporter | None = None,
    download_semaphore: asyncio.Semaphore | None = None,
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
                if on_status_change is not None:
                    on_status_change(batch_id, "ready")
                if download_semaphore is None:
                    if on_status_change is not None:
                        on_status_change(batch_id, "downloading")
                    await _download_response_to_file(
                        session=session,
                        url=output_url,
                        headers=headers,
                        destination_path=destination_path,
                    )
                else:
                    async with download_semaphore:
                        if on_status_change is not None:
                            on_status_change(batch_id, "downloading")
                        await _download_response_to_file(
                            session=session,
                            url=output_url,
                            headers=headers,
                            destination_path=destination_path,
                        )
                return batch_id

            if status in ("failed", "expired", "cancelled"):
                raise ValueError(f"OpenAI batch {batch_id} failed with status: {status}")

            if on_status_change is not None:
                if status in ("validating", "in_progress", "finalizing", "cancelling"):
                    on_status_change(batch_id, "in_progress")
                else:
                    on_status_change(batch_id, "other")

        await asyncio.sleep(poll_interval)


async def wait_and_download_anthropic_batch_async(
    session,
    batch_id: str,
    destination_path: Path,
    poll_interval: int,
    on_status_change: BatchStatusReporter | None = None,
    download_semaphore: asyncio.Semaphore | None = None,
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
                if on_status_change is not None:
                    on_status_change(batch_id, "ready")
                if download_semaphore is None:
                    if on_status_change is not None:
                        on_status_change(batch_id, "downloading")
                    await _download_response_to_file(
                        session=session,
                        url=output_url,
                        headers=headers,
                        destination_path=destination_path,
                    )
                else:
                    async with download_semaphore:
                        if on_status_change is not None:
                            on_status_change(batch_id, "downloading")
                        await _download_response_to_file(
                            session=session,
                            url=output_url,
                            headers=headers,
                            destination_path=destination_path,
                        )
                return batch_id

            if status in ("canceled", "expired"):
                raise ValueError(f"Anthropic batch {batch_id} failed with status: {status}")

            if on_status_change is not None:
                if status in ("in_progress", "canceling"):
                    on_status_change(batch_id, "in_progress")
                else:
                    on_status_change(batch_id, "other")

        await asyncio.sleep(poll_interval)


async def wait_for_batch_downloads_async(
    batch_ids: list[str],
    provider: str,
    results_dir: Path,
    timeout: int | None,
    poll_interval: int,
    reporter: Callable[[str], None],
    skip_failed_batches: bool,
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
        tasks: list[asyncio.Task[tuple[str, Exception | None]]] = []
        ready_batch_ids: set[str] = set()
        downloading_batch_ids: set[str] = set()
        in_progress_batch_ids: set[str] = set()
        other_batch_ids: set[str] = set(pending_batch_ids)
        completed_batch_ids: set[str] = set()
        last_status_snapshot: tuple[int, int, int, int, int] | None = None
        total_pending = len(pending_batch_ids)
        download_semaphore = asyncio.Semaphore(min(DEFAULT_MAX_CONCURRENT_DOWNLOADS, total_pending))
        use_rich_live = False
        rich_live = None
        rich_progress = None
        rich_task_id = None

        if reporter in (click.echo, click.secho) and sys.stdout.isatty():
            try:
                from rich.console import Console
                from rich.live import Live
                from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

                rich_console = Console(highlight=False)
                rich_progress = Progress(
                    SpinnerColumn(style="cyan"),
                    TextColumn("[bold cyan]Batch retrieval[/bold cyan]"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("• [green]completed[/green] {task.completed:,.0f}"),
                    TextColumn("• [yellow]ready[/yellow] {task.fields[ready]}"),
                    TextColumn("• [blue]downloading[/blue] {task.fields[downloading]}"),
                    TextColumn("• [magenta]in-progress[/magenta] {task.fields[in_progress]}"),
                    TextColumn("• [red]other[/red] {task.fields[other]}"),
                )
                rich_task_id = rich_progress.add_task(
                    "download",
                    total=total_pending,
                    completed=0,
                    ready="0",
                    downloading="0",
                    in_progress="0",
                    other=f"{total_pending:,}",
                )
                rich_live = Live(rich_progress, console=rich_console, refresh_per_second=10, transient=False)
                rich_live.start()
                use_rich_live = True
            except Exception:
                use_rich_live = False
                rich_live = None
                rich_progress = None
                rich_task_id = None

        def _report_status(force: bool = False) -> None:
            nonlocal last_status_snapshot
            snapshot = (
                len(completed_batch_ids),
                len(ready_batch_ids),
                len(downloading_batch_ids),
                len(in_progress_batch_ids),
                len(other_batch_ids),
            )
            if not force and snapshot == last_status_snapshot:
                return
            last_status_snapshot = snapshot

            if use_rich_live and rich_progress is not None and rich_task_id is not None:
                rich_progress.update(
                    rich_task_id,
                    completed=snapshot[0],
                    total=total_pending,
                    ready=f"{snapshot[1]:,}",
                    downloading=f"{snapshot[2]:,}",
                    in_progress=f"{snapshot[3]:,}",
                    other=f"{snapshot[4]:,}",
                )
                return

            reporter(
                _format_batch_status_line(
                    completed=snapshot[0],
                    ready=snapshot[1],
                    downloading=snapshot[2],
                    in_progress=snapshot[3],
                    other=snapshot[4],
                    total=total_pending,
                )
            )

        def _set_batch_status(batch_id: str, status: BatchDownloadStatus) -> None:
            if batch_id in completed_batch_ids and status != "completed":
                return

            ready_batch_ids.discard(batch_id)
            downloading_batch_ids.discard(batch_id)
            in_progress_batch_ids.discard(batch_id)
            other_batch_ids.discard(batch_id)

            if status == "completed":
                completed_batch_ids.add(batch_id)
            elif status == "ready":
                ready_batch_ids.add(batch_id)
            elif status == "downloading":
                downloading_batch_ids.add(batch_id)
            elif status == "in_progress":
                in_progress_batch_ids.add(batch_id)
            else:
                other_batch_ids.add(batch_id)

        def _on_status_change(batch_id: str, status: BatchDownloadStatus) -> None:
            _set_batch_status(batch_id, status)
            _report_status()

        async def _download_with_error_capture(batch_id: str) -> tuple[str, Exception | None]:
            destination_path = result_path_for_batch(results_dir=results_dir, batch_id=batch_id)
            try:
                if provider == "openai":
                    await wait_and_download_openai_batch_async(
                        session=session,
                        batch_id=batch_id,
                        destination_path=destination_path,
                        poll_interval=poll_interval,
                        on_status_change=_on_status_change,
                        download_semaphore=download_semaphore,
                    )
                else:
                    await wait_and_download_anthropic_batch_async(
                        session=session,
                        batch_id=batch_id,
                        destination_path=destination_path,
                        poll_interval=poll_interval,
                        on_status_change=_on_status_change,
                        download_semaphore=download_semaphore,
                    )
                return batch_id, None
            except Exception as exc:
                return batch_id, exc

        for batch_id in pending_batch_ids:
            tasks.append(asyncio.create_task(_download_with_error_capture(batch_id)))

        async def _wait_and_report() -> None:
            _report_status(force=True)
            for finished_task in asyncio.as_completed(tasks):
                finished_batch_id, error = await finished_task
                if error is not None:
                    if not skip_failed_batches:
                        for pending_task in tasks:
                            if not pending_task.done():
                                pending_task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise error
                    _set_batch_status(finished_batch_id, "completed")
                    error_text = str(error).strip() or error.__class__.__name__
                    skipped_line = (
                        f"  {click.style('Skipped', fg='yellow', bold=True)} "
                        f"{click.style(f'{len(completed_batch_ids):,}/{total_pending:,}', fg='yellow')}: "
                        f"{finished_batch_id} ({error_text})"
                    )
                    if use_rich_live and rich_live is not None:
                        rich_live.console.print(skipped_line)
                    else:
                        reporter(skipped_line)
                else:
                    _set_batch_status(finished_batch_id, "completed")
                    downloaded_line = (
                        f"  {click.style('Downloaded', fg='green', bold=True)} "
                        f"{click.style(f'{len(completed_batch_ids):,}/{total_pending:,}', fg='green')}: "
                        f"{finished_batch_id}{RESULT_SUFFIX}"
                    )
                    if use_rich_live and rich_live is not None:
                        rich_live.console.print(downloaded_line)
                    else:
                        reporter(downloaded_line)
                _report_status()

        try:
            if timeout is not None:
                await asyncio.wait_for(_wait_and_report(), timeout=timeout)
            else:
                await _wait_and_report()
        finally:
            if rich_live is not None:
                rich_live.stop()

    result_paths = [
        result_path_for_batch(results_dir=results_dir, batch_id=batch_id) for batch_id in normalized_batch_ids
    ]
    return [result_path for result_path in result_paths if result_path.exists()]


def wait_for_batch_downloads(
    batch_ids: list[str],
    provider: str,
    results_dir: Path,
    timeout: int | None,
    poll_interval: int = 10,
    reporter: Callable[[str], None] = click.echo,
    skip_failed_batches: bool = False,
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
                skip_failed_batches=skip_failed_batches,
            )
        )
    except asyncio.TimeoutError as exc:
        raise click.ClickException(f"Batch wait timed out after {timeout:,}s") from exc
