#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "boto3",
#     "click",
# ]
# ///
"""Build an S3 Batch Operations manifest from S3 patterns stored in a CSV column.

The input column should contain S3 URIs using slash-separated glob segments, for
example:

    s3://my-bucket/path/to/*.jsonl.gz
    s3://my-bucket/path/to/quality_*/shard_*.jsonl.gz
    s3://my-bucket/path/to/prefix/

Cells may contain one pattern or multiple newline-separated patterns. Patterns
ending with `/` are treated as recursive prefixes and include every object
below the matched prefix. All output rows are written in the CSV format
expected by S3 Batch Operations manifests: `bucket,key`, with keys URL-encoded.

Examples:
    uv run scripts/build_s3_batch_manifest.py input.csv s3_pattern > manifest.csv
    uv run scripts/build_s3_batch_manifest.py input.csv s3_pattern -o manifest.csv
    uv run scripts/build_s3_batch_manifest.py input.csv s3_pattern --workers 32
"""

from __future__ import annotations

import csv
import fnmatch
import os
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import quote, urlparse

import boto3
import click
from botocore.config import Config
from botocore.exceptions import ClientError

DEFAULT_WORKERS = min(64, max(4, (os.cpu_count() or 4) * 4))
_THREAD_LOCAL = threading.local()
_S3_CONFIG = Config(max_pool_connections=DEFAULT_WORKERS)


class ManifestError(RuntimeError):
    """Raised when a pattern cannot be expanded."""


class ManifestStore:
    """Small on-disk store for deduplicated manifest rows."""

    def __init__(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="s3-batch-manifest-")
        self._db_path = Path(self._temp_dir.name) / "manifest.sqlite3"
        self._connection = sqlite3.connect(self._db_path)
        self._connection.execute(
            """
            CREATE TABLE manifest (
                bucket TEXT NOT NULL,
                key TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                PRIMARY KEY (bucket, key)
            ) WITHOUT ROWID
            """
        )

    def add_many(self, rows: Iterable[tuple[str, str, int]]) -> int:
        before = self._connection.total_changes
        self._connection.executemany(
            "INSERT OR IGNORE INTO manifest(bucket, key, size_bytes) VALUES (?, ?, ?)",
            rows,
        )
        return self._connection.total_changes - before

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM manifest").fetchone()
        return int(row[0]) if row is not None else 0

    def total_size_bytes(self) -> int:
        row = self._connection.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM manifest").fetchone()
        return int(row[0]) if row is not None else 0

    def iter_rows(self) -> Iterator[tuple[str, str]]:
        for bucket, key in self._connection.execute("SELECT bucket, key FROM manifest ORDER BY bucket, key"):
            yield str(bucket), str(key)

    def close(self) -> None:
        self._connection.close()
        self._temp_dir.cleanup()

    def __enter__(self) -> "ManifestStore":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()


def set_s3_client_pool_size(workers: int) -> None:
    global _S3_CONFIG
    _S3_CONFIG = Config(max_pool_connections=max(4, workers))


def get_s3_client():
    client = getattr(_THREAD_LOCAL, "s3_client", None)
    if client is None:
        client = boto3.session.Session().client("s3", config=_S3_CONFIG)
        _THREAD_LOCAL.s3_client = client
    return client


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ManifestError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def has_glob(text: str) -> bool:
    return any(char in text for char in "*?[")


def segment_has_glob(segment: str) -> bool:
    return any(char in segment for char in "*?[")


def iter_list_pages(bucket: str, prefix: str, delimiter: str | None = None) -> Iterator[dict]:
    paginator = get_s3_client().get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    if delimiter is not None:
        kwargs["Delimiter"] = delimiter

    for page in paginator.paginate(**kwargs):
        yield page


def iter_common_prefixes(bucket: str, prefix: str) -> Iterator[str]:
    for page in iter_list_pages(bucket, prefix, delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            child_prefix = entry["Prefix"]
            if child_prefix != prefix:
                yield child_prefix


def iter_direct_child_objects(bucket: str, prefix: str) -> Iterator[tuple[str, int]]:
    for page in iter_list_pages(bucket, prefix, delimiter="/"):
        for entry in page.get("Contents", []):
            key = entry["Key"]
            relative_name = key[len(prefix) :]
            if not relative_name or "/" in relative_name or key.endswith("/"):
                continue
            yield key, int(entry["Size"])


def iter_all_objects(bucket: str, prefix: str) -> Iterator[tuple[str, int]]:
    for page in iter_list_pages(bucket, prefix):
        for entry in page.get("Contents", []):
            key = entry["Key"]
            if key.endswith("/"):
                continue
            yield key, int(entry["Size"])


def get_object_size(bucket: str, key: str) -> int | None:
    try:
        response = get_s3_client().head_object(Bucket=bucket, Key=key)
        return int(response["ContentLength"])
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def resolve_directory_prefixes(bucket: str, directory_segments: list[str]) -> list[str]:
    prefixes = [""]

    for segment in directory_segments:
        if not prefixes:
            break

        if segment_has_glob(segment):
            next_prefixes: list[str] = []
            for prefix in prefixes:
                for child_prefix in iter_common_prefixes(bucket, prefix):
                    child_name = child_prefix[len(prefix) :].rstrip("/")
                    if fnmatch.fnmatchcase(child_name, segment):
                        next_prefixes.append(child_prefix)
            prefixes = dedupe_preserving_order(next_prefixes)
            continue

        prefixes = [f"{prefix}{segment}/" for prefix in prefixes]

    return prefixes


def expand_s3_pattern(pattern: str) -> list[tuple[str, str, int]]:
    bucket, key_pattern = parse_s3_uri(pattern)
    recursive_prefix = key_pattern.endswith("/")
    normalized_pattern = key_pattern.rstrip("/")

    if recursive_prefix and not normalized_pattern:
        return [(bucket, key, size_bytes) for key, size_bytes in iter_all_objects(bucket, "")]

    if not normalized_pattern:
        raise ManifestError(f"S3 pattern is missing a key: {pattern}")

    if not has_glob(normalized_pattern):
        if recursive_prefix:
            prefix = f"{normalized_pattern}/"
            return [(bucket, key, size_bytes) for key, size_bytes in iter_all_objects(bucket, prefix)]

        object_size = get_object_size(bucket, normalized_pattern)
        if object_size is not None:
            return [(bucket, normalized_pattern, object_size)]
        return []

    segments = normalized_pattern.split("/")

    if recursive_prefix:
        prefixes = resolve_directory_prefixes(bucket, segments)
        matches: list[tuple[str, str, int]] = []
        for prefix in prefixes:
            matches.extend((bucket, key, size_bytes) for key, size_bytes in iter_all_objects(bucket, prefix))
        return matches

    directory_segments = segments[:-1]
    file_segment = segments[-1]
    prefixes = resolve_directory_prefixes(bucket, directory_segments)
    matches: list[tuple[str, str, int]] = []

    if segment_has_glob(file_segment):
        for prefix in prefixes:
            for key, size_bytes in iter_direct_child_objects(bucket, prefix):
                if fnmatch.fnmatchcase(key[len(prefix) :], file_segment):
                    matches.append((bucket, key, size_bytes))
        return matches

    for prefix in prefixes:
        key = f"{prefix}{file_segment}"
        object_size = get_object_size(bucket, key)
        if object_size is not None:
            matches.append((bucket, key, object_size))

    return matches


def format_row_numbers(row_numbers: list[int], max_items: int = 6) -> str:
    preview = ", ".join(str(value) for value in row_numbers[:max_items])
    if len(row_numbers) > max_items:
        return f"{preview}, ..."
    return preview


def format_size(size_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(size_bytes)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value):,} {unit}"
            return f"{value:.1f} {unit} ({size_bytes:,} bytes)"
        value /= 1024

    return f"{size_bytes:,} B"


def split_cell_patterns(raw_value: str) -> list[str]:
    normalized_value = raw_value.replace("\\n", "\n")
    return [part.strip() for part in normalized_value.splitlines() if part.strip()]


def load_patterns(csv_path: Path, column_name: str) -> tuple[int, dict[str, list[int]]]:
    pattern_rows: dict[str, list[int]] = {}
    non_empty_values = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise click.ClickException(f"{csv_path} does not contain a CSV header row.")
        if column_name not in reader.fieldnames:
            available_columns = ", ".join(reader.fieldnames)
            raise click.ClickException(
                f"Column '{column_name}' was not found in {csv_path}. Available columns: {available_columns}"
            )

        for row_number, row in enumerate(reader, start=2):
            raw_value = row.get(column_name, "")
            if raw_value is None:
                continue

            for pattern in split_cell_patterns(raw_value):
                non_empty_values += 1
                pattern_rows.setdefault(pattern, []).append(row_number)

    if not pattern_rows:
        raise click.ClickException(f"No non-empty values were found in column '{column_name}'.")

    return non_empty_values, pattern_rows


def write_manifest(store: ManifestStore, output_path: Path | None) -> None:
    output_handle = None
    should_close = False

    try:
        if output_path is None:
            output_handle = click.get_text_stream("stdout")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("w", encoding="utf-8", newline="")
            should_close = True

        writer = csv.writer(output_handle, lineterminator="\n")
        for bucket, key in store.iter_rows():
            # S3 Batch Operations manifest CSV requires URL-encoded keys.
            writer.writerow([bucket, quote(key, safe="/")])
    finally:
        if should_close and output_handle is not None:
            output_handle.close()


@click.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("column_name")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the manifest to this path. Defaults to stdout.",
)
@click.option(
    "-w",
    "--workers",
    type=click.IntRange(1, None),
    default=DEFAULT_WORKERS,
    show_default=True,
    help="Number of worker threads used to expand patterns in parallel.",
)
@click.option(
    "--allow-empty-matches/--fail-on-empty-matches",
    default=False,
    show_default=True,
    help="Exit successfully even when some patterns do not match any S3 objects.",
)
@click.option("-v", "--verbose", is_flag=True, help="Print per-pattern progress to stderr.")
def main(
    csv_path: Path, column_name: str, output: Path | None, workers: int, allow_empty_matches: bool, verbose: bool
):
    """Read S3 patterns from CSV_PATH:COLUMN_NAME and emit an S3 Batch manifest CSV."""

    started_at = time.perf_counter()
    non_empty_values, pattern_rows = load_patterns(csv_path, column_name)
    unique_patterns = list(pattern_rows)
    worker_count = min(workers, len(unique_patterns))
    set_s3_client_pool_size(worker_count)

    click.echo(
        (
            f"Loaded {non_empty_values:,} non-empty values from '{column_name}' in {csv_path} "
            f"({len(unique_patterns):,} unique patterns)."
        ),
        err=True,
    )
    click.echo(f"Expanding patterns with {worker_count} worker thread(s).", err=True)

    total_raw_matches = 0
    total_raw_size_bytes = 0
    failures: dict[str, str] = {}
    empty_patterns: list[str] = []
    progress_interval = 1 if verbose or len(unique_patterns) <= 10 else max(1, len(unique_patterns) // 10)

    with ManifestStore() as store:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_pattern = {
                executor.submit(expand_s3_pattern, pattern): pattern for pattern in unique_patterns
            }

            for completed_count, future in enumerate(as_completed(future_to_pattern), start=1):
                pattern = future_to_pattern[future]
                row_numbers = pattern_rows[pattern]

                try:
                    matches = future.result()
                except Exception as exc:
                    failures[pattern] = str(exc)
                    click.echo(
                        (
                            f"[{completed_count}/{len(unique_patterns)}] ERROR rows {format_row_numbers(row_numbers)} "
                            f"{pattern}: {exc}"
                        ),
                        err=True,
                    )
                    continue

                match_size_bytes = sum(size_bytes for _, _, size_bytes in matches)
                total_raw_matches += len(matches)
                total_raw_size_bytes += match_size_bytes
                store.add_many(matches)

                if not matches:
                    empty_patterns.append(pattern)

                if verbose or completed_count % progress_interval == 0 or completed_count == len(unique_patterns):
                    click.echo(
                        (
                            f"[{completed_count}/{len(unique_patterns)}] rows {format_row_numbers(row_numbers)} "
                            f"matched {len(matches):,} object(s), {format_size(match_size_bytes)}"
                        ),
                        err=True,
                    )

        unique_match_count = store.count()
        unique_total_size_bytes = store.total_size_bytes()
        if unique_match_count == 0:
            raise click.ClickException("No S3 objects matched any pattern.")

        write_manifest(store, output)

    destination = str(output) if output is not None else "stdout"
    elapsed = time.perf_counter() - started_at
    click.echo(
        (
            f"Wrote {unique_match_count:,} unique manifest row(s) "
            f"totaling {format_size(unique_total_size_bytes)} "
            f"from {total_raw_matches:,} raw match(es) totaling {format_size(total_raw_size_bytes)} "
            f"to {destination} in {elapsed:.1f}s."
        ),
        err=True,
    )

    for pattern in empty_patterns:
        click.echo(
            f"Empty match rows {format_row_numbers(pattern_rows[pattern])}: {pattern}",
            err=True,
        )

    for pattern, error_message in failures.items():
        click.echo(
            f"Failed rows {format_row_numbers(pattern_rows[pattern])}: {pattern} -> {error_message}",
            err=True,
        )

    if failures or (empty_patterns and not allow_empty_matches):
        raise click.exceptions.Exit(1)


if __name__ == "__main__":
    main()
