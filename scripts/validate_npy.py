#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "boto3",
#     "click",
#     "numpy",
#     "tokenizers",
# ]
# ///
"""Validate tokenized Dolma shards stored as `.npy` plus `.csv.gz` pairs.

This validator is designed for tokenized Dolma-style shards where:
- the `.csv.gz` file stores `(start, end, doc_id, source_path, source_line_number)`
- the `.npy` payload is typically raw little-endian uint32 tokens, despite the extension
- each sampled document should round-trip through the Dolma2 tokenizer

Examples:
    uv run scripts/validate_npy.py /path/to/tokenized-dir
    uv run scripts/validate_npy.py 's3://ai2-llm/preprocessed/.../allenai/dolma2-tokenizer/*.npy'
"""

from __future__ import annotations

import ast
import csv
import fnmatch
import gzip
import io
import json
import logging
import os
import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator
from urllib.parse import urlparse

import boto3
import click
import numpy as np
from tokenizers import Tokenizer

LOGGER = logging.getLogger("validate_npy")
NPY_MAGIC = b"\x93NUMPY"
LOCAL_SOURCE_ROOT = Path("/mnt/raid0/ai2-llm/pretraining-data/sources")
DEFAULT_SOURCE_S3_ROOT = "s3://ai2-llm/pretraining-data/sources"
TOKENIZER_NAME = "allenai/dolma2-tokenizer"
VALID_EXTENSIONS = (".npy", ".csv.gz")


@dataclass(frozen=True)
class CsvRow:
    start: int
    end: int
    doc_id: str
    source_path: str
    source_line_number: int
    csv_row_number: int


@dataclass(frozen=True)
class ShardPair:
    base: str
    npy_location: str
    csv_location: str


@dataclass
class PairReport:
    pair: ShardPair
    ok: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    row_count: int = 0
    token_count: int = 0
    csv_last_end: int = 0
    source_count: int = 0
    token_format: str = "unknown"
    source_exact_matches: int = 0
    source_trimmed_whitespace_matches: int = 0
    source_checks_attempted: int = 0
    sample_rows_validated: int = 0

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)


class ValidationError(RuntimeError):
    """Raised when a shard cannot be validated."""


def configure_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def is_s3_uri(location: str) -> bool:
    return location.startswith("s3://")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValidationError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


def drop_suffix(location: str) -> str:
    if location.endswith(".csv.gz"):
        return location[: -len(".csv.gz")]
    if location.endswith(".npy"):
        return location[: -len(".npy")]
    raise ValidationError(f"Unsupported shard filename: {location}")


def matches_extension(location: str) -> bool:
    return location.endswith(".npy") or location.endswith(".csv.gz")


def fixed_prefix_for_pattern(pattern: str, separator: str) -> str:
    wildcard_positions = [pattern.find(ch) for ch in "*?[" if ch in pattern]
    if not wildcard_positions:
        return pattern
    prefix = pattern[: min(wildcard_positions)]
    if separator in prefix:
        return prefix.rsplit(separator, 1)[0] + separator
    return ""


def normalize_local_pattern(pattern: str) -> str:
    path = Path(pattern)
    if path.is_absolute():
        return str(path)
    return str(Path.cwd() / pattern)


def iter_local_candidates(target: str, recursive: bool) -> Iterator[str]:
    target_path = Path(target)
    if target_path.exists():
        if target_path.is_dir():
            iterator = target_path.rglob("*") if recursive else target_path.iterdir()
            for path in iterator:
                if path.is_file() and matches_extension(path.name):
                    yield str(path.resolve())
            return
        if target_path.is_file():
            parent = target_path.parent
            iterator = parent.rglob("*") if recursive else parent.iterdir()
            for path in iterator:
                if path.is_file() and matches_extension(path.name):
                    yield str(path.resolve())
            return

    if not has_glob(target):
        raise ValidationError(f"Local path does not exist: {target}")

    normalized_pattern = normalize_local_pattern(target)
    search_root = Path(fixed_prefix_for_pattern(normalized_pattern, os.sep) or Path.cwd())
    iterator = search_root.rglob("*") if recursive else search_root.iterdir()
    for path in iterator:
        if path.is_file() and matches_extension(path.name):
            yield str(path.resolve())


def iter_s3_candidates(target: str, recursive: bool, s3_client) -> Iterator[str]:
    bucket, key_pattern = parse_s3_uri(target)
    if not has_glob(key_pattern) and key_pattern.endswith(tuple(VALID_EXTENSIONS)):
        prefix = key_pattern.rsplit("/", 1)[0] + "/" if "/" in key_pattern else ""
    elif not has_glob(key_pattern):
        prefix = key_pattern.rstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"
    else:
        prefix = fixed_prefix_for_pattern(key_pattern, "/")

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for entry in page.get("Contents", []):
            key = entry["Key"]
            if key.endswith(tuple(VALID_EXTENSIONS)):
                yield f"s3://{bucket}/{key}"
            elif not recursive and key.rstrip("/") == prefix.rstrip("/"):
                yield f"s3://{bucket}/{key}"


def discover_pairs(target: str, recursive: bool, s3_client) -> list[ShardPair]:
    all_candidates = list(
        iter_s3_candidates(target, recursive, s3_client)
        if is_s3_uri(target)
        else iter_local_candidates(target, recursive)
    )
    if not all_candidates:
        raise ValidationError(f"No .npy/.csv.gz files found under {target}")

    if is_s3_uri(target):
        wildcard_or_exact = target if has_glob(target) or target.endswith(tuple(VALID_EXTENSIONS)) else None
    else:
        normalized_target = normalize_local_pattern(target)
        wildcard_or_exact = normalized_target if has_glob(target) else None
        if Path(target).exists() and Path(target).is_file():
            wildcard_or_exact = str(Path(target).resolve())

    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    matched_bases: set[str] = set()

    for candidate in all_candidates:
        base = drop_suffix(candidate)
        suffix_key = "npy" if candidate.endswith(".npy") else "csv"
        grouped[base][suffix_key] = candidate

        if wildcard_or_exact is None:
            matched_bases.add(base)
        elif fnmatch.fnmatch(candidate, wildcard_or_exact):
            matched_bases.add(base)

    pairs: list[ShardPair] = []
    for base in sorted(matched_bases):
        match = grouped[base]
        if "npy" not in match or "csv" not in match:
            missing = ".npy" if "npy" not in match else ".csv.gz"
            raise ValidationError(f"Missing {missing} companion for shard base {base}")
        pairs.append(
            ShardPair(
                base=base,
                npy_location=match["npy"],
                csv_location=match["csv"],
            )
        )

    if not pairs:
        raise ValidationError(f"No complete shard pairs matched {target}")
    return pairs


@contextmanager
def open_binary_stream(location: str, s3_client) -> Iterator[BinaryIO]:
    if is_s3_uri(location):
        bucket, key = parse_s3_uri(location)
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            yield body
        finally:
            body.close()
        return

    with Path(location).open("rb") as handle:
        yield handle


def read_location_range(location: str, start: int, length: int, s3_client) -> bytes:
    if length <= 0:
        return b""
    if is_s3_uri(location):
        bucket, key = parse_s3_uri(location)
        response = s3_client.get_object(
            Bucket=bucket,
            Key=key,
            Range=f"bytes={start}-{start + length - 1}",
        )
        body = response["Body"]
        try:
            data = body.read()
        finally:
            body.close()
        return data

    with Path(location).open("rb") as handle:
        handle.seek(start)
        return handle.read(length)


def get_location_size(location: str, s3_client) -> int:
    if is_s3_uri(location):
        bucket, key = parse_s3_uri(location)
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return int(response["ContentLength"])
    return Path(location).stat().st_size


class TokenFile:
    def __init__(self, location: str, s3_client):
        self.location = location
        self.s3_client = s3_client
        self.byte_size = get_location_size(location, s3_client)
        self.data_offset = 0
        self.dtype = np.dtype("<u4")
        self.token_count = 0
        self.format_name = "raw-uint32"
        self._inspect()

    def _inspect(self) -> None:
        prefix = read_location_range(self.location, 0, min(self.byte_size, 256), self.s3_client)
        if prefix.startswith(NPY_MAGIC):
            self._load_npy_header(prefix)
            return
        if self.byte_size % 4 != 0:
            raise ValidationError(
                f"{self.location} is not a NumPy file and its size ({self.byte_size}) is not divisible by 4"
            )
        self.data_offset = 0
        self.dtype = np.dtype("<u4")
        self.token_count = self.byte_size // 4
        self.format_name = "raw-uint32"

    def _load_npy_header(self, prefix: bytes) -> None:
        major = prefix[6]
        header_len_offset = 8
        if major == 1:
            header_len = int.from_bytes(prefix[header_len_offset : header_len_offset + 2], "little")
            data_offset = 10 + header_len
        elif major in {2, 3}:
            header_len = int.from_bytes(prefix[header_len_offset : header_len_offset + 4], "little")
            data_offset = 12 + header_len
        else:
            raise ValidationError(f"Unsupported NumPy header version in {self.location}: {major}")

        if len(prefix) < data_offset:
            prefix = read_location_range(self.location, 0, data_offset, self.s3_client)

        header_text = prefix[data_offset - header_len : data_offset].decode("latin1").strip()
        header = ast.literal_eval(header_text)
        shape = header["shape"]
        fortran_order = header["fortran_order"]
        dtype = np.dtype(header["descr"])

        if fortran_order:
            raise ValidationError(f"{self.location} is Fortran-ordered; expected a flat C-contiguous token buffer")
        if dtype.itemsize != 4 or dtype.kind != "u":
            raise ValidationError(f"{self.location} dtype is {dtype}, expected uint32-compatible tokens")
        if len(shape) != 1:
            raise ValidationError(f"{self.location} shape is {shape}, expected a 1-D token array")

        self.data_offset = data_offset
        self.dtype = dtype
        self.token_count = int(shape[0])
        self.format_name = f"npy[{dtype}]"

    def read_span(self, start: int, end: int) -> list[int]:
        if start < 0 or end < start or end > self.token_count:
            raise ValidationError(f"Requested invalid token span [{start}, {end}) from {self.location}")
        byte_start = self.data_offset + start * self.dtype.itemsize
        byte_length = (end - start) * self.dtype.itemsize
        data = read_location_range(self.location, byte_start, byte_length, self.s3_client)
        if len(data) != byte_length:
            raise ValidationError(
                f"Short read from {self.location}: expected {byte_length} bytes for [{start}, {end}), got {len(data)}"
            )
        values = np.frombuffer(data, dtype=self.dtype)
        return values.astype(np.uint32, copy=False).tolist()


@contextmanager
def open_text_reader(location: str, s3_client) -> Iterator[io.TextIOBase]:
    with open_binary_stream(location, s3_client) as handle:
        if location.endswith(".gz"):
            with gzip.GzipFile(fileobj=handle, mode="rb") as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as reader:
                    yield reader
            return
        with io.TextIOWrapper(handle, encoding="utf-8") as reader:
            yield reader


def reservoir_sample(
    rows: list[CsvRow], sample_size: int, candidate: CsvRow, rng: random.Random, count: int
) -> None:
    if sample_size <= 0:
        return
    if len(rows) < sample_size:
        rows.append(candidate)
        return
    replacement_index = rng.randrange(count)
    if replacement_index < sample_size:
        rows[replacement_index] = candidate


def scan_csv(
    location: str,
    sample_size: int,
    seed: int,
    report: PairReport,
    s3_client,
) -> tuple[list[CsvRow], int]:
    rng = random.Random(seed)
    samples: list[CsvRow] = []
    first_row: CsvRow | None = None
    last_row: CsvRow | None = None
    previous_end: int | None = None
    source_paths: set[str] = set()

    with open_text_reader(location, s3_client) as reader:
        csv_reader = csv.reader(reader)
        for row_number, raw_row in enumerate(csv_reader, start=1):
            if len(raw_row) != 5:
                report.fail(f"{location} row {row_number} has {len(raw_row)} columns; expected 5")
                continue

            start_raw, end_raw, doc_id, source_path, line_number_raw = raw_row
            try:
                start = int(start_raw)
                end = int(end_raw)
                source_line_number = int(line_number_raw)
            except ValueError as exc:
                report.fail(f"{location} row {row_number} has non-integer offsets or line number: {exc}")
                continue

            if start < 0 or end <= start:
                report.fail(f"{location} row {row_number} has invalid span [{start}, {end})")
                continue
            if source_line_number <= 0:
                report.fail(f"{location} row {row_number} has invalid source line number {source_line_number}")
                continue

            row = CsvRow(
                start=start,
                end=end,
                doc_id=doc_id,
                source_path=source_path,
                source_line_number=source_line_number,
                csv_row_number=row_number,
            )

            if first_row is None:
                first_row = row
                if row.start != 0:
                    report.fail(f"{location} first row starts at {row.start}, expected 0")

            if previous_end is not None:
                if row.start != previous_end:
                    report.fail(
                        f"{location} row {row_number} starts at {row.start}, expected contiguous offset {previous_end}"
                    )
            previous_end = row.end
            last_row = row
            source_paths.add(source_path)
            report.row_count += 1
            reservoir_sample(samples, sample_size, row, rng, report.row_count)

    if report.row_count == 0:
        report.fail(f"{location} contained no rows")
        return [], 0

    report.source_count = len(source_paths)
    report.csv_last_end = last_row.end if last_row else 0

    selected_rows: dict[int, CsvRow] = {row.csv_row_number: row for row in samples}
    if first_row is not None:
        selected_rows[first_row.csv_row_number] = first_row
    if last_row is not None:
        selected_rows[last_row.csv_row_number] = last_row
    return sorted(selected_rows.values(), key=lambda row: row.csv_row_number), report.csv_last_end


def load_tokenizer(tokenizer_path: str | None) -> Tokenizer:
    if tokenizer_path:
        LOGGER.info("Loading tokenizer from %s", tokenizer_path)
        return Tokenizer.from_file(tokenizer_path)

    repo_copy = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "bonepick"
        / "data"
        / "tokenizers_config"
        / "dolma2_tokenizer.json"
    )
    if repo_copy.exists():
        LOGGER.info("Loading vendored %s tokenizer from %s", TOKENIZER_NAME, repo_copy)
        return Tokenizer.from_file(str(repo_copy))

    LOGGER.info("Loading tokenizer from Hugging Face: %s", TOKENIZER_NAME)
    return Tokenizer.from_pretrained(TOKENIZER_NAME)


def validate_sample_rows(
    sample_rows: list[CsvRow],
    token_file: TokenFile,
    tokenizer: Tokenizer,
    eos_id: int,
    vocab_size: int,
    report: PairReport,
) -> dict[int, list[int]]:
    token_cache: dict[int, list[int]] = {}
    for row in sample_rows:
        tokens = token_file.read_span(row.start, row.end)
        token_cache[row.csv_row_number] = tokens
        if len(tokens) != row.end - row.start:
            report.fail(
                f"Sample row {row.csv_row_number} span length mismatch: "
                f"CSV says {row.end - row.start}, binary read returned {len(tokens)}"
            )
            continue
        if not tokens:
            report.fail(f"Sample row {row.csv_row_number} decoded to an empty token span")
            continue
        invalid_token = next((token for token in tokens if token < 0 or token >= vocab_size), None)
        if invalid_token is not None:
            report.fail(
                f"Sample row {row.csv_row_number} contains token id {invalid_token}, outside tokenizer vocab [0, {vocab_size})"
            )
            continue
        if tokens[-1] != eos_id:
            report.fail(
                f"Sample row {row.csv_row_number} does not terminate with EOS id {eos_id}; got {tokens[-1]}"
            )
            continue

        core_tokens = tokens[:-1]
        decoded = tokenizer.decode(core_tokens)
        roundtrip = tokenizer.encode(decoded).ids
        if roundtrip != core_tokens:
            report.fail(
                f"Sample row {row.csv_row_number} failed tokenizer round-trip after removing EOS "
                f"(decoded chars={len(decoded)}, token_count={len(core_tokens)})"
            )
            continue
        report.sample_rows_validated += 1

    return token_cache


def resolve_source_location(source_path: str, source_s3_root: str) -> str | None:
    local_path = Path(source_path)
    if local_path.exists():
        return str(local_path)
    if is_s3_uri(source_path):
        return source_path
    try:
        relative = local_path.relative_to(LOCAL_SOURCE_ROOT)
    except ValueError:
        return None
    return f"{source_s3_root.rstrip('/')}/{relative.as_posix()}"


def read_selected_jsonl_rows(
    location: str,
    line_numbers: set[int],
    s3_client,
) -> dict[int, dict]:
    if not line_numbers:
        return {}

    targets = sorted(line_numbers)
    target_index = 0
    found: dict[int, dict] = {}
    max_target = targets[-1]

    with open_text_reader(location, s3_client) as reader:
        for line_number, line in enumerate(reader, start=1):
            if line_number > max_target or target_index >= len(targets):
                break
            if line_number != targets[target_index]:
                continue
            found[line_number] = json.loads(line)
            target_index += 1

    return found


def validate_against_sources(
    sample_rows: list[CsvRow],
    token_cache: dict[int, list[int]],
    tokenizer: Tokenizer,
    eos_id: int,
    source_s3_root: str,
    report: PairReport,
    s3_client,
) -> None:
    rows_by_source: dict[str, list[CsvRow]] = defaultdict(list)
    unresolved_sources: set[str] = set()

    for row in sample_rows:
        resolved = resolve_source_location(row.source_path, source_s3_root)
        if resolved is None:
            unresolved_sources.add(row.source_path)
            continue
        rows_by_source[resolved].append(row)

    for source_path in sorted(unresolved_sources):
        report.warn(f"Could not resolve source path for source-backed verification: {source_path}")

    for source_location, rows in rows_by_source.items():
        line_numbers = {row.source_line_number for row in rows}
        try:
            source_rows = read_selected_jsonl_rows(source_location, line_numbers, s3_client)
        except Exception as exc:  # noqa: BLE001
            report.warn(f"Failed to read source shard {source_location}: {exc}")
            continue

        for row in rows:
            report.source_checks_attempted += 1
            source_row = source_rows.get(row.source_line_number)
            if source_row is None:
                report.fail(
                    f"Source shard {source_location} does not contain line {row.source_line_number} "
                    f"referenced by CSV row {row.csv_row_number}"
                )
                continue
            if "text" not in source_row:
                report.fail(f"Source shard {source_location} line {row.source_line_number} lacks a `text` field")
                continue

            source_id = str(source_row.get("id"))
            if source_id != row.doc_id:
                report.fail(
                    f"CSV row {row.csv_row_number} doc_id={row.doc_id} does not match source id={source_id} "
                    f"at {source_location}:{row.source_line_number}"
                )
                continue

            stored_tokens = token_cache[row.csv_row_number]
            source_text = source_row["text"]
            expected_exact = tokenizer.encode(source_text).ids + [eos_id]
            if stored_tokens == expected_exact:
                report.source_exact_matches += 1
                continue

            expected_trimmed = tokenizer.encode(source_text.rstrip()).ids + [eos_id]
            if stored_tokens == expected_trimmed:
                report.source_trimmed_whitespace_matches += 1
                continue

            report.fail(
                f"CSV row {row.csv_row_number} tokens do not match source text tokenization "
                f"for {source_location}:{row.source_line_number}"
            )


def format_pair_name(pair: ShardPair) -> str:
    if is_s3_uri(pair.base):
        return PurePosixPath(pair.base).name
    return Path(pair.base).name


def validate_pair(
    pair: ShardPair,
    tokenizer: Tokenizer,
    sample_size: int,
    seed: int,
    source_s3_root: str,
    check_sources: bool,
    s3_client,
) -> PairReport:
    report = PairReport(pair=pair)
    pair_name = format_pair_name(pair)

    LOGGER.info("Validating shard %s", pair_name)
    token_file = TokenFile(pair.npy_location, s3_client)
    report.token_count = token_file.token_count
    report.token_format = token_file.format_name

    sample_rows, last_end = scan_csv(
        pair.csv_location, sample_size=sample_size, seed=seed, report=report, s3_client=s3_client
    )
    if last_end != token_file.token_count:
        report.fail(
            f"{pair.csv_location} ends at token {last_end}, but {pair.npy_location} contains {token_file.token_count} tokens"
        )

    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValidationError("Tokenizer is missing the <|endoftext|> token")
    vocab_size = tokenizer.get_vocab_size()
    token_cache = validate_sample_rows(sample_rows, token_file, tokenizer, eos_id, vocab_size, report)

    if check_sources and sample_rows:
        validate_against_sources(
            sample_rows=sample_rows,
            token_cache=token_cache,
            tokenizer=tokenizer,
            eos_id=eos_id,
            source_s3_root=source_s3_root,
            report=report,
            s3_client=s3_client,
        )

    LOGGER.info(
        "Shard %s summary: rows=%s tokens=%s format=%s sampled=%s source_checks=%s",
        pair_name,
        f"{report.row_count:,}",
        f"{report.token_count:,}",
        report.token_format,
        report.sample_rows_validated,
        report.source_checks_attempted,
    )
    return report


@click.command()
@click.argument("target")
@click.option(
    "--sample-size", default=8, show_default=True, type=int, help="Number of CSV rows to sample per shard."
)
@click.option("--max-pairs", default=None, type=int, help="Optional limit on how many shard pairs to validate.")
@click.option("--seed", default=0, show_default=True, type=int, help="Random seed for reservoir sampling.")
@click.option(
    "--tokenizer-path",
    default=None,
    help="Optional local tokenizer JSON path. Defaults to the vendored Dolma2 tokenizer in this repo.",
)
@click.option(
    "--source-s3-root",
    default=DEFAULT_SOURCE_S3_ROOT,
    show_default=True,
    help="S3 root used to resolve source paths recorded in the CSV metadata.",
)
@click.option(
    "--check-sources/--skip-sources",
    default=True,
    show_default=True,
    help="Whether to fetch sampled source rows and compare stored tokens against source text tokenization.",
)
@click.option(
    "--recursive/--no-recursive", default=True, show_default=True, help="Search local/S3 targets recursively."
)
@click.option(
    "--strict-warnings/--allow-warnings", default=False, show_default=True, help="Exit non-zero on warnings."
)
@click.option("-v", "--verbose", count=True, help="Increase logging verbosity.")
def main(
    target: str,
    sample_size: int,
    max_pairs: int | None,
    seed: int,
    tokenizer_path: str | None,
    source_s3_root: str,
    check_sources: bool,
    recursive: bool,
    strict_warnings: bool,
    verbose: int,
) -> None:
    """Validate local or S3 token shards made of `.npy` plus `.csv.gz` pairs."""

    configure_logging(verbose)
    if sample_size < 0:
        raise click.ClickException("--sample-size must be >= 0")
    if max_pairs is not None and max_pairs <= 0:
        raise click.ClickException("--max-pairs must be > 0")

    s3_client = boto3.client("s3")
    tokenizer = load_tokenizer(tokenizer_path)
    pairs = discover_pairs(target, recursive=recursive, s3_client=s3_client)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    LOGGER.info("Discovered %s shard pair(s) under %s", len(pairs), target)
    reports = [
        validate_pair(
            pair=pair,
            tokenizer=tokenizer,
            sample_size=sample_size,
            seed=seed,
            source_s3_root=source_s3_root,
            check_sources=check_sources,
            s3_client=s3_client,
        )
        for pair in pairs
    ]

    passed = 0
    failed = 0
    warning_count = 0
    for report in reports:
        pair_name = format_pair_name(report.pair)
        if report.ok:
            passed += 1
            click.echo(
                "PASS "
                f"{pair_name}: rows={report.row_count:,}, tokens={report.token_count:,}, "
                f"format={report.token_format}, sampled={report.sample_rows_validated}, "
                f"source_checks={report.source_checks_attempted}"
            )
        else:
            failed += 1
            click.echo(
                "FAIL "
                f"{pair_name}: rows={report.row_count:,}, tokens={report.token_count:,}, "
                f"format={report.token_format}, sampled={report.sample_rows_validated}, "
                f"source_checks={report.source_checks_attempted}"
            )

        for warning in report.warnings:
            warning_count += 1
            click.echo(f"  WARNING {warning}")
        for error in report.errors:
            click.echo(f"  ERROR {error}")
        if report.source_checks_attempted:
            click.echo(
                "  Source matches: "
                f"exact={report.source_exact_matches}, trimmed_whitespace={report.source_trimmed_whitespace_matches}"
            )

    click.echo(
        f"Summary: {passed} passed, {failed} failed, {warning_count} warning(s), {len(reports)} pair(s) checked"
    )

    if failed > 0 or (strict_warnings and warning_count > 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
