import gzip
import os
from pathlib import Path

from backports.zstd import open as zstd_open


def estimate_rows_in_file(path: Path, min_compressed_bytes: int = 131_072) -> int:
    """Estimate the number of rows in a JSONL file by reading a sample and extrapolating.

    For compressed files, reads decompressed chunks until at least 10% of the compressed
    file has been consumed (minimum min_compressed_bytes). For uncompressed files, reads
    1MB chunks and extrapolates from bytes read vs file size.

    Args:
        path: Path to a JSONL file (possibly compressed with .zst, .zstd, .gz, .gzip).
        min_compressed_bytes: Minimum compressed bytes to consume before extrapolating.

    Returns:
        Estimated number of rows in the file.
    """
    file_size = path.stat().st_size
    if file_size == 0:
        return 0

    suffix = "".join(path.suffixes[-2:])
    chunk_size = 1_048_576  # 1MB decompressed chunks

    if suffix.endswith((".zst", ".zstd")):
        return _estimate_rows_compressed_zstd(path, file_size, chunk_size, min_compressed_bytes)
    elif suffix.endswith((".gz", ".gzip")):
        return _estimate_rows_compressed_gzip(path, file_size, chunk_size, min_compressed_bytes)
    else:
        return _estimate_rows_uncompressed(path, file_size, chunk_size)


def _estimate_rows_compressed_zstd(path: Path, file_size: int, chunk_size: int, min_compressed_bytes: int) -> int:
    lines_read = 0
    remainder = b""
    raw_fh = open(path, "rb")
    try:
        decomp_fh = zstd_open(raw_fh, "rb")
        try:
            while True:
                chunk = decomp_fh.read(chunk_size)
                if not chunk:
                    # Consumed entire file — return exact count
                    if remainder:
                        lines_read += 1
                    return lines_read

                data = remainder + chunk
                parts = data.split(b"\n")
                # Last element is incomplete line (or empty if chunk ended with newline)
                remainder = parts[-1]
                lines_read += len(parts) - 1

                compressed_consumed = raw_fh.tell()
                min_threshold = max(min_compressed_bytes, int(file_size * 0.10))
                if compressed_consumed >= min_threshold:
                    if lines_read == 0:
                        return 0
                    return int(lines_read * (file_size / compressed_consumed))
        finally:
            decomp_fh.close()
    finally:
        raw_fh.close()


def _estimate_rows_compressed_gzip(path: Path, file_size: int, chunk_size: int, min_compressed_bytes: int) -> int:
    lines_read = 0
    remainder = b""
    raw_fh = open(path, "rb")
    try:
        decomp_fh = gzip.open(raw_fh, "rb")
        try:
            while True:
                chunk = decomp_fh.read(chunk_size)
                if not chunk:
                    if remainder:
                        lines_read += 1
                    return lines_read

                data = remainder + chunk
                parts = data.split(b"\n")
                remainder = parts[-1]
                lines_read += len(parts) - 1

                compressed_consumed = raw_fh.tell()
                min_threshold = max(min_compressed_bytes, int(file_size * 0.10))
                if compressed_consumed >= min_threshold:
                    if lines_read == 0:
                        return 0
                    return int(lines_read * (file_size / compressed_consumed))
        finally:
            decomp_fh.close()
    finally:
        raw_fh.close()


def _estimate_rows_uncompressed(path: Path, file_size: int, chunk_size: int) -> int:
    lines_read = 0
    bytes_read = 0
    remainder = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                if remainder:
                    lines_read += 1
                return lines_read

            data = remainder + chunk
            parts = data.split(b"\n")
            remainder = parts[-1]
            lines_read += len(parts) - 1
            bytes_read = fh.tell()

            if bytes_read >= file_size:
                if remainder:
                    lines_read += 1
                return lines_read

            if lines_read > 0:
                return int(lines_read * (file_size / bytes_read))

    return lines_read


def group_files_by_min_rows(
    files: list[Path],
    relative_paths: list[str],
    estimated_rows: list[int],
    min_rows_per_group: int,
) -> list[list[tuple[Path, str]]]:
    """Group files greedily until each group reaches a minimum row count.

    Iterates files in order, accumulating estimated rows. When the accumulated
    count reaches min_rows_per_group, the current group is finalized. The last
    group is merged into the previous one if it's smaller than the threshold
    (unless it's the only group).

    Args:
        files: List of file paths.
        relative_paths: List of relative path strings (parallel to files).
        estimated_rows: List of estimated row counts (parallel to files).
        min_rows_per_group: Minimum estimated rows per group.

    Returns:
        List of groups, each a list of (file_path, relative_path) tuples.
    """
    if not files:
        return []

    groups: list[list[tuple[Path, str]]] = []
    current_group: list[tuple[Path, str]] = []
    current_rows = 0

    for file_path, rel_path, est_rows in zip(files, relative_paths, estimated_rows):
        current_group.append((file_path, rel_path))
        current_rows += est_rows

        if current_rows >= min_rows_per_group:
            groups.append(current_group)
            current_group = []
            current_rows = 0

    # Handle remaining files
    if current_group:
        if groups:
            # Merge into previous group
            groups[-1].extend(current_group)
        else:
            # Only group
            groups.append(current_group)

    return groups
