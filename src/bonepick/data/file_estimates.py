import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from backports.zstd import open as zstd_open
from fsspec.caching import NamedTuple
from tqdm import tqdm


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


class FileRowCount(NamedTuple):
    path: Path
    est: int


def group_files_by_min_rows(
    files: list[Path], min_rows_per_group: int, num_proc: int = 1
) -> list[list[FileRowCount]]:
    if not files:
        return []
    if min_rows_per_group <= 0:
        raise ValueError(f"min_rows_per_group must be > 0, got {min_rows_per_group}")
    if num_proc <= 0:
        raise ValueError(f"num_proc must be > 0, got {num_proc}")

    all_estimated_rows: list[int] = [0] * len(files)
    pool = ThreadPoolExecutor(max_workers=num_proc)
    futures = {}
    try:
        futures = {pool.submit(estimate_rows_in_file, f): (i, f) for i, f in enumerate(files)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Estimating rows", unit="file"):
            idx, file_path = futures[future]
            try:
                all_estimated_rows[idx] = future.result()
            except Exception as e:
                raise RuntimeError(f"Failed to estimate rows for {file_path}: {e}") from e
    except BaseException:
        for pending in futures:
            pending.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)

    groups: list[list[FileRowCount]] = [[]]
    sizes: list[int] = [0]

    for file_path, estimated_rows in zip(files, all_estimated_rows):
        groups[-1].append(FileRowCount(path=file_path, est=estimated_rows))
        sizes[-1] += estimated_rows

        if sizes[-1] >= min_rows_per_group:
            groups.append([])
            sizes.append(0)

    if groups[-1] == []:
        groups.pop()

    return groups
