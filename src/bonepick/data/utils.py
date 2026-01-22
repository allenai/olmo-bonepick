import dataclasses as dt
import hashlib
import os
import pickle
import random
import re
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from functools import cached_property
from math import floor, log10
from pathlib import Path
from typing import ClassVar, Generator, Generic, Self, TypeVar

import msgspec
import smart_open
from tqdm import tqdm

from bonepick.data.expressions import compile_jq
from bonepick.data.normalizers import get_normalizer
from bonepick.logger import LOGGER

FILE_SUFFIXES = [
    f"{type_}{compr}" for type_ in (".jsonl", ".json") for compr in (".zst", ".zstd", ".gz", ".gzip", "")
]
DEFAULT_SUFFIX = ".jsonl.zst"


def make_uuid() -> str:
    """Platform-agnostic UUID generation. See https://stackoverflow.com/questions/2759644/python-multiprocessing-doesnt-play-nicely-with-uuid-uuid4 for more details."""
    return str(uuid.UUID(bytes=os.urandom(16), version=4))


def batch_save_hf_dataset(
    batch: dict[str, list],
    indices: list[int],
    destination_dir: Path,
):
    destination_file = destination_dir / f"shard_{indices[0]}{DEFAULT_SUFFIX}"

    encoder = msgspec.json.Encoder()

    with smart_open.open(str(destination_file), "wb") as f:  # pyright: ignore
        for idx, loc in enumerate(indices):
            sample = {k: v[idx] for k, v in batch.items()}
            f.write(encoder.encode(sample) + b"\n")

    return batch


def transform_single_file(
    source_path: Path,
    destination_path: Path,
    text_transform: str,
    label_transform: str,
):
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    decoder = msgspec.json.Decoder()
    encoder = msgspec.json.Encoder()

    text_transform_fn = compile_jq(text_transform)
    label_transform_fn = compile_jq(label_transform)

    with ExitStack() as stack:
        source_file = stack.enter_context(smart_open.open(source_path, "rb"))  # pyright: ignore
        destination_file = stack.enter_context(smart_open.open(destination_path, "wb"))  # pyright: ignore

        for line in source_file:
            row = decoder.decode(line)
            new_row = {**text_transform_fn(row), **label_transform_fn(row)}
            destination_file.write(encoder.encode(new_row) + b"\n")


def normalize_single_file(
    source_path: Path,
    destination_path: Path,
    text_field: str,
    label_field: str,
    normalization: str,
):
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    normalizer = get_normalizer(normalization)
    decoder = msgspec.json.Decoder()
    encoder = msgspec.json.Encoder()

    with ExitStack() as stack:
        source_file = stack.enter_context(smart_open.open(source_path, "rb"))  # pyright: ignore
        destination_file = stack.enter_context(smart_open.open(destination_path, "wb"))  # pyright: ignore

        for line in source_file:
            row = decoder.decode(line)
            row[text_field] = normalizer.normalize(str(row[text_field]))
            row[label_field] = normalize_label(str(row[label_field]))
            destination_file.write(encoder.encode(row) + b"\n")


@dt.dataclass(frozen=True)
class DatasetSplit:
    text: list[str]
    label: list[str | None]

    @classmethod
    def new(cls):
        return cls(text=[], label=[])

    def __iter__(self) -> Generator[tuple[str, str | None], None, None]:
        for text, label in zip(self.text, self.label):
            yield text, label

    def __len__(self) -> int:
        assert len(self.text) == len(self.label), "Text and label lists must have the same length"
        return len(self.text)

    def shuffle(self, rng: random.Random | None = None) -> Self:
        rng = rng or random.Random()
        indices = list(range(len(self)))
        rng.shuffle(indices)
        return self.__class__(text=[self.text[i] for i in indices], label=[self.label[i] for i in indices])

    @cached_property
    def signature(self) -> str:
        h = hashlib.sha256()
        for text, label in zip(self.text, self.label):
            h.update(f"{text}\t{label}".encode() if label is not None else text.encode())
        return h.hexdigest()


@dt.dataclass(frozen=True)
class DatasetTuple:
    train: DatasetSplit
    valid: DatasetSplit
    test: DatasetSplit

    @cached_property
    def signature(self) -> str:
        h = hashlib.sha256()
        h.update(self.train.signature.encode())
        h.update(self.test.signature.encode())
        return h.hexdigest()

    def __iter__(self) -> Generator[tuple[str, DatasetSplit], None, None]:
        for field in dt.fields(self):
            yield field.name, getattr(self, field.name)

    @classmethod
    def new(cls):
        return cls(train=DatasetSplit.new(), valid=DatasetSplit.new(), test=DatasetSplit.new())


def _load_single_json_dataset_file(
    file_path: Path,
    text_field_expression: str,
    label_field_expression: str,
    allow_missing_label: bool = False,
    normalizer_name: str | None = None,
    text_max_length: int | None = None,
) -> tuple[list[str], list[str | None]]:
    texts: list[str] = []
    labels: list[str | None] = []

    text_field_selector = compile_jq(text_field_expression)
    label_field_selector = compile_jq(label_field_expression)
    decoder = msgspec.json.Decoder()
    normalizer = get_normalizer(normalizer_name) if normalizer_name is not None else None

    with smart_open.open(file_path, "rb") as f:  # pyright: ignore
        for line in f:
            row = decoder.decode(line)

            # Step 1: process text
            text_value = str(text_field_selector(row))
            if text_max_length is not None and len(text_value) > text_max_length:
                text_value = text_value[:text_max_length]
            if normalizer is not None:
                text_value = normalizer.normalize(text_value)
            texts.append(text_value)

            # Step 2: process label
            label_value = label_field_selector(row)
            if allow_missing_label:
                label_value = str(label_value) if label_value is not None else None
            elif label_value is not None:
                label_value = str(label_value)
            else:
                raise ValueError(f"Label expression {label_field_expression} yielded None for row {row}")

            labels.append(label_value)

    return texts, labels


def load_jsonl_dataset(
    dataset_dirs: Path | list[Path],
    train_split_name: str = "train",
    valid_split_name: str = "valid",
    test_split_name: str = "test",
    text_field_expression: str = ".text",
    label_field_expression: str = ".label",
    train_split_required: bool = True,
    valid_split_required: bool = False,
    test_split_required: bool = True,
    allow_missing_label: bool = False,
    max_workers: int | None = None,
    normalizer_name: str | None = None,
    text_max_length: int | None = None,
) -> DatasetTuple:
    """Load dataset from one or more directories.

    Each directory should have train/ and test/ subdirectories containing JSONL files.
    When multiple directories are provided, the data is concatenated.
    """

    # use all available workers by default
    max_workers = max_workers or os.cpu_count() or 1

    # Normalize to list
    if isinstance(dataset_dirs, Path):
        dataset_dirs = [dataset_dirs]

    dataset_tuple = DatasetTuple.new()

    for dataset_split, split_name, is_required in (
        (dataset_tuple.train, train_split_name, train_split_required),
        (dataset_tuple.valid, valid_split_name, valid_split_required),
        (dataset_tuple.test, test_split_name, test_split_required),
    ):
        all_files: list[Path] = []

        for dataset_dir in dataset_dirs:
            split_path = dataset_dir / split_name
            if split_path.exists():
                if not split_path.is_dir():
                    raise NotADirectoryError(f"Split path {split_path} is not a directory")
            elif is_required and not split_path.exists():
                raise FileNotFoundError(f"Split path {split_path} does not exist")

            for root, _, files in os.walk(split_path):
                for file in files:
                    file_path = Path(root) / file
                    if "".join(file_path.suffixes) not in FILE_SUFFIXES:
                        continue
                    all_files.append(file_path)

            if is_required and not all_files:
                raise FileNotFoundError(f"No files found for {split_name} split")

        futures = []
        with ExitStack() as stack:
            pool_cls = ProcessPoolExecutor if max_workers > 1 else ThreadPoolExecutor
            pool = stack.enter_context(pool_cls(max_workers=max_workers))
            pbar = stack.enter_context(
                tqdm(total=len(all_files), desc=f"Loading {split_name} split", unit=" files", unit_scale=True)
            )

            for file_path in all_files:
                future = pool.submit(
                    _load_single_json_dataset_file,
                    file_path=file_path,
                    text_field_expression=text_field_expression,
                    label_field_expression=label_field_expression,
                    allow_missing_label=allow_missing_label,
                    normalizer_name=normalizer_name,
                    text_max_length=text_max_length,
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    texts, labels = future.result()
                    dataset_split.text.extend(texts)
                    dataset_split.label.extend(labels)
                except Exception as e:
                    for future in futures:
                        future.cancel()
                    raise e
                pbar.update(1)

    return dataset_tuple


def write_dataset(
    dataset: DatasetTuple,
    destination_dir: Path,
    text_field_name: str,
    label_field_name: str,
    max_rows_per_file: int = 100_000,
):
    destination_dir.mkdir(parents=True, exist_ok=True)
    encoder = msgspec.json.Encoder()

    with ExitStack() as stack:
        pbar = stack.enter_context(tqdm(desc="Writing dataset files", unit=" files", unit_scale=True))

        for split_name, split_data in dataset:
            if len(split_data) == 0:
                continue

            split_path = destination_dir / split_name
            split_path.mkdir(parents=True, exist_ok=True)

            def make_new_file():
                return stack.enter_context(
                    smart_open.open(split_path / f"shard_{uuid.uuid4()}.jsonl.zst", "wb")  # pyright: ignore
                )

            current_file = make_new_file()
            count = 0
            for text, label in split_data:
                pbar.set_postfix(split=split_name, elements=len(split_data))
                current_file.write(encoder.encode({text_field_name: text, label_field_name: label}) + b"\n")
                count += 1
                if count >= max_rows_per_file:
                    current_file.close()
                    current_file = make_new_file()
                    count = 0
                    pbar.update(1)


def normalize_label(label: str) -> str:
    label = re.sub(r"^__label__", "", label)
    label = re.sub(r"\s+", "_", label)
    return label.lower().strip()


def pretty_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60):,}m {seconds % 60:.2f}s"
    else:
        return f"{int(seconds // 3600):,}h {int(seconds % 3600 // 60):,}m {seconds % 60:.2f}s"


def pretty_length(length: int) -> int:
    """Round it to closest bucket values"""
    if length < 0:
        raise ValueError(f"Length must be non-negative, got {length}")

    pos = max(0, floor(log10(length) - 1))
    bucket = 10**pos
    return round(length / bucket) * bucket


def convert_single_file_to_fasttext(
    source_path: Path,
    text_expression: str,
    label_expression: str,
    normalization: str,
    print_progress: bool = True,
    print_every: int = 5_000,
    max_length: int | None = None,
    label_mapper: tuple[list[float], list[str]] | None = None,
    multi_label: bool = False,
) -> list[str]:
    """Convert a JSONL file to FastText format.

    Args:
        source_path: Path to the source JSONL file.
        text_expression: JQ expression to extract text field.
        label_expression: JQ expression to extract label field.
        normalization: Text normalization method to apply.
        print_progress: Whether to print progress messages.
        print_every: Print progress every N rows.
        max_length: Maximum length of text to process (truncate if longer).
        label_mapper: Optional tuple of (bin_edges, bin_labels) for auto-binning.
            bin_edges is a sorted list of N+1 floats defining N bins.
            bin_labels is a list of N strings for the bin labels.
            Labels are assigned based on which bin the numeric value falls into.
        multi_label: If True, label_expression should return a dict {criterion_name: 0_or_1}.
            Generates __label__<criterion> for each criterion with value 1.

    Returns:
        List of FastText formatted rows (__label__<label> <text>).
    """
    decoder = msgspec.json.Decoder()
    rows: list[str] = []

    normalizer = get_normalizer(normalization)
    re_remove_extra_labels = re.compile(r"\b__label__(\S+)\b")

    start_time = time.time()
    count = 0

    text_field_selector = compile_jq(text_expression)
    label_field_selector = compile_jq(label_expression)

    with smart_open.open(source_path, "rb") as f:  # pyright: ignore
        for line in f:
            count += 1
            if print_progress and count % print_every == 0:
                now = time.time()
                elapsed = pretty_time(now - start_time)
                throughput = f"{count / (now - start_time):.2f}"
                LOGGER.info(f"Converted {count:,} rows in {elapsed} ({throughput} rows/s)")

            row = decoder.decode(line)
            text_value = str(text_field_selector(row))
            if max_length is not None and len(text_value) > max_length:
                text_value = text_value[:max_length]

            text = normalizer.normalize(text_value)

            if "__label__" in text:
                text = re_remove_extra_labels.sub(r"\1", text)

            label_value = label_field_selector(row)

            # Multi-label mode: label_value is a dict {criterion_name: 0_or_1_or_bool}
            if multi_label:
                if not isinstance(label_value, dict):
                    raise ValueError(
                        f"Multi-label mode requires label expression to return a dict, "
                        f"got {type(label_value).__name__}: {label_value}"
                    )
                # Generate labels for each criterion with truthy value (handles 0/1, True/False, etc.)
                labels = [
                    f"__label__{normalize_label(str(criterion))}"
                    for criterion, value in sorted(label_value.items())
                    if bool(value)
                ]
                if labels:
                    rows.append(f"{' '.join(labels)} {text}")
                # Skip rows with no true criteria (no labels)
                continue

            # Apply label mapping if provided
            if label_mapper is not None:
                bin_edges, bin_labels = label_mapper
                numeric_value = float(label_value)
                # Find the bin this value belongs to
                label = bin_labels[-1]  # Default to last bin
                for i in range(len(bin_edges) - 1):
                    lower, upper = bin_edges[i], bin_edges[i + 1]
                    # Handle single-value bins (lower == upper) with exact match
                    if lower == upper:
                        if numeric_value == lower:
                            label = bin_labels[i]
                            break
                    elif lower <= numeric_value < upper:
                        label = bin_labels[i]
                        break
                # Handle edge case: value equals the max edge
                if numeric_value == bin_edges[-1]:
                    label = bin_labels[-1]
            else:
                label = normalize_label(str(label_value))

            rows.append(f"__label__{label} {text}")
    return rows


def count_labels_in_file(
    source_path: Path,
    label_field: str,
) -> dict[str, int]:
    """Count labels in a single file."""
    decoder = msgspec.json.Decoder()
    counts: dict[str, int] = {}

    with smart_open.open(source_path, "rb") as f:  # pyright: ignore
        for line in f:
            row = decoder.decode(line)
            label = str(row[label_field])
            counts[label] = counts.get(label, 0) + 1

    return counts


def extract_numeric_labels_from_file(
    source_path: Path,
    label_expression: str,
) -> list[float]:
    """Extract numeric labels from a single file using a jq expression."""
    decoder = msgspec.json.Decoder()
    label_selector = compile_jq(label_expression)
    labels: list[float] = []

    with smart_open.open(source_path, "rb") as f:  # pyright: ignore
        for line in f:
            row = decoder.decode(line)
            label_value = label_selector(row)
            try:
                labels.append(float(label_value))
            except (TypeError, ValueError) as e:
                raise ValueError(f"Label value {label_value!r} is not numeric: {e}") from e

    return labels


def read_samples_from_file(
    source_path: Path,
    label_field: str,
) -> dict[str, list[dict]]:
    """Read all samples from a file, grouped by label."""
    decoder = msgspec.json.Decoder()
    samples_by_label: dict[str, list[dict]] = {}

    with smart_open.open(source_path, "rb") as f:  # pyright: ignore
        for line in f:
            row = decoder.decode(line)
            label = str(row[label_field])
            if label not in samples_by_label:
                samples_by_label[label] = []
            samples_by_label[label].append(row)

    return samples_by_label


def count_tokens_in_file(
    source_path: Path,
    tokenizer_json: str,
    input_field_expression: str,
    batch_size: int = 2000,
) -> int:
    """Count tokens in a single file.

    Args:
        source_path: Path to source file
        tokenizer_json: Serialized tokenizer JSON string
        input_field_expression: JQ expression to extract text field
        batch_size: Process in batches of this size

    Returns:
        Total token count for the file
    """
    from tokenizers import Tokenizer

    decoder = msgspec.json.Decoder()
    tokenizer = Tokenizer.from_str(tokenizer_json)
    transform_fn = compile_jq(input_field_expression)

    total_tokens = 0
    batch: list[str] = []
    with smart_open.open(source_path, "rb") as f:  # pyright: ignore
        for line in f:
            row = decoder.decode(line)
            transformed = transform_fn(row)
            batch.append(str(transformed))

            if len(batch) >= batch_size:
                encoded = tokenizer.encode_batch_fast(batch, add_special_tokens=False)
                total_tokens += sum(len(encoded_text.ids) for encoded_text in encoded)
                del encoded
                batch = []

    if batch:
        encoded = tokenizer.encode_batch_fast(batch, add_special_tokens=False)
        total_tokens += sum(len(encoded_text.ids) for encoded_text in encoded)

    return total_tokens


def sample_single_file(
    source_path: Path,
    destination_path: Path,
    target_size: int,
    seed: int = 42,
):
    """Sample a file to approximately target size in bytes.

    Args:
        source_path: Path to source file
        destination_path: Path to destination file
        target_size: Target size in bytes for the output file
        seed: Random seed for reproducibility
    """
    import random

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    # Get file size
    source_size = source_path.stat().st_size

    # If file is already smaller or equal, just copy it
    if source_size <= target_size:
        import shutil

        shutil.copy2(source_path, destination_path)
        return

    # Calculate sampling ratio
    sampling_ratio = target_size / source_size
    rng = random.Random(seed)

    # Optimization: If sampling ratio is very low (<5%), use byte counting approach
    # instead of probabilistic sampling to be more precise and efficient
    MIN_SAMPLING_RATIO = 0.05

    if sampling_ratio < MIN_SAMPLING_RATIO:
        # Use byte counting approach: sample with higher ratio, stop when we reach target
        # Use 5% sampling ratio to ensure we don't process the entire file
        effective_sampling_ratio = MIN_SAMPLING_RATIO

        with ExitStack() as stack:
            source_file = stack.enter_context(smart_open.open(source_path, "rb"))  # pyright: ignore
            destination_file = stack.enter_context(smart_open.open(destination_path, "wb"))  # pyright: ignore

            bytes_written = 0
            for line in source_file:
                if bytes_written >= target_size:
                    break

                if rng.random() < effective_sampling_ratio:
                    destination_file.write(line)
                    bytes_written += len(line)
    else:
        # Normal probabilistic sampling for higher ratios (>=5%)
        with ExitStack() as stack:
            source_file = stack.enter_context(smart_open.open(source_path, "rb"))  # pyright: ignore
            destination_file = stack.enter_context(smart_open.open(destination_path, "wb"))  # pyright: ignore

            for line in source_file:
                if rng.random() < sampling_ratio:
                    destination_file.write(line)


T = TypeVar("T")


@dt.dataclass(frozen=True)
class ChunkedDatasetStructManager(Generic[T]):
    type_: type[T]
    tempdir: Path

    TYPE_PATH_NAME: ClassVar[str] = "type.pickle"

    @property
    def struct_cls(self) -> type[msgspec.Struct]:
        return msgspec.defstruct(uuid.uuid4().hex, [("element", self.type_)])

    @classmethod
    def make(cls, type_: type[T], tempdir: Path) -> Self:
        assert tempdir.is_dir() or not tempdir.exists(), f"Tempdir {tempdir} is not a directory"
        tempdir.mkdir(parents=True, exist_ok=True)
        type_path = tempdir / cls.TYPE_PATH_NAME
        with smart_open.open(type_path, "wb") as f:  # pyright: ignore
            pickle.dump(type_, f)

        return cls(type_=type_, tempdir=tempdir)

    @classmethod
    def load(cls, tempdir: Path) -> Self:
        type_path = tempdir / cls.TYPE_PATH_NAME
        if not type_path.exists():
            raise FileNotFoundError(f"Type path {type_path} does not exist")
        with smart_open.open(type_path, "rb") as f:  # pyright: ignore
            type_ = pickle.load(f)
        return cls(type_=type_, tempdir=tempdir)

    @classmethod
    def make_or_load(cls, type_: type[T], tempdir: Path) -> Self:
        try:
            return cls.load(tempdir)
        except FileNotFoundError:
            return cls.make(type_, tempdir)


@dt.dataclass(frozen=True)
class ChunkedDatasetPath(Generic[T]):
    chunk_path: Path

    def __iter__(self) -> Generator[T, None, None]:
        struct_manager = ChunkedDatasetStructManager.load(self.chunk_path.parent)
        decoder = msgspec.json.Decoder(struct_manager.struct_cls)
        with smart_open.open(self.chunk_path, "rb") as f:  # pyright: ignore
            for line in f:
                row = decoder.decode(line)
                yield getattr(row, "element")


@dt.dataclass(frozen=True)
class ChunkedDataset(Generic[T]):
    tempdir: Path | None = None

    def get_tempdir(self) -> Path:
        assert self.tempdir is not None, "Tempdir not created, have you entered the context?"
        return self.tempdir

    def add_chunk(self, chunk: list[T], encoder: msgspec.json.Encoder | None = None):
        element_type = type(chunk[0])
        struct_cls = ChunkedDatasetStructManager.make_or_load(element_type, self.get_tempdir()).struct_cls
        encoder = encoder or msgspec.json.Encoder()
        chunk_path = self.get_tempdir() / f"chunk_{uuid.uuid4()}.jsonl.zst"
        with smart_open.open(chunk_path, "wb") as f:  # pyright: ignore
            for element in chunk:
                f.write(encoder.encode(struct_cls(element=element)) + b"\n")

    def add_dataset(
        self,
        dataset: list[T],
        chunk_size: int | float | None = None,
        chunk_number: int | None = None,
    ):
        if chunk_size is not None:
            chunk_size = float(chunk_size)
        elif chunk_number is not None:
            chunk_size = float(len(dataset) / chunk_number)
        else:
            raise ValueError("Either chunk_size or chunk_number must be provided")

        start_idx = 0
        encoder = msgspec.json.Encoder()
        with tqdm(total=len(dataset), desc="Chunking dataset", unit_scale=True) as pbar:
            while start_idx < len(dataset):
                end_idx = round(start_idx + chunk_size)
                chunk = dataset[start_idx:end_idx]
                self.add_chunk(chunk=chunk, encoder=encoder)
                start_idx = end_idx
                pbar.update(len(chunk))
            pbar.close()

    def __enter__(self):
        return ChunkedDataset(tempdir=Path(tempfile.mkdtemp()))

    def __len__(self) -> int:
        assert self.tempdir is not None, "Tempdir not created, have you entered the context?"
        return len(list(self.tempdir.iterdir()))

    def __iter__(self) -> Generator[ChunkedDatasetPath[T], None, None]:
        assert self.tempdir is not None, "Tempdir not created, have you entered the context?"
        try:
            struct_manager = ChunkedDatasetStructManager.load(self.tempdir)
        except FileNotFoundError:
            raise FileNotFoundError("Tempdir does not contain a valid chunked dataset")

        for chunk_path in self.tempdir.iterdir():
            if chunk_path.is_file() and chunk_path.name != struct_manager.TYPE_PATH_NAME:
                yield ChunkedDatasetPath(chunk_path=chunk_path)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tempdir is not None:
            shutil.rmtree(self.tempdir)


@dt.dataclass(frozen=True)
class FasttextElement:
    text: str
    label: str


@dt.dataclass(frozen=True)
class FasttextDatasetSplit:
    path: Path

    def __len__(self) -> int:
        # count lines in file
        with smart_open.open(self.path, "rt", encoding="utf-8") as f:  # pyright: ignore
            return sum(1 for _ in f)

    def __iter__(self) -> Generator[FasttextElement, None, None]:
        with smart_open.open(self.path, "rt", encoding="utf-8") as f:  # pyright: ignore
            for line in f:
                label, text = line.strip().split(" ", 1)
                yield FasttextElement(text=text, label=label)

    @classmethod
    def merge(cls, splits: list[Self], split_name: str, tempdir: Path | None = None) -> Self:
        if len(splits) == 0:
            raise ValueError("No splits provided")
        elif len(splits) == 1:
            return splits[0]
        elif tempdir is None:
            raise ValueError("Tempdir is required if there are multiple splits")

        tempdir.mkdir(parents=True, exist_ok=True)
        split_path = tempdir / f"{split_name}.txt"
        with smart_open.open(split_path, "wt", encoding="utf-8") as f:  # pyright: ignore
            for split in splits:
                for element in split:
                    f.write(f"{element.text} {element.label}\n")
        return cls(path=split_path)


@dt.dataclass(frozen=True)
class FasttextDataset:
    train: FasttextDatasetSplit
    test: FasttextDatasetSplit
    valid: FasttextDatasetSplit | None = None

    @classmethod
    def from_splits(
        cls,
        train: list[FasttextDatasetSplit],
        test: list[FasttextDatasetSplit],
        valid: list[FasttextDatasetSplit] | None = None,
        tempdir: Path | None = None,
    ) -> Self:
        train_split = FasttextDatasetSplit.merge(splits=train, split_name="train", tempdir=tempdir)
        test_split = FasttextDatasetSplit.merge(splits=test, split_name="test", tempdir=tempdir)
        valid_split = (
            FasttextDatasetSplit.merge(splits=valid, split_name="valid", tempdir=tempdir)
            if valid is not None
            else None
        )
        return cls(train=train_split, test=test_split, valid=valid_split)


def load_fasttext_dataset(
    dataset_dirs: list[Path],
    tempdir: Path,
) -> FasttextDataset:
    collected_splits: dict[str, list[FasttextDatasetSplit]] = {}

    for dataset_dir in dataset_dirs:
        for split_name in ("train", "test", "valid"):
            split_path = dataset_dir / f"{split_name}.txt"

            if not split_path.exists():
                # skip if the split does not exist
                continue

            collected_splits.setdefault(split_name, []).append(FasttextDatasetSplit(path=split_path))

    return FasttextDataset.from_splits(
        train=collected_splits.get("train", []),
        test=collected_splits.get("test", []),
        valid=valid_split if len(valid_split := collected_splits.get("valid", [])) > 0 else None,
        tempdir=tempdir,
    )


def pretty_size(size: int | float, precision: int = 2, unit: str = "B") -> str:
    mappings = [unit, f"K{unit}", f"M{unit}", f"G{unit}", f"T{unit}", f"P{unit}"]
    i = 0
    while i < len(mappings):
        if size < 1024**i:
            break
        i += 1

    return f"{size / 1024 ** (i - 1):.{precision}f} {mappings[i - 1]}"
