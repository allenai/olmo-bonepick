import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from functools import partial

from tokenizers import Tokenizer
from tqdm import tqdm

from bonepick.data.utils import ChunkedDataset, ChunkedDatasetPath


def tokenize_chunk(
    chunk_file: ChunkedDatasetPath[str],
    tokenizer_json: str,
    max_length: int,
    truncate_length: int,
    output_chunks: ChunkedDataset[list[int]],
) -> int:
    """Tokenize a chunk of texts in a separate process.

    :param chunk_file: The chunk of texts to tokenize.
    :param tokenizer_json: The tokenizer JSON string.
    :param max_length: The maximum token length.
    :param truncate_length: The character truncation length before tokenizing.
    :param output_chunks: The output dataset to write tokenized results to.
    :return: The number of texts tokenized.
    """
    tokenizer = Tokenizer.from_str(tokenizer_json)
    tokenizer.enable_truncation(max_length=max_length)

    to_tokenize_batch = [text[:truncate_length] for text in chunk_file]
    batch_output = tokenizer.encode_batch_fast(to_tokenize_batch, add_special_tokens=False)
    output_chunks.add_chunk([tokens_sequence.ids for tokens_sequence in batch_output])
    return len(to_tokenize_batch)


def parallel_tokenize(
    texts: list[str],
    tokenizer: Tokenizer,
    max_length: int = 512,
    num_proc: int | None = None,
    max_chunk_size: int = 20_000,
) -> list[list[int]]:
    """Tokenize texts in parallel using multiple processes.

    :param texts: The list of texts to tokenize.
    :param tokenizer: The tokenizer to use.
    :param max_length: The maximum token length.
    :param num_proc: Number of processes to use. Defaults to CPU count.
    :param max_chunk_size: Maximum size of each chunk.
    :return: List of tokenized texts (each is a list of token IDs).
    """
    if num_proc is None:
        num_proc = os.cpu_count() or 1

    truncate_length = max_length * 10
    n_samples = len(texts)
    tokenized: list[list[int]] = []

    tokenizer_json = tokenizer.to_str()
    chunk_size = min(max_chunk_size, (n_samples + num_proc - 1) // num_proc)

    with ExitStack() as stack:
        input_chunks = stack.enter_context(ChunkedDataset())
        input_chunks.add_dataset(texts, chunk_size=chunk_size)
        output_chunks = stack.enter_context(ChunkedDataset())
        del texts
        pbar = stack.enter_context(tqdm(total=n_samples, desc="Tokenizing dataset", unit_scale=True))
        pool = stack.enter_context(
            ProcessPoolExecutor(max_workers=num_proc) if num_proc > 1 else ThreadPoolExecutor(max_workers=1)
        )
        futures = []

        tokenize_fn = partial(
            tokenize_chunk,
            tokenizer_json=tokenizer_json,
            max_length=max_length,
            truncate_length=truncate_length,
            output_chunks=output_chunks,
        )

        for chunk in input_chunks:
            future = pool.submit(tokenize_fn, chunk)
            futures.append(future)
        for future in as_completed(futures):
            n_processed = future.result()
            pbar.update(n_processed)

        pbar.close()

        for chunk in tqdm(output_chunks, desc="Loading tokenized chunks"):
            tokenized.extend(chunk)

    return tokenized
