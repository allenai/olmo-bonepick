from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import cast
from contextlib import ExitStack

import torch
from model2vec.train import StaticModelForClassification
from model2vec.train.classifier import LabelType
from model2vec.train.base import TextDataset
from tokenizers import Tokenizer
from tqdm import tqdm

from bonepick.train.data_utils import ChunkedDataset, ChunkedDatasetPath

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

_RANDOM_SEED = 42


class BetterStaticModelForClassification(StaticModelForClassification):
    @staticmethod
    def _tokenize_chunk(
        chunk_file: ChunkedDatasetPath[str],
        tokenizer_json: str,
        max_length: int,
        truncate_length: int,
        output_chunks: ChunkedDataset,
        # encode_batch_size: int = 20_000,
    ) -> int:
        """Tokenize a chunk of texts in a separate process."""

        tokenizer = Tokenizer.from_str(tokenizer_json)
        tokenizer.enable_truncation(max_length=max_length)

        to_tokenize_batch = [text[:truncate_length] for text in chunk_file]
        batch_output = tokenizer.encode_batch_fast(to_tokenize_batch, add_special_tokens=False)
        output_chunks.add_chunk([tokens_sequence.ids for tokens_sequence in batch_output])
        return len(to_tokenize_batch)

    def _faster_tokenize(
        self,
        X: list[str],
        max_length: int = 512,
        num_proc: int | None = None,
        max_chunk_size: int = 20_000
    ) -> list[list[int]]:
        if num_proc is None:
            num_proc = os.cpu_count() or 1

        truncate_length = max_length * 10
        n_samples = len(X)
        tokenized: list[list[int]] = []

        # Parallel tokenization for large datasets
        tokenizer_json = self.tokenizer.to_str()
        chunk_size = min(max_chunk_size, (n_samples + num_proc - 1) // num_proc)

        with ExitStack() as stack:
            input_chunks = stack.enter_context(ChunkedDataset())
            input_chunks.add_dataset(X, chunk_size=chunk_size)
            output_chunks = stack.enter_context(ChunkedDataset())
            del X
            pbar = stack.enter_context(tqdm(total=n_samples, desc="Tokenizing dataset", unit_scale=True))
            pool = stack.enter_context(
                ProcessPoolExecutor(max_workers=num_proc) if num_proc > 1 else ThreadPoolExecutor(max_workers=1)
            )
            futures = []

            tokenize_fn = partial(
                self._tokenize_chunk,
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

    def _prepare_dataset(self, X: list[str], y: "LabelType", max_length: int = 512) -> "TextDataset":
        """
        Prepare a dataset. For multilabel classification, each target is converted into a multi-hot vector.

        :param X: The texts.
        :param y: The labels.
        :param max_length: The maximum length of the input.
        :return: A TextDataset.
        """
        # This is a speed optimization.
        # assumes a mean token length of 10, which is really high, so safe.
        tokenized = self._faster_tokenize(X, max_length=max_length)
        if self.multilabel:
            # Convert labels to multi-hot vectors
            num_classes = len(self.classes_)
            labels_tensor = torch.zeros(len(y), num_classes, dtype=torch.float)
            mapping = {label: idx for idx, label in enumerate(self.classes_)}
            for i, sample_labels in enumerate(y):
                indices = [mapping[label] for label in sample_labels]
                labels_tensor[i, indices] = 1.0
        else:
            labels_tensor = torch.tensor([self.classes_.index(label) for label in cast(list[str], y)], dtype=torch.long)
        return TextDataset(tokenized, labels_tensor)
