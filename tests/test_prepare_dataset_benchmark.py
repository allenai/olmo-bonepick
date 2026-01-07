"""Benchmark tests for _prepare_contrastive_dataset."""

import time
import numpy as np
import pytest

from bonepick.train.train_utils import HingeLossModelForClassification


def generate_fake_data(n_samples: int, text_length: int = 5000) -> tuple[list[str], list[str], np.ndarray]:
    """Generate fake data for benchmarking.

    Default text_length=5000 chars (~1000 words) simulates realistic web documents.
    """
    # Generate random text (words)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "cat",
             "runs", "fast", "slow", "big", "small", "good", "bad", "new", "old", "first",
             "second", "third", "last", "next", "best", "worst", "more", "less", "very",
             "much", "little", "great", "poor", "rich", "high", "low", "long", "short"]

    rng = np.random.default_rng(42)
    texts = []
    for _ in range(n_samples):
        n_words = text_length // 5  # ~5 chars per word
        text = " ".join(rng.choice(words, size=n_words))
        texts.append(text)

    # Generate binary labels
    labels = [str(rng.integers(0, 2)) for _ in range(n_samples)]

    # Generate cluster IDs
    n_clusters = 100
    cluster_ids = rng.integers(0, n_clusters, size=n_samples)

    return texts, labels, cluster_ids


@pytest.fixture
def model():
    """Create a model instance for testing."""
    model = HingeLossModelForClassification.from_pretrained(
        model_name="minishlab/potion-base-8M",  # smaller model for faster tests
        n_clusters=100,
    )
    # Initialize with dummy labels to set up classes_
    model._initialize(["0", "1"])
    return model


@pytest.mark.parametrize("n_samples", [1000, 10000, 100000])
def test_prepare_contrastive_dataset_benchmark(model, n_samples: int):
    """Benchmark _prepare_contrastive_dataset with different dataset sizes."""
    texts, labels, cluster_ids = generate_fake_data(n_samples)

    start_time = time.perf_counter()
    dataset = model._prepare_contrastive_dataset(texts, labels, cluster_ids)
    elapsed = time.perf_counter() - start_time

    # Print timing info
    print(f"\n_prepare_contrastive_dataset with {n_samples:,} samples: {elapsed:.3f}s ({n_samples/elapsed:.0f} samples/sec)")

    # Basic correctness checks
    assert len(dataset) == n_samples
    assert len(dataset.tokenized) == n_samples
    assert len(dataset.labels) == n_samples
    assert len(dataset.cluster_ids) == n_samples


@pytest.mark.parametrize("n_samples", [10000])
def test_prepare_contrastive_dataset_profile(model, n_samples: int):
    """Profile individual steps of _prepare_contrastive_dataset."""
    # Use very long texts (50k chars) to simulate real web documents
    texts, labels, cluster_ids = generate_fake_data(n_samples, text_length=50000)

    # Profile tokenization WITHOUT pre-truncation
    model.tokenizer.enable_truncation(max_length=512)
    start = time.perf_counter()
    tokenizer_output = model.tokenizer.encode_batch_fast(texts, add_special_tokens=False)
    tokenize_time = time.perf_counter() - start

    # Profile tokenization WITH pre-truncation (chars)
    max_length = 512
    truncate_length = max_length * 10  # ~10 chars per token estimate
    texts_truncated = [t[:truncate_length] for t in texts]
    start = time.perf_counter()
    tokenizer_output_trunc = model.tokenizer.encode_batch_fast(texts_truncated, add_special_tokens=False)
    tokenize_time_trunc = time.perf_counter() - start

    # Profile extracting token IDs
    start = time.perf_counter()
    tokens_ids = [tokens_sequence.ids for tokens_sequence in tokenizer_output]
    extract_time = time.perf_counter() - start

    # Profile label conversion (current implementation)
    start = time.perf_counter()
    import torch
    labels_tensor = torch.tensor([model.classes_.index(str(label)) for label in labels], dtype=torch.long)
    label_time = time.perf_counter() - start

    # Profile label conversion (optimized with dict)
    start = time.perf_counter()
    class_to_idx = {cls: idx for idx, cls in enumerate(model.classes_)}
    labels_tensor_opt = torch.tensor([class_to_idx[str(label)] for label in labels], dtype=torch.long)
    label_time_opt = time.perf_counter() - start

    print(f"\nProfile for {n_samples:,} samples:")
    print(f"  Tokenization (no pre-trunc):  {tokenize_time:.3f}s")
    print(f"  Tokenization (pre-truncated): {tokenize_time_trunc:.3f}s  ({tokenize_time/tokenize_time_trunc:.1f}x faster)")
    print(f"  Extract token IDs:            {extract_time:.3f}s")
    print(f"  Label conversion (list):      {label_time:.3f}s")
    print(f"  Label conversion (dict):      {label_time_opt:.3f}s")

    # Verify both approaches produce same result
    assert torch.equal(labels_tensor, labels_tensor_opt)


@pytest.mark.parametrize("n_samples", [10000])
def test_prepare_contrastive_dataset_long_texts(model, n_samples: int):
    """Benchmark with very long texts (50k chars) to verify pre-truncation optimization."""
    texts, labels, cluster_ids = generate_fake_data(n_samples, text_length=50000)

    start_time = time.perf_counter()
    dataset = model._prepare_contrastive_dataset(texts, labels, cluster_ids)
    elapsed = time.perf_counter() - start_time

    print(f"\n_prepare_contrastive_dataset with {n_samples:,} samples (50k chars each): {elapsed:.3f}s ({n_samples/elapsed:.0f} samples/sec)")

    assert len(dataset) == n_samples


@pytest.mark.parametrize("n_samples", [50000])
def test_prepare_contrastive_dataset_multiprocessing(model, n_samples: int):
    """Compare single-process vs multi-process tokenization."""
    import os
    texts, labels, cluster_ids = generate_fake_data(n_samples, text_length=5000)

    # Single process
    start_time = time.perf_counter()
    dataset1 = model._prepare_contrastive_dataset(texts, labels, cluster_ids, num_proc=1)
    single_time = time.perf_counter() - start_time

    # Multi-process (use all CPUs)
    num_cpus = os.cpu_count() or 1
    start_time = time.perf_counter()
    dataset2 = model._prepare_contrastive_dataset(texts, labels, cluster_ids, num_proc=num_cpus)
    multi_time = time.perf_counter() - start_time

    speedup = single_time / multi_time if multi_time > 0 else 0

    print(f"\n_prepare_contrastive_dataset with {n_samples:,} samples:")
    print(f"  Single process: {single_time:.3f}s ({n_samples/single_time:.0f} samples/sec)")
    print(f"  Multi process ({num_cpus} CPUs): {multi_time:.3f}s ({n_samples/multi_time:.0f} samples/sec)")
    print(f"  Speedup: {speedup:.2f}x")

    assert len(dataset1) == n_samples
    assert len(dataset2) == n_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
