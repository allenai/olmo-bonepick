# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Always use `uv run python3` instead of `python3` directly.

```bash
# Run CLI commands
uv run bonepick <command>

# Available commands
uv run bonepick --help
uv run bonepick import-hf-dataset --help
uv run bonepick transform-dataset --help
uv run bonepick balance-dataset --help
uv run bonepick normalize-dataset --help
uv run bonepick convert-to-fasttext --help
uv run bonepick train-model2vec --help
uv run bonepick train-contrastive --help
uv run bonepick train-fasttext --help
uv run bonepick eval-model2vec --help
uv run bonepick eval-fasttext --help
```

## Architecture

This is a CLI tool for training efficient quality classifiers (Model2Vec and FastText) on text data.

### Data Pipeline

1. **Import**: `import-hf-dataset` - Downloads HuggingFace datasets to local JSONL format with train/test splits
2. **Transform**: `transform-dataset` - Applies jq expressions to reshape fields (e.g., binarize labels)
3. **Balance**: `balance-dataset` - Balances datasets so each label has equal representation (supports multiple input directories)
4. **Normalize**: `normalize-dataset` - Applies text normalization (whitespace, plsfix, tokenizer-based, ultrafine, potion)
5. **Convert**: `convert-to-fasttext` - Converts JSONL to FastText format (`__label__<label> <text>`)

### Training Methods

- **train-model2vec**: Standard classification using Model2Vec static embeddings with sklearn-style `.fit()`
- **train-contrastive**: Contrastive/ranking training using hinge loss within semantic clusters (PyTorch Lightning)
- **train-fasttext**: Shells out to the fasttext binary for training

### Evaluation System

Both `eval-model2vec` and `eval-fasttext` compute detailed classification metrics:

- **Metrics**: Precision, recall, F1-score, and AUC for each class, plus macro averages
- **Implementation**:
  - Model2Vec: Uses `pipeline.predict_proba()` to get probability distributions
  - FastText: Shells out to `fasttext predict-prob` command with `-1` flag (all classes)
- **Label Encoding**: Uses `sklearn.preprocessing.LabelEncoder` for consistent label encoding
- **Output Format**: YAML files saved to model directory as `results_<dataset_signature>.yaml`
- **Multi-dataset**: Supports multiple `--dataset-dir` options; results computed on combined test sets

Key functions in `eval_loop.py`:
- `_compute_metrics_from_predictions()`: Shared helper that computes all metrics from probability predictions
- `compute_detailed_metrics()`: Model2Vec evaluation wrapper
- `compute_detailed_metrics_fasttext()`: FastText evaluation wrapper with subprocess handling
- `result_to_text()`: Formats results as YAML with dataset paths, macro metrics, and per-class breakdowns

### Key Components

- `train_loop.py`: Training CLI commands (`train-model2vec`, `train-contrastive`, `train-fasttext`)
- `eval_loop.py`: Evaluation CLI commands (`eval-model2vec`, `eval-fasttext`) with detailed probability-based metrics
- `model2vec_utils.py`: `HingeLossModelForClassification` - extends Model2Vec's `StaticModelForClassification` with contrastive training that clusters documents by semantic similarity, then trains with pairwise hinge loss
- `data.py`: Dataset loading, transformation, balancing, and format conversion CLI commands
- `data_utils.py`: Helper functions for file I/O, label counting, and sample reading; includes `load_jsonl_dataset()` and `load_fasttext_dataset()` with support for multiple dataset directories
- `normalizers.py`: Text normalizer registry with implementations (whitespace, plsfix, tokenizer, ultrafine, ultrafine-plus, potion)
- `fasttext_utils.py`: FastText binary detection and dataset signature utilities
- `cli.py`: Click CLI setup and custom parameter types

### Data Format

Datasets are stored as compressed JSONL files (`.jsonl.zst`) in `train/` and `test/` subdirectories. Each row must have text and label fields (configurable via `--text-field` and `--label-field`).

All training and evaluation commands support multiple `--dataset-dir` options to combine data from multiple directories.


### Testing

- Test data is stored in `tests/data/`
- Test output should be written to `tests/output/` (gitignored)

### Git commands

Use git commands directly, e.g., `git status` not `git -C /home/lucas/oe-data-internal/bonepick status`.
