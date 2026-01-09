# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Always use `uv run python3` instead of `python3` directly.

```bash
# Run CLI commands
uv run bonepick <command>

# Available commands
uv run bonepick --help

# Data Pipeline
uv run bonepick import-hf-dataset --help
uv run bonepick transform-dataset --help
uv run bonepick balance-dataset --help
uv run bonepick sample-dataset --help
uv run bonepick normalize-dataset --help
uv run bonepick convert-to-fasttext --help

# Training
uv run bonepick train-model2vec --help
uv run bonepick train-fasttext --help
uv run bonepick distill-model --help

# Evaluation
uv run bonepick eval-model2vec --help
uv run bonepick eval-fasttext --help

# Annotation (requires --extra annotate)
uv run bonepick annotate-dataset --help
uv run bonepick list-prompts --help

# Utility
uv run bonepick version
```

## Architecture

This is a CLI tool for training efficient quality classifiers (Model2Vec and FastText) on text data.

### Data Pipeline

1. **Import**: `import-hf-dataset` - Downloads HuggingFace datasets to local JSONL format with train/test splits
2. **Transform**: `transform-dataset` - Applies jq expressions to reshape fields (e.g., binarize labels)
3. **Balance**: `balance-dataset` - Balances datasets so each label has equal representation (supports multiple input directories)
4. **Sample**: `sample-dataset` - Creates a random sample of a dataset by sampling rate or target size (supports multiple input directories)
5. **Normalize**: `normalize-dataset` - Applies text normalization (whitespace, plsfix, tokenizer-based, ultrafine, potion)
6. **Convert**: `convert-to-fasttext` - Converts JSONL to FastText format (`__label__<label> <text>`)

### Training Methods

- **train-model2vec**: Standard classification using Model2Vec static embeddings with sklearn-style `.fit()`
- **train-fasttext**: Shells out to the fasttext binary for training
- **distill-model**: Distills a Sentence Transformer model to a Model2Vec static embedding model

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

### Annotation System (Optional)

Requires `uv sync --extra annotate` to enable. Uses `lm-deluge` library for async LLM requests.

- **annotate-dataset**: Annotates datasets using LLM APIs (OpenAI, etc.) with configurable prompts
- **list-prompts**: Lists available task and system prompts for annotation
- Key features: rate limiting, caching (SQLite), batch processing, supports both text and conversation formats

### Key Components

**Core Modules:**
- `train/train_loop.py`: Training CLI commands (`train-model2vec`, `train-fasttext`)
- `train/distill_loop.py`: Model distillation command (`distill-model`)
- `train/eval_loop.py`: Evaluation CLI commands (`eval-model2vec`, `eval-fasttext`) with detailed probability-based metrics
- `train/data_loop.py`: Dataset loading, transformation, balancing, sampling, and format conversion CLI commands
- `train/data_utils.py`: Helper functions for file I/O, label counting, sample reading, and file sampling; includes `load_jsonl_dataset()` and `load_fasttext_dataset()` with support for multiple dataset directories; `sample_single_file()` for random sampling
- `train/normalizers.py`: Text normalizer registry with implementations (whitespace, plsfix, tokenizer, ultrafine, ultrafine-plus, potion)
- `train/fasttext_utils.py`: FastText binary detection and dataset signature utilities

**Annotation Modules (Optional):**
- `annotate/annotate_loop.py`: Annotation CLI commands (`annotate-dataset`, `list-prompts`)
- `annotate/annotate_utils.py`: Annotation helper functions
- `annotate/prompts.py`: Base classes for annotation prompts
- `annotate/deluge_utils.py`: LM-deluge integration and caching utilities

**CLI Infrastructure:**
- `cli.py`: Click CLI setup and custom parameter types (PathParamType, FloatOrIntParamType, ByteSizeParamType, PCADimTypeParamType)
- `__init__.py`: Command registration hub

### Data Format

Datasets are stored as compressed JSONL files (`.jsonl.zst`) in `train/` and `test/` subdirectories. Each row must have text and label fields (configurable via `--text-field` and `--label-field`).

All training and evaluation commands support multiple `--dataset-dir` options to combine data from multiple directories.


### Data Format Details

- **Compression**: Files can be `.jsonl.zst`, `.jsonl.gz`, or `.jsonl`
- **Directory structure**: Must have `train/` and `test/` subdirectories
- **Field names**: Configurable via `--text-field` and `--label-field` (defaults: `text` and `label`)
- **Multiple datasets**: Most commands support multiple `-d/--dataset-dir` options to combine datasets

### Normalizers

Available text normalizers (used with `normalize-dataset` and `convert-to-fasttext`):
- `whitespace`: Basic whitespace normalization
- `plsfix`: PlsFix normalization
- `tokenizer`: Tokenizer-based normalization
- `ultrafine`: Ultrafine normalization
- `ultrafine-plus`: Enhanced ultrafine normalization
- `potion`: Potion normalization

### Model Types

**Model2Vec:**
- Static embeddings (no GPU needed for inference)
- Fast classification head training with sklearn
- Supports custom normalizers
- Probability-based evaluation

**FastText:**
- Requires `fasttext` binary in PATH
- Extremely fast training and inference
- N-gram character features
- Shell-based training and evaluation

### Testing

- Test data is stored in `tests/data/`
- Test output should be written to `tests/output/` (gitignored)

### Common Workflows

1. **Quick binary classifier**: import → transform (binarize) → normalize → train-model2vec → eval
2. **Balanced training**: import → transform → balance → normalize → train → eval
3. **Sampling for experiments**: import → sample → normalize → train → eval
4. **Distilling custom embeddings**: distill-model from Sentence Transformer → use in train-model2vec
5. **LLM annotation pipeline**: import → annotate-dataset → balance → normalize → train → eval

### Tips

- Always use `uv run bonepick` not `python -m bonepick` or direct python execution
- For Model2Vec, normalize BEFORE training (not during)
- For FastText, normalize during `convert-to-fasttext`
- Use `--help` on any command to see all options
- Evaluation results are saved as YAML in the model directory
- Multiple dataset directories are concatenated before processing
- jq expressions in `transform-dataset` can reshape any field structure
