# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run CLI commands
uv run bonepick <command>
uv run bonepick --help

# Run tests
uv run pytest
uv run pytest tests/test_indent_utils.py  # single file
uv run pytest -k "test_name"              # single test

# Format code
uv run ruff format .

# Commit (Claude must add co-author)
git commit -m "message

Co-Authored-By: Claude <model> <noreply@anthropic.com>"
```

**Important**: Claude must always add co-author trailer to commits, including the model name (e.g., `Claude Opus 4.5`).

## Architecture

CLI tool for training efficient text quality classifiers (Model2Vec and FastText) on text data.

### Module Structure

```
src/bonepick/
├── __init__.py      # Command registration hub (all commands added to cli here)
├── cli.py           # Click CLI setup + custom param types (PathParamType, ByteSizeParamType, etc.)
├── data/
│   ├── commands.py  # Data pipeline CLI commands
│   ├── utils.py     # DatasetSplit, DatasetTuple, file I/O helpers
│   ├── normalizers.py  # Text normalizer registry
│   └── expressions.py  # JQ expression utilities
├── model2vec/
│   ├── commands.py  # train/eval/distill commands
│   └── utils.py     # StaticModelForClassification, StaticModelForRegression
├── fasttext/
│   ├── commands.py  # train/eval/infer commands
│   └── utils.py     # FastText binary helpers
├── evals/
│   ├── commands.py  # eval-calibration command
│   └── utils.py     # Shared metrics computation
└── annotate/        # Optional LLM annotation (requires --extra annotate)
    ├── annotate_loop.py
    ├── analysis_loop.py
    └── prompt_collections/
```

### Code Style

- **CLI pattern**: Each command is a `@click.command()` function in a `commands.py` file, registered in `__init__.py`
- **Custom Click types**: Use `PathParamType`, `FloatOrIntParamType`, `ByteSizeParamType` from `cli.py`
- **Parallel processing**: Use `ProcessPoolExecutor`/`ThreadPoolExecutor` with `as_completed()` pattern
- **File I/O**: Use `smart_open` for transparent compression (.zst, .gz) support
- **JSON**: Use `msgspec.json.Encoder/Decoder` for fast serialization
- **JQ expressions**: Use `compile_jq()` from `data/expressions.py` for field transforms
- **Options pattern**: Use `multiple=True` for options that accept multiple values (e.g., `-d/--dataset-dir`)

### Data Pipeline

1. `import-hf-dataset` - Downloads HuggingFace datasets to local JSONL
2. `transform-dataset` - Applies jq expressions to reshape fields
3. `balance-dataset` - Balances datasets by label
4. `sample-dataset` - Random sampling by rate or target size
5. `reshard-dataset` - Combines files into evenly-sized shards
6. `normalize-dataset` - Text normalization
7. `convert-to-fasttext` - Converts JSONL to FastText format
8. `count-tokens` - Token counting with specified tokenizer

### Data Format

- Compressed JSONL (`.jsonl.zst`, `.jsonl.gz`, `.jsonl`) in `train/` and `test/` subdirectories
- Each row needs `text` and `label` fields (configurable via `--text-field`, `--label-field`)
- Multiple `-d/--dataset-dir` options supported for combining datasets

### Key Data Structures

- `DatasetSplit`: Holds parallel `text` and `label` lists for a single split
- `DatasetTuple`: Contains `train`, `valid`, `test` splits with signature hashing

## Normalizers

`whitespace`, `plsfix`, `tokenizer`, `ultrafine`, `hyperfine`, `hyperfine-code`, `potion`, `potion-code`

## Testing

- Test data: `tests/data/`
- Test output: `tests/output/` (gitignored)

## Tips

- For Model2Vec: normalize BEFORE training
- For FastText: normalize during `convert-to-fasttext`
