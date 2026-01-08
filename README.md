
<p align="center">
  <img src="assets/logo.png" alt="Olmo Bonepick library logo" width="500"/>
</p>



`bonepick` trains efficient text quality classifiers that run on GPU. It supports training [**Model2Vec**][1] (static embeddings) and [**FastText**][2] classifiers.

⚠️ **Warning: Claude-generated documentation below.** ⚠️

## Installation

```shell
uv sync
```

### Optional Dependencies

The `annotate` extra provides tools for using LLM APIs to label data:

```shell
uv sync --extra annotate
```

**Note:** Support for the `annotate` feature is coming soon. This will include tools to classify and label text data using various LLM providers (OpenAI, Anthropic, etc.).

## Data Format

Datasets are stored as compressed JSONL files (`.jsonl.zst`, `.jsonl.gz`, or `.jsonl`) in `train/` and `test/` subdirectories. Each row must have a text field and a label field.

```
dataset/
├── train/
│   ├── shard_0.jsonl.zst
│   └── shard_100000.jsonl.zst
└── test/
    └── shard_0.jsonl.zst
```

## Data Preparation Pipeline

### 1. Import from HuggingFace

Download a HuggingFace dataset to local JSONL format:

```shell
uv run bonepick import-hf-dataset \
    -n HuggingFaceFW/fineweb-edu-llama3-annotations \
    -o data/fineweb-edu-llama3-annotations \
    --test-split 0.1
```

### 2. Transform Labels (Optional)

Use jq expressions to reshape fields. Common use case: binarize multi-class labels.

```shell
# Binarize scores: 0-1 → 0 (low quality), 2-5 → 1 (high quality)
uv run bonepick transform-dataset \
    --input-dir data/fineweb-edu-llama3-annotations \
    --output-dir data/fineweb-edu-binary \
    -l '{score: (if .score < 2 then 0 else 1 end)}'

# Or use string labels
uv run bonepick transform-dataset \
    --input-dir data/fineweb-edu-llama3-annotations \
    --output-dir data/fineweb-edu-binary \
    -l '{score: (if .score < 2 then "neg" else "pos" end)}'
```

### 3. Balance Dataset (Optional)

Balance the dataset so each label has equal representation. Useful when one class significantly outnumbers others:

```shell
uv run bonepick balance-dataset \
    --input-dir data/fineweb-edu-binary \
    --output-dir data/fineweb-edu-binary-balanced \
    --seed 42
```

Supports multiple input directories:

```shell
uv run bonepick balance-dataset \
    -i data/dataset1 \
    -i data/dataset2 \
    -o data/combined-balanced \
    --seed 42
```

### 4a. Normalize Text (for Model2Vec)

Apply text normalization before training Model2Vec classifiers:

```shell
uv run bonepick normalize-dataset \
    --input-dir data/fineweb-edu-binary \
    --output-dir data/fineweb-edu-binary-normalized \
    -n plsfix
```

Available normalizers: `whitespace`, `plsfix`, `tokenizer`, `ultrafine`, `ultrafine-plus`

### 4b. Convert to FastText Format (for FastText)

Convert JSONL to FastText's `__label__<label> <text>` format:

```shell
uv run bonepick convert-to-fasttext \
    --input-dir data/fineweb-edu-binary \
    --output-dir data/fasttext-fineweb-edu-binary \
    -n ultrafine
```

## Training

### Model2Vec Classifier

Trains a classifier head on top of frozen Model2Vec static embeddings:

```shell
uv run bonepick train-model2vec \
    -d data/fineweb-edu-binary-normalized \
    -o models/model2vec-classifier
```

### Model2Vec with Contrastive Loss

Clusters documents by semantic similarity, then trains using pairwise hinge loss within clusters. Better for ranking/quality scoring:

```shell
uv run bonepick train-contrastive \
    --dataset-dir data/fineweb-edu-binary-normalized \
    --output-dir models/contrastive-classifier \
    --n-clusters 100
```

### FastText Classifier

Trains a FastText classifier (requires `fasttext` binary in PATH):

```shell
uv run bonepick train-fasttext \
    -d data/fasttext-fineweb-edu-binary \
    -o models/fasttext-classifier
```

### Training on Multiple Datasets

All training commands support combining data from multiple directories using repeated `-d` flags:

```shell
# Combine multiple datasets for training
uv run bonepick train-contrastive \
    -d data/dataset1-normalized \
    -d data/dataset2-normalized \
    -d data/dataset3-normalized \
    -o models/combined-classifier
```

Data from all directories is concatenated before training. Each directory must have `train/` and `test/` subdirectories.

## Evaluation

Both evaluation commands compute detailed classification metrics using probability predictions (`predict_proba` for Model2Vec, `predict-prob` for FastText). Results include precision, recall, F1-score, and AUC for each class, plus macro averages.

### Model2Vec Evaluation

```shell
uv run bonepick eval-model2vec \
    -d data/fineweb-edu-binary-normalized \
    -m models/contrastive-classifier \
    --text-field text \
    --label-field score
```

### FastText Evaluation

```shell
uv run bonepick eval-fasttext \
    -d data/fasttext-fineweb-edu-binary \
    -m models/fasttext-classifier \
    --text-field text \
    --label-field score
```

### Multi-Dataset Evaluation

Evaluate on multiple datasets simultaneously. Results are computed on the combined test sets:

```shell
uv run bonepick eval-model2vec \
    -d data/dataset1-normalized \
    -d data/dataset2-normalized \
    -d data/dataset3-normalized \
    -m models/combined-classifier
```

### Output Format

Results are saved as YAML files in the model directory with the naming pattern `results_<dataset_signature>.yaml`:

```yaml
dataset_dir:
  - data/fineweb-edu-binary-normalized
model_dir: models/contrastive-classifier
overall_results:
  macro_precision: 0.8734
  macro_recall: 0.8621
  macro_f1: 0.8677
  macro_auc: 0.9245
per_class_metrics:
  - class_name: '0'
    precision: 0.8512
    recall: 0.8823
    f1: 0.8665
    support: 1523
    auc: 0.9245
  - class_name: '1'
    precision: 0.8956
    recall: 0.8419
    f1: 0.8679
    support: 1477
    auc: 0.9245
```

### Metrics Explained

- **Precision**: Of all predictions for a class, how many were correct
- **Recall**: Of all actual instances of a class, how many were predicted correctly
- **F1**: Harmonic mean of precision and recall
- **AUC**: Area Under the ROC Curve (one-vs-rest for multi-class)
- **Macro averages**: Unweighted mean across all classes
- **Support**: Number of true instances for each class in the test set

### Custom Field Names

Both evaluation commands support custom field names if your dataset uses different column names:

```shell
uv run bonepick eval-model2vec \
    -d data/custom-dataset \
    -m models/my-classifier \
    --text-field document \
    --label-field quality_score
```

## CLI Reference

```shell
uv run bonepick --help
uv run bonepick <command> --help
```

| Command | Description |
|---------|-------------|
| `import-hf-dataset` | Download HuggingFace dataset to local JSONL |
| `transform-dataset` | Apply jq transforms to reshape fields |
| `balance-dataset` | Balance dataset so each label has equal representation |
| `normalize-dataset` | Normalize text (for Model2Vec) |
| `convert-to-fasttext` | Convert JSONL to FastText format |
| `train-model2vec` | Train Model2Vec classifier |
| `train-contrastive` | Train Model2Vec with contrastive/hinge loss |
| `train-fasttext` | Train FastText classifier |
| `eval-model2vec` | Evaluate Model2Vec classifier |
| `eval-fasttext` | Evaluate FastText classifier |

[1]: https://github.com/MinishLab/model2vec
[2]: https://fasttext.cc
