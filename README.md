
<p align="center">
  <img src="assets/logo.png" alt="Olmo Bonepick library logo" width="400"/>
</p>



`bonepick` trains efficient text quality classifiers that run on GPU. It supports training [**Model2Vec**][1] (static embeddings) and [**FastText**][2] classifiers.

⚠️ **Warning: Claude-generated documentation below.** ⚠️

## Installation

```shell
uv sync
```

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

```shell
# Evaluate Model2Vec classifier
uv run bonepick eval-model2vec \
    -d data/fineweb-edu-binary-normalized \
    -m models/contrastive-classifier

# Evaluate FastText classifier
uv run bonepick eval-fasttext \
    -d data/fasttext-fineweb-edu-binary \
    -m models/fasttext-classifier

# Evaluate on multiple datasets (results combined)
uv run bonepick eval-model2vec \
    -d data/dataset1-normalized \
    -d data/dataset2-normalized \
    -m models/combined-classifier
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


## Example

#### Step 1: manage data

##### Step 1a: import FineWeb dataset

```shell
uv run bonepick import-hf-dataset \
    --name HuggingFaceFW/fineweb-edu-llama3-annotations \
    --output-dir tmp/data/fineweb-edu-llama3-annotations \
    --test-split 10000
```

##### Step 1b: copy FineWeb++ annotations

```shell
s5cmd cp -sp \
    's3://ai2-llm/pretraining-data/sources/WebOrganizer/v0/batch_api/fw_pp_ref_mini_o4-mini-batch_medium/quality_jun1/retrieved/*' \
    tmp/data/fw_pp_ref_mini_o4-mini-batch_medium

s5cmd cp -sp \
    's3://ai2-llm/pretraining-data/sources/WebOrganizer/v0/quality/fw_pp_ref/*' \
    tmp/data/fw_pp_ref
```

#### Step 2: binarized dataset

##### Step 2a: binarize fineweb

```shell
uv run bonepick transform-dataset \
    --input-dir tmp/data/fineweb-edu-llama3-annotations \
    --output-dir tmp/data/fineweb-edu-llama3-annotations-binary-pos-neg \
    --label-transform '{score: (if .score < 3 then "neg" else "pos" end)}'
```

##### Step 2b: binarize FineWeb++

```shell
uv run bonepick transform-dataset \
    --input-dir tmp/data/fw_pp_ref_mini_o4-mini-batch_medium \
    --output-dir tmp/data/fw_pp_ref_mini_o4-mini-batch_medium-pos-neg \
    --label-transform '{score: (if .metadata."fw_pp_ref_mini_o4-mini-batch_medium".score < 3 then "neg" else "pos" end)}'

uv run bonepick transform-dataset \
    --input-dir tmp/data/fw_pp_ref \
    --output-dir tmp/data/fw_pp_ref-pos-neg \
    --label-transform '{score: (if .metadata."fw_pp_ref".score < 3 then "neg" else "pos" end)}'
```

#### Step 3: convert and normalize data

##### Step 3a: FastText format, `ultrafine` normalizer

```shell
uv run bonepick convert-to-fasttext \
    --input-dir tmp/data/fineweb-edu-llama3-annotations-binary-pos-neg \
    --output-dir tmp/data/fasttext-fineweb-edu-llama3-annotations-binary-pos-neg-ultrafine \
    --normalization ultrafine

uv run bonepick convert-to-fasttext \
    --input-dir tmp/data/fw_pp_ref_mini_o4-mini-batch_medium-pos-neg \
    --output-dir tmp/data/fasttext-fw_pp_ref_mini_o4-mini-batch_medium-pos-neg-ultrafine \
    --normalization ultrafine

uv run bonepick convert-to-fasttext \
    --input-dir tmp/data/fw_pp_ref-pos-neg \
    --output-dir tmp/data/fasttext-fw_pp_ref-pos-neg-ultrafine \
    --normalization ultrafine
```

##### Step 3b: Model2Vec format, `potion` normalizer

```shell
uv run bonepick normalize-dataset \
    --input-dir tmp/data/fineweb-edu-llama3-annotations-binary-pos-neg \
    --output-dir tmp/data/fineweb-edu-llama3-annotations-binary-pos-neg-normalized-potion \
    -n potion

uv run bonepick normalize-dataset \
    --input-dir tmp/data/fw_pp_ref_mini_o4-mini-batch_medium-pos-neg \
    --output-dir tmp/data/fw_pp_ref_mini_o4-mini-batch_medium-pos-neg-normalized-potion \
    -n potion

uv run bonepick normalize-dataset \
    --input-dir tmp/data/fw_pp_ref-pos-neg \
    --output-dir tmp/data/fw_pp_ref-pos-neg-normalized-potion \
    -n potion
```

#### Step 4: train models


##### Step 4a: train FastText model (only fineweb)

```shell
uv run bonepick train-fasttext \
    --dataset-dir tmp/data/fasttext-fineweb-edu-llama3-annotations-binary-pos-neg-ultrafine \
    --output-dir tmp/models/fasttext-fineweb-edu-llama3-annotations-binary-pos-neg-ultrafine
```

##### Step 4b: train a Model2vec model (only fineweb)


```shell
uv run bonepick train-model2vec \
    --dataset-dir tmp/data/fineweb-edu-llama3-annotations-binary-pos-neg-normalized-potion \
    --output-dir tmp/models/potion-32M-fineweb-edu-llama3-annotations-binary-pos-neg-normalized-potion
```

##### Step 4c: train a Model2Vec model with extra annotations

```shell
uv run bonepick train-model2vec \
    --dataset-dir tmp/data/fineweb-edu-llama3-annotations-binary-pos-neg-normalized-potion \
    --dataset-dir tmp/data/fw_pp_ref_mini_o4-mini-batch_medium-pos-neg-normalized-potion \
    --dataset-dir tmp/data/fw_pp_ref-pos-neg-normalized-potion \
    --output-dir tmp/models/potion-32M-fw-fw_pp-fw_pp_o4-posneg-normalized-potion

```

#### Step 5: eval models

##### Step 5a: eval FastText model

```shell
uv run bonepick eval-fasttext \
    --dataset-dir tmp/data/fasttext-fineweb-edu-llama3-annotations-binary-pos-neg-ultrafine \
    --model-dir tmp/models/fasttext-fineweb-edu-llama3-annotations-binary-pos-neg-ultrafine
```

Results:

```text
Test results:
F1-Score : 0.964600  Precision : 0.946825  Recall : 0.983055   __label__neg
F1-Score : 0.513274  Precision : 0.691849  Recall : 0.407972   __label__pos
N       10000
P@1     0.934
R@1     0.934
```

##### Step 5b: eval Model2Vec model


```shell
uv run bonepick eval-model2vec \
    --dataset-dir tmp/data/fineweb-edu-llama3-annotations-binary-pos-neg-normalized-potion \
    --model-dir tmp/models/potion-32M-fineweb-edu-llama3-annotations-binary-pos-neg-normalized-potion
```

Results

```text
Evaluation results:
              precision    recall  f1-score   support

         neg       0.94      0.99      0.97      9147
         pos       0.80      0.33      0.47       853

    accuracy                           0.94     10000
   macro avg       0.87      0.66      0.72     10000
weighted avg       0.93      0.94      0.92     10000
```

##### Step 5c: eval UltraFineWeb

```shell
uv run --with=huggingface-hub \
    hf download openbmb/Ultra-FineWeb-classifier \
    --local-dir /tmp/openbmb/Ultra-FineWeb-classifier

mkdir -p tmp/models/openbmb_Ultra-FineWeb-classifier_en
mv /tmp/openbmb/Ultra-FineWeb-classifier/classifiers/ultra_fineweb_en.bin tmp/models/openbmb_Ultra-FineWeb-classifier_en/model.bin
rm -rf /tmp/openbmb

uv run bonepick eval-fasttext \
    --dataset-dir tmp/data/fasttext-fineweb-edu-llama3-annotations-binary-pos-neg-ultrafine \
    --model-dir tmp/models/openbmb_Ultra-FineWeb-classifier_en
```

Results


```text
Test results:
F1-Score : 0.834956  Precision : 0.736533  Recall : 0.963742   __label__neg
F1-Score : 0.356601  Precision : 0.744376  Recall : 0.234461   __label__pos
N       10000
P@1     0.737
R@1     0.737
```


#### Step 6: balance dataset

This ensures same number of positive and negative items

```shell
uv run bonepick balance-dataset \
    --input-dir tmp/data/fineweb-edu-llama3-annotations-binary-pos-neg-normalized-potion \
    --input-dir tmp/data/fw_pp_ref_mini_o4-mini-batch_medium-pos-neg-normalized-potion \
    --input-dir tmp/data/fw_pp_ref-pos-neg-normalized-potion \
    --output-dir tmp/data/fw-fw_pp-fw_pp_o4-posneg-normalized-potion-balanced
```

#### Step 7: train on balanced dataset

```shell
uv run bonepick train-model2vec \
    --dataset-dir tmp/data/fw-fw_pp-fw_pp_o4-posneg-normalized-potion-balanced \
    --output-dir tmp/models/potion-32M-fw-fw_pp-fw_pp_o4-posneg-normalized-potion-balanced
```
