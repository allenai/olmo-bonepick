# WORKLOG

This is where I ([@soldni](https://soldaini.net)) keep track of my work on the project.

## General Quality

### Step 1: manage data

#### Step 1a: import FineWeb dataset

```shell
uv run bonepick import-hf-dataset \
    --name HuggingFaceFW/fineweb-edu-llama3-annotations \
    --output-dir tmp/data/fineweb-edu-llama3-annotations \
    --test-split 10000
```

#### Step 1b: copy FineWeb++ annotations

```shell
s5cmd cp -sp \
    's3://ai2-llm/pretraining-data/sources/WebOrganizer/v0/batch_api/fw_pp_ref_mini_o4-mini-batch_medium/quality_jun1/retrieved/*' \
    tmp/data/fw_pp_ref_mini_o4-mini-batch_medium

s5cmd cp -sp \
    's3://ai2-llm/pretraining-data/sources/WebOrganizer/v0/quality/fw_pp_ref/*' \
    tmp/data/fw_pp_ref
```

### Step 2: binarized dataset

#### Step 2a: binarize fineweb

```shell
uv run bonepick transform-dataset \
    --input-dir tmp/data/fineweb-edu-llama3-annotations \
    --output-dir tmp/data/fineweb-edu-llama3-annotations-binary-pos-neg \
    --label-transform '{score: (if .score < 3 then "neg" else "pos" end)}'
```

#### Step 2b: binarize FineWeb++

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

### Step 3: convert and normalize data

#### Step 3a: FastText format, `ultrafine` normalizer

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

#### Step 3b: Model2Vec format, `potion` normalizer

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

### Step 4: train models


#### Step 4a: train FastText model (only fineweb)

```shell
uv run bonepick train-fasttext \
    --dataset-dir tmp/data/fasttext-fineweb-edu-llama3-annotations-binary-pos-neg-ultrafine \
    --output-dir tmp/models/fasttext-fineweb-edu-llama3-annotations-binary-pos-neg-ultrafine
```

#### Step 4b: train a Model2vec model (only fineweb)


```shell
uv run bonepick train-model2vec \
    --dataset-dir tmp/data/fineweb-edu-llama3-annotations-binary-pos-neg-normalized-potion \
    --output-dir tmp/models/potion-32M-fineweb-edu-llama3-annotations-binary-pos-neg-normalized-potion
```

#### Step 4c: train a Model2Vec model with extra annotations

```shell
uv run bonepick train-model2vec \
    --dataset-dir tmp/data/fineweb-edu-llama3-annotations-binary-pos-neg-normalized-potion \
    --dataset-dir tmp/data/fw_pp_ref_mini_o4-mini-batch_medium-pos-neg-normalized-potion \
    --dataset-dir tmp/data/fw_pp_ref-pos-neg-normalized-potion \
    --output-dir tmp/models/potion-32M-fw-fw_pp-fw_pp_o4-posneg-normalized-potion

```

### Step 5: eval models

#### Step 5a: eval FastText model

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

#### Step 5b: eval Model2Vec model


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

#### Step 5c: eval UltraFineWeb

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


### Step 6: balance dataset

This ensures same number of positive and negative items

```shell
uv run bonepick balance-dataset \
    --input-dir tmp/data/fineweb-edu-llama3-annotations-binary-pos-neg-normalized-potion \
    --input-dir tmp/data/fw_pp_ref_mini_o4-mini-batch_medium-pos-neg-normalized-potion \
    --input-dir tmp/data/fw_pp_ref-pos-neg-normalized-potion \
    --output-dir tmp/data/fw-fw_pp-fw_pp_o4-posneg-normalized-potion-balanced
```

### Step 7: train on balanced dataset

```shell
uv run bonepick train-model2vec \
    --dataset-dir tmp/data/fw-fw_pp-fw_pp_o4-posneg-normalized-potion-balanced \
    --output-dir tmp/models/potion-32M-fw-fw_pp-fw_pp_o4-posneg-normalized-potion-balanced
```

## Code Quality

### Step 1: Some test code

```shell
s5cmd cp -sp \
    's3://ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2_annotated_reshard/*/step_final/shard_00000*' \
    "$HOME/ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2_annotated_reshard/"
```

### Step 2: sample about 5GB of data

```shell
uv run bonepick sample-dataset \
    --dataset-dir "${HOME}/ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2_annotated_reshard/" \
    --output-dir "${HOME}/ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2_5GB_sample_to_annotate" \
    --target-size 5GB
```


### Step 2: lets run annotation pipeline

```shell
uv run --extra=annotate bonepick annotate-dataset \
    --dataset-dir tmp/data/spring2code_python \
    --output-dir tmp/data/spring2code_python-annotated \
    --model-name gpt-5-mini \
    --service-tier flex \
    --annotation-task-prompt 'claude_rubric_code' \
    --annotation-system-prompt 'code_system' \
    --num-proc 1 \
    --reasoning-effort low \
    --max-text-length 30000
```
