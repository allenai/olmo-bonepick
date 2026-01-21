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
ROOT_DIR="/mnt/raid0"
BASE_DIR="${ROOT_DIR}/ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2_annotated/pruned"
s5cmd cp -sp \
    's3://ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2_annotated/pruned/*/step_final/sorted_chunk_0000*' \
    "${BASE_DIR}/"
```


### Step 2: reshard the sampled data

```shell
for pl in $(ls --color=never ${BASE_DIR}); do
    echo "Processing ${pl}..."
    uv run bonepick reshard-dataset \
        --dataset-dir "${BASE_DIR}/${pl}" \
        --output-dir "${BASE_DIR}_resharded/${pl}" \
        --num-files 32
done
```

### Step 3: sample about 1GB of data per PL

```shell
for pl in $(ls --color=never ${BASE_DIR}_resharded); do
    echo "Processing ${pl}..."
    uv run bonepick sample-dataset \
        --dataset-dir "${BASE_DIR}_resharded/${pl}" \
        --output-dir "${BASE_DIR}_1GB_sample_to_annotate/${pl}" \
        --target-size 1GB
done
```

### Step 4: lets run annotation pipeline

```shell
RUBRIC_PROMPT="countup_criteria_v2"
MODEL_NAME="gpt-5-mini"
MAX_LENGTH=10_000
LIMIT_ROWS=500_000
CACHE_LOCATION="${ROOT_DIR}/bonepick.annotate"

for pl in "Markdown" "Python"; do
    echo "Processing ${pl}..."
    uv run --extra=annotate bonepick annotate-dataset \
        --dataset-dir "${BASE_DIR}_1GB_sample_to_annotate/${pl}" \
        --output-dir "${BASE_DIR}_1GB_sample_annotated_${MODEL_NAME}_${RUBRIC_PROMPT}_${MAX_LENGTH}/${pl}" \
        --model-name "${MODEL_NAME}" \
        --service-tier flex \
        --annotation-task-prompt "${RUBRIC_PROMPT}" \
        --max-concurrent-requests 5_000 \
        --max-new-tokens 4096 \
        --annotation-system-prompt 'code_system' \
        --max-text-length ${MAX_LENGTH} \
        --limit-rows ${LIMIT_ROWS} \
        --cache-location ${CACHE_LOCATION}
done
```

Doing another round with 100_000 but all PLs

```shell
LIMIT_ROWS=100_000

for pl in $(ls --color=never ${BASE_DIR}_1GB_sample_to_annotate); do
    if [[ "${pl}" == "Markdown" ]] || [[ "${pl}" == "Python" ]]; then
        echo "Skipping ${pl}..."
        continue
    fi

    echo "Processing ${pl}..."
    uv run --extra=annotate bonepick annotate-dataset \
        --dataset-dir "${BASE_DIR}_1GB_sample_to_annotate/${pl}" \
        --output-dir "${BASE_DIR}_1GB_sample_annotated_${MODEL_NAME}_${RUBRIC_PROMPT}_${MAX_LENGTH}/${pl}" \
        --model-name "${MODEL_NAME}" \
        --service-tier flex \
        --annotation-task-prompt "${RUBRIC_PROMPT}" \
        --max-concurrent-requests 5_000 \
        --max-new-tokens 4096 \
        --annotation-system-prompt 'code_system' \
        --max-text-length ${MAX_LENGTH} \
        --limit-rows ${LIMIT_ROWS} \
        --cache-location ${CACHE_LOCATION}
done
```

### Step 5: upload the annotated data to S3

```shell
s5cmd cp -sp \
    "${ROOT_DIR}/ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2_annotated/pruned_1GB_sample_annotated_${MODEL_NAME}_${RUBRIC_PROMPT}_${MAX_LENGTH}/*" \
    "s3://ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2_annotated/pruned_1GB_sample_annotated_${MODEL_NAME}_${RUBRIC_PROMPT}_${MAX_LENGTH}/"
```

------------------------------------------------------------------------------------------------

## Comparing different annotation prompts


Copy the one python file we have been using for testing

```shell
s5cmd cp -sp 's3://ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2_annotated_reshard/Python/step_final/shard_00007001.jsonl.zst' "tmp/data/spring2code_python/train/shard_00007001.jsonl.zst"
```

### Prompt: claude_rubric_code


We are gonna test annotating with v1 of the claude prompt and compare GPT-5.2, GPT-5.2-medium and GPT-5.2-mini.

```shell
# 5.2
uv run --extra=annotate bonepick annotate-dataset \
    --dataset-dir tmp/data/spring2code_python \
    --output-dir tmp/data/spring2code_python-annotated-gpt-5.2 \
    --model-name gpt-5.2 \
    --service-tier flex \
    --annotation-task-prompt 'claude_rubric_code' \
    --annotation-system-prompt 'code_system' \
    --max-concurrent-requests 1000 \
    --max-requests-per-minute 3_000 \
    --limit-rows 5000 \
    --cache-location /tmp/bonepick/gpt-5.2

# 5 mini
uv run --extra=annotate bonepick annotate-dataset \
    --dataset-dir tmp/data/spring2code_python \
    --output-dir tmp/data/spring2code_python-annotated-gpt-5-mini \
    --model-name gpt-5-mini \
    --service-tier flex \
    --annotation-task-prompt 'claude_rubric_code' \
    --annotation-system-prompt 'code_system' \
    --max-concurrent-requests 1000 \
    --max-requests-per-minute 3_000 \
    --limit-rows 5000 \
    --cache-location /tmp/bonepick/gpt-5-mini

# 5.2 medium
uv run --extra=annotate bonepick annotate-dataset \
    --dataset-dir tmp/data/spring2code_python \
    --output-dir tmp/data/spring2code_python-annotated-gpt-5.2-medium \
    --model-name gpt-5.2-medium \
    --service-tier flex \
    --annotation-task-prompt 'claude_rubric_code' \
    --annotation-system-prompt 'code_system' \
    --max-concurrent-requests 1000 \
    --reasoning-effort medium \
    --max-requests-per-minute 3_000 \
    --limit-rows 5000 \
    --cache-location /tmp/bonepick/gpt-5.2-medium
```

Okay now we compare full agreement:

```shell
# Binary agreement

uv run bonepick annotation-agreement \
    --dataset1 tmp/data/spring2code_python-annotated-gpt-5.2-medium \
    --dataset2 tmp/data/spring2code_python-annotated-gpt-5-mini \
    --label-expression '(if .claude_rubric_code.score < 3 then "neg" else "pos" end)' \
    --key-expression '.text'
```

Output:
```text
Annotation Agreement Analysis

Dataset 1: tmp/data/spring2code_python-annotated-gpt-5.2-medium
Dataset 2: tmp/data/spring2code_python-annotated-gpt-5-mini
Label expression: (if .claude_rubric_code.score < 3 then "neg" else "pos" end)
Key expression: .text

Loading annotations from dataset 1...
Loaded 2,000 annotations from dataset 1

Loading annotations from dataset 2...
Loaded 2,000 annotations from dataset 2

Dataset Coverage:
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric               ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Samples in dataset 1 │ 2,000 │
│ Samples in dataset 2 │ 2,000 │
│ Common samples       │ 2,000 │
│ Only in dataset 1    │     0 │
│ Only in dataset 2    │     0 │
└──────────────────────┴───────┘

Computing agreement metrics...

╭──── Agreement Metrics ─────╮
│ Agreement Rate: 77.20%     │
│ Cohen's Kappa: 0.5381      │
│ Agreements: 1,544 / 2,000  │
│ Disagreements: 456 / 2,000 │
╰────────────────────────────╯

Cohen's Kappa interpretation: Moderate

Label Distribution:
┏━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Label ┃     Dataset 1 ┃     Dataset 2 ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ neg   │   955 (47.8%) │   717 (35.9%) │
│ pos   │ 1,045 (52.2%) │ 1,283 (64.1%) │
└───────┴───────────────┴───────────────┘

Confusion Matrix:
(rows=dataset1, columns=dataset2)

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━┓
┃ Dataset 1 \ Dataset 2 ┃ neg ┃ pos ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━┩
│ neg                   │ 608 │ 347 │
│ pos                   │ 109 │ 936 │
└───────────────────────┴─────┴─────┘
```

Okay now we compare scores:

```shell
# Scores agreement

uv run bonepick annotation-agreement \
    --dataset1 tmp/data/spring2code_python-annotated-gpt-5.2-medium \
    --dataset2 tmp/data/spring2code_python-annotated-gpt-5-mini \
    --label-expression '.claude_rubric_code.score' \
    --key-expression '.text'
```

```text
Annotation Agreement Analysis

Dataset 1: tmp/data/spring2code_python-annotated-gpt-5.2-medium
Dataset 2: tmp/data/spring2code_python-annotated-gpt-5-mini
Label expression: .claude_rubric_code.score
Key expression: .text

Loading annotations from dataset 1...
Loaded 2,000 annotations from dataset 1

Loading annotations from dataset 2...
Loaded 2,000 annotations from dataset 2

Dataset Coverage:
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric               ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Samples in dataset 1 │ 2,000 │
│ Samples in dataset 2 │ 2,000 │
│ Common samples       │ 2,000 │
│ Only in dataset 1    │     0 │
│ Only in dataset 2    │     0 │
└──────────────────────┴───────┘

Computing agreement metrics...

╭──── Agreement Metrics ─────╮
│ Agreement Rate: 55.80%     │
│ Cohen's Kappa: 0.3739      │
│ Agreements: 1,116 / 2,000  │
│ Disagreements: 884 / 2,000 │
╰────────────────────────────╯

Cohen's Kappa interpretation: Fair

Label Distribution:
┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Label ┃   Dataset 1 ┃   Dataset 2 ┃
┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 0     │    1 (0.1%) │   14 (0.7%) │
│ 1     │ 315 (15.8%) │  171 (8.6%) │
│ 2     │ 639 (31.9%) │ 532 (26.6%) │
│ 3     │ 734 (36.7%) │ 916 (45.8%) │
│ 4     │ 306 (15.3%) │ 359 (17.9%) │
│ 5     │    5 (0.2%) │    8 (0.4%) │
└───────┴─────────────┴─────────────┘

Confusion Matrix:
(rows=dataset1, columns=dataset2)

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━┓
┃ Dataset 1 \ Dataset 2 ┃  0 ┃   1 ┃   2 ┃   3 ┃   4 ┃ 5 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━┩
│ 0                     │  0 │   1 │   0 │   0 │   0 │ 0 │
│ 1                     │ 12 │ 122 │ 133 │  46 │   2 │ 0 │
│ 2                     │  1 │  43 │ 296 │ 260 │  39 │ 0 │
│ 3                     │  1 │   5 │  97 │ 505 │ 124 │ 2 │
│ 4                     │  0 │   0 │   6 │ 105 │ 191 │ 4 │
│ 5                     │  0 │   0 │   0 │   0 │   3 │ 2 │
└───────────────────────┴────┴─────┴─────┴─────┴─────┴───┘
```

Let's compare against non medium:


```text
Annotation Agreement Analysis

Dataset 1: tmp/data/spring2code_python-annotated-gpt-5.2
Dataset 2: tmp/data/spring2code_python-annotated-gpt-5-mini
Label expression: .claude_rubric_code.score
Key expression: .text

Loading annotations from dataset 1...
Loaded 2,000 annotations from dataset 1

Loading annotations from dataset 2...
Loaded 2,000 annotations from dataset 2

Dataset Coverage:
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric               ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Samples in dataset 1 │ 2,000 │
│ Samples in dataset 2 │ 2,000 │
│ Common samples       │ 2,000 │
│ Only in dataset 1    │     0 │
│ Only in dataset 2    │     0 │
└──────────────────────┴───────┘

Computing agreement metrics...

╭──── Agreement Metrics ─────╮
│ Agreement Rate: 57.60%     │
│ Cohen's Kappa: 0.4017      │
│ Agreements: 1,152 / 2,000  │
│ Disagreements: 848 / 2,000 │
╰────────────────────────────╯

Cohen's Kappa interpretation: Moderate

Label Distribution:
┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Label ┃   Dataset 1 ┃   Dataset 2 ┃
┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 0     │   30 (1.5%) │   14 (0.7%) │
│ 1     │ 318 (15.9%) │  171 (8.6%) │
│ 2     │ 669 (33.5%) │ 532 (26.6%) │
│ 3     │ 725 (36.2%) │ 916 (45.8%) │
│ 4     │ 252 (12.6%) │ 359 (17.9%) │
│ 5     │    6 (0.3%) │    8 (0.4%) │
└───────┴─────────────┴─────────────┘

Confusion Matrix:
(rows=dataset1, columns=dataset2)

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━┓
┃ Dataset 1 \ Dataset 2 ┃ 0 ┃   1 ┃   2 ┃   3 ┃   4 ┃ 5 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━┩
│ 0                     │ 7 │  12 │   9 │   1 │   1 │ 0 │
│ 1                     │ 6 │ 120 │ 122 │  67 │   3 │ 0 │
│ 2                     │ 1 │  35 │ 322 │ 271 │  39 │ 1 │
│ 3                     │ 0 │   4 │  75 │ 516 │ 129 │ 1 │
│ 4                     │ 0 │   0 │   4 │  61 │ 184 │ 3 │
│ 5                     │ 0 │   0 │   0 │   0 │   3 │ 3 │
└───────────────────────┴───┴─────┴─────┴─────┴─────┴───┘
```



Now we compare gpt 5.2 and gpt 5.2 medium (full scores agreement):

```shell
# Scores agreement

uv run bonepick annotation-agreement \
    --dataset1 tmp/data/spring2code_python-annotated-gpt-5.2-medium \
    --dataset2 tmp/data/spring2code_python-annotated-gpt-5.2 \
    --label-expression '.claude_rubric_code.score' \
    --key-expression '.text'
```

result:
```text
Annotation Agreement Analysis

Dataset 1: tmp/data/spring2code_python-annotated-gpt-5.2-medium
Dataset 2: tmp/data/spring2code_python-annotated-gpt-5.2
Label expression: .claude_rubric_code.score
Key expression: .text

Loading annotations from dataset 1...
Loaded 2,000 annotations from dataset 1

Loading annotations from dataset 2...
Loaded 2,000 annotations from dataset 2

Dataset Coverage:
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric               ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Samples in dataset 1 │ 2,000 │
│ Samples in dataset 2 │ 2,000 │
│ Common samples       │ 2,000 │
│ Only in dataset 1    │     0 │
│ Only in dataset 2    │     0 │
└──────────────────────┴───────┘

Computing agreement metrics...

╭──── Agreement Metrics ─────╮
│ Agreement Rate: 68.55%     │
│ Cohen's Kappa: 0.5606      │
│ Agreements: 1,371 / 2,000  │
│ Disagreements: 629 / 2,000 │
╰────────────────────────────╯

Cohen's Kappa interpretation: Moderate

Label Distribution:
┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Label ┃   Dataset 1 ┃   Dataset 2 ┃
┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 0     │    1 (0.1%) │   30 (1.5%) │
│ 1     │ 315 (15.8%) │ 318 (15.9%) │
│ 2     │ 639 (31.9%) │ 669 (33.5%) │
│ 3     │ 734 (36.7%) │ 725 (36.2%) │
│ 4     │ 306 (15.3%) │ 252 (12.6%) │
│ 5     │    5 (0.2%) │    6 (0.3%) │
└───────┴─────────────┴─────────────┘

Confusion Matrix:
(rows=dataset1, columns=dataset2)

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━┓
┃ Dataset 1 \ Dataset 2 ┃  0 ┃   1 ┃   2 ┃   3 ┃   4 ┃ 5 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━┩
│ 0                     │  1 │   0 │   0 │   0 │   0 │ 0 │
│ 1                     │ 27 │ 208 │  75 │   5 │   0 │ 0 │
│ 2                     │  2 │  97 │ 443 │  93 │   4 │ 0 │
│ 3                     │  0 │  13 │ 137 │ 527 │  57 │ 0 │
│ 4                     │  0 │   0 │  14 │ 100 │ 189 │ 3 │
│ 5                     │  0 │   0 │   0 │   0 │   2 │ 3 │
└───────────────────────┴────┴─────┴─────┴─────┴─────┴───┘
```

Much tighter! What's the agreement if we run gpt-5.2 twice?

```shell
 uv run --extra=annotate bonepick annotate-dataset \
    --dataset-dir tmp/data/spring2code_python \
    --output-dir tmp/data/spring2code_python-annotated-gpt-5.2-twice \
    --model-name gpt-5.2 \
    --service-tier flex \
    --annotation-task-prompt 'claude_rubric_code' \
    --annotation-system-prompt 'code_system' \
    --max-concurrent-requests 1000 \
    --max-requests-per-minute 3_000 \
    --limit-rows 5000 \
    --cache-location /tmp/bonepick/gpt-5.2-twice
```

result:
```text
Annotation Agreement Analysis

Dataset 1: tmp/data/spring2code_python-annotated-gpt-5.2
Dataset 2: tmp/data/spring2code_python-annotated-gpt-5.2-twice
Label expression: .claude_rubric_code.score
Key expression: .text

Loading annotations from dataset 1...
Loaded 2,000 annotations from dataset 1

Loading annotations from dataset 2...
Loaded 2,000 annotations from dataset 2

Dataset Coverage:
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric               ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Samples in dataset 1 │ 2,000 │
│ Samples in dataset 2 │ 2,000 │
│ Common samples       │ 2,000 │
│ Only in dataset 1    │     0 │
│ Only in dataset 2    │     0 │
└──────────────────────┴───────┘

Computing agreement metrics...

╭──── Agreement Metrics ─────╮
│ Agreement Rate: 75.00%     │
│ Cohen's Kappa: 0.6514      │
│ Agreements: 1,500 / 2,000  │
│ Disagreements: 500 / 2,000 │
╰────────────────────────────╯

Cohen's Kappa interpretation: Substantial

Label Distribution:
┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Label ┃   Dataset 1 ┃   Dataset 2 ┃
┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 0     │   30 (1.5%) │   25 (1.2%) │
│ 1     │ 318 (15.9%) │ 347 (17.3%) │
│ 2     │ 669 (33.5%) │ 670 (33.5%) │
│ 3     │ 725 (36.2%) │ 703 (35.1%) │
│ 4     │ 252 (12.6%) │ 246 (12.3%) │
│ 5     │    6 (0.3%) │    9 (0.4%) │
└───────┴─────────────┴─────────────┘

Confusion Matrix:
(rows=dataset1, columns=dataset2)

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━┓
┃ Dataset 1 \ Dataset 2 ┃  0 ┃   1 ┃   2 ┃   3 ┃   4 ┃ 5 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━┩
│ 0                     │ 18 │  12 │   0 │   0 │   0 │ 0 │
│ 1                     │  5 │ 247 │  63 │   3 │   0 │ 0 │
│ 2                     │  2 │  83 │ 497 │  78 │   9 │ 0 │
│ 3                     │  0 │   5 │ 103 │ 558 │  59 │ 0 │
│ 4                     │  0 │   0 │   7 │  64 │ 176 │ 5 │
│ 5                     │  0 │   0 │   0 │   0 │   2 │ 4 │
└───────────────────────┴────┴─────┴─────┴─────┴─────┴───┘
```

So 75% agreement on this labeling task is best we can do.

### Prompt: claude_progressive_rubric_code

We are gonna test annotating with v2 of the claude prompt and compare GPT-5.2, GPT-5.2-medium and GPT-5.2-mini.

```shell
# 5.2

export RUBRIC_PROMPT="claude_progressive_rubric_code"

models=(
    "gpt-5.2/"
    "gpt-5-mini/"
    "gpt-5.2/medium"
    "gpt-5.2/minimal"
)

for model in "${models[@]}"; do
    model_name=$(echo "${model}" | cut -d '/' -f 1)
    effort_name=$(echo "${model}" | cut -d '/' -f 2)

    ## check if effort name is empty string, if not, set effort_flag="--reasoning-effort ${effort_name}"
    if [[ -z "${effort_name}" ]]; then
        effort_flag=""
    else
        effort_flag="--reasoning-effort ${effort_name}"
    fi

    destination_dir="tmp/data/spring2code_python-annotated-${RUBRIC_PROMPT}-${model}"
    cache_dir="tmp/cache/${model}"

    echo "Annotating ${model_name} with reasoning effort: ${effort_flag}"

    uv run --extra=annotate bonepick annotate-dataset \
        --dataset-dir tmp/data/spring2code_python \
        --output-dir "${destination_dir}" \
        --model-name "${model_name}" \
        --service-tier default \
        --annotation-task-prompt "${RUBRIC_PROMPT}" \
        --annotation-system-prompt 'code_system' \
        ${effort_flag} \
        --max-concurrent-requests 1_000 \
        --max-requests-per-minute 5_000 \
        --limit-rows 5_000 \
        --cache-location
done
```


Calculate agreement between 5.2 and 5-mini:

```shell
uv run bonepick annotation-agreement \
    --dataset1 tmp/data/spring2code_python-annotated-${RUBRIC_PROMPT}-gpt-5.2 \
    --dataset2 tmp/data/spring2code_python-annotated-${RUBRIC_PROMPT}-gpt-5-mini \
    --label-expression ".${RUBRIC_PROMPT}.score" \
    --key-expression '.text'
```

```text
Annotation Agreement Analysis

Dataset 1: tmp/data/spring2code_python-annotated-claude_progressive_rubric_code-gpt-5.2
Dataset 2: tmp/data/spring2code_python-annotated-claude_progressive_rubric_code-gpt-5-mini
Label expression: .claude_progressive_rubric_code.score
Key expression: .text

Loading annotations from dataset 1...
Loaded 2,000 annotations from dataset 1

Loading annotations from dataset 2...
Loaded 2,000 annotations from dataset 2

Dataset Coverage:
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric               ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Samples in dataset 1 │ 2,000 │
│ Samples in dataset 2 │ 2,000 │
│ Common samples       │ 2,000 │
│ Only in dataset 1    │     0 │
│ Only in dataset 2    │     0 │
└──────────────────────┴───────┘

Computing agreement metrics...

╭──── Agreement Metrics ─────╮
│ Agreement Rate: 62.95%     │
│ Cohen's Kappa: 0.3664      │
│ Agreements: 1,259 / 2,000  │
│ Disagreements: 741 / 2,000 │
╰────────────────────────────╯

Cohen's Kappa interpretation: Fair

Label Distribution:
┏━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Label ┃     Dataset 1 ┃     Dataset 2 ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 0     │     50 (2.5%) │     37 (1.8%) │
│ 1     │   491 (24.6%) │   208 (10.4%) │
│ 2     │ 1,163 (58.1%) │ 1,226 (61.3%) │
│ 3     │   262 (13.1%) │   497 (24.9%) │
│ 4     │     34 (1.7%) │     32 (1.6%) │
└───────┴───────────────┴───────────────┘

Confusion Matrix:
(rows=dataset1, columns=dataset2)

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━┓
┃ Dataset 1 \ Dataset 2 ┃  0 ┃   1 ┃   2 ┃   3 ┃  4 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━┩
│ 0                     │  2 │   5 │  28 │  15 │  0 │
│ 1                     │ 32 │ 166 │ 236 │  57 │  0 │
│ 2                     │  3 │  37 │ 894 │ 222 │  7 │
│ 3                     │  0 │   0 │  68 │ 183 │ 11 │
│ 4                     │  0 │   0 │   0 │  20 │ 14 │
└───────────────────────┴────┴─────┴─────┴─────┴────┘
```

Going from 57.6% to 62.9% agreement!


### Prompt: simplified_code_rubric

```shell
export RUBRIC_PROMPT="simplified_code_rubric"
export MAX_TEXT_LENGTH=10000

models=(
    "gpt-5.2/"
    "gpt-5-mini/"
    "gpt-5.2/medium"
    "gpt-5.2/minimal"
)

for model in "${models[@]}"; do
    model_name=$(echo "${model}" | cut -d '/' -f 1)
    effort_name=$(echo "${model}" | cut -d '/' -f 2)

    destination_dir="tmp/data/spring2code_python-annotated-${RUBRIC_PROMPT}-${model_name}-${MAX_TEXT_LENGTH}"
    cache_dir="tmp/cache/${model}"

    ## check if effort name is empty string, if not, set effort_flag="--reasoning-effort ${effort_name}"
    if [[ -z "${effort_name}" ]]; then
        effort_flag=""
    else
        effort_flag="--reasoning-effort ${effort_name}"
        destination_dir="${destination_dir}-${effort_name}"
        cache_dir="${cache_dir}-${effort_name}"
    fi

    cache_dir="tmp/cache/${model}"

    echo "Annotating ${model_name} with reasoning effort: ${effort_flag}"

    uv run --extra=annotate bonepick annotate-dataset \
        --dataset-dir tmp/data/spring2code_python \
        --output-dir "${destination_dir}" \
        --model-name "${model_name}" \
        --service-tier default \
        --annotation-task-prompt "${RUBRIC_PROMPT}" \
        --annotation-system-prompt 'code_system' \
        ${effort_flag} \
        --max-concurrent-requests 1_000 \
        --max-requests-per-minute 5_000 \
        --limit-rows 5_000 \
        --max-text-length ${MAX_TEXT_LENGTH} \
        --cache-location "${cache_dir}"
done
```


### Prompt: `inv_codedoc_verysimple` (like this one!)

```shell
export RUBRIC_PROMPT="inv_codedoc_verysimple"
export MAX_TEXT_LENGTH=10000

models=(
    "gpt-5.2/"
    "gpt-5-mini/"
    "gpt-5.2/medium"
    "gpt-5.2/minimal"
)

for model in "${models[@]}"; do
    model_name=$(echo "${model}" | cut -d '/' -f 1)
    effort_name=$(echo "${model}" | cut -d '/' -f 2)

    destination_dir="tmp/data/spring2code_python-annotated-${RUBRIC_PROMPT}-${model_name}"
    cache_dir="tmp/cache/${model}"

    ## check if effort name is empty string, if not, set effort_flag="--reasoning-effort ${effort_name}"
    if [[ -z "${effort_name}" ]]; then
        effort_flag=""
    else
        effort_flag="--reasoning-effort ${effort_name}"
        destination_dir="${destination_dir}-${effort_name}"
        cache_dir="${cache_dir}-${effort_name}"
    fi

    echo "Annotating ${model_name} with reasoning effort: ${effort_flag}"

    uv run --extra=annotate bonepick annotate-dataset \
        --dataset-dir tmp/data/spring2code_python \
        --output-dir "${destination_dir}" \
        --model-name "${model_name}" \
        --service-tier flex \
        --annotation-task-prompt "${RUBRIC_PROMPT}" \
        --annotation-system-prompt 'code_system' \
        ${effort_flag} \
        --max-concurrent-requests 1_000 \
        --max-requests-per-minute 5_000 \
        --limit-rows 5_000 \
        --max-text-length ${MAX_TEXT_LENGTH} \
        --cache-location "${cache_dir}"
done
```

Now we check full and binary agreement between 5.2 and 5-mini:

```shell
uv run bonepick annotation-agreement \
    --dataset-dir tmp/data/spring2code_python-annotated-${RUBRIC_PROMPT}-gpt-5.2 \
    --dataset-dir tmp/data/spring2code_python-annotated-${RUBRIC_PROMPT}-gpt-5-mini \
    --label-expression ".${RUBRIC_PROMPT}.score" \
    --key-expression '.text'
```

Output for full agreement:

```text
Annotation Agreement Analysis

Dataset 1:
  - path:       tmp/data/spring2code_python-annotated-inv_codedoc_verysimple-gpt-5.2-10000
  - label expr: .inv_codedoc_verysimple.score
  - key expr:   .text

Dataset 2:
  - path:       tmp/data/spring2code_python-annotated-inv_codedoc_verysimple-gpt-5-mini-10000
  - label expr: .inv_codedoc_verysimple.score
  - key expr:   .text

Loading annotations from dataset 1...
Loaded 2,000 annotations from dataset 1

Loading annotations from dataset 2...
Loaded 2,000 annotations from dataset 2

Dataset Coverage:
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric               ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Samples in dataset 1 │ 2,000 │
│ Samples in dataset 2 │ 2,000 │
│ Common samples       │ 2,000 │
│ Only in dataset 1    │     0 │
│ Only in dataset 2    │     0 │
└──────────────────────┴───────┘

Computing agreement metrics...

╭──── Agreement Metrics ─────╮
│ Agreement Rate: 60.00%     │
│ Cohen's Kappa: 0.4190      │
│ Agreements: 1,200 / 2,000  │
│ Disagreements: 800 / 2,000 │
╰────────────────────────────╯

Cohen's Kappa interpretation: Moderate

Label Distribution:
┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Label ┃   Dataset 1 ┃   Dataset 2 ┃
┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 0     │   19 (0.9%) │    3 (0.1%) │
│ 1     │  132 (6.6%) │   68 (3.4%) │
│ 2     │ 445 (22.2%) │ 295 (14.8%) │
│ 3     │ 611 (30.6%) │ 865 (43.2%) │
│ 4     │ 783 (39.1%) │ 737 (36.9%) │
│ 5     │   10 (0.5%) │   32 (1.6%) │
└───────┴─────────────┴─────────────┘

Confusion Matrix:
(rows=dataset1, columns=dataset2)

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━┳━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━┓
┃ Dataset 1 \ Dataset 2 ┃ 0 ┃  1 ┃   2 ┃   3 ┃   4 ┃  5 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━╇━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━┩
│ 0                     │ 1 │  7 │   6 │   5 │   0 │  0 │
│ 1                     │ 2 │ 39 │  45 │  43 │   3 │  0 │
│ 2                     │ 0 │ 17 │ 172 │ 216 │  40 │  0 │
│ 3                     │ 0 │  5 │  54 │ 422 │ 130 │  0 │
│ 4                     │ 0 │  0 │  18 │ 179 │ 560 │ 26 │
│ 5                     │ 0 │  0 │   0 │   0 │   4 │  6 │
└───────────────────────┴───┴────┴─────┴─────┴─────┴────┘
```

Now onto binary agreement:

```shell
uv run bonepick annotation-agreement \
    --dataset-dir tmp/data/spring2code_python-annotated-${RUBRIC_PROMPT}-gpt-5.2 \
    --dataset-dir tmp/data/spring2code_python-annotated-${RUBRIC_PROMPT}-gpt-5-mini \
    --label-expression "(if .${RUBRIC_PROMPT}.score > 3 then \"pos\" else \"neg\" end)" \
    --key-expression '.text'
```

Output for binary agreement:

```text
Annotation Agreement Analysis

Dataset 1:
  - path:       tmp/data/spring2code_python-annotated-inv_codedoc_verysimple-gpt-5.2
  - label expr: (if .inv_codedoc_verysimple.score > 3 then "pos" else "neg" end)
  - key expr:   .text

Dataset 2:
  - path:       tmp/data/spring2code_python-annotated-inv_codedoc_verysimple-gpt-5-mini
  - label expr: (if .inv_codedoc_verysimple.score > 3 then "pos" else "neg" end)
  - key expr:   .text

Loading annotations from dataset 1...
Loaded 2,000 annotations from dataset 1

Loading annotations from dataset 2...
Loaded 2,000 annotations from dataset 2

Dataset Coverage:
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric               ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Samples in dataset 1 │ 2,000 │
│ Samples in dataset 2 │ 2,000 │
│ Common samples       │ 2,000 │
│ Only in dataset 1    │     0 │
│ Only in dataset 2    │     0 │
└──────────────────────┴───────┘

Computing agreement metrics...

╭──── Agreement Metrics ─────╮
│ Agreement Rate: 81.50%     │
│ Cohen's Kappa: 0.6114      │
│ Agreements: 1,630 / 2,000  │
│ Disagreements: 370 / 2,000 │
╰────────────────────────────╯

Cohen's Kappa interpretation: Substantial

Label Distribution:
┏━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Label ┃     Dataset 1 ┃     Dataset 2 ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ neg   │ 1,207 (60.4%) │ 1,231 (61.6%) │
│ pos   │   793 (39.6%) │   769 (38.5%) │
└───────┴───────────────┴───────────────┘

Confusion Matrix:
(rows=dataset1, columns=dataset2)

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━┓
┃ Dataset 1 \ Dataset 2 ┃   neg ┃ pos ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━┩
│ neg                   │ 1,034 │ 173 │
│ pos                   │   197 │ 596 │
└───────────────────────┴───────┴─────┘
```


### Prompt: `countup_criteria` (like this one!)

```shell
export RUBRIC_PROMPT="countup_criteria"
export MAX_TEXT_LENGTH=10000

models=(
    "gpt-5.2/"
    "gpt-5-mini/"
)

for model in "${models[@]}"; do
    model_name=$(echo "${model}" | cut -d '/' -f 1)
    effort_name=$(echo "${model}" | cut -d '/' -f 2)

    destination_dir="tmp/data/spring2code_python-annotated-${RUBRIC_PROMPT}-${model_name}"
    cache_dir="tmp/cache/${model}"

    ## check if effort name is empty string, if not, set effort_flag="--reasoning-effort ${effort_name}"
    if [[ -z "${effort_name}" ]]; then
        effort_flag=""
    else
        effort_flag="--reasoning-effort ${effort_name}"
        destination_dir="${destination_dir}-${effort_name}"
        cache_dir="${cache_dir}-${effort_name}"
    fi

    echo "Annotating ${model_name} with reasoning effort: ${effort_flag}"

    uv run --extra=annotate bonepick annotate-dataset \
        --dataset-dir tmp/data/spring2code_python \
        --output-dir "${destination_dir}" \
        --model-name "${model_name}" \
        --service-tier standard \
        --annotation-task-prompt "${RUBRIC_PROMPT}" \
        --annotation-system-prompt 'code_system' \
        ${effort_flag} \
        --max-concurrent-requests 1_000 \
        --max-requests-per-minute 5_000 \
        --limit-rows 2_000 \
        --max-text-length ${MAX_TEXT_LENGTH} \
        --cache-location "${cache_dir}"
done
```

#### Reviewing agreement by label

```json
{
    "code_purpose": "...",    // a short description of the purpose of the snippet.
    "programming_language": "...",    // the programming language of the snippet in lowercase.
    "criteria": {
        "basic_validity": {
            "has_clear_purpose": bool,          // 1993/2000, 0.6657, remove
            "not_mostly_empty": bool,           //  970/2000, 0.0500, clarify, done
            "no_syntax_errors": bool,           // 1851/2000, 0.3694, review, done
            "has_executable_logic": bool,       // 1934/2000, 0.6561, review, done
            "not_procedurally_generated": bool, // 1907/2000, 0.9084, keep
        },
        "code_cleanliness": {
            "no_boilerplate": bool,             // 1715/2000, 0.5954, tighten?
            "no_binary_data": bool,             // 1992/2000, 0.5865, tighten?
            "no_commented_out_code": bool,      // 1585/2000, 0.8681, keep
            "no_placeholder_text": bool,        // 1740/2000, 0.4750, review, low agreement!
            "no_debug_artifacts": bool,         // 1234/2000, 0.7259, tighten?
            "no_repetition": bool,              // 1470/2000, 0.3463, remove
        },
        "security": {
            "no_hardcoded_secrets": bool,      // 1923/2000, 0.7310, keep
            "no_vulnerabilities": bool,        // 1663/2000, 0.6002, review
        },
        "documentation_and_readability": {
            "has_comments": bool,              // 1253/2000, 0.6316, keep/review
            "has_docstrings": bool,            //  467/2000, 0.8406, keep
            "good_grammar": bool,              // 1713/2000, 0.4220, clarify
            "good_naming": bool,               // 1474/2000, 0.4599, remove
            "has_type_hints": bool,            //   97/2000, 0.8305, keep
        },
        "structure_and_organization": {
            "good_logical_flow": bool,         // 1931/2000, 0.5651, review/remove
            "only_shallow_nesting": bool,      // 1987/2000, -0.002, remove
            "is_concise": bool,                // 1972/2000, 0.3268, remove
            "is_modular": bool,                //  918/2000, 0.5003, clarify or remove
            "no_hardcoded_values": bool,       //  139/2000, 0.4524, change to no hardcoded inputs?
        },
        "robustness_and_performance": {
            "has_error_handling": bool,        //  346/2000, 0.6529, keep/review
            "minimal_side_effects": bool,      //  616/2000, 0.6851, keep
            "is_efficient": bool,              // 1630/2000, 0.3530, remove
        },
    },
    "overall_assessment": "...",    // a final explanation of the overall assessment of the snippet.
    "score": int    // the final score between 0 and 26 (inclusive); counts the number of criteria that are true
}
```


### We are gonna use countup_criteria_v2 for now.

```text
Annotation Agreement Analysis

Dataset 1:
  - path:       tmp/data/spring2code_python-annotated-countup_criteria_v2-gpt-5.2-high
  - label expr: .countup_criteria_v2.score
  - key expr:   .text

Dataset 2:
  - path:       tmp/data/spring2code_python-annotated-countup_criteria_v2-gpt-5-mini
  - label expr: .countup_criteria_v2.score
  - key expr:   .text

Loading annotations from dataset 1...
Loaded 1,996 annotations from dataset 1

Loading annotations from dataset 2...
Loaded 2,000 annotations from dataset 2

Dataset Coverage:
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric               ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Samples in dataset 1 │ 1,996 │
│ Samples in dataset 2 │ 2,000 │
│ Common samples       │ 1,996 │
│ Only in dataset 1    │     0 │
│ Only in dataset 2    │     4 │
└──────────────────────┴───────┘

Computing agreement metrics...

╭────────── Agreement Metrics ───────────╮
│ Agreement Rate: 23.60%                 │
│ Weighted Kappa (quadratic): 0.6144     │
│ Mean Absolute Error (MAE): 1.3282      │
│ Root Mean Squared Error (RMSE): 1.7357 │
│ Pearson Correlation: 0.6147            │
│ Agreements: 471 / 1,996                │
│ Disagreements: 1,525 / 1,996           │
╰────────────────────────────────────────╯

Weighted Kappa interpretation: Substantial

Difference Histogram (Dataset 2 - Dataset 1):
Negative values: Dataset 1 rated higher | Positive values: Dataset 2 rated higher

   Difference     Count  Bar
           -7  1 (0.1%)
           -6  2 (0.1%)
           -5  3 (0.2%)
           -4        36  ████
                 (1.8%)
           -3        93  ███████████
                 (4.7%)
           -2       221  ████████████████████████████
                (11.1%)
           -1       416  ████████████████████████████████████████████████████
                (20.8%)
           +0       471  ████████████████████████████████████████████████████████████
                (23.6%)
           +1       391  █████████████████████████████████████████████████
                (19.6%)
           +2       205  ██████████████████████████
                (10.3%)
           +3       107  █████████████
                 (5.4%)
           +4        38  ████
                 (1.9%)
           +5        10  █
                 (0.5%)
           +6  2 (0.1%)

Label Distribution:
┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Label ┃   Dataset 1 ┃   Dataset 2 ┃
┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 6     │    0 (0.0%) │    4 (0.2%) │
│ 7     │   10 (0.5%) │   17 (0.9%) │
│ 8     │   21 (1.1%) │   26 (1.3%) │
│ 9     │   68 (3.4%) │   79 (4.0%) │
│ 10    │  130 (6.5%) │   55 (2.8%) │
│ 11    │ 295 (14.8%) │ 254 (12.7%) │
│ 12    │ 363 (18.2%) │ 406 (20.3%) │
│ 13    │ 388 (19.4%) │ 462 (23.1%) │
│ 14    │ 293 (14.7%) │ 264 (13.2%) │
│ 15    │ 238 (11.9%) │ 303 (15.2%) │
│ 16    │  138 (6.9%) │   87 (4.4%) │
│ 17    │   42 (2.1%) │   30 (1.5%) │
│ 18    │    9 (0.5%) │    7 (0.4%) │
│ 19    │    1 (0.1%) │    2 (0.1%) │
└───────┴─────────────┴─────────────┘

Confusion Matrix:
(rows=dataset1, columns=dataset2)

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━┳━━━┳━━━┳━━━━┳━━━━┳━━━━┳━━━━━┳━━━━━┳━━━━┳━━━━┳━━━━┳━━━━┳━━━━┳━━━━┓
┃ Dataset 1 \ Dataset 2 ┃ 6 ┃ 7 ┃ 8 ┃  9 ┃ 10 ┃ 11 ┃  12 ┃  13 ┃ 14 ┃ 15 ┃ 16 ┃ 17 ┃ 18 ┃ 19 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━╇━━━╇━━━╇━━━━╇━━━━╇━━━━╇━━━━━╇━━━━━╇━━━━╇━━━━╇━━━━╇━━━━╇━━━━╇━━━━┩
│ 6                     │ 0 │ 0 │ 0 │  0 │  0 │  0 │   0 │   0 │  0 │  0 │  0 │  0 │  0 │  0 │
│ 7                     │ 1 │ 1 │ 1 │  1 │  2 │  3 │   1 │   0 │  0 │  0 │  0 │  0 │  0 │  0 │
│ 8                     │ 3 │ 5 │ 3 │  3 │  0 │  3 │   2 │   0 │  2 │  0 │  0 │  0 │  0 │  0 │
│ 9                     │ 0 │ 5 │ 7 │  9 │  7 │ 14 │  15 │  11 │  0 │  0 │  0 │  0 │  0 │  0 │
│ 10                    │ 0 │ 4 │ 2 │ 17 │  3 │ 37 │  27 │  27 │  6 │  7 │  0 │  0 │  0 │  0 │
│ 11                    │ 0 │ 2 │ 6 │ 29 │ 13 │ 51 │  85 │  64 │ 29 │ 14 │  2 │  0 │  0 │  0 │
│ 12                    │ 0 │ 0 │ 3 │ 13 │ 17 │ 68 │ 106 │ 101 │ 32 │ 21 │  2 │  0 │  0 │  0 │
│ 13                    │ 0 │ 0 │ 1 │  7 │ 10 │ 43 │  99 │ 115 │ 61 │ 46 │  6 │  0 │  0 │  0 │
│ 14                    │ 0 │ 0 │ 2 │  0 │  3 │ 22 │  42 │  80 │ 63 │ 65 │ 12 │  4 │  0 │  0 │
│ 15                    │ 0 │ 0 │ 1 │  0 │  0 │ 11 │  20 │  51 │ 53 │ 81 │ 18 │  3 │  0 │  0 │
│ 16                    │ 0 │ 0 │ 0 │  0 │  0 │  2 │   9 │  12 │ 15 │ 54 │ 31 │ 11 │  4 │  0 │
│ 17                    │ 0 │ 0 │ 0 │  0 │  0 │  0 │   0 │   1 │  3 │ 12 │ 14 │  8 │  2 │  2 │
│ 18                    │ 0 │ 0 │ 0 │  0 │  0 │  0 │   0 │   0 │  0 │  3 │  2 │  4 │  0 │  0 │
│ 19                    │ 0 │ 0 │ 0 │  0 │  0 │  0 │   0 │   0 │  0 │  0 │  0 │  0 │  1 │  0 │
└───────────────────────┴───┴───┴───┴────┴────┴────┴─────┴─────┴────┴────┴────┴────┴────┴────┘
```

## Data storing

```shell
 s5cmd cp -sp 'tmp/data/*' 's3://ai2-lucas/annotations-code-bonepick-diff-prompts/'
 ```


## Let's train models on Python/Markdown data

### Step 0: Let's first store data in a proper location (one time operation)

```shell
export LOCAL_BASE_DIR='/mnt/raid0'
export S3_BASE_DIR='s3://ai2-llm/classifiers/code-quality'

export BASE_SRC_DIR="${LOCAL_BASE_DIR}/ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2_annotated/pruned_1GB_sample"
export BASE_DST_DIR="${S3_BASE_DIR}/data/the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB"

export SRC_TO_ANNOTATE_DIR="${BASE_SRC_DIR}_to_annotate"
export DST_ANNOTATED_DIR="${BASE_DST_DIR}/raw"
export SRC_CLAUDE_RUBRIC_DIR="${BASE_SRC_DIR}_annotated_gpt-5-mini_claude_rubric_code_32000"
export DST_CLAUDE_RUBRIC_DIR="${BASE_DST_DIR}/claude_rubric_code/gpt-5-mini/32k_trimmed"
export SRC_VERY_SIMPLE_DIR="${BASE_SRC_DIR}_annotated_gpt-5-mini_inv_codedoc_verysimple_10_000"
export DST_VERY_SIMPLE_DIR="${BASE_DST_DIR}/inv_codedoc_verysimple/gpt-5-mini/10k_trimmed"
export SRC_COUNTUP_CRITERIA_V2_DIR="${BASE_SRC_DIR}_annotated_gpt-5-mini_countup_criteria_v2_10_000"
export DST_COUNTUP_CRITERIA_V2_DIR="${BASE_DST_DIR}/countup_criteria_v2/gpt-5-mini/10k_trimmed"

s5cmd cp -sp "${SRC_TO_ANNOTATE_DIR}/*" "${DST_ANNOTATED_DIR}/"
s5cmd cp -sp "${SRC_CLAUDE_RUBRIC_DIR}/*" "${DST_CLAUDE_RUBRIC_DIR}/"
s5cmd cp -sp "${SRC_VERY_SIMPLE_DIR}/*" "${DST_VERY_SIMPLE_DIR}/"
s5cmd cp -sp "${SRC_COUNTUP_CRITERIA_V2_DIR}/*" "${DST_COUNTUP_CRITERIA_V2_DIR}/"
```

### Step 1: Copy down the data from S3

```shell
export LOCAL_BASE_DIR="${HOME}/ai2-llm/classifiers/code-quality"
export S3_BASE_DIR='s3://ai2-llm/classifiers/code-quality'
export BASE_NAME_PREFIX="the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB"
export COUNTUP_CRITERIA_V2_DIR="${LOCAL_BASE_DIR}/data/${BASE_NAME_PREFIX}/countup_criteria_v2/gpt-5-mini/10k_trimmed"

s5cmd cp -sp "${S3_BASE_DIR}/data/*" "${LOCAL_BASE_DIR}/data/"
s5cmd cp -sp "${S3_BASE_DIR}/models/*" "${LOCAL_BASE_DIR}/models/"

# we don't use ones below here.
# export CLAUDE_RUBRIC_DIR="${LOCAL_BASE_DIR}/data/${BASE_NAME_PREFIX}/claude_rubric_code/gpt-5-mini/32k_trimmed"
# export VERY_SIMPLE_DIR="${LOCAL_BASE_DIR}/data/${BASE_NAME_PREFIX}/inv_codedoc_verysimple/gpt-5-mini/10k_trimmed"
```


### Step 2: Make train/test/valid splits and preprocess the data

```shell
export RUBRIC_PROMPT="countup_criteria_v2"
export LABEL_NAME="${RUBRIC_PROMPT}/gpt-5-mini/10k_trimmed"
export DATASET_DIR_UNSPLIT="${LOCAL_BASE_DIR}/data/${BASE_NAME_PREFIX}/${LABEL_NAME}"
export DATASET_DIR_SPLIT="${LOCAL_BASE_DIR}/data-train_test_split/${BASE_NAME_PREFIX}/${LABEL_NAME}"
for pl in $(ls --color=never ${DATASET_DIR_UNSPLIT}); do
    echo "Processing ${pl}..."
    uv run bonepick reshard-dataset \
        --dataset-dir "${DATASET_DIR_UNSPLIT}/${pl}" \
        --output-dir "${DATASET_DIR_SPLIT}/${pl}" \
        --num-files 5 \
        --test-split-frac 10_000 \
        --valid-split-frac 10_000
done
```


### Step 3: do we wanna binarize the data?


I've looked at stats to get maybe 50/50?

For Python data:

```shell
$ zstdcat Python/* | grep -oP '"score":\d+' | sed 's/"score"://g' | uv run --with=tqdm tqdm | sort | uniq -c | sort -k2,2n | awk '
{scores[NR]=$2; counts[NR]=$1; total+=$1}
END {
  print "score", "count", "cumul", "cdf"
  cum=0
  for(i=1; i<=NR; i++) {
    cum += counts[i]
    printf "%d %d %d %.6f\n", scores[i], counts[i], cum, cum/total
  }
}'

score count cumul cdf
0 3 3 0.000006
3 2 5 0.000010
4 7 12 0.000024
5 44 56 0.000112
6 1243 1299 0.002598
7 2320 3619 0.007238
8 6837 10456 0.020912
9 16012 26468 0.052936
10 13958 40426 0.080852
11 60264 100690 0.201381
12 98476 199166 0.398334
13 114091 313257 0.626518
14 67144 380401 0.760807
15 83388 463789 0.927584
16 25126 488915 0.977836
17 7129 496044 0.992094
18 2939 498983 0.997972
19 1014 499997 1.000000
```


for Markdown data:
```shell
$ zstdcat Markdown/* | grep -oP '"score":\d+' | sed 's/"score"://g' | uv run --with=tqdm tqdm | sort | uniq -c | sort -k2,2n | awk '
{scores[NR]=$2; counts[NR]=$1; total+=$1}
END {
  print "score", "count", "cumul", "cdf"
  cum=0
  for(i=1; i<=NR; i++) {
    cum += counts[i]
    printf "%d %d %d %.6f\n", scores[i], counts[i], cum, cum/total
  }
}'

score count cumul cdf
3 17 17 0.000034
4 190 207 0.000414
5 979 1186 0.002372
6 18020 19206 0.038412
7 33123 52329 0.104658
8 71425 123754 0.247509
9 117139 240893 0.481788
10 59020 299913 0.599828
11 114894 414807 0.829617
12 53041 467848 0.935700
13 19211 487059 0.974122
14 5450 492509 0.985022
15 5512 498021 0.996046
16 1609 499630 0.999264
17 326 499956 0.999916
18 36 499992 0.999988
19 6 499998 1.000000
```

It's different! if you really want, we probably want `>12` for Python and `>9` for Markdown. But we should try regression too.

```zsh
export DATASET_DIR_JSONL="${LOCAL_BASE_DIR}/preprocessed/${BASE_NAME_PREFIX}/${LABEL_NAME}/binary_threshold/jsonl"
for pl in $(ls --color=never ${DATASET_DIR_SPLIT}); do
    if [[ "${pl}" == "Python" ]]; then
        THRESHOLD=12
    elif [[ "${pl}" == "Markdown" ]]; then
        THRESHOLD=9
    else
        echo "Unknown programming language: ${pl}"
        exit 1
    fi
    echo "Processing ${pl} (threshold: ${THRESHOLD})..."
    uv run bonepick transform-dataset \
        --input-dir "${DATASET_DIR_SPLIT}/${pl}" \
        --output-dir "${DATASET_DIR_JSONL}/${pl}" \
        --label-transform "{score: (if .${RUBRIC_PROMPT}.score > ${THRESHOLD} then \"pos\" else \"neg\" end)}"
done
```

### Step 4: convert to FastText and Model2Vec format

```shell
export FASTTEXT_NORMALIZATION="ultrafine"
export DATASET_DIR_JSONL_FASTTEXT="${LOCAL_BASE_DIR}/preprocessed/${BASE_NAME_PREFIX}/${LABEL_NAME}/binary_threshold/fasttext/${FASTTEXT_NORMALIZATION}"

for pl in $(ls --color=never ${DATASET_DIR_JSONL}); do
    echo "Processing ${pl}..."
    uv run bonepick convert-to-fasttext \
        --input-dir "${DATASET_DIR_JSONL}/${pl}" \
        --output-dir "${DATASET_DIR_JSONL_FASTTEXT}/${pl}" \
        --normalization ${FASTTEXT_NORMALIZATION}
done

export MODEL2VEC_NORMALIZATION="potion-code"
export DATASET_DIR_JSONL_MODEL2VEC="${LOCAL_BASE_DIR}/preprocessed/${BASE_NAME_PREFIX}/${LABEL_NAME}/binary_threshold/model2vec/${MODEL2VEC_NORMALIZATION}"

for pl in $(ls --color=never ${DATASET_DIR_JSONL}); do
    echo "Processing ${pl}..."
    uv run bonepick normalize-dataset \
        --input-dir "${DATASET_DIR_JSONL}/${pl}" \
        --output-dir "${DATASET_DIR_JSONL_MODEL2VEC}/${pl}" \
        --normalization ${MODEL2VEC_NORMALIZATION}
done
```

### Step 3: Train/eval a model2vec model

```shell
# model_paths=(
#     "minishlab/potion-base-32M"
#     "${LOCAL_BASE_DIR}/models/model2vec/Qwen3-Embedding-4B"
# )

# programming_languages=(
#     "Python"
#     "Markdown"
# )

# all_loss_class_weights=(
#     "sqrt"
#     "uniform"
# )

export MODEL2VEC_MODEL_DIR="${LOCAL_BASE_DIR}/trained_models/model2vec"

programming_language="Python"
model_path="minishlab/potion-base-32M"
loss_class_weights="uniform"

model_name=$(echo "${model_path}" | awk -F'/' '{print $NF}')
dataset_name=$(echo "${DATASET_DIR_JSONL_MODEL2VEC#"${LOCAL_BASE_DIR}/preprocessed/"}" | tr '/' '_')
model_dir="${MODEL2VEC_MODEL_DIR}/${dataset_name}/${programming_language}/${model_name}/${loss_class_weights}"

uv run bonepick train-model2vec \
    --dataset-dir ${DATASET_DIR_JSONL_MODEL2VEC}/${programming_language} \
    --model-name "${model_path}" \
    --output-dir "${model_dir}" \
    --loss-class-weight "${loss_class_weights}"
```

Now we eval the models:

```shell
uv run bonepick eval-model2vec \
    --dataset-dir ${DATASET_DIR_JSONL_MODEL2VEC}/${programming_language} \
    --model-dir "${model_dir}"
```


### Step 4: Train/eval a fasttext model

First we train the models:

```shell
export FASTTEXT_MODEL_DIR="${LOCAL_BASE_DIR}/trained_models/fasttext"

for pl in "${programming_languages[@]}"; do
    dataset_name=$(echo "${DATASET_DIR_JSONL_FASTTEXT#"${LOCAL_BASE_DIR}/preprocessed/"}" | tr '/' '_')

    uv run bonepick train-fasttext \
        --dataset-dir "${DATASET_DIR_JSONL_FASTTEXT}/${pl}" \
        --output-dir "${FASTTEXT_MODEL_DIR}/${dataset_name}/${pl}"
done
```

Now we eval the models:

```shell
for pl in "${programming_languages[@]}"; do
    dataset_name=$(echo "${DATASET_DIR_JSONL_FASTTEXT#"${LOCAL_BASE_DIR}/preprocessed/"}" | tr '/' '_')
    model_dir="${FASTTEXT_MODEL_DIR}/${dataset_name}/${pl}"

    uv run bonepick eval-fasttext \
        --dataset-dir "${DATASET_DIR_JSONL_FASTTEXT}/${pl}" \
        --model-dir "${model_dir}"
done
```


### Step 5: Train a regression model

Let's train a regression model too:

```shell
export REGRESSION_MODEL_DIR="${LOCAL_BASE_DIR}/trained_models/model2vec_regression"

model_path="minishlab/potion-base-32M"
pl="Python"

model_name=$(echo "${model_path}" | awk -F'/' '{print $NF}')
dataset_name=$(echo "${DATASET_DIR_SPLIT#"${LOCAL_BASE_DIR}/data-train_test_split/"}" | tr '/' '_')
output_dir="${REGRESSION_MODEL_DIR}/${dataset_name}/${pl}/${model_name}"

uv run bonepick train-model2vec \
    --dataset-dir ${DATASET_DIR_SPLIT}/${pl} \
    --model-name "${model_path}" \
    --output-dir "${output_dir}" \
    --regression \
    --label-expression ".${RUBRIC_PROMPT}.score"
```


## New set of commands


All variables (DGX Spark):


```shell
export LOCAL_BASE_DIR="${HOME}/ai2-llm/classifiers/code-quality"
export RUBRIC_PROMPT="countup_criteria_v2"
export LABEL_NAME="${RUBRIC_PROMPT}/gpt-5-mini/10k_trimmed"
export DATASET_DIR_UNSPLIT="${LOCAL_BASE_DIR}/data/${BASE_NAME_PREFIX}/${LABEL_NAME}"
export DATASET_DIR_SPLIT="${LOCAL_BASE_DIR}/data-train_test_split/${BASE_NAME_PREFIX}/${LABEL_NAME}"

export PROGRAMMING_LANGUAGE="Python"
export LABEL_THRESHOLD=13
export LOSS_CLASS_WEIGHTS="uniform"
export MODEL2VEC_NORMALIZER="plsfix"
export FASTTEXT_NORMALIZER="hyperfine"
export TEXT_MAX_LENGTH=10_000
export LABEL_EXPRESSION="(if .${RUBRIC_PROMPT}.score > ${LABEL_THRESHOLD} then \"pos\" else \"neg\" end)"
export MODEL2VEC_MODEL="minishlab/potion-base-32M"

export MODEL2VEC_MODEL_DIR="${LOCAL_BASE_DIR}/trained_models/model2vec"
export MODEL2VEC_MODEL_NAME=$(echo "${MODEL2VEC_MODEL}" | awk -F'/' '{print $NF}')
export MODEL2VEC_DATASET_NAME=$(echo "${DATASET_DIR_SPLIT#"${LOCAL_BASE_DIR}/data-train_test_split/"}" | tr '/' '_')
export MODEL2VEC_OUTPUT_DIR="${MODEL2VEC_MODEL_DIR}/${MODEL2VEC_DATASET_NAME}/${PROGRAMMING_LANGUAGE}/${MODEL2VEC_MODEL_NAME}/${LOSS_CLASS_WEIGHTS}"

export DATASET_DIR_FASTTEXT="${LOCAL_BASE_DIR}/preprocessed/${BASE_NAME_PREFIX}/${LABEL_NAME}/fasttext/${FASTTEXT_NORMALIZER}_thr${LABEL_THRESHOLD}"
export FASTTEXT_DATASET_NAME=$(echo "${DATASET_DIR_FASTTEXT#"${LOCAL_BASE_DIR}/preprocessed/"}" | tr '/' '_')
export FASTTEXT_OUTPUT_DIR="${LOCAL_BASE_DIR}/trained_models/fasttext/${FASTTEXT_DATASET_NAME}/${PROGRAMMING_LANGUAGE}"
```


Reshard the data:


```shell
for pl in $(ls --color=never ${DATASET_DIR_UNSPLIT}); do
    echo "Processing ${pl}..."
    uv run bonepick reshard-dataset \
        --dataset-dir "${DATASET_DIR_UNSPLIT}/${pl}" \
        --output-dir "${DATASET_DIR_SPLIT}/${pl}" \
        --num-files 20 \
        --test-split-frac 10_000 \
        --valid-split-frac 10_000
done
```

Train a model2vec model directly:

```shell
uv run bonepick train-model2vec \
    --dataset-dir "${DATASET_DIR_SPLIT}/${PROGRAMMING_LANGUAGE}" \
    --model-name "${MODEL2VEC_MODEL}" \
    --loss-class-weight "${LOSS_CLASS_WEIGHTS}" \
    --normalizer "${MODEL2VEC_NORMALIZER}" \
    --max-length "${TEXT_MAX_LENGTH}" \
    --label-expression "${LABEL_EXPRESSION}" \
    --output-dir "${MODEL2VEC_OUTPUT_DIR}"
```

Now we eval the models:

```shell
uv run bonepick eval-model2vec \
    --dataset-dir "${DATASET_DIR_SPLIT}/${PROGRAMMING_LANGUAGE}" \
    --model-dir "${MODEL2VEC_OUTPUT_DIR}" \
    --label-expression "${LABEL_EXPRESSION}" \
    --max-length "${TEXT_MAX_LENGTH}" \
    --normalizer "${MODEL2VEC_NORMALIZER}"
```

Make fasttext dataset:

```shell
uv run bonepick convert-to-fasttext \
    --input-dir "${DATASET_DIR_SPLIT}/${PROGRAMMING_LANGUAGE}" \
    --output-dir "${DATASET_DIR_FASTTEXT}/${PROGRAMMING_LANGUAGE}" \
    --normalization "${FASTTEXT_NORMALIZER}" \
    --label-expression "${LABEL_EXPRESSION}" \
    --max-length "${TEXT_MAX_LENGTH}"
```


Train a fasttext model directly:

```shell
uv run bonepick train-fasttext \
    --dataset-dir "${DATASET_DIR_FASTTEXT}/${PROGRAMMING_LANGUAGE}" \
    --output-dir "${FASTTEXT_OUTPUT_DIR}"
```

Now we eval the models:

```shell
uv run bonepick eval-fasttext \
    --dataset-dir "${DATASET_DIR_FASTTEXT}/${PROGRAMMING_LANGUAGE}" \
    --model-dir "${FASTTEXT_OUTPUT_DIR}"
```


### Some fasttext results

#### Markdown, threshold >10, hyperfine normalization

```yaml
dataset_dir:
- /home/lucas/ai2-llm/classifiers/code-quality/preprocessed/the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB/countup_criteria_v2/gpt-5-mini/10k_trimmed/fasttext/hyperfine_thr10/Markdown
model_dir: /home/lucas/ai2-llm/classifiers/code-quality/trained_models/fasttext/the-stack-v2_spring2code_v2_minhash_v2_annotated_sample_1GB_countup_criteria_v2_gpt-5-mini_10k_trimmed_fasttext_hyperfine_thr10/Markdown
overall_results:
  macro_precision: 0.7827
  macro_recall: 0.7868
  macro_f1: 0.7844
  macro_auc: 0.8831
per_class_metrics:
- class_name: __label__neg
  precision: 0.835
  recall: 0.8051
  f1: 0.8198
  support: 5926
  auc: null
- class_name: __label__pos
  precision: 0.7305
  recall: 0.7685
  f1: 0.749
  support: 4074
  auc: 0.8831
```

#### Python, threshold >13, hyperfine normalization

(ultrafine-plus normalization is same as hyperfine; renamed it to hyperfine recently)

```yaml
dataset_dir:
- /home/lucas/ai2-llm/classifiers/code-quality/preprocessed/the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB/countup_criteria_v2/gpt-5-mini/10k_trimmed/fasttext/ultrafine-plus_thr13/Python
model_dir: /home/lucas/ai2-llm/classifiers/code-quality/trained_models/fasttext/the-stack-v2_spring2code_v2_minhash_v2_annotated_sample_1GB_countup_criteria_v2_gpt-5-mini_10k_trimmed_fasttext_ultrafine-plus_thr13/Python
overall_results:
  macro_precision: 0.8218
  macro_recall: 0.8171
  macro_f1: 0.8193
  macro_auc: 0.9054
per_class_metrics:
- class_name: __label__neg
  precision: 0.8592
  recall: 0.8765
  f1: 0.8678
  support: 6277
  auc: null
- class_name: __label__pos
  precision: 0.7845
  recall: 0.7577
  f1: 0.7709
  support: 3723
  auc: 0.9054
```


## Tracking differences datamap_rs & bonepick

datamap-rs

```text
import os \\n import random \\n import time \\n import gui as gui \\n \\n def main (): \\n gui . gui (). open window () \\n \\n if __ name __ == "__ main __ ": \\n main () \\n
```

bonepick

```text
import os \n import random \n import time \n import gui as gui \n \n def main (): \n gui . gui (). open window () \n \n if __ name __ == "__ main __ ": \n main () \n
```

official

```text
import os \n import random \n import time \n import gui as gui \n \n def main (): \n gui . gui (). open window () \n \n if __ name __ == "__ main __ ": \n main () \n
```

----

bonepick

```text
import os \n import random \n import time \n import gui as gui \n \n def main (): \n gui . gui (). open window () \n \n if __ name __ == "__ main __ ": \n main () \n
```

datamap-rs

```text
import os \n import random \n import time \n import gui as gui \n \n def main (): \n gui . gui (). open window () \n \n if __ name __ == "__ main __ ": \n main () \n
```


## Comparing ranked prediciton

First infer...

```bash
uv run bonepick infer-fasttext \
    -i ~/ai2-llm/classifiers/code-quality/data-train_test_split/the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB/countup_criteria_v2/gpt-5-mini/10k_trimmed/Python/test \
    --o tmp/sample_1GB_countup_Python_test \
    --normalizer ultrafine \
    -m ~/ai2-llm/classifiers/code-quality/trained_models/fasttext/the-stack-v2_spring2code_v2_minhash_v2_annotated_sample_1GB_countup_criteria_v2_gpt-5-mini_10k_trimmed_fasttext_ultrafine_thr13/Python \
    -c "code_quality" \
    --max-length 10000
```

...then compare:


```bash
uv run bonepick eval-predictions \
    -d  tmp/sample_1GB_countup_Python_test \
    -p '.metadata.code_quality.__label__pos' \
    -l '.countup_criteria_v2.score'
```
