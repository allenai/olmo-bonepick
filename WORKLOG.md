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
RUBRIC_PROMPT="claude_rubric_code"
MODEL_NAME="gpt-5-mini"
MAX_LENGTH=32000
LIMIT_ROWS=100000
CACHE_LOCATION="${ROOT_DIR}/bonepick.annotate"

for pl in $(ls --color=never ${BASE_DIR}_1GB_sample_to_annotate); do
    # skip markdown files, they need custom prompts
    if [[ "${pl}" == *.md ]]; then
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
        --max-concurrent-requests 2000 \
        --max-new-tokens 2048 \
        --annotation-system-prompt 'code_system' \
        --max-text-length ${MAX_LENGTH} \
        --limit-rows ${LIMIT_ROWS} \
        --cache-location ${CACHE_LOCATION}
done
```

### Step 5: upload the annotated data to S3

```shell
s5cmd cp -sp "${ROOT_DIR}/ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2_annotated/pruned_1GB_sample_annotated_gpt-5-mini_claude_rubric_code_32000/"* "s3://ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/minhash_v2_annotated/pruned_1GB_sample_annotated_gpt-5-mini_claude_rubric_code_32000/"
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


### Prompt: claude_progressive_rubric_code_v2

```shell
export RUBRIC_PROMPT="claude_progressive_rubric_code_v2"

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

    ## check if effort name is empty string, if not, set effort_flag="--reasoning-effort ${effort_name}"
    if [[ -z "${effort_name}" ]]; then
        effort_flag=""
    else
        effort_flag="--reasoning-effort ${effort_name}"
        destination_dir="${destination_dir}-${effort_name}"
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
        --cache-location "${cache_dir}"
done
```


Compare agreement between 5.2 and 5-mini:

```shell
uv run bonepick annotation-agreement \
    --dataset1 tmp/data/spring2code_python-annotated-${RUBRIC_PROMPT}-gpt-5.2 \
    --dataset2 tmp/data/spring2code_python-annotated-${RUBRIC_PROMPT}-gpt-5-mini \
    --label-expression ".${RUBRIC_PROMPT}.score" \
    --key-expression '.text'
```
