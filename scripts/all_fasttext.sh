#!/usr/bin/env bash

###############################################################################
#                            SCRIPT CONFIGURATION                             #
###############################################################################

LOCAL_BASE_DIR="${HOME}/ai2-llm/classifiers/code-quality"
RUBRIC_PROMPT="countup_criteria_v2"
FASTTEXT_NORMALIZER="ultrafine"
TEXT_MAX_LENGTH=10_000
DATASET_NAME="the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB"
RUBRIC_MODEL_NAME="gpt-5-mini/10k_trimmed"
LANGUAGES=(
    "C"
    "C++"
    "C-Sharp"
    "Go"
    "Java"
    "JavaScript"
    "Markdown"
    "PHP"
    "Python"
    "Ruby"
    "Rust"
    "Shell"
    "SQL"
    "Swift"
    "TypeScript"
)

# See scripts/threshold_repot.txt for how these thresholds are pick. They are
# the first bucket where cumulative distribution is >50%. We pick the bucket
# after that as positive class
THRESHOLDS=(
    13
    13
    14
    15
    14
    14
    10
    13
    13
    14
    15
    12
    12
    14
    14
)

###############################################################################
#                        END OF SCRIPT CONFIGURATION                          #
###############################################################################

set -ex

# Step 1: make train/dev/test split for all LANGUAGES
DATASET_DIR_UNSPLIT="${LOCAL_BASE_DIR}/data/${DATASET_NAME}/${RUBRIC_PROMPT}/${RUBRIC_MODEL_NAME}"
DATASET_DIR_SPLIT="${LOCAL_BASE_DIR}/data-train_test_split/${DATASET_NAME}/${RUBRIC_PROMPT}/${RUBRIC_MODEL_NAME}"

for LANGUAGE in "${LANGUAGES[@]}"; do
    LANGUAGE_DATASET_DIR_SPLIT="${DATASET_DIR_SPLIT}/${LANGUAGE}"

    if [ -d "${LANGUAGE_DATASET_DIR_SPLIT}" ]; then
        echo "Skipping lang=${LANGUAGE} as it is already split"
        continue
    fi

    echo "Splitting lang=${LANGUAGE}"
    LANGUAGE_DATASET_DIR_UNSPLIT="${DATASET_DIR_UNSPLIT}/${LANGUAGE}"

    uv run bonepick reshard-dataset \
        --dataset-dir "${LANGUAGE_DATASET_DIR_UNSPLIT}" \
        --output-dir "${LANGUAGE_DATASET_DIR_SPLIT}" \
        --num-files 20 \
        --test-split-frac 10_000 \
        --valid-split-frac 10_000
    done

echo "All languages split"


# Step 2: create fasttext data for all LANGUAGES
DATASET_DIR_FASTTEXT="${LOCAL_BASE_DIR}/preprocessed/${DATASET_NAME}/${RUBRIC_NAME}/${RUBRIC_MODEL_NAME}/fasttext/${FASTTEXT_NORMALIZER}"

for i in "${!LANGUAGES[@]}"; do
    LANGUAGE="${LANGUAGES[$i]}"
    THRESHOLD="${THRESHOLDS[$i]}"

    LANGUAGE_DATASET_DIR_FASTTEXT="${DATASET_DIR_FASTTEXT}_thr${THRESHOLD}/${LANGUAGE}"

    if [ -d "${LANGUAGE_DATASET_DIR_FASTTEXT}" ]; then
        echo "Skipping lang=${LANGUAGE} as it is already processed"
        continue
    fi

    echo "Processing lang=${LANGUAGE} with threshold=${THRESHOLD}"
    LANGUAGE_DATASET_DIR_SPLIT="${DATASET_DIR_SPLIT}/${LANGUAGE}"
    LABEL_EXPRESSION="(if .${RUBRIC_PROMPT}.score > ${THRESHOLD} then \"pos\" else \"neg\" end)"

    uv run bonepick convert-to-fasttext \
        --input-dir "${LANGUAGE_DATASET_DIR_SPLIT}" \
        --output-dir "${LANGUAGE_DATASET_DIR_FASTTEXT}" \
        --normalization "${FASTTEXT_NORMALIZER}" \
        --label-expression "${LABEL_EXPRESSION}" \
        --max-length "${TEXT_MAX_LENGTH}"
done

echo "All languages processed"


# Step 3: train fasttext models for all LANGUAGES
DATASET_DIR_FASTTEXT_TRAIN="${DATASET_DIR_FASTTEXT}/train"
DATASET_DIR_FASTTEXT_TEST="${DATASET_DIR_FASTTEXT}/test"
TRAINED_MODEL_PATH=()

for i in "${!LANGUAGES[@]}"; do
    LANGUAGE="${LANGUAGES[$i]}"
    THRESHOLD="${THRESHOLDS[$i]}"

    LANGUAGE_DATASET_DIR_FASTTEXT="${DATASET_DIR_FASTTEXT}_thr${THRESHOLD}/${LANGUAGE}"
    FASTTEXT_DATASET_NAME=$(echo "${LANGUAGE_DATASET_DIR_FASTTEXT#"${LOCAL_BASE_DIR}/preprocessed/"}" | tr '/' '_')
    FASTTEXT_OUTPUT_DIR="${LOCAL_BASE_DIR}/trained_models/fasttext/${FASTTEXT_DATASET_NAME}/${LANGUAGE}"

    if [ -d "${FASTTEXT_OUTPUT_DIR}" ]; then
        echo "Skipping lang=${LANGUAGE} as it is already trained"
        continue
    fi

    echo "Training ${LANGUAGE} with threshold ${THRESHOLD}"

    uv run bonepick train-fasttext \
        --dataset-dir "${LANGUAGE_DATASET_DIR_FASTTEXT}" \
        --output-dir "${FASTTEXT_OUTPUT_DIR}"
done

echo "All languages trained"


# Step 4: Evaluate trained models
REPORT_DESTINATION=$(dirname "${0}")/all_fasttext_evals.txt
echo "=========================" > "${REPORT_DESTINATION}"

for i in "${!LANGUAGES[@]}"; do
    LANGUAGE="${LANGUAGES[$i]}"
    THRESHOLD="${THRESHOLDS[$i]}"

    LANGUAGE_DATASET_DIR_FASTTEXT="${DATASET_DIR_FASTTEXT}_thr${THRESHOLD}/${LANGUAGE}"
    FASTTEXT_DATASET_NAME=$(echo "${LANGUAGE_DATASET_DIR_FASTTEXT#"${LOCAL_BASE_DIR}/preprocessed/"}" | tr '/' '_')
    FASTTEXT_OUTPUT_DIR="${LOCAL_BASE_DIR}/trained_models/fasttext/${FASTTEXT_DATASET_NAME}/${LANGUAGE}"

    echo "Evaluating ${LANGUAGE} with threshold ${THRESHOLD}" | tee -a "${REPORT_DESTINATION}"
    echo "-------------------------" | tee -a "${REPORT_DESTINATION}"

    uv run bonepick eval-fasttext \
        --dataset-dir "${LANGUAGE_DATASET_DIR_FASTTEXT}" \
        --model-dir "${FASTTEXT_OUTPUT_DIR}" | tee -a "${REPORT_DESTINATION}"

    echo "=========================" >> "${REPORT_DESTINATION}"
done

echo "All models evaluated"
