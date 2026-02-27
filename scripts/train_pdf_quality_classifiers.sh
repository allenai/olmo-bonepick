#!/bin/bash
# Train FastText classifiers for PDF quality rubrics.
#
# Data sources:
#   - s3://ai2-llm/classifiers/pdf-quality/data/finepdfish_dclm
#   - s3://ai2-llm/classifiers/pdf-quality/data/finepdfish_edu
#   - s3://ai2-llm/classifiers/pdf-quality/data/finepdfish_ocr
#
# Pipeline (per classifier):
# 1. Download annotated data from S3
# 2. Split into train/valid/test using reshard-dataset
# 3. Convert to FastText format (5 bins)
# 4. Train FastText classifier
# 5. Run inference on valid/test sets
# 6. Fit linear calibration on valid (or train if valid is too small)
# 7. Evaluate calibration on test set
#
# ==============================================================================
# USAGE
# ==============================================================================
#
# Basic usage:
#   ./scripts/train_pdf_quality_classifiers.sh
#
# With custom local base directory:
#   LOCAL_BASE_DIR=/path/to/workdir ./scripts/train_pdf_quality_classifiers.sh
#
# Environment variables:
#   LOCAL_BASE_DIR              - Base directory for local data/outputs
#                                 (default: ~/ai2-llm/classifiers/pdf-quality)
#   S3_BASE                     - S3 base path for input datasets
#                                 (default: s3://ai2-llm/classifiers/pdf-quality)
#   S3_FINEPDFISH_DCLM_INPUT    - Optional override for dclm dataset path
#   S3_FINEPDFISH_EDU_INPUT     - Optional override for edu dataset path
#   S3_FINEPDFISH_OCR_INPUT     - Optional override for ocr dataset path
#
# ==============================================================================

set -xeuo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

# Training parameters
NUM_FILES=$(($(nproc) + 2))
TEST_SPLIT_SIZE=10000
MAX_TEXT_LENGTH=10000

# FastText hyperparameters
WORD_NGRAMS=5
WINDOW_SIZE=10
EPOCHS=10
DIMENSION=512

# Normalizer and text field
NORMALIZER="ultrafine"
TEXT_EXPRESSION=".text"

# Minimum valid samples to use valid set for calibration (otherwise use train)
MIN_VALID_FOR_CALIBRATION=5000

# Task names (also rubric field names in annotations)
CLASSIFIERS=("finepdfish_dclm" "finepdfish_edu" "finepdfish_ocr")

# Paths
S3_BASE="${S3_BASE:-s3://ai2-llm/classifiers/pdf-quality}"
LOCAL_BASE_DIR="${LOCAL_BASE_DIR:-${HOME}/ai2-llm/classifiers/pdf-quality}"

declare -A S3_INPUTS=(
    ["finepdfish_dclm"]="${S3_FINEPDFISH_DCLM_INPUT:-${S3_BASE}/data/finepdfish_dclm}"
    ["finepdfish_edu"]="${S3_FINEPDFISH_EDU_INPUT:-${S3_BASE}/data/finepdfish_edu}"
    ["finepdfish_ocr"]="${S3_FINEPDFISH_OCR_INPUT:-${S3_BASE}/data/finepdfish_ocr}"
)

declare -A DATA_DIRS
declare -A SPLIT_DATA_DIRS
declare -A PREPROCESSED_DIRS
declare -A MODEL_DIRS
declare -A CALIBRATION_DIRS
declare -A METADATA_FIELDS

for classifier in "${CLASSIFIERS[@]}"; do
    DATA_DIRS["${classifier}"]="${LOCAL_BASE_DIR}/data/${classifier}"
    SPLIT_DATA_DIRS["${classifier}"]="${LOCAL_BASE_DIR}/data-train_test_split/${classifier}"
    PREPROCESSED_DIRS["${classifier}"]="${LOCAL_BASE_DIR}/preprocessed/${classifier}/fasttext/${NORMALIZER}_bin5"
    MODEL_DIRS["${classifier}"]="${LOCAL_BASE_DIR}/trained_models/fasttext/${classifier}_${NORMALIZER}_bin5"
    CALIBRATION_DIRS["${classifier}"]="${LOCAL_BASE_DIR}/calibration/${classifier}"
    METADATA_FIELDS["${classifier}"]="${classifier}_${NORMALIZER}_bin5"
done

# ==============================================================================
# Helper functions
# ==============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is required but not installed."
        exit 1
    fi
}

# ==============================================================================
# Main script
# ==============================================================================

log "Starting PDF quality classifier training pipeline"
log "Local base: ${LOCAL_BASE_DIR}"
for classifier in "${CLASSIFIERS[@]}"; do
    log "  ${classifier} input: ${S3_INPUTS[${classifier}]}"
done

# Check required tools
check_command uv
check_command zstd
check_command s5cmd

# Create top-level output directories
for classifier in "${CLASSIFIERS[@]}"; do
    mkdir -p "${CALIBRATION_DIRS[${classifier}]}"
done

# ==============================================================================
# Step 1: Download annotated data from S3
# ==============================================================================

log "Step 1: Downloading annotated data from S3..."

for classifier in "${CLASSIFIERS[@]}"; do
    data_dir="${DATA_DIRS[${classifier}]}"
    s3_input="${S3_INPUTS[${classifier}]}"

    log "  ${classifier}: ${s3_input}"

    if [[ -d "${data_dir}" ]] && [[ -n "$(ls -A "${data_dir}" 2>/dev/null)" ]]; then
        log "    Skipping download - data already exists at ${data_dir}"
    else
        mkdir -p "${data_dir}"
        s5cmd cp -sp "${s3_input}/*" "${data_dir}/"
        log "    Downloaded to ${data_dir}"
    fi

    file_count=$(find "${data_dir}" -type f \( -name "*.jsonl.zst" -o -name "*.jsonl.gz" -o -name "*.jsonl" \) | wc -l)
    if [[ "${file_count}" -eq 0 ]]; then
        echo "Error: No JSONL files found for ${classifier} in ${data_dir}"
        exit 1
    fi
done

log "Step 1 complete."

# ==============================================================================
# Step 2: Split train/valid/test for each classifier
# ==============================================================================

log "Step 2: Splitting data into train/valid/test..."

for classifier in "${CLASSIFIERS[@]}"; do
    log "  Processing ${classifier}..."

    input_dir="${DATA_DIRS[${classifier}]}"
    split_dir="${SPLIT_DATA_DIRS[${classifier}]}"

    if [[ -d "${split_dir}/train" ]] && [[ -d "${split_dir}/valid" ]] && [[ -d "${split_dir}/test" ]]; then
        log "    Skipping ${classifier} - already split"
        continue
    fi

    # For small datasets, avoid requesting more test rows than available.
    test_split_size_this="${TEST_SPLIT_SIZE}"
    dir_size_mb=$(du -sm "${input_dir}" 2>/dev/null | cut -f1)
    if [[ "${dir_size_mb}" -lt 100 ]]; then
        log "    Small dataset detected (${dir_size_mb}MB), checking row count..."
        line_count=$(find "${input_dir}" -type f \( -name "*.jsonl.zst" -o -name "*.jsonl.gz" -o -name "*.jsonl" \) 2>/dev/null \
            | xargs -I{} sh -c 'case "{}" in *.zst) zstdcat "{}";; *.gz) zcat "{}";; *) cat "{}";; esac' 2>/dev/null \
            | wc -l)
        log "    Row count: ${line_count}"

        threshold=$((line_count / 10))
        if [[ "${TEST_SPLIT_SIZE}" -gt "${threshold}" ]]; then
            test_split_size_this="0.1"
            log "    Using 10% split (${test_split_size_this}) instead of ${TEST_SPLIT_SIZE}"
        fi
    fi

    uv run bonepick reshard-dataset \
        --dataset-dir "${input_dir}" \
        --output-dir "${split_dir}" \
        --num-files "${NUM_FILES}" \
        --test-split-frac "${test_split_size_this}" \
        --valid-split-frac "${test_split_size_this}"
done

log "Step 2 complete."

# ==============================================================================
# Step 3: Convert to FastText format with 5 bins
# ==============================================================================

log "Step 3: Converting to FastText format with 5 bins..."

for classifier in "${CLASSIFIERS[@]}"; do
    log "  Processing ${classifier}..."

    split_dir="${SPLIT_DATA_DIRS[${classifier}]}"
    preprocessed_dir="${PREPROCESSED_DIRS[${classifier}]}"

    if [[ -f "${preprocessed_dir}/train.txt" ]]; then
        log "    Skipping ${classifier} - already converted"
        continue
    fi

    # Convert score to bin1-bin5, clamping score into [1, 5].
    label_expr='"bin\([[.'"${classifier}"'.score // 1, 1] | max, 5] | min)"'

    uv run bonepick convert-to-fasttext \
        --input-dir "${split_dir}" \
        --output-dir "${preprocessed_dir}" \
        --normalization "${NORMALIZER}" \
        --text-expression "${TEXT_EXPRESSION}" \
        --label-expression "${label_expr}" \
        --max-length "${MAX_TEXT_LENGTH}"
done

log "Step 3 complete."

# ==============================================================================
# Step 4: Train FastText classifiers
# ==============================================================================

log "Step 4: Training FastText classifiers..."

for classifier in "${CLASSIFIERS[@]}"; do
    log "  Training ${classifier}..."

    preprocessed_dir="${PREPROCESSED_DIRS[${classifier}]}"
    model_dir="${MODEL_DIRS[${classifier}]}"

    if [[ -f "${model_dir}/model.bin" ]]; then
        log "    Skipping ${classifier} - model already exists"
        continue
    fi

    uv run bonepick train-fasttext \
        --dataset-dir "${preprocessed_dir}" \
        --output-dir "${model_dir}" \
        --word-ngrams "${WORD_NGRAMS}" \
        --window-size "${WINDOW_SIZE}" \
        --epoch "${EPOCHS}" \
        --dimension "${DIMENSION}"
done

log "Step 4 complete."

# ==============================================================================
# Step 5: Run inference on valid and test sets
# ==============================================================================

log "Step 5: Running inference on valid and test sets..."

declare -A CALIBRATION_SPLITS

for classifier in "${CLASSIFIERS[@]}"; do
    log "  Inference for ${classifier}..."

    split_dir="${SPLIT_DATA_DIRS[${classifier}]}"
    model_dir="${MODEL_DIRS[${classifier}]}"
    calibration_dir="${CALIBRATION_DIRS[${classifier}]}"
    metadata_field="${METADATA_FIELDS[${classifier}]}"

    valid_count=$(find "${split_dir}/valid" -name "*.jsonl.zst" -o -name "*.jsonl.gz" -o -name "*.jsonl" 2>/dev/null \
        | xargs -I{} sh -c 'case "{}" in *.zst) zstdcat "{}";; *.gz) zcat "{}";; *) cat "{}";; esac' 2>/dev/null \
        | wc -l)

    if [[ "${valid_count}" -lt "${MIN_VALID_FOR_CALIBRATION}" ]]; then
        CALIBRATION_SPLITS["${classifier}"]="train"
        log "    Valid set has ${valid_count} samples (< ${MIN_VALID_FOR_CALIBRATION}), will calibrate on train set"
    else
        CALIBRATION_SPLITS["${classifier}"]="valid"
        log "    Valid set has ${valid_count} samples, will calibrate on valid set"
    fi

    splits=("valid" "test")
    if [[ "${CALIBRATION_SPLITS[${classifier}]}" == "train" ]]; then
        splits=("train" "valid" "test")
    fi

    for split in "${splits[@]}"; do
        input_dir="${split_dir}/${split}"
        output_dir="${calibration_dir}/${split}"

        if [[ -d "${output_dir}" ]] && [[ -n "$(ls -A "${output_dir}" 2>/dev/null)" ]]; then
            log "    Skipping ${classifier}/${split} - already inferred"
            continue
        fi

        uv run bonepick infer-fasttext \
            --input-dir "${input_dir}" \
            --output-dir "${output_dir}" \
            --normalizer "${NORMALIZER}" \
            --model-dir "${model_dir}" \
            --classifier-name "${metadata_field}" \
            --text-expression "${TEXT_EXPRESSION}" \
            --max-length "${MAX_TEXT_LENGTH}"
    done
done

log "Step 5 complete."

# ==============================================================================
# Step 6: Train calibration model
# ==============================================================================

log "Step 6: Training calibration models..."

declare -A CALIBRATION_EXPRESSIONS

for classifier in "${CLASSIFIERS[@]}"; do
    log "  Training calibration for ${classifier}..."

    model_dir="${MODEL_DIRS[${classifier}]}"
    calibration_dir="${CALIBRATION_DIRS[${classifier}]}"
    metadata_field="${METADATA_FIELDS[${classifier}]}"
    cal_split="${CALIBRATION_SPLITS[${classifier}]:-valid}"
    calibration_file="${model_dir}/calibration.yaml"

    log "    Using ${cal_split} set for calibration"

    if [[ -f "${calibration_file}" ]]; then
        log "    Loading existing calibration for ${classifier}"
        CALIBRATION_EXPRESSIONS["${classifier}"]=$(uv run --with=pyyaml python3 -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['jq_expression'])" "${calibration_file}")
        continue
    fi

    uv run bonepick train-calibration \
        -d "${calibration_dir}/${cal_split}" \
        -p ".metadata.${metadata_field}" \
        -l ".${classifier}.score" \
        --output-file "${calibration_file}"

    jq_expr=$(uv run --with=pyyaml python3 -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['jq_expression'])" "${calibration_file}")
    CALIBRATION_EXPRESSIONS["${classifier}"]="${jq_expr}"
    log "    Calibration expression: ${jq_expr}"
done

log "Step 6 complete."

# ==============================================================================
# Step 7: Evaluate calibration on test sets
# ==============================================================================

log "Step 7: Evaluating calibration on test sets..."

for classifier in "${CLASSIFIERS[@]}"; do
    log "  Evaluating ${classifier}..."

    model_dir="${MODEL_DIRS[${classifier}]}"
    calibration_dir="${CALIBRATION_DIRS[${classifier}]}"
    calibration_file="${model_dir}/calibration.yaml"
    results_file="${model_dir}/calibration_results.txt"

    if [[ ! -f "${calibration_file}" ]]; then
        log "    Skipping ${classifier} - no calibration model"
        continue
    fi

    jq_expr=$(uv run --with=pyyaml python3 -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['jq_expression'])" "${calibration_file}")

    echo "Calibration Results - ${classifier}" > "${results_file}"
    echo "========================================" >> "${results_file}"
    echo "" >> "${results_file}"
    echo "Rubric field: ${classifier}" >> "${results_file}"
    echo "Calibration expression: ${jq_expr}" >> "${results_file}"
    echo "" >> "${results_file}"

    uv run bonepick eval-calibration \
        -d "${calibration_dir}/test" \
        -p "${jq_expr}" \
        -l ".${classifier}.score" 2>&1 | tee -a "${results_file}"
done

log "Step 7 complete."

# ==============================================================================
# Summary
# ==============================================================================

log "Pipeline complete!"
log ""
for classifier in "${CLASSIFIERS[@]}"; do
    log "${classifier}:"
    log "  Downloaded data: ${DATA_DIRS[${classifier}]}"
    log "  Split data:      ${SPLIT_DATA_DIRS[${classifier}]}"
    log "  Preprocessed:    ${PREPROCESSED_DIRS[${classifier}]}"
    log "  Model:           ${MODEL_DIRS[${classifier}]}"
    log "  Calibration:     ${CALIBRATION_DIRS[${classifier}]}"
    log "  Results:         ${MODEL_DIRS[${classifier}]}/calibration_results.txt"
    log ""
done

log "To use a trained model for inference on new data:"
log ""
log "  uv run bonepick infer-fasttext \\"
log "      -i <input_dir> \\"
log "      -o <output_dir> \\"
log "      --normalizer ${NORMALIZER} \\"
log "      -m ${LOCAL_BASE_DIR}/trained_models/fasttext/<classifier>_${NORMALIZER}_bin5 \\"
log "      -c <classifier>_${NORMALIZER}_bin5 \\"
log "      --text-expression '${TEXT_EXPRESSION}' \\"
log "      --max-length ${MAX_TEXT_LENGTH}"
