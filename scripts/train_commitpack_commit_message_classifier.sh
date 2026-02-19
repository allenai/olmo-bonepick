#!/bin/bash
# Train a single FastText classifier for commitpack commit message quality
# across ALL programming languages combined.
#
# Data source: s3://ai2-llm/classifiers/code-quality/data/bigcode_commitpack/dolma-3_5-languages_annotated
# Annotation rubric: stack_edu_commit_message (scores 0-5)
# Text field: .message (the commit message, not the code)
#
# Pipeline:
# 1. Download annotated data from S3
# 2. Split combined (all languages) into train/valid/test using reshard-dataset
# 3. Convert to FastText format with 5 bins
# 4. Train FastText classifier
# 5. Run inference on valid/test sets
# 6. Fit calibration on valid set
# 7. Evaluate calibration on test set
# 8. Upload all artifacts to S3
#
# ==============================================================================
# USAGE
# ==============================================================================
#
# Basic usage (uses default paths):
#   ./scripts/train_commitpack_commit_message_classifier.sh
#
# With custom local base directory:
#   LOCAL_BASE_DIR=/path/to/workdir ./scripts/train_commitpack_commit_message_classifier.sh
#
# Environment variables:
#   LOCAL_BASE_DIR    - Base directory for all local data/outputs (default: ~/ai2-llm/classifiers/code-quality)
#   S3_INPUT          - S3 path to annotated data (default: see below)
#   S3_BASE           - S3 base path for all outputs (default: s3://ai2-llm/classifiers/code-quality)
#   S3_OUTPUT         - S3 path for trained model (default: ${S3_BASE}/trained_models/...)
#
# ==============================================================================

set -e

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

# Normalizer
NORMALIZER="ultrafine-commits"

# Rubric field (from stack_edu_commit_message annotation)
RUBRIC_FIELD="stack_edu_commit_message"

# Text expression: commit message is in .message field
TEXT_EXPRESSION=".message"

# Metadata field name for inference output
METADATA_FIELD="commitpack_commit_message_ultrafine_bin5"

# Minimum valid samples to use valid set for calibration (otherwise use train)
MIN_VALID_FOR_CALIBRATION=5000

# S3 paths
S3_INPUT="${S3_INPUT:-s3://ai2-llm/classifiers/code-quality/data/bigcode_commitpack/dolma-3_5-languages_annotated}"
S3_BASE="${S3_BASE:-s3://ai2-llm/classifiers/code-quality}"
S3_OUTPUT="${S3_OUTPUT:-${S3_BASE}/trained_models/fasttext/commitpack_commit_message_${NORMALIZER}_bin5}"
S3_SPLIT_DATA="${S3_SPLIT_DATA:-${S3_BASE}/data-train_test_split/bigcode_commitpack/commitpack_commit_message}"
S3_PREPROCESSED="${S3_PREPROCESSED:-${S3_BASE}/preprocessed/bigcode_commitpack/commitpack_commit_message/fasttext/ultrafine_bin5}"
S3_CALIBRATION="${S3_CALIBRATION:-${S3_BASE}/calibration/commitpack_commit_message}"

# Local paths
LOCAL_BASE_DIR="${LOCAL_BASE_DIR:-${HOME}/ai2-llm/classifiers/code-quality}"
DATA_DIR="${LOCAL_BASE_DIR}/data/bigcode_commitpack/dolma-3_5-languages_annotated"
SPLIT_DATA_DIR="${LOCAL_BASE_DIR}/data-train_test_split/bigcode_commitpack/commitpack_commit_message"
PREPROCESSED_DIR="${LOCAL_BASE_DIR}/preprocessed/bigcode_commitpack/commitpack_commit_message/fasttext/ultrafine_bin5"
MODELS_DIR="${LOCAL_BASE_DIR}/trained_models/fasttext/commitpack_commit_message_${NORMALIZER}_bin5"
CALIBRATION_DIR="${LOCAL_BASE_DIR}/calibration/commitpack_commit_message"


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

log "Starting commitpack commit message classifier training pipeline"
log "S3 input:  ${S3_INPUT}"
log "S3 output: ${S3_OUTPUT}"
log "Local base: ${LOCAL_BASE_DIR}"

# Check required tools
check_command uv
check_command zstd
check_command s5cmd

# Create output directories
mkdir -p "${CALIBRATION_DIR}"

# ==============================================================================
# Step 1: Download annotated data from S3
# ==============================================================================

log "Step 1: Downloading annotated data from S3..."

if [[ -d "${DATA_DIR}" ]] && [[ -n "$(ls -A "${DATA_DIR}" 2>/dev/null)" ]]; then
    log "  Skipping download - data already exists at ${DATA_DIR}"
else
    mkdir -p "${DATA_DIR}"
    s5cmd cp -sp "${S3_INPUT}/*" "${DATA_DIR}/"
    log "  Downloaded to ${DATA_DIR}"
fi

# Discover language directories
LANGUAGES=($(ls --color=never "${DATA_DIR}" 2>/dev/null | grep -v '^\.' || true))

if [[ ${#LANGUAGES[@]} -eq 0 ]]; then
    echo "Error: No language directories found in ${DATA_DIR}"
    exit 1
fi

log "Found ${#LANGUAGES[@]} languages: ${LANGUAGES[*]}"
log "Step 1 complete."

# ==============================================================================
# Step 2: Split combined data into train/valid/test
# ==============================================================================

log "Step 2: Splitting combined data into train/valid/test..."

if [[ -d "${SPLIT_DATA_DIR}/train" ]] && [[ -d "${SPLIT_DATA_DIR}/valid" ]] && [[ -d "${SPLIT_DATA_DIR}/test" ]]; then
    log "  Skipping - already split"
else
    # Build -d flags for all language directories
    DATASET_DIR_FLAGS=()
    for lang in "${LANGUAGES[@]}"; do
        DATASET_DIR_FLAGS+=("--dataset-dir" "${DATA_DIR}/${lang}")
    done

    uv run bonepick reshard-dataset \
        "${DATASET_DIR_FLAGS[@]}" \
        --output-dir "${SPLIT_DATA_DIR}" \
        --num-files "${NUM_FILES}" \
        --test-split-frac "${TEST_SPLIT_SIZE}" \
        --valid-split-frac "${TEST_SPLIT_SIZE}"
fi

log "Step 2 complete."

# ==============================================================================
# Step 3: Convert to FastText format with 5 bins
# ==============================================================================

log "Step 3: Converting to FastText format with 5 bins..."

if [[ -f "${PREPROCESSED_DIR}/train.txt" ]]; then
    log "  Skipping - already converted"
else
    # Label expression: bin scores 0-5 into bin1-bin5
    # Scores 0 and 1 both map to bin1; scores 2-5 map to bin2-bin5
    LABEL_EXPR="\"bin\\([[.${RUBRIC_FIELD}.score // 1, 1] | max, 5] | min)\""

    uv run bonepick convert-to-fasttext \
        --input-dir "${SPLIT_DATA_DIR}" \
        --output-dir "${PREPROCESSED_DIR}" \
        --normalization "${NORMALIZER}" \
        --text-expression "${TEXT_EXPRESSION}" \
        --label-expression "${LABEL_EXPR}" \
        --max-length "${MAX_TEXT_LENGTH}"
fi

log "Step 3 complete."

# ==============================================================================
# Step 4: Train FastText classifier
# ==============================================================================

log "Step 4: Training FastText classifier..."

if [[ -f "${MODELS_DIR}/model.bin" ]]; then
    log "  Skipping - model already exists"
else
    uv run bonepick train-fasttext \
        --dataset-dir "${PREPROCESSED_DIR}" \
        --output-dir "${MODELS_DIR}" \
        --word-ngrams "${WORD_NGRAMS}" \
        --window-size "${WINDOW_SIZE}" \
        --epoch "${EPOCHS}" \
        --dimension "${DIMENSION}"
fi

log "Step 4 complete."

# ==============================================================================
# Step 5: Run inference on valid and test sets
# ==============================================================================

log "Step 5: Running inference on valid and test sets..."

# Count valid set size to decide calibration split
valid_count=$(find "${SPLIT_DATA_DIR}/valid" -name "*.jsonl.zst" -o -name "*.jsonl.gz" -o -name "*.jsonl" 2>/dev/null \
    | xargs -I{} sh -c 'case "{}" in *.zst) zstdcat "{}";; *.gz) zcat "{}";; *) cat "{}";; esac' 2>/dev/null \
    | wc -l)

if [[ "${valid_count}" -lt "${MIN_VALID_FOR_CALIBRATION}" ]]; then
    CALIBRATION_SPLIT="train"
    log "  Valid set has ${valid_count} samples (< ${MIN_VALID_FOR_CALIBRATION}), will calibrate on train set"
else
    CALIBRATION_SPLIT="valid"
    log "  Valid set has ${valid_count} samples, will calibrate on valid set"
fi

# Determine which splits need inference
SPLITS=("valid" "test")
if [[ "${CALIBRATION_SPLIT}" == "train" ]]; then
    SPLITS=("train" "valid" "test")
fi

for split in "${SPLITS[@]}"; do
    input_dir="${SPLIT_DATA_DIR}/${split}"
    output_dir="${CALIBRATION_DIR}/${split}"

    if [[ -d "${output_dir}" ]] && [[ -n "$(ls -A "${output_dir}" 2>/dev/null)" ]]; then
        log "  Skipping ${split} - already inferred"
        continue
    fi

    log "  Running inference on ${split}..."

    uv run bonepick infer-fasttext \
        --input-dir "${input_dir}" \
        --output-dir "${output_dir}" \
        --normalizer "${NORMALIZER}" \
        --model-dir "${MODELS_DIR}" \
        --classifier-name "${METADATA_FIELD}" \
        --text-expression "${TEXT_EXPRESSION}" \
        --max-length "${MAX_TEXT_LENGTH}"
done

log "Step 5 complete."

# ==============================================================================
# Step 6: Train calibration model on valid (or train) set
# ==============================================================================

log "Step 6: Training calibration model..."

calibration_file="${MODELS_DIR}/calibration.yaml"

if [[ -f "${calibration_file}" ]]; then
    log "  Loading existing calibration"
else
    cal_dir="${CALIBRATION_DIR}/${CALIBRATION_SPLIT}"

    log "  Using ${CALIBRATION_SPLIT} set for calibration"

    uv run bonepick train-calibration \
        -d "${cal_dir}" \
        -p ".metadata.${METADATA_FIELD}" \
        -l "[[.${RUBRIC_FIELD}.score // 1, 1] | max, 5] | min" \
        --output-file "${calibration_file}"
fi

# Extract JQ expression from calibration YAML
jq_expr=$(uv run --with=pyyaml python3 -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['jq_expression'])" "${calibration_file}")
log "  Calibration expression: ${jq_expr}"

log "Step 6 complete."

# ==============================================================================
# Step 7: Evaluate calibration on test set
# ==============================================================================

log "Step 7: Evaluating calibration on test set..."

results_file="${MODELS_DIR}/calibration_results.txt"

echo "Calibration Results - Commitpack Commit Message Classifier" > "${results_file}"
echo "==========================================================" >> "${results_file}"
echo "" >> "${results_file}"
echo "Rubric field: ${RUBRIC_FIELD}" >> "${results_file}"
echo "Calibration expression: ${jq_expr}" >> "${results_file}"
echo "" >> "${results_file}"

uv run bonepick eval-calibration \
    -d "${CALIBRATION_DIR}/test" \
    -p "${jq_expr}" \
    -l "[[.${RUBRIC_FIELD}.score // 1, 1] | max, 5] | min" 2>&1 | tee -a "${results_file}"

log "Step 7 complete."

# ==============================================================================
# Step 8: Upload all artifacts to S3
# ==============================================================================

log "Step 8: Uploading artifacts to S3..."

log "  Uploading split data..."
s5cmd sync "${SPLIT_DATA_DIR}/*" "${S3_SPLIT_DATA}/"

log "  Uploading preprocessed data..."
s5cmd sync "${PREPROCESSED_DIR}/*" "${S3_PREPROCESSED}/"

log "  Uploading trained model..."
s5cmd sync "${MODELS_DIR}/*" "${S3_OUTPUT}/"

log "  Uploading calibration data..."
s5cmd sync "${CALIBRATION_DIR}/*" "${S3_CALIBRATION}/"

log "Step 8 complete."

# ==============================================================================
# Summary
# ==============================================================================

log "Pipeline complete!"
log ""
log "Local outputs:"
log "  Downloaded data:  ${DATA_DIR}"
log "  Split data:       ${SPLIT_DATA_DIR}"
log "  Preprocessed:     ${PREPROCESSED_DIR}"
log "  Model:            ${MODELS_DIR}"
log "  Calibration:      ${CALIBRATION_DIR}"
log "  Results:          ${results_file}"
log ""
log "S3 outputs:"
log "  Annotated data:   ${S3_INPUT}"
log "  Split data:       ${S3_SPLIT_DATA}"
log "  Preprocessed:     ${S3_PREPROCESSED}"
log "  Model:            ${S3_OUTPUT}"
log "  Calibration:      ${S3_CALIBRATION}"
log ""
log "To use the trained model for inference on new data:"
log ""
log "  uv run bonepick infer-fasttext \\"
log "      -i <input_dir> \\"
log "      -o <output_dir> \\"
log "      --normalizer ${NORMALIZER} \\"
log "      -m ${MODELS_DIR} \\"
log "      -c ${METADATA_FIELD} \\"
log "      --text-expression '${TEXT_EXPRESSION}' \\"
log "      --max-length ${MAX_TEXT_LENGTH}"
