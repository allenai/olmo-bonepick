#!/bin/bash
# Train quality classifiers for stack_edu_redux data
#
# This script follows the procedure from WORKLOG.md:
# 1. Split train/test/valid using reshard-dataset
# 2. Preprocess using ultrafine formatter and convert to FastText format (5 bins)
# 3. Train FastText classifier
# 4. Run inference on valid/test sets
# 5. Fit linear regression (calibration) on valid set
# 6. Evaluate calibration on test set
#
# ==============================================================================
# USAGE
# ==============================================================================
#
# Basic usage (uses default paths):
#   ./scripts/train_stack_edu_redux_classifier.sh
#
# With custom input data directory:
#   INPUT_DATA_DIR=/path/to/stack_edu_redux ./scripts/train_stack_edu_redux_classifier.sh
#
# Example with the specific path from the task:
#   INPUT_DATA_DIR=/home/ec2-user/ai2-llm/classifiers/code-quality/data/the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB/stack_edu_redux \
#   LOCAL_BASE_DIR=/home/ec2-user/ai2-llm/classifiers/code-quality \
#   ./scripts/train_stack_edu_redux_classifier.sh
#
# Environment variables:
#   INPUT_DATA_DIR    - Directory containing annotated data (with language subdirs)
#   LOCAL_BASE_DIR    - Base directory for all outputs
#   RUBRIC_PREFIX     - Prefix for rubric field names (default: stack_edu_redux)
#
# Expected input directory structure:
#   ${INPUT_DATA_DIR}/
#     Python/
#       train/
#         *.jsonl.zst
#     JavaScript/
#       train/
#         *.jsonl.zst
#     ...
#
# Each JSONL file should have:
#   - "text" field with the code
#   - "<rubric_prefix>_<language>.score" field with the score (1-5)
#
# ==============================================================================

set -e

# ==============================================================================
# Configuration
# ==============================================================================

# Base paths - adjust these for your environment
LOCAL_BASE_DIR="${LOCAL_BASE_DIR:-${HOME}/ai2-llm/classifiers/code-quality}"
BASE_NAME_PREFIX="the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB"

# Input data directory containing annotated data with stack_edu_redux scores
INPUT_DATA_DIR="${INPUT_DATA_DIR:-${LOCAL_BASE_DIR}/data/${BASE_NAME_PREFIX}/stack_edu_redux}"

# Output directories
SPLIT_DATA_DIR="${LOCAL_BASE_DIR}/data-train_test_split/${BASE_NAME_PREFIX}/stack_edu_redux"
PREPROCESSED_DIR="${LOCAL_BASE_DIR}/preprocessed/${BASE_NAME_PREFIX}/stack_edu_redux/fasttext/ultrafine_bin5"
MODELS_DIR="${LOCAL_BASE_DIR}/trained_models/fasttext/stack_edu_redux_ultrafine_bin5"
CALIBRATION_DIR="${LOCAL_BASE_DIR}/calibration/stack_edu_redux"

# Training parameters
NUM_FILES=$(($(nproc) + 2))  # Number of training files after resharding (CPU cores + 2)
TEST_SPLIT_SIZE=10000  # Number of test samples
VALID_SPLIT_SIZE=10000 # Number of validation samples
MAX_TEXT_LENGTH=10000  # Max text length for FastText

# FastText hyperparameters
WORD_NGRAMS=5
WINDOW_SIZE=10
EPOCHS=10
DIMENSION=512

# Normalizer
NORMALIZER="ultrafine"

# Metadata field name for inference output
METADATA_FIELD=$(basename "${MODELS_DIR}")

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
# Detect rubric name from directory structure
# ==============================================================================

# The label expression needs to match the annotation field name
# For stack_edu_redux, each language has its own annotation field like:
#   .stack_edu_redux_python.score
#   .stack_edu_redux_javascript.score
# etc.
#
# Set RUBRIC_PREFIX to override (e.g., "stack_edu" for older annotations)
RUBRIC_PREFIX="${RUBRIC_PREFIX:-stack_edu_redux}"

get_rubric_field() {
    local lang="$1"
    # Convert language name to lowercase and replace hyphens/spaces with underscores
    local lang_lower=$(echo "$lang" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr ' ' '_')

    # Handle special cases (C++, C#)
    if [[ "$lang_lower" == "c++" ]]; then
        lang_lower="cpp"
    elif [[ "$lang_lower" == "c#" ]] || [[ "$lang_lower" == "c_sharp" ]]; then
        lang_lower="csharp"
    fi

    echo "${RUBRIC_PREFIX}_${lang_lower}"
}

# Auto-detect rubric field from data if possible
auto_detect_rubric_field() {
    local data_dir="$1"
    local lang="$2"

    # Try to find a sample file and extract the rubric field
    local sample_file=$(find "${data_dir}" -name "*.jsonl.zst" -o -name "*.jsonl.gz" -o -name "*.jsonl" 2>/dev/null | head -1)

    if [[ -n "${sample_file}" ]]; then
        local fields
        if [[ "${sample_file}" == *.zst ]]; then
            fields=$(zstdcat "${sample_file}" 2>/dev/null | head -1 | jq -r 'keys[]' 2>/dev/null | grep -E "stack_edu.*score" | head -1 || true)
        elif [[ "${sample_file}" == *.gz ]]; then
            fields=$(zcat "${sample_file}" 2>/dev/null | head -1 | jq -r 'keys[]' 2>/dev/null | grep -E "stack_edu" | head -1 || true)
        else
            fields=$(head -1 "${sample_file}" 2>/dev/null | jq -r 'keys[]' 2>/dev/null | grep -E "stack_edu" | head -1 || true)
        fi

        if [[ -n "${fields}" ]]; then
            echo "${fields}"
            return
        fi
    fi

    # Fall back to computed field name
    get_rubric_field "${lang}"
}

# ==============================================================================
# Main script
# ==============================================================================

log "Starting stack_edu_redux classifier training pipeline"
log "Input data: ${INPUT_DATA_DIR}"

# Check required tools
check_command uv
check_command zstd

# Create output directories
mkdir -p "${CALIBRATION_DIR}"

# Get list of programming languages from input directory
if [[ ! -d "${INPUT_DATA_DIR}" ]]; then
    echo "Error: Input data directory not found: ${INPUT_DATA_DIR}"
    echo ""
    echo "Expected directory structure:"
    echo "  ${INPUT_DATA_DIR}/"
    echo "    Python/"
    echo "      train/"
    echo "        *.jsonl.zst"
    echo "    JavaScript/"
    echo "      train/"
    echo "        *.jsonl.zst"
    echo "    ..."
    exit 1
fi

LANGUAGES=($(ls --color=never "${INPUT_DATA_DIR}" 2>/dev/null | grep -v '^\.' || true))

if [[ ${#LANGUAGES[@]} -eq 0 ]]; then
    echo "Error: No language directories found in ${INPUT_DATA_DIR}"
    exit 1
fi

log "Found ${#LANGUAGES[@]} languages: ${LANGUAGES[*]}"

# ==============================================================================
# Step 1: Split train/valid/test for each language
# ==============================================================================

log "Step 1: Splitting data into train/valid/test..."

for lang in "${LANGUAGES[@]}"; do
    log "  Processing ${lang}..."

    input_lang_dir="${INPUT_DATA_DIR}/${lang}"
    output_lang_dir="${SPLIT_DATA_DIR}/${lang}"

    if [[ -d "${output_lang_dir}/train" ]] && [[ -d "${output_lang_dir}/valid" ]] && [[ -d "${output_lang_dir}/test" ]]; then
        log "    Skipping ${lang} - already split"
        continue
    fi

    uv run bonepick reshard-dataset \
        --dataset-dir "${input_lang_dir}" \
        --output-dir "${output_lang_dir}" \
        --num-files "${NUM_FILES}" \
        --test-split-frac "${TEST_SPLIT_SIZE}" \
        --valid-split-frac "${VALID_SPLIT_SIZE}"
done

log "Step 1 complete."

# ==============================================================================
# Step 2: Convert to FastText format with 5 bins
# ==============================================================================

log "Step 2: Converting to FastText format with 5 bins..."

for lang in "${LANGUAGES[@]}"; do
    log "  Processing ${lang}..."

    input_lang_dir="${SPLIT_DATA_DIR}/${lang}"
    output_lang_dir="${PREPROCESSED_DIR}/${lang}"

    if [[ -f "${output_lang_dir}/train.txt" ]]; then

        log "    Skipping ${lang} - already converted"
        continue
    fi

    # Try to auto-detect the rubric field from the data
    rubric_field=$(auto_detect_rubric_field "${input_lang_dir}" "${lang}")
    log "    Using rubric field: ${rubric_field}"

    # Label expression: bin scores 1-5 into bin1-bin5
    # The expression ensures scores are clamped to [1,5] range
    label_expr='"bin\([[.'"${rubric_field}"'.score // 1, 1] | max, 5] | min)"'

    uv run bonepick convert-to-fasttext \
        --input-dir "${input_lang_dir}" \
        --output-dir "${output_lang_dir}" \
        --normalization "${NORMALIZER}" \
        --label-expression "${label_expr}" \
        --max-length "${MAX_TEXT_LENGTH}"

    # Save the rubric field for later steps
    echo "${rubric_field}" > "${output_lang_dir}/rubric_field.txt"
done

log "Step 2 complete."

# ==============================================================================
# Step 3: Train FastText classifiers
# ==============================================================================

log "Step 3: Training FastText classifiers..."

for lang in "${LANGUAGES[@]}"; do
    log "  Training ${lang}..."

    dataset_dir="${PREPROCESSED_DIR}/${lang}"
    model_dir="${MODELS_DIR}/${lang}"

    if [[ -f "${model_dir}/model.bin" ]]; then
        log "    Skipping ${lang} - model already exists"
        continue
    fi

    uv run bonepick train-fasttext \
        --dataset-dir "${dataset_dir}" \
        --output-dir "${model_dir}" \
        --word-ngrams "${WORD_NGRAMS}" \
        --window-size "${WINDOW_SIZE}" \
        --epoch "${EPOCHS}" \
        --dimension "${DIMENSION}"
done

log "Step 3 complete."

# ==============================================================================
# Step 4: Run inference on valid and test sets
# ==============================================================================

log "Step 4: Running inference on valid and test sets..."

for lang in "${LANGUAGES[@]}"; do
    log "  Inference for ${lang}..."

    model_dir="${MODELS_DIR}/${lang}"

    for split in "valid" "test"; do
        input_dir="${SPLIT_DATA_DIR}/${lang}/${split}"
        output_dir="${CALIBRATION_DIR}/${lang}_${split}"

        if [[ -d "${output_dir}" ]] && [[ -n "$(ls -A ${output_dir} 2>/dev/null)" ]]; then
            log "    Skipping ${lang}/${split} - already inferred"
            continue
        fi

        uv run bonepick infer-fasttext \
            --input-dir "${input_dir}" \
            --output-dir "${output_dir}" \
            --normalizer "${NORMALIZER}" \
            --model-dir "${model_dir}" \
            --classifier-name "${METADATA_FIELD}" \
            --max-length "${MAX_TEXT_LENGTH}"
    done
done

log "Step 4 complete."

# ==============================================================================
# Step 5: Train calibration model on valid set
# ==============================================================================

log "Step 5: Training calibration models..."

# Store calibration expressions for each language
declare -A CALIBRATION_EXPRESSIONS

for lang in "${LANGUAGES[@]}"; do
    log "  Training calibration for ${lang}..."

    valid_dir="${CALIBRATION_DIR}/${lang}_valid"
    calibration_file="${MODELS_DIR}/${lang}/calibration.yaml"

    # Load the rubric field from preprocessing step, or compute it
    rubric_field_file="${PREPROCESSED_DIR}/${lang}/rubric_field.txt"
    if [[ -f "${rubric_field_file}" ]]; then
        rubric_field=$(cat "${rubric_field_file}")
    else
        rubric_field=$(get_rubric_field "${lang}")
    fi

    if [[ -f "${calibration_file}" ]]; then
        log "    Loading existing calibration for ${lang}"
        CALIBRATION_EXPRESSIONS["${lang}"]=$(uv run --with=pyyaml python3 -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['jq_expression'])" "${calibration_file}")
        continue
    fi

    log "    Using rubric field: ${rubric_field}"

    # Run calibration training with output file
    uv run bonepick train-calibration \
        -d "${valid_dir}" \
        -p ".metadata.${METADATA_FIELD}" \
        -l ".${rubric_field}.score" \
        --output-file "${calibration_file}"

    # Extract JQ expression from YAML output
    jq_expr=$(uv run --with=pyyaml python3 -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['jq_expression'])" "${calibration_file}")

    if [[ -n "${jq_expr}" ]]; then
        CALIBRATION_EXPRESSIONS["${lang}"]="${jq_expr}"
        log "    Calibration expression: ${jq_expr}"
    else
        log "    Warning: Could not extract calibration expression for ${lang}"
    fi
done

log "Step 5 complete."

# ==============================================================================
# Step 6: Evaluate calibration on test set
# ==============================================================================

log "Step 6: Evaluating calibration on test sets..."

results_file="${MODELS_DIR}/calibration_results.txt"
echo "Calibration Results Summary" > "${results_file}"
echo "============================" >> "${results_file}"
echo "" >> "${results_file}"

for lang in "${LANGUAGES[@]}"; do
    log "  Evaluating ${lang}..."

    test_dir="${CALIBRATION_DIR}/${lang}_test"
    calibration_file="${MODELS_DIR}/${lang}/calibration.yaml"

    # Load the rubric field from preprocessing step, or compute it
    rubric_field_file="${PREPROCESSED_DIR}/${lang}/rubric_field.txt"
    if [[ -f "${rubric_field_file}" ]]; then
        rubric_field=$(cat "${rubric_field_file}")
    else
        rubric_field=$(get_rubric_field "${lang}")
    fi

    if [[ ! -f "${calibration_file}" ]]; then
        log "    Skipping ${lang} - no calibration model"
        continue
    fi

    jq_expr=$(uv run --with=pyyaml python3 -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['jq_expression'])" "${calibration_file}")

    echo "Language: ${lang}" >> "${results_file}"
    echo "----------------" >> "${results_file}"
    echo "Rubric field: ${rubric_field}" >> "${results_file}"

    uv run bonepick eval-calibration \
        -d "${test_dir}" \
        -p "${jq_expr}" \
        -l ".${rubric_field}.score" 2>&1 | tee -a "${results_file}"

    echo "" >> "${results_file}"
    echo "" >> "${results_file}"
done

log "Step 6 complete."

# ==============================================================================
# Summary
# ==============================================================================

log "Pipeline complete!"
log ""
log "Outputs:"
log "  Split data:      ${SPLIT_DATA_DIR}"
log "  Preprocessed:    ${PREPROCESSED_DIR}"
log "  Models:          ${MODELS_DIR}"
log "  Calibration tmp: ${CALIBRATION_DIR}"
log "  Results:         ${results_file}"
log ""
log "To use a trained model for inference on new data:"
log ""
log "  uv run bonepick infer-fasttext \\"
log "      -i <input_dir> \\"
log "      -o <output_dir> \\"
log "      --normalizer ${NORMALIZER} \\"
log "      -m ${MODELS_DIR}/<language> \\"
log "      -c ${METADATA_FIELD} \\"
log "      --max-length ${MAX_TEXT_LENGTH}"
