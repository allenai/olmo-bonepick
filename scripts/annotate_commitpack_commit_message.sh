#!/bin/bash
# Annotates all bigcode_commitpack files using the stack_edu_commit_message rubric.
#
# Input: /mnt/raid0/ai2-llm/pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages/
# Output: ~/ai2-llm/classifiers/code-quality/data/bigcode_commitpack/dolma-3_5-languages_annotated/
# S3: s3://ai2-llm/classifiers/code-quality/data/bigcode_commitpack/dolma-3_5-languages_annotated/
#
# Usage:
#   ./scripts/annotate_commitpack_commit_message.sh

set -euo pipefail

# Configuration
INPUT_DIR="/mnt/raid0/ai2-llm/pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages"
OUTPUT_BASE="/mnt/raid0/ai2-llm/classifiers/code-quality/data/bigcode_commitpack/dolma-3_5-languages_annotated"
S3_OUTPUT="s3://ai2-llm/classifiers/code-quality/data/bigcode_commitpack/dolma-3_5-languages_annotated"
MODEL_NAME="gpt-5-mini"
SERVICE_TIER="default"
MAX_TEXT_LENGTH=10000
MAX_CONCURRENT_REQUESTS=5000
LIMIT_ROWS=1000000
CACHE_LOCATION="/mnt/raid0/annotation_cache.db"
ANNOTATION_TASK="stack_edu_commit_message"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Commitpack Commit Message Annotation"
echo "========================================"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_BASE}"
echo "S3 destination: ${S3_OUTPUT}"
echo "Model: ${MODEL_NAME}"
echo "Annotation task: ${ANNOTATION_TASK}"
echo "Max text length: ${MAX_TEXT_LENGTH}"
echo "Cache location: ${CACHE_LOCATION}"
echo "Service tier: ${SERVICE_TIER}"
echo "========================================"
echo ""

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_BASE}"

# Process each programming language
for pl_dir in "${INPUT_DIR}"/*; do
    if [[ ! -d "${pl_dir}" ]]; then
        continue
    fi

    pl=$(basename "${pl_dir}")
    output_pl_dir="${OUTPUT_BASE}/${pl}"

    # Skip if output directory already exists
    if [[ -d "${output_pl_dir}" ]]; then
        echo -e "${YELLOW}Skipping ${pl}: output directory already exists${NC}"
        continue
    fi

    echo -e "${GREEN}Processing ${pl}...${NC}"

    uv run --extra=annotate bonepick annotate-dataset \
        --dataset-dir "${pl_dir}" \
        --output-dir "${output_pl_dir}" \
        --model-name "${MODEL_NAME}" \
        --service-tier ${SERVICE_TIER} \
        --annotation-task-prompt "${ANNOTATION_TASK}" \
        --input-field-expression '.message' \
        --max-concurrent-requests ${MAX_CONCURRENT_REQUESTS} \
        --max-new-tokens 4096 \
        --max-text-length ${MAX_TEXT_LENGTH} \
        --cache-location "${CACHE_LOCATION}" \
        --limit-rows ${LIMIT_ROWS}

    echo -e "${GREEN}Completed ${pl}${NC}"
    echo ""
done

echo "========================================"
echo -e "${GREEN}All languages annotated!${NC}"
echo "========================================"

# Upload to S3
echo ""
echo "========================================"
echo "Uploading to S3..."
echo "========================================"
s5cmd cp -sp "${OUTPUT_BASE}/*" "${S3_OUTPUT}/"
echo -e "${GREEN}Upload complete: ${S3_OUTPUT}${NC}"
echo "========================================"
