#!/usr/bin/env bash
set -euo pipefail

# Annotate bigcode_commitpack data using batch LLM annotation.
# Processes one programming language at a time, skipping languages
# whose output already exists on S3.
#
# Steps per language:
#   1. Download language data from S3 (skip if batch already submitted)
#   2. Submit batch annotation job (skip if batch already submitted)
#   3. Wait for batch to complete and retrieve results (1h timeout, non-fatal)
#   4. Upload annotated data back to S3

LOCAL_BASE_DIR=${LOCAL_BASE_DIR:-"/mnt/raid0/ai2-llm"}
REMOTE_BASE_DIR=${REMOTE_BASE_DIR:-"s3://ai2-llm"}
DATA_DIR=${DATA_DIR:-"pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages_tagged_resharded"}
BATCH_DIR=${BATCH_DIR:-"pretraining-data/sources/bigcode_commitpack/batch_commitpack_rewrite"}
OUTPUT_DIR=${OUTPUT_DIR:-"pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten"}

# ====================== #

S3_DATA_DIR="${REMOTE_BASE_DIR}/${DATA_DIR}"
S3_OUTPUT_DIR="${REMOTE_BASE_DIR}/${OUTPUT_DIR}"
LOCAL_SRC_DIR="${LOCAL_BASE_DIR}/${DATA_DIR}"
LOCAL_BATCH_DIR="${LOCAL_BASE_DIR}/${BATCH_DIR}"
LOCAL_OUTPUT_DIR="${LOCAL_BASE_DIR}/${OUTPUT_DIR}"

MODEL="gpt-5-nano"
TASK_PROMPT="commit_to_request_short"
SYSTEM_PROMPT="code_system"
INPUT_FIELD=".message"

RETRIEVE_TIMEOUT=3600  # 1 hour

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track languages that fail to retrieve
failed_languages=()

# List available languages from S3 source
echo "=== Discovering languages from S3 ==="
languages=$(aws s3 ls "${S3_DATA_DIR}/" | awk '{print $NF}' | sed 's:/$::')
echo "Found languages: ${languages}"
echo ""

for pl in ${languages}; do
    # Check if output already exists on S3
    if aws s3 ls "${S3_OUTPUT_DIR}/${pl}/" > /dev/null 2>&1; then
        echo -e "${YELLOW}Skipping ${pl}: output already exists on S3${NC}"
        continue
    fi

    echo -e "${GREEN}=== Processing ${pl} ===${NC}"

    # Check if batch was already submitted (batch dir exists)
    if [[ -d "${LOCAL_BATCH_DIR}/${pl}" ]]; then
        echo -e "${YELLOW}  Batch already submitted for ${pl}, skipping to retrieval${NC}"
    else
        # Step 1: Download language data from S3
        echo "  Downloading ${pl} from S3..."
        s5cmd cp -sp "${S3_DATA_DIR}/${pl}/*" "${LOCAL_SRC_DIR}/${pl}/"
        echo "  Download complete."

        # Step 2: Submit batch annotation job
        echo "  Submitting batch annotation job..."
        uv run --extra=annotate bonepick batch-annotate-submit \
            -d "${LOCAL_SRC_DIR}/${pl}" \
            -b "${LOCAL_BATCH_DIR}/${pl}" \
            -m "${MODEL}" \
            -T "${TASK_PROMPT}" \
            -S "${SYSTEM_PROMPT}" \
            -i "${INPUT_FIELD}"
        echo "  Batch submitted."
    fi

    # Step 3: Retrieve batch results (non-fatal, 1h timeout)
    echo "  Retrieving batch results (timeout: ${RETRIEVE_TIMEOUT}s)..."
    if uv run --extra=annotate bonepick batch-annotate-retrieve \
        -b "${LOCAL_BATCH_DIR}/${pl}" \
        -o "${LOCAL_OUTPUT_DIR}/${pl}" \
        --timeout "${RETRIEVE_TIMEOUT}"; then
        echo "  Batch results retrieved."

        # Step 4: Upload language results to S3
        echo "  Uploading ${pl} results to S3..."
        s5cmd cp -sp "${LOCAL_OUTPUT_DIR}/${pl}/*" "${S3_OUTPUT_DIR}/${pl}/"
        echo "  Upload complete."

        echo -e "${GREEN}=== Completed ${pl} ===${NC}"
    else
        echo -e "${RED}  Failed to retrieve batch results for ${pl}, moving on${NC}"
        failed_languages+=("${pl}")
    fi
    echo ""
done

# Report failed languages
if [[ ${#failed_languages[@]} -gt 0 ]]; then
    echo -e "${RED}=== Failed to retrieve results for the following languages ===${NC}"
    for pl in "${failed_languages[@]}"; do
        echo -e "${RED}  - ${pl}${NC}"
    done
    echo ""
fi

echo "=== Done ==="
