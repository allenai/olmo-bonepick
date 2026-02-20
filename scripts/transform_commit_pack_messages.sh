#!/usr/bin/env bash
set -euo pipefail

# Annotate bigcode_commitpack data using batch LLM annotation.
# Processes one programming language at a time, skipping languages
# whose output already exists on S3.
#
# Steps per language:
#   1. Download language data from S3
#   2. Submit batch annotation job
#   3. Wait for batch to complete and retrieve results
#   4. Upload annotated data back to S3

S3_SRC="s3://ai2-llm/pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages_tagged_resharded"
S3_DST="s3://ai2-llm/pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten"
LOCAL_DATA="/mnt/raid0/ai2-llm/pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages_tagged_resharded"
BATCH_DIR="/mnt/raid0/ai2-llm/pretraining-data/sources/bigcode_commitpack/batch_commitpack_rewrite"
OUTPUT_DIR="/mnt/raid0/ai2-llm/pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten"

MODEL="gpt-5-nano"
TASK_PROMPT="commit_to_request_short"
SYSTEM_PROMPT="code_system"
INPUT_FIELD=".message"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# List available languages from S3 source
echo "=== Discovering languages from S3 ==="
languages=$(aws s3 ls "${S3_SRC}/" | awk '{print $NF}' | sed 's:/$::')
echo "Found languages: ${languages}"
echo ""

for pl in ${languages}; do
    # Check if output already exists on S3
    if aws s3 ls "${S3_DST}/${pl}/" > /dev/null 2>&1; then
        echo -e "${YELLOW}Skipping ${pl}: output already exists on S3${NC}"
        continue
    fi

    echo -e "${GREEN}=== Processing ${pl} ===${NC}"

    # Step 1: Download language data from S3
    echo "  Downloading ${pl} from S3..."
    mkdir -p "${LOCAL_DATA}/${pl}"
    aws s3 sync "${S3_SRC}/${pl}/" "${LOCAL_DATA}/${pl}/" --no-progress
    echo "  Download complete."

    # Step 2: Submit batch annotation job
    echo "  Submitting batch annotation job..."
    uv run --extra=annotate bonepick batch-annotate-submit \
        -d "${LOCAL_DATA}/${pl}" \
        -b "${BATCH_DIR}/${pl}" \
        -m "${MODEL}" \
        -T "${TASK_PROMPT}" \
        -S "${SYSTEM_PROMPT}" \
        -i "${INPUT_FIELD}"
    echo "  Batch submitted."

    # Step 3: Retrieve batch results
    echo "  Retrieving batch results..."
    uv run --extra=annotate bonepick batch-annotate-retrieve \
        -b "${BATCH_DIR}/${pl}" \
        -o "${OUTPUT_DIR}/${pl}"
    echo "  Batch results retrieved."

    # Step 4: Upload language results to S3
    echo "  Uploading ${pl} results to S3..."
    aws s3 sync "${OUTPUT_DIR}/${pl}/" "${S3_DST}/${pl}/" --no-progress
    echo "  Upload complete."

    echo -e "${GREEN}=== Completed ${pl} ===${NC}"
    echo ""
done

echo "=== Done ==="
