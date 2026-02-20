#!/usr/bin/env bash
set -euo pipefail

# Annotate bigcode_commitpack data using batch LLM annotation.
# Steps:
#   1. Download data from S3
#   2. Submit batch annotation job
#   3. Wait for batch to complete and retrieve results
#   4. Upload annotated data back to S3

S3_SRC="s3://ai2-llm/pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages_tagged_resharded/"
S3_DST="s3://ai2-llm/pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten/"
LOCAL_DATA="/mnt/raid0/ai2-llm/pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages_tagged_resharded"
BATCH_DIR="/mnt/raid0/ai2-llm/pretraining-data/sources/bigcode_commitpack/batch_commitpack_rewrite"
OUTPUT_DIR="/mnt/raid0/ai2-llm/pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten"

MODEL="gpt-5-nano"
TASK_PROMPT="commit_to_request_short"
SYSTEM_PROMPT="code_system"
INPUT_FIELD=".message"

echo "=== Step 1: Download data from S3 ==="
mkdir -p "${LOCAL_DATA}"
aws s3 sync "${S3_SRC}" "${LOCAL_DATA}/" --no-progress
echo "Download complete."

echo ""
echo "=== Step 2: Submit batch annotation job ==="
uv run bonepick batch-annotate-submit \
    -d "${LOCAL_DATA}" \
    -b "${BATCH_DIR}" \
    -m "${MODEL}" \
    -T "${TASK_PROMPT}" \
    -S "${SYSTEM_PROMPT}" \
    -i "${INPUT_FIELD}"
echo "Batch submitted."

echo ""
echo "=== Step 3: Retrieve batch results ==="
uv run bonepick batch-annotate-retrieve \
    -b "${BATCH_DIR}" \
    -o "${OUTPUT_DIR}"
echo "Batch results retrieved."

echo ""
echo "=== Step 4: Upload results to S3 ==="
aws s3 sync "${OUTPUT_DIR}/" "${S3_DST}" --no-progress
echo "Upload complete."

echo ""
echo "=== Done ==="
