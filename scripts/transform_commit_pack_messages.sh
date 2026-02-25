#!/usr/bin/env bash
set -euo pipefail

# Annotate bigcode_commitpack data using batch LLM annotation.
# Skips languages whose output already exists on S3.
#
# Usage: ./transform_commit_pack_messages.sh [submit|retrieve]
#   submit   - Only run Phase 1 (download data and submit batch jobs)
#   retrieve - Only run Phase 2 (retrieve results and upload to S3)
#   (no arg) - Run both phases
#
# Phase 1 (submit): For each language, download data and submit batch job
# Phase 2 (retrieve): For each submitted language, retrieve results and upload to S3

# ============= Parse arguments ============= #

MODE="${1:-all}"
if [[ "${MODE}" != "all" && "${MODE}" != "submit" && "${MODE}" != "retrieve" ]]; then
    echo "Usage: $0 [submit|retrieve]"
    echo "  submit   - Only submit batch jobs (Phase 1)"
    echo "  retrieve - Only retrieve batch results (Phase 2)"
    echo "  (no arg) - Run both phases"
    exit 1
fi

DO_SUBMIT=false
DO_RETRIEVE=false
if [[ "${MODE}" == "all" || "${MODE}" == "submit" ]]; then
    DO_SUBMIT=true
fi
if [[ "${MODE}" == "all" || "${MODE}" == "retrieve" ]]; then
    DO_RETRIEVE=true
fi

# ============= Configure paths ============= #

LOCAL_BASE_DIR=${LOCAL_BASE_DIR:-"/mnt/raid0/ai2-llm"}
REMOTE_BASE_DIR=${REMOTE_BASE_DIR:-"s3://ai2-llm"}
DATA_DIR=${DATA_DIR:-"pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages_tagged_resharded"}
BATCH_DIR=${BATCH_DIR:-"pretraining-data/sources/bigcode_commitpack/batch_commitpack_rewrite"}
OUTPUT_DIR=${OUTPUT_DIR:-"pretraining-data/sources/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten"}


# ============ Configure options ============ #

MODEL=${MODEL:-"gpt-5-nano"}
TASK_PROMPT=${TASK_PROMPT:-"commit_to_request_short"}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"code_system"}
INPUT_FIELD=${INPUT_FIELD:-".message"}
RETRIEVE_TIMEOUT=${RETRIEVE_TIMEOUT:-3600}
NUM_PROC=${NUM_PROC:-$(nproc)}
ALLOW_SKIP_BATCHES=${ALLOW_SKIP_BATCHES:-false}

SKIP_FAILED_BATCHES_FLAG=()
case "${ALLOW_SKIP_BATCHES,,}" in
    1|true|yes|y)
        SKIP_FAILED_BATCHES_FLAG=(--skip-failed-batches)
        ;;
    0|false|no|n|"")
        ;;
    *)
        echo "Invalid ALLOW_SKIP_BATCHES value: '${ALLOW_SKIP_BATCHES}' (expected true/false)"
        exit 1
        ;;
esac

# =========== Beginning of script =========== #

# derive all paths from base directories and data/output dirs
S3_DATA_DIR="${REMOTE_BASE_DIR}/${DATA_DIR}"
S3_OUTPUT_DIR="${REMOTE_BASE_DIR}/${OUTPUT_DIR}"
LOCAL_SRC_DIR="${LOCAL_BASE_DIR}/${DATA_DIR}"
LOCAL_BATCH_DIR="${LOCAL_BASE_DIR}/${BATCH_DIR}"
LOCAL_OUTPUT_DIR="${LOCAL_BASE_DIR}/${OUTPUT_DIR}"

# print configuration
echo "=== Configuration ==="
echo "Mode: ${MODE}"
echo "Model: ${MODEL}"
echo "Task prompt: ${TASK_PROMPT}"
echo "System prompt: ${SYSTEM_PROMPT}"
echo "Input field: ${INPUT_FIELD}"
echo "S3 data directory: ${S3_DATA_DIR}"
echo "S3 output directory: ${S3_OUTPUT_DIR}"
echo "Local source directory: ${LOCAL_SRC_DIR}"
echo "Local batch directory: ${LOCAL_BATCH_DIR}"
echo "Local output directory: ${LOCAL_OUTPUT_DIR}"
echo "Retrieve timeout (s): ${RETRIEVE_TIMEOUT}"
echo "Allow skip batches: ${ALLOW_SKIP_BATCHES}"
echo "Number of processes: ${NUM_PROC}"
echo "====================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track languages that need retrieval
submitted_languages=()
failed_languages=()

# List available languages from S3 source
echo "=== Discovering languages from S3 ==="
languages=$(aws s3 ls "${S3_DATA_DIR}/" | awk '{print $NF}' | sed 's:/$::')
echo "Found languages: ${languages}"
echo ""

# ====== Phase 1: Submit all batch jobs ====== #

if ${DO_SUBMIT}; then
    echo "=== Phase 1: Submitting batch jobs ==="
    echo ""

    for pl in ${languages}; do
        # Check if output already exists on S3
        if aws s3 ls "${S3_OUTPUT_DIR}/${pl}/" > /dev/null 2>&1; then
            echo -e "${YELLOW}Skipping ${pl}: output already exists on S3${NC}"
            continue
        fi

        # Check if batch was already submitted (batch dir exists)
        if [[ -d "${LOCAL_BATCH_DIR}/${pl}" ]]; then
            echo -e "${YELLOW}  Batch already submitted for ${pl}, skipping download/submit${NC}"
            submitted_languages+=("${pl}")
            continue
        fi

        echo -e "${GREEN}=== Submitting ${pl} ===${NC}"

        # Download language data from S3
        echo "  Downloading ${pl} from S3..."
        s5cmd cp -sp "${S3_DATA_DIR}/${pl}/step_final/*" "${LOCAL_SRC_DIR}/${pl}/step_final/"
        echo "  Download complete."

        # Submit batch annotation job
        echo "  Submitting batch annotation job..."
        uv run --extra=annotate bonepick batch-annotate-submit \
            -d "${LOCAL_SRC_DIR}/${pl}" \
            -b "${LOCAL_BATCH_DIR}/${pl}" \
            -m "${MODEL}" \
            -T "${TASK_PROMPT}" \
            -S "${SYSTEM_PROMPT}" \
            -i "${INPUT_FIELD}" \
            --num-proc ${NUM_PROC}
        echo "  Batch submitted."

        submitted_languages+=("${pl}")
        echo ""
    done

    echo ""
    echo "=== Phase 1 complete: ${#submitted_languages[@]} language(s) submitted ==="
    echo ""
fi

# ====== Phase 2: Retrieve all batch results ====== #

if ${DO_RETRIEVE}; then
    # When running retrieve-only, discover languages from existing batch directories
    if ! ${DO_SUBMIT}; then
        echo "=== Discovering submitted languages from batch directory ==="
        for pl in ${languages}; do
            if [[ -d "${LOCAL_BATCH_DIR}/${pl}" ]]; then
                # Skip languages whose output already exists on S3
                if aws s3 ls "${S3_OUTPUT_DIR}/${pl}/" > /dev/null 2>&1; then
                    echo -e "${YELLOW}Skipping ${pl}: output already exists on S3${NC}"
                    continue
                fi
                submitted_languages+=("${pl}")
            fi
        done
        echo "Found ${#submitted_languages[@]} language(s) with pending batches"
        echo ""
    fi

    echo "=== Phase 2: Retrieving batch results ==="
    echo ""

    for pl in "${submitted_languages[@]}"; do
        echo -e "${GREEN}=== Retrieving ${pl} ===${NC}"

        # Retrieve batch results (non-fatal, 1h timeout)
        echo "  Retrieving batch results (timeout: ${RETRIEVE_TIMEOUT}s)..."
        if uv run --extra=annotate bonepick batch-annotate-retrieve \
            -b "${LOCAL_BATCH_DIR}/${pl}" \
            -o "${LOCAL_OUTPUT_DIR}/${pl}" \
            --timeout "${RETRIEVE_TIMEOUT}" \
            "${SKIP_FAILED_BATCHES_FLAG[@]}"; then
            echo "  Batch results retrieved."

            # Upload language results to S3
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
fi

echo "=== Done ==="
