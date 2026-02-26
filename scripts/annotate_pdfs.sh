#!/bin/bash

set -euo pipefail

# Configuration
LOCAL_PREFIX="${LOCAL_PREFIX:-/mnt/raid0/ai2-llm}"
REMOTE_PREFIX="${REMOTE_PREFIX:-s3://ai2-llm}"
REL_INPUT_DIR="${INPUT_DIR:-classifiers/pdf-quality/samples}"
REL_OUTPUT_DIR="${OUTPUT_DIR:-classifiers/pdf-quality/data}"

TASK_PROMPTS=(
    "finepdfish_edu"
    "finepdfish_ocr"
    "finepdfish_dclm"
)
MODEL_NAME="gpt-5-mini"
MAX_TEXT_LENGTH=10000
LIMIT_ROWS=1000000
INPUT_FIELD=".text"
SYSTEM_PROMPT="general_rubric_system"
NUM_PROC=${NUM_PROC:-$(nproc)}
RETRIEVE_TIMEOUT=${RETRIEVE_TIMEOUT:-3600}

# cli parsing
MODE="${1:-all}"
if [[ "${MODE}" != "all" && "${MODE}" != "submit" && "${MODE}" != "retrieve" ]]; then
    echo "Usage: $0 [submit|retrieve]"
    echo "  submit   - Only submit batch jobs (Phase 1)"
    echo "  retrieve - Only retrieve batch results (Phase 2)"
    echo "  (no arg) - Run both phases"
    exit 1
fi

# are we okay if some batches fail?
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


if [[ "${MODE}" == "all" || "${MODE}" == "submit" ]]; then
    # Step 1: Download the data
    s5cmd sync "${REMOTE_PREFIX}/${REL_INPUT_DIR}/*" "${LOCAL_PREFIX}/${REL_INPUT_DIR}/"

    # Step 2: Submit for annotation
    for TASK_PROMPT in "${TASK_PROMPTS[@]}"; do
        uv run --extra=annotate bonepick batch-annotate-submit \
            -d "${LOCAL_PREFIX}/${REL_INPUT_DIR}" \
            -b "${LOCAL_PREFIX}/${REL_OUTPUT_DIR}" \
            -m "${MODEL_NAME}" \
            -T "${TASK_PROMPT}" \
            -S "${SYSTEM_PROMPT}" \
            -i "${INPUT_FIELD}" \
            --num-proc ${NUM_PROC}
    done
fi

if [[ "${MODE}" == "all" || "${MODE}" == "retrieve" ]]; then
    # Step 3: Retrieve the results

    for TASK_PROMPT in "${TASK_PROMPTS[@]}"; do
        uv run --extra=annotate bonepick batch-annotate-retrieve \
            -b "${LOCAL_PREFIX}/${REL_OUTPUT_DIR}/${TASK_PROMPT}" \
            -o "${LOCAL_PREFIX}/${REL_OUTPUT_DIR}/${TASK_PROMPT}" \
            --timeout "${RETRIEVE_TIMEOUT}" \
            "${SKIP_FAILED_BATCHES_FLAG[@]}"
    done
fi
