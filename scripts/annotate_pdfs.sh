#!/usr/bin/env bash
set -eEuo pipefail

# Annotate PDF samples using batch LLM annotation.
#
# Usage: ./annotate_pdfs.sh [submit|retrieve]
#   submit   - Only run Phase 1 (sync inputs + submit batch jobs)
#   retrieve - Only run Phase 2 (retrieve batch results)
#   (no arg) - Run both phases

# ============= Logging and error handling ============= #

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] [INFO] $*"
}

warn() {
    echo -e "${YELLOW}[$(timestamp)] [WARN] $*${NC}" >&2
}

error() {
    echo -e "${RED}[$(timestamp)] [ERROR] $*${NC}" >&2
}

die() {
    error "$*"
    exit 1
}

on_error() {
    local exit_code=$?
    error "Command failed (exit ${exit_code}) at line ${BASH_LINENO[0]}: ${BASH_COMMAND}"
}

trap on_error ERR
trap 'warn "Interrupted by user"; exit 130' INT TERM

for cmd in uv s5cmd; do
    command -v "${cmd}" > /dev/null || die "Required command not found: ${cmd}"
done

# ============= Parse arguments ============= #

MODE="${1:-all}"
if [[ "${MODE}" != "all" && "${MODE}" != "submit" && "${MODE}" != "retrieve" ]]; then
    echo "Usage: $0 [submit|retrieve]"
    echo "  submit   - Only run Phase 1 (sync inputs + submit batch jobs)"
    echo "  retrieve - Only run Phase 2 (retrieve batch results)"
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

# ============= Configure paths/options ============= #

LOCAL_PREFIX="${LOCAL_PREFIX:-/mnt/raid0/ai2-llm}"
REMOTE_PREFIX="${REMOTE_PREFIX:-s3://ai2-llm}"
REL_INPUT_DIR="${INPUT_DIR:-classifiers/pdf-quality/samples}"
REL_BATCH_DIR="${BATCH_DIR:-classifiers/pdf-quality/batch}"
REL_OUTPUT_DIR="${OUTPUT_DIR:-classifiers/pdf-quality/data}"

LOCAL_INPUT_DIR="${LOCAL_PREFIX}/${REL_INPUT_DIR}"
LOCAL_BATCH_DIR="${LOCAL_PREFIX}/${REL_BATCH_DIR}"
LOCAL_OUTPUT_DIR="${LOCAL_PREFIX}/${REL_OUTPUT_DIR}"
REMOTE_INPUT_DIR="${REMOTE_PREFIX}/${REL_INPUT_DIR}"

TASK_PROMPTS=(
    "finepdfish_edu"
    "finepdfish_ocr"
    "finepdfish_dclm"
)

MODEL_NAME="${MODEL_NAME:-gpt-5-mini}"
INPUT_FIELD="${INPUT_FIELD:-.text}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-general_rubric_system}"
if command -v nproc > /dev/null; then
    DEFAULT_NUM_PROC="$(nproc)"
else
    DEFAULT_NUM_PROC="1"
fi
NUM_PROC="${NUM_PROC:-${DEFAULT_NUM_PROC}}"
RETRIEVE_TIMEOUT="${RETRIEVE_TIMEOUT:-3600}"
ALLOW_SKIP_BATCHES="${ALLOW_SKIP_BATCHES:-false}"

SKIP_FAILED_BATCHES_FLAG=()
case "${ALLOW_SKIP_BATCHES,,}" in
    1|true|yes|y)
        SKIP_FAILED_BATCHES_FLAG=(--skip-failed-batches)
        ;;
    0|false|no|n|"")
        ;;
    *)
        die "Invalid ALLOW_SKIP_BATCHES value: '${ALLOW_SKIP_BATCHES}' (expected true/false)"
        ;;
esac

mkdir -p "${LOCAL_INPUT_DIR}" "${LOCAL_BATCH_DIR}" "${LOCAL_OUTPUT_DIR}"

log "=== Configuration ==="
log "Mode: ${MODE}"
log "Model: ${MODEL_NAME}"
log "Task prompts: ${TASK_PROMPTS[*]}"
log "System prompt: ${SYSTEM_PROMPT}"
log "Input field: ${INPUT_FIELD}"
log "Remote input directory: ${REMOTE_INPUT_DIR}"
log "Local input directory: ${LOCAL_INPUT_DIR}"
log "Local batch directory: ${LOCAL_BATCH_DIR}"
log "Local output directory: ${LOCAL_OUTPUT_DIR}"
log "Retrieve timeout (s): ${RETRIEVE_TIMEOUT}"
log "Allow skip batches: ${ALLOW_SKIP_BATCHES}"
log "Number of processes: ${NUM_PROC}"
log "====================="

submitted_prompts=()
retrieved_prompts=()
failed_prompts=()

# ====== Phase 1: Sync + submit batch jobs ====== #

if ${DO_SUBMIT}; then
    log "=== Phase 1: Syncing source data ==="
    s5cmd sync "${REMOTE_INPUT_DIR}/*" "${LOCAL_INPUT_DIR}/"
    log "Source sync complete."

    log "=== Phase 1: Submitting batch jobs ==="
    for task_prompt in "${TASK_PROMPTS[@]}"; do
        log "Submitting task prompt: ${task_prompt}"
        uv run --extra=annotate bonepick batch-annotate-submit \
            -d "${LOCAL_INPUT_DIR}" \
            -b "${LOCAL_BATCH_DIR}" \
            -m "${MODEL_NAME}" \
            -T "${task_prompt}" \
            -S "${SYSTEM_PROMPT}" \
            -i "${INPUT_FIELD}" \
            --num-proc "${NUM_PROC}"
        submitted_prompts+=("${task_prompt}")
        log "Submitted task prompt: ${task_prompt}"
    done
fi

# ====== Phase 2: Retrieve batch results ====== #

if ${DO_RETRIEVE}; then
    log "=== Phase 2: Retrieving batch results ==="
    for task_prompt in "${TASK_PROMPTS[@]}"; do
        prompt_batch_dir="${LOCAL_BATCH_DIR}/${task_prompt}"
        prompt_output_dir="${LOCAL_OUTPUT_DIR}/${task_prompt}"

        if [[ ! -d "${prompt_batch_dir}" ]]; then
            warn "Skipping ${task_prompt}: batch directory not found at ${prompt_batch_dir}"
            failed_prompts+=("${task_prompt}")
            continue
        fi

        mkdir -p "${prompt_output_dir}"
        log "Retrieving task prompt: ${task_prompt} (timeout: ${RETRIEVE_TIMEOUT}s)"
        if uv run --extra=annotate bonepick batch-annotate-retrieve \
            -b "${prompt_batch_dir}" \
            -o "${prompt_output_dir}" \
            --timeout "${RETRIEVE_TIMEOUT}" \
            "${SKIP_FAILED_BATCHES_FLAG[@]}"; then
            retrieved_prompts+=("${task_prompt}")
            log "Retrieved task prompt: ${task_prompt}"
        else
            failed_prompts+=("${task_prompt}")
            warn "Failed retrieval for ${task_prompt}; continuing with remaining prompts"
        fi
    done
fi

log "=== Summary ==="
log "Submitted prompts: ${#submitted_prompts[@]}"
if [[ ${#submitted_prompts[@]} -gt 0 ]]; then
    log "Submitted list: ${submitted_prompts[*]}"
fi
log "Retrieved prompts: ${#retrieved_prompts[@]}"
if [[ ${#retrieved_prompts[@]} -gt 0 ]]; then
    log "Retrieved list: ${retrieved_prompts[*]}"
fi

if [[ ${#failed_prompts[@]} -gt 0 ]]; then
    warn "Failed prompts: ${#failed_prompts[@]}"
    warn "Failed list: ${failed_prompts[*]}"
    exit 1
fi

log "Done."
