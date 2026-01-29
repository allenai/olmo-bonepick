#!/bin/bash
# Samples ~1GB per programming language from S3, deduplicates, and annotates using stack_edu_redux rubrics.
#
# This script handles three scenarios based on S3 data size:
#   - <1GB: Copy all data directly (no sampling needed, still deduplicate)
#   - 1GB-10GB: Download, deduplicate, and use bonepick sample-dataset
#   - >10GB: Sample ~10GB of files from S3 first, deduplicate, then use bonepick sample-dataset
#
# Usage:
#   ./scripts/sample_and_annotate_stack_edu_redux.sh
#
# Environment variables:
#   BASE_DIR              - Base directory for local data (default: $HOME)
#   S3_SOURCE_PREFIX      - S3 prefix for source data
#   TARGET_SAMPLE_SIZE    - Target sample size in bytes (default: 1GB)
#   MODEL_NAME            - Model for annotation (default: gpt-5-mini)
#   SERVICE_TIER          - OpenAI service tier (default: flex)
#   SKIP_DEDUP            - Set to "true" to skip deduplication step

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

BASE_DIR="${BASE_DIR:-${HOME}}"
S3_SOURCE_PREFIX="${S3_SOURCE_PREFIX:-s3://ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/data}"

# Duplodocus configuration
DUPLODOCUS_DIR="${BASE_DIR}/duplodocus"
DUPLODOCUS_BIN="${DUPLODOCUS_DIR}/target/release/duplodocus"

# Local directories
LOCAL_DATA_DIR="${BASE_DIR}/ai2-llm/pretraining-data/sources/the-stack-v2/spring2code_v2/data"
OUTPUT_BASE_DIR="${BASE_DIR}/ai2-llm/classifiers/code-quality/data/the-stack-v2/spring2code_v2/stack_edu_redux_additional"
DEDUPED_DIR="${OUTPUT_BASE_DIR}/deduped"
SAMPLE_DIR="${OUTPUT_BASE_DIR}/sampled"
ANNOTATED_DIR="${OUTPUT_BASE_DIR}/annotated"
CACHE_LOCATION="${BASE_DIR}/annotation_cache"
DEDUP_WORK_DIR="${OUTPUT_BASE_DIR}/dedup_work"

# Sampling parameters
TARGET_SAMPLE_SIZE=$((1 * 1024 * 1024 * 1024))  # 1GB in bytes
SMALL_THRESHOLD=$((1 * 1024 * 1024 * 1024))     # 1GB
MEDIUM_THRESHOLD=$((10 * 1024 * 1024 * 1024))   # 10GB
LARGE_SAMPLE_SIZE=$((10 * 1024 * 1024 * 1024))  # 10GB for pre-sampling large datasets

# Deduplication parameters (MinHash LSH)
MINHASH_NUM_BUCKETS=20
MINHASH_BUCKET_SIZE=5
MINHASH_NGRAM_SIZE=5

# Annotation parameters
MODEL_NAME="${MODEL_NAME:-gpt-5-mini}"
SERVICE_TIER="${SERVICE_TIER:-flex}"
MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-5000}
MAX_NEW_TOKENS=4096
MAX_TEXT_LENGTH=10000
LIMIT_ROWS=500000

# Skip deduplication flag
SKIP_DEDUP="${SKIP_DEDUP:-false}"

# Languages to process with their rubric mappings
# Format: "FolderName:rubric_name"
declare -a LANGUAGE_MAPPINGS=(
    "Blade:stack_edu_redux_blade"
    "Bluespec:stack_edu_redux_bluespec"
    "Clojure:stack_edu_redux_clojure"
    "Common_Lisp:stack_edu_redux_common_lisp"
    "CSS:stack_edu_redux_css"
    "Cuda:stack_edu_redux_cuda"
    "Dart:stack_edu_redux_dart"
    "Erlang:stack_edu_redux_erlang"
    "Fortran:stack_edu_redux_fortran"
    "Fortran_Free_Form:stack_edu_redux_fortran_free_form"
    "Haskell:stack_edu_redux_haskell"
    "HTML:stack_edu_redux_html"
    "Java_Server_Pages:stack_edu_redux_java_server_pages"
    "Julia:stack_edu_redux_julia"
    "Kotlin:stack_edu_redux_kotlin"
    "Lua:stack_edu_redux_lua"
    "Mathematica:stack_edu_redux_mathematica"
    "MATLAB:stack_edu_redux_matlab"
    "Objective-C:stack_edu_redux_objective_c"
    "OCaml:stack_edu_redux_ocaml"
    "OpenCL:stack_edu_redux_opencl"
    "Pascal:stack_edu_redux_pascal"
    "Perl:stack_edu_redux_perl"
    "R:stack_edu_redux_r"
    "RMarkdown:stack_edu_redux_rmarkdown"
    "Scala:stack_edu_redux_scala"
    "Scheme:stack_edu_redux_scheme"
    "SCSS:stack_edu_redux_scss"
    "SystemVerilog:stack_edu_redux_systemverilog"
    "Tcl:stack_edu_redux_tcl"
    "Verilog:stack_edu_redux_verilog"
    "VHDL:stack_edu_redux_vhdl"
    "Vue:stack_edu_redux_vue"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# Helper functions
# ==============================================================================

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_info() {
    log "${BLUE}INFO:${NC} $*"
}

log_success() {
    log "${GREEN}SUCCESS:${NC} $*"
}

log_warning() {
    log "${YELLOW}WARNING:${NC} $*"
}

log_error() {
    log "${RED}ERROR:${NC} $*"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is required but not installed."
        exit 1
    fi
}

human_readable_size() {
    local bytes=$1
    if [[ $bytes -ge $((1024 * 1024 * 1024)) ]]; then
        echo "$(echo "scale=2; $bytes / 1024 / 1024 / 1024" | bc)GB"
    elif [[ $bytes -ge $((1024 * 1024)) ]]; then
        echo "$(echo "scale=2; $bytes / 1024 / 1024" | bc)MB"
    elif [[ $bytes -ge 1024 ]]; then
        echo "$(echo "scale=2; $bytes / 1024" | bc)KB"
    else
        echo "${bytes}B"
    fi
}

# Get total size of S3 prefix in bytes
get_s3_size() {
    local s3_path="$1"
    local total_bytes=0

    # Use s5cmd for faster listing if available, otherwise fall back to aws cli
    # s5cmd ls output format: "date time size filename" - size is column 3
    # aws s3 ls output format: "date time size filename" - size is column 3
    if command -v s5cmd &> /dev/null; then
        total_bytes=$(s5cmd ls "${s3_path}/*" 2>/dev/null | awk '{sum += $3} END {print sum+0}')
    else
        total_bytes=$(aws s3 ls --recursive "${s3_path}/" 2>/dev/null | awk '{sum += $3} END {print sum+0}')
    fi

    echo "$total_bytes"
}

# Get local directory size in bytes
get_local_size() {
    local path="$1"
    if [[ -d "$path" ]]; then
        du -sb "$path" 2>/dev/null | cut -f1
    else
        echo "0"
    fi
}

# Copy all files from S3 to local
copy_all_from_s3() {
    local s3_path="$1"
    local local_path="$2"

    mkdir -p "$local_path"
    if command -v s5cmd &> /dev/null; then
        s5cmd cp -sp "${s3_path}/*" "${local_path}/"
    else
        aws s3 cp --recursive "${s3_path}/" "${local_path}/"
    fi
}

# Sample files from S3 (download a subset of files up to target size)
sample_files_from_s3() {
    local s3_path="$1"
    local local_path="$2"
    local target_size="$3"

    mkdir -p "$local_path"

    local cumulative_size=0
    local files_to_download=()

    # List files with sizes and accumulate until target size
    # s5cmd ls output format: "date time size filename" - size is $3, filename is $4 (just basename)
    # aws s3 ls output format: "date time size filename" - size is $3, filename is $4 (relative path)
    if command -v s5cmd &> /dev/null; then
        while IFS= read -r line; do
            local size=$(echo "$line" | awk '{print $3}')
            local file=$(echo "$line" | awk '{print $4}')

            if [[ -z "$size" ]] || [[ "$size" == "0" ]] || [[ -z "$file" ]]; then
                continue
            fi

            cumulative_size=$((cumulative_size + size))
            # s5cmd ls only returns the basename, so reconstruct full S3 path
            files_to_download+=("${s3_path}/${file}")

            if [[ $cumulative_size -ge $target_size ]]; then
                break
            fi
        done < <(s5cmd ls "${s3_path}/*" 2>/dev/null | sort -R)
    else
        while IFS= read -r line; do
            local size=$(echo "$line" | awk '{print $3}')
            local file=$(echo "$line" | awk '{print $4}')

            if [[ -z "$size" ]] || [[ "$size" == "0" ]] || [[ -z "$file" ]]; then
                continue
            fi

            cumulative_size=$((cumulative_size + size))
            files_to_download+=("s3://$(echo "$s3_path" | sed 's|s3://||')/${file##*/}")

            if [[ $cumulative_size -ge $target_size ]]; then
                break
            fi
        done < <(aws s3 ls --recursive "${s3_path}/" 2>/dev/null | sort -R)
    fi

    log_info "Downloading ${#files_to_download[@]} files ($(human_readable_size $cumulative_size))"

    # Download selected files in parallel
    if command -v s5cmd &> /dev/null; then
        # Use s5cmd run for parallel downloads - generate cp commands and pipe to s5cmd run
        printf '%s\n' "${files_to_download[@]}" | \
            sed "s|.*|cp & ${local_path}/|" | \
            s5cmd run
    else
        # Fallback to sequential aws s3 cp
        for file in "${files_to_download[@]}"; do
            aws s3 cp "$file" "${local_path}/"
        done
    fi
}

# Run fuzzy deduplication using Duplodocus
run_deduplication() {
    local input_dir="$1"
    local output_dir="$2"
    local pl_name="$3"

    local work_dir="${DEDUP_WORK_DIR}/${pl_name}"
    mkdir -p "$work_dir"
    mkdir -p "$output_dir"

    log_info "  Running MinHash fuzzy deduplication..."

    "${DUPLODOCUS_BIN}" minhash-memory \
        --input-dir "$input_dir" \
        --storage-dir "$work_dir" \
        --output-dir "$output_dir" \
        --text-key "text" \
        --tokenizer cl100k \
        --num-buckets ${MINHASH_NUM_BUCKETS} \
        --bucket-size ${MINHASH_BUCKET_SIZE} \
        --ngram-size ${MINHASH_NGRAM_SIZE} \
        --remove-duplicates true \
        --cleanup-storage

    # Count documents before and after
    local docs_before=$(find "$input_dir" -name "*.jsonl*" -exec zstdcat {} 2>/dev/null \; 2>/dev/null | wc -l || echo "?")
    local docs_after=$(find "$output_dir" -name "*.jsonl*" -exec zstdcat {} 2>/dev/null \; 2>/dev/null | wc -l || echo "?")

    log_info "  Deduplication complete: ${docs_before} -> ${docs_after} documents"
}

# ==============================================================================
# Setup: Clone and build Duplodocus
# ==============================================================================

setup_duplodocus() {
    log_info "Setting up Duplodocus..."

    # Check if already built
    if [[ -f "${DUPLODOCUS_BIN}" ]]; then
        log_info "Duplodocus already built at ${DUPLODOCUS_BIN}"
        return 0
    fi

    # Check for Rust
    if ! command -v cargo &> /dev/null; then
        log_info "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    # Clone if not present
    if [[ ! -d "${DUPLODOCUS_DIR}" ]]; then
        log_info "Cloning Duplodocus repository..."
        git clone https://github.com/allenai/duplodocus.git "${DUPLODOCUS_DIR}"
    fi

    # Build
    log_info "Building Duplodocus (this may take a few minutes)..."
    cd "${DUPLODOCUS_DIR}"
    cargo build --release
    cd - > /dev/null

    if [[ ! -f "${DUPLODOCUS_BIN}" ]]; then
        log_error "Failed to build Duplodocus"
        exit 1
    fi

    log_success "Duplodocus built successfully"
}

# ==============================================================================
# Main script
# ==============================================================================

echo "========================================"
echo "Stack Edu Redux Sampling and Annotation"
echo "========================================"
echo "S3 source: ${S3_SOURCE_PREFIX}"
echo "Local base: ${BASE_DIR}"
echo "Deduped output: ${DEDUPED_DIR}"
echo "Sample output: ${SAMPLE_DIR}"
echo "Annotated output: ${ANNOTATED_DIR}"
echo "Target sample size: $(human_readable_size $TARGET_SAMPLE_SIZE)"
echo "Model: ${MODEL_NAME}"
echo "Service tier: ${SERVICE_TIER}"
echo "Skip dedup: ${SKIP_DEDUP}"
echo "========================================"
echo ""

# Check required tools
check_command uv
check_command bc
check_command git
if ! command -v s5cmd &> /dev/null && ! command -v aws &> /dev/null; then
    log_error "Either s5cmd or aws CLI is required"
    exit 1
fi

# Setup Duplodocus (unless skipping dedup)
if [[ "${SKIP_DEDUP}" != "true" ]]; then
    setup_duplodocus
fi

# Create output directories
mkdir -p "${DEDUPED_DIR}"
mkdir -p "${SAMPLE_DIR}"
mkdir -p "${ANNOTATED_DIR}"
mkdir -p "${LOCAL_DATA_DIR}"
mkdir -p "${DEDUP_WORK_DIR}"

# ==============================================================================
# Phase 1: Download and Deduplicate
# ==============================================================================

log_info "Phase 1: Downloading and deduplicating data from S3"

for mapping in "${LANGUAGE_MAPPINGS[@]}"; do
    pl_folder=$(echo "$mapping" | cut -d: -f1)
    rubric=$(echo "$mapping" | cut -d: -f2)

    deduped_output_dir="${DEDUPED_DIR}/${pl_folder}"
    sample_output_dir="${SAMPLE_DIR}/${pl_folder}"

    # Skip if deduped data already exists
    if [[ -d "${deduped_output_dir}" ]] && [[ -n "$(ls -A "${deduped_output_dir}" 2>/dev/null)" ]]; then
        log_warning "Skipping ${pl_folder}: deduped data already exists"
        continue
    fi

    s3_path="${S3_SOURCE_PREFIX}/${pl_folder}"
    local_download_dir="${LOCAL_DATA_DIR}/${pl_folder}"

    log_info "Processing ${pl_folder}..."

    # Get S3 size
    s3_size=$(get_s3_size "$s3_path")

    if [[ "$s3_size" == "0" ]] || [[ -z "$s3_size" ]]; then
        log_warning "No data found for ${pl_folder} at ${s3_path}"
        continue
    fi

    log_info "  S3 size: $(human_readable_size $s3_size)"

    # Step 1: Download data
    if [[ $s3_size -lt $SMALL_THRESHOLD ]]; then
        # Case 1: <1GB - copy all
        log_info "  Strategy: Copy all (small dataset)"
        if [[ ! -d "${local_download_dir}" ]] || [[ -z "$(ls -A "${local_download_dir}" 2>/dev/null)" ]]; then
            copy_all_from_s3 "$s3_path" "$local_download_dir"
        else
            log_info "  Using existing download at ${local_download_dir}"
        fi

    elif [[ $s3_size -lt $MEDIUM_THRESHOLD ]]; then
        # Case 2: 1GB-10GB - download all
        log_info "  Strategy: Download all, deduplicate, then sample"
        if [[ ! -d "${local_download_dir}" ]] || [[ -z "$(ls -A "${local_download_dir}" 2>/dev/null)" ]]; then
            log_info "  Downloading to ${local_download_dir}..."
            copy_all_from_s3 "$s3_path" "$local_download_dir"
        else
            log_info "  Using existing download at ${local_download_dir}"
        fi

    else
        # Case 3: >10GB - sample files from S3 first
        log_info "  Strategy: Pre-sample from S3, deduplicate, then sample"
        if [[ ! -d "${local_download_dir}" ]] || [[ -z "$(ls -A "${local_download_dir}" 2>/dev/null)" ]]; then
            log_info "  Pre-sampling ~$(human_readable_size $LARGE_SAMPLE_SIZE) from S3..."
            sample_files_from_s3 "$s3_path" "$local_download_dir" "$LARGE_SAMPLE_SIZE"
        else
            log_info "  Using existing pre-sample at ${local_download_dir}"
        fi
    fi

    # Step 2: Deduplicate
    if [[ "${SKIP_DEDUP}" != "true" ]]; then
        run_deduplication "$local_download_dir" "$deduped_output_dir" "$pl_folder"
    else
        log_info "  Skipping deduplication (SKIP_DEDUP=true)"
        # Just copy/link the data
        mkdir -p "$deduped_output_dir"
        cp -r "${local_download_dir}"/* "$deduped_output_dir/" 2>/dev/null || true
    fi

    log_success "Completed download and dedup for ${pl_folder}"
done

log_success "Phase 1 complete: All languages downloaded and deduplicated"

# ==============================================================================
# Phase 2: Sampling
# ==============================================================================

log_info "Phase 2: Sampling deduplicated data"

for mapping in "${LANGUAGE_MAPPINGS[@]}"; do
    pl_folder=$(echo "$mapping" | cut -d: -f1)
    rubric=$(echo "$mapping" | cut -d: -f2)

    deduped_input_dir="${DEDUPED_DIR}/${pl_folder}"
    sample_output_dir="${SAMPLE_DIR}/${pl_folder}"

    # Skip if sample already exists
    if [[ -d "${sample_output_dir}" ]] && [[ -n "$(ls -A "${sample_output_dir}" 2>/dev/null)" ]]; then
        log_warning "Skipping ${pl_folder}: sample already exists"
        continue
    fi

    # Skip if no deduped data
    if [[ ! -d "${deduped_input_dir}" ]] || [[ -z "$(ls -A "${deduped_input_dir}" 2>/dev/null)" ]]; then
        log_warning "Skipping ${pl_folder}: no deduplicated data found"
        continue
    fi

    log_info "Sampling ${pl_folder}..."

    # Get deduped size
    deduped_size=$(get_local_size "$deduped_input_dir")
    log_info "  Deduped size: $(human_readable_size $deduped_size)"

    if [[ $deduped_size -le $TARGET_SAMPLE_SIZE ]]; then
        # Data is already small enough, just copy
        log_info "  Data smaller than target, copying all..."
        mkdir -p "$sample_output_dir"
        cp -r "${deduped_input_dir}"/* "$sample_output_dir/"
    else
        # Sample using bonepick
        log_info "  Sampling to $(human_readable_size $TARGET_SAMPLE_SIZE)..."
        uv run bonepick sample-dataset \
            --dataset-dir "${deduped_input_dir}" \
            --output-dir "${sample_output_dir}" \
            --target-size "${TARGET_SAMPLE_SIZE}"
    fi

    log_success "Completed sampling for ${pl_folder}"
done

log_success "Phase 2 complete: All languages sampled"

# ==============================================================================
# Phase 3: Annotation
# ==============================================================================

log_info "Phase 3: Annotating sampled data"

for mapping in "${LANGUAGE_MAPPINGS[@]}"; do
    pl_folder=$(echo "$mapping" | cut -d: -f1)
    rubric=$(echo "$mapping" | cut -d: -f2)

    sample_dir="${SAMPLE_DIR}/${pl_folder}"
    annotated_output_dir="${ANNOTATED_DIR}/${pl_folder}"

    # Skip if sample doesn't exist
    if [[ ! -d "${sample_dir}" ]] || [[ -z "$(ls -A "${sample_dir}" 2>/dev/null)" ]]; then
        log_warning "Skipping ${pl_folder}: no sampled data found"
        continue
    fi

    # Skip if already annotated
    if [[ -d "${annotated_output_dir}" ]] && [[ -n "$(ls -A "${annotated_output_dir}" 2>/dev/null)" ]]; then
        log_warning "Skipping ${pl_folder}: already annotated"
        continue
    fi

    log_info "Annotating ${pl_folder} with rubric ${rubric}..."

    uv run --extra=annotate bonepick annotate-dataset \
        --dataset-dir "${sample_dir}" \
        --output-dir "${annotated_output_dir}" \
        --model-name "${MODEL_NAME}" \
        --service-tier "${SERVICE_TIER}" \
        --annotation-task-prompt "${rubric}" \
        --max-concurrent-requests ${MAX_CONCURRENT_REQUESTS} \
        --max-new-tokens ${MAX_NEW_TOKENS} \
        --annotation-system-prompt 'code_system' \
        --max-text-length ${MAX_TEXT_LENGTH} \
        --limit-rows ${LIMIT_ROWS} \
        --cache-location "${CACHE_LOCATION}"

    log_success "Completed annotation for ${pl_folder}"
done

log_success "Phase 3 complete: All languages annotated"

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Deduplicated data: ${DEDUPED_DIR}"
echo "Sampled data: ${SAMPLE_DIR}"
echo "Annotated data: ${ANNOTATED_DIR}"
echo ""

# Count successful outputs at each stage
deduped_count=$(ls -d "${DEDUPED_DIR}"/*/ 2>/dev/null | wc -l || echo "0")
sample_count=$(ls -d "${SAMPLE_DIR}"/*/ 2>/dev/null | wc -l || echo "0")
annotated_count=$(ls -d "${ANNOTATED_DIR}"/*/ 2>/dev/null | wc -l || echo "0")

echo "Languages deduplicated: ${deduped_count}"
echo "Languages sampled: ${sample_count}"
echo "Languages annotated: ${annotated_count}"
echo "========================================"
