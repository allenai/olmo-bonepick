#!/bin/bash
#
# Validate a fixed set of S3 token shard patterns with scripts/validate_npy.py.
#
# Default behavior:
# - expands the hard-coded S3 globs below via s5cmd
# - validates each concrete .npy shard in parallel
# - writes manifests, logs, and summaries under /mnt/raid0/validate_npy
#
# Useful overrides:
#   JOBS=64 SAMPLE_SIZE=4 ./scripts/validate_npy.sh
#   CHECK_SOURCES=0 JOBS=96 ./scripts/validate_npy.sh
#   MAX_SHARDS=10 DRY_RUN=1 ./scripts/validate_npy.sh
#   RUN_ROOT=/mnt/raid0/custom-validate ./scripts/validate_npy.sh

set -euo pipefail

check_command() {
    local cmd="$1"
    if ! command -v "$cmd" > /dev/null 2>&1; then
        echo "Missing required command: $cmd" >&2
        exit 1
    fi
}

default_jobs() {
    local cpu_count
    cpu_count="$(nproc)"
    if [[ "$cpu_count" -gt 32 ]]; then
        echo 32
    else
        echo "$cpu_count"
    fi
}

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="${VALIDATOR:-${SCRIPT_DIR}/validate_npy.py}"
RUN_ROOT="${RUN_ROOT:-/mnt/raid0/validate_npy}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/raid0/.cache/uv}"
JOBS="${JOBS:-$(default_jobs)}"
SAMPLE_SIZE="${SAMPLE_SIZE:-8}"
CHECK_SOURCES="${CHECK_SOURCES:-1}"
STRICT_WARNINGS="${STRICT_WARNINGS:-0}"
VERBOSE="${VERBOSE:-0}"
MAX_SHARDS="${MAX_SHARDS:-0}"
DRY_RUN="${DRY_RUN:-0}"

readonly SCRIPT_DIR VALIDATOR RUN_ROOT UV_CACHE_DIR JOBS SAMPLE_SIZE CHECK_SOURCES STRICT_WARNINGS VERBOSE MAX_SHARDS DRY_RUN
export UV_CACHE_DIR

check_command uv
check_command s5cmd
check_command awk
check_command sort
check_command cksum

if [[ ! -f "$VALIDATOR" ]]; then
    echo "Validator script not found: $VALIDATOR" >&2
    exit 1
fi

timestamp="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${RUN_ROOT}/run_${timestamp}"
LOG_DIR="${RUN_DIR}/logs"
LISTING_DIR="${RUN_DIR}/listings"
MANIFEST_RAW="${RUN_DIR}/manifest_raw.txt"
MANIFEST="${RUN_DIR}/manifest.txt"
RESULTS="${RUN_DIR}/results.tsv"
SUMMARY="${RUN_DIR}/summary.txt"
RUN_LOG="${RUN_DIR}/run.log"

mkdir -p "$LOG_DIR" "$LISTING_DIR" "$UV_CACHE_DIR"
touch "$MANIFEST_RAW" "$RESULTS" "$SUMMARY"

exec > >(tee -a "$RUN_LOG") 2>&1

log "Run directory: $RUN_DIR"
log "Validator: $VALIDATOR"
log "UV cache: $UV_CACHE_DIR"
log "Parallel jobs: $JOBS"
log "Sample size per shard: $SAMPLE_SIZE"

PATTERNS=(
    "s3://ai2-llm/preprocessed/dolma4pdfs/dolma4pdfs_full_deduped_partitioned_resharded_qualitytagged_partitioned_decon/finepdfs_wo_partitioned_qual_ngram_filtered/__label__*/qual_00*/allenai/dolma2-tokenizer/*.npy"
    "s3://ai2-llm/preprocessed/dolma4pdfs/dolma4pdfs_full_deduped_partitioned_resharded_qualitytagged_partitioned_decon/olmo-crawled-pdfs_ngram_filtered/__label__*/qual_00*/allenai/dolma2-tokenizer/*.npy"
    "s3://ai2-llm/preprocessed/dolma4pdfs/dolma4pdfs_full_deduped_partitioned_resharded_qualitytagged_partitioned_decon/s2orcforolmo_nogpl_ngram_filtered/__label__*/qual_00*/allenai/dolma2-tokenizer/*.npy"
    "s3://ai2-llm/preprocessed/common-pile_wikish/v0/common-pile_*_decon_ngram_filtered/allenai/dolma2-tokenizer/*.npy"
    "s3://ai2-llm/preprocessed/common-pile_texbookish/v0/common-pile_*_decon_ngram_filtered/allenai/dolma2-tokenizer/*.npy"
    "s3://ai2-llm/preprocessed/common-pile_codetextish/v0/common-pile_*_decon_ngram_filtered/allenai/dolma2-tokenizer/*.npy"
    "s3://ai2-llm/preprocessed/sponge_211_code_prose_code_prose_tagged_partitioned_decon_ngram_filter/quality_*/allenai/dolma2-tokenizer/*.npy"
    "s3://ai2-llm/preprocessed/sponge_211_non-software-development_code_prose_code_prose_tagged_partitioned_decon_ngram_filter/quality_*/allenai/dolma2-tokenizer/*.npy"
    "s3://ai2-llm/preprocessed/the-stack-v2/spring2code_v2/minhash_filter_v2_2026_stack_edu_redux_tagged_decon_reshard_partitioned_gzip_ngram_filtered/*/quality_*/allenai/dolma2-tokenizer/*.npy"
    "s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v5/topic_vigintiles_v2_decon/*/vigintile_00*/allenai/dolma2-tokenizer/*.npy"
    "s3://ai2-llm/preprocessed/tokyotech-llm_swallow-math-v2/stage3-texbook_decon_ngram_filtered/allenai/dolma2-tokenizer/*.npy"
    "s3://ai2-llm/preprocessed/tokyotech-llm_swallow-math-v2/stage3-qa_decon_ngram_filtered/allenai/dolma2-tokenizer/*.npy"
    "s3://ai2-llm/preprocessed/tokyotech-llm_swallow-code-v2/stage5_python_medium_decon_ngram_filtered/allenai/dolma2-tokenizer/*.npy"
)

pattern_failures=0

for index in "${!PATTERNS[@]}"; do
    pattern="${PATTERNS[$index]}"
    listing_file="${LISTING_DIR}/pattern_$(printf '%02d' "$((index + 1))").txt"

    log "Listing pattern $((index + 1))/${#PATTERNS[@]}: $pattern"
    if ! s5cmd ls "$pattern" > "$listing_file"; then
        log "ERROR listing failed for pattern: $pattern"
        pattern_failures=$((pattern_failures + 1))
        continue
    fi

    match_count="$(awk 'END {print NR + 0}' "$listing_file")"
    if [[ "$match_count" -eq 0 ]]; then
        log "ERROR no shards matched pattern: $pattern"
        pattern_failures=$((pattern_failures + 1))
        continue
    fi

    awk '{print $4}' "$listing_file" >> "$MANIFEST_RAW"
    log "Matched $match_count shard(s)"
done

sort -u "$MANIFEST_RAW" > "$MANIFEST"
total_shards="$(awk 'END {print NR + 0}' "$MANIFEST")"

if [[ "$MAX_SHARDS" -gt 0 ]] && [[ "$total_shards" -gt "$MAX_SHARDS" ]]; then
    log "Limiting manifest from $total_shards shard(s) to MAX_SHARDS=$MAX_SHARDS"
    head -n "$MAX_SHARDS" "$MANIFEST" > "${MANIFEST}.tmp"
    mv "${MANIFEST}.tmp" "$MANIFEST"
    total_shards="$(awk 'END {print NR + 0}' "$MANIFEST")"
fi

log "Concrete shard manifest: $MANIFEST"
log "Total concrete shards: $total_shards"

if [[ "$total_shards" -eq 0 ]]; then
    log "ERROR no shards were discovered"
    exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Dry run requested. Manifest created without launching validation jobs."
    exit 0
fi

log "Priming validator environment"
uv run "$VALIDATOR" --help > /dev/null

export VALIDATOR
export LOG_DIR
export RESULTS
export SAMPLE_SIZE
export CHECK_SOURCES
export STRICT_WARNINGS
export VERBOSE
export UV_CACHE_DIR

log "Launching validation jobs"

set +e
cat "$MANIFEST" | xargs -I{} -P "$JOBS" bash -c '
uri="$1"
base_name="$(basename "${uri%.npy}")"
checksum="$(printf "%s" "$uri" | cksum | awk "{print \$1}")"
log_file="${LOG_DIR}/${base_name}_${checksum}.log"

cmd=(uv run "$VALIDATOR" "$uri" "--sample-size" "$SAMPLE_SIZE")
if [[ "$CHECK_SOURCES" != "1" ]]; then
    cmd+=("--skip-sources")
fi
if [[ "$STRICT_WARNINGS" == "1" ]]; then
    cmd+=("--strict-warnings")
fi
for ((i = 0; i < VERBOSE; i++)); do
    cmd+=("-v")
done

{
    printf "URI: %s\n" "$uri"
    printf "Command:"
    for part in "${cmd[@]}"; do
        printf " %q" "$part"
    done
    printf "\n\n"
} > "$log_file"

set +e
"${cmd[@]}" >> "$log_file" 2>&1
status=$?
set -e

printf "%s\t%s\t%s\n" "$status" "$uri" "$log_file" >> "$RESULTS"
if [[ "$status" -eq 0 ]]; then
    printf "[PASS] %s\n" "$uri"
else
    printf "[FAIL] %s (see %s)\n" "$uri" "$log_file" >&2
fi

exit 0
' _ {}
xargs_status=$?
set -e

completed_shards="$(awk 'END {print NR + 0}' "$RESULTS")"
passed_shards="$(awk -F'\t' '$1 == 0 {count += 1} END {print count + 0}' "$RESULTS")"
failed_shards="$(awk -F'\t' '$1 != 0 {count += 1} END {print count + 0}' "$RESULTS")"
missing_shards=$((total_shards - completed_shards))

{
    echo "Run directory: $RUN_DIR"
    echo "Manifest: $MANIFEST"
    echo "Results: $RESULTS"
    echo "Total shards: $total_shards"
    echo "Completed shards: $completed_shards"
    echo "Passed shards: $passed_shards"
    echo "Failed shards: $failed_shards"
    echo "Missing shard results: $missing_shards"
    echo "Pattern listing failures: $pattern_failures"
    echo "xargs status: $xargs_status"
} | tee "$SUMMARY"

if [[ "$failed_shards" -gt 0 ]]; then
    echo
    echo "Failed shard logs:"
    awk -F'\t' '$1 != 0 {printf "  %s\t%s\n", $2, $3}' "$RESULTS" | tee -a "$SUMMARY"
fi

if [[ "$pattern_failures" -gt 0 ]] || [[ "$failed_shards" -gt 0 ]] || [[ "$missing_shards" -gt 0 ]]; then
    exit 1
fi

log "All shard validations passed"
