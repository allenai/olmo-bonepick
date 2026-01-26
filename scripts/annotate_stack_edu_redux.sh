#!/bin/bash
# Annotates 500,000 examples per programming language using stack_edu_redux rubrics.
#
# Input: ~/ai2-llm/classifiers/code-quality/data/the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB/raw/
# Output: ~/ai2-llm/classifiers/code-quality/data/the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB/stack_edu_redux/
#
# Usage:
#   ./scripts/annotate_stack_edu_redux.sh
#
# Each programming language folder is processed with its corresponding rubric from
# src/bonepick/annotate/prompt_collections/stack_edu_rubrics.py

set -euo pipefail

# Configuration
BASE_DIR="$HOME/ai2-llm/classifiers/code-quality/data/the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB"
INPUT_DIR="${BASE_DIR}/raw"
OUTPUT_DIR="${BASE_DIR}/stack_edu_redux"
MODEL_NAME="gpt-5-mini"
SERVICE_TIER="flex"
LIMIT_ROWS=500000
MAX_TEXT_LENGTH=10000
MAX_CONCURRENT_REQUESTS=5000
CACHE_LOCATION="${BASE_DIR}/annotation_cache.db"

# Mapping from folder names to rubric names
declare -A RUBRIC_MAP=(
    ["C"]="stack_edu_redux_c"
    ["C++"]="stack_edu_redux_cpp"
    ["C-Sharp"]="stack_edu_redux_csharp"
    ["Go"]="stack_edu_redux_go"
    ["Java"]="stack_edu_redux_java"
    ["JavaScript"]="stack_edu_redux_javascript"
    ["Markdown"]="stack_edu_redux_markdown"
    ["PHP"]="stack_edu_redux_php"
    ["Python"]="stack_edu_redux_python"
    ["Ruby"]="stack_edu_redux_ruby"
    ["Rust"]="stack_edu_redux_rust"
    ["Shell"]="stack_edu_redux_shell"
    ["SQL"]="stack_edu_redux_sql"
    ["Swift"]="stack_edu_redux_swift"
    ["TypeScript"]="stack_edu_redux_typescript"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Stack Edu Redux Annotation Script"
echo "========================================"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Limit rows per language: ${LIMIT_ROWS}"
echo "Max text length: ${MAX_TEXT_LENGTH}"
echo "Cache location: ${CACHE_LOCATION}"
echo "Service tier: ${SERVICE_TIER}"
echo "========================================"
echo ""

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Process each programming language
for pl_dir in "${INPUT_DIR}"/*; do
    if [[ ! -d "${pl_dir}" ]]; then
        continue
    fi

    pl=$(basename "${pl_dir}")

    # Check if we have a rubric for this language
    if [[ -z "${RUBRIC_MAP[$pl]:-}" ]]; then
        echo -e "${YELLOW}Warning: No rubric found for ${pl}, skipping...${NC}"
        continue
    fi

    rubric="${RUBRIC_MAP[$pl]}"
    output_pl_dir="${OUTPUT_DIR}/${pl}"

    # Skip if output directory already exists
    if [[ -d "${output_pl_dir}" ]]; then
        echo -e "${YELLOW}Skipping ${pl}: output directory already exists${NC}"
        continue
    fi

    echo -e "${GREEN}Processing ${pl} with rubric ${rubric}...${NC}"

    uv run --extra=annotate bonepick annotate-dataset \
        --dataset-dir "${pl_dir}" \
        --output-dir "${output_pl_dir}" \
        --model-name "${MODEL_NAME}" \
        --service-tier ${SERVICE_TIER} \
        --annotation-task-prompt "${rubric}" \
        --max-concurrent-requests ${MAX_CONCURRENT_REQUESTS} \
        --max-new-tokens 4096 \
        --annotation-system-prompt 'code_system' \
        --max-text-length ${MAX_TEXT_LENGTH} \
        --limit-rows ${LIMIT_ROWS} \
        --cache-location "${CACHE_LOCATION}"

    echo -e "${GREEN}Completed ${pl}${NC}"
    echo ""
done

echo "========================================"
echo -e "${GREEN}All languages processed!${NC}"
echo "========================================"
