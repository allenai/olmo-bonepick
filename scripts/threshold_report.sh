#!/bin/bash

BASE_DIR=$HOME/ai2-llm/classifiers/code-quality/data/the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB/countup_criteria_v2/gpt-5-mini/10k_trimmed

DEST_REPORT=$(dirname $0)/threshold_report.txt

# reset contents and create temp dir
echo "" > "$DEST_REPORT"

# sync once before parallel runs
uv sync --extra annotate


# launch all jobs in parallel
for lang_dir in "$BASE_DIR"/*/; do
    (
        echo "========================================"
        echo "Processing: $lang"
        echo "========================================"
        uv run --no-sync --frozen bonepick label-distribution \
            -d "$lang_dir" \
            -l '.countup_criteria_v2.score' \
            -k '.text' \
            -t ordinal \
            -b 10
        echo ""
    ) 2>&1 | tee -a "$DEST_REPORT"
done

# finalize report
echo "Report written to $DEST_REPORT"
