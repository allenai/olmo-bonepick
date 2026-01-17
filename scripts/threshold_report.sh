#!/bin/bash

BASE_DIR=~/ai2-llm/classifiers/code-quality/data/the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB/countup_criteria_v2/gpt-5-mini/10k_trimmed

DEST_REPORT=$(dirname $0)/threshold_report.txt

# reset contents
echo "" > "$DEST_REPORT"

for lang_dir in "$BASE_DIR"/*/; do
    lang=$(basename "$lang_dir")
    echo "========================================" | tee -a "$DEST_REPORT"
    echo "Processing: $lang" | tee -a "$DEST_REPORT"
    echo "========================================" | tee -a "$DEST_REPORT"
    uv run bonepick label-distribution \
        -d "$lang_dir" \
        -l '.countup_criteria_v2.score' \
        -k '.text' \
        -t ordinal \
        -b 10 | tee -a "$DEST_REPORT"
    echo "" | tee -a "$DEST_REPORT"
done
