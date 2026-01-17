#!/bin/bash

BASE_DIR=~/ai2-llm/classifiers/code-quality/data/the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB/countup_criteria_v2/gpt-5-mini/10k_trimmed

DEST_REPORT=$(dirname $0)/threshold_report.txt
TMP_DIR=/tmp/threshold_report

# reset contents and create temp dir
echo "" > "$DEST_REPORT"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

# collect all language names for ordering
langs=()

# launch all jobs in parallel
for lang_dir in "$BASE_DIR"/*/; do
    lang=$(basename "$lang_dir")
    langs+=("$lang")
    tmp_file="$TMP_DIR/${lang}.txt"

    (
        echo "========================================"
        echo "Processing: $lang"
        echo "========================================"
        uv run bonepick label-distribution \
            -d "$lang_dir" \
            -l '.countup_criteria_v2.score' \
            -k '.text' \
            -t ordinal \
            -b 10
        echo ""
    ) 2>&1 | tee "$tmp_file" &
done

# wait for all background jobs to finish
wait

# merge all temp files into final report in order
for lang in "${langs[@]}"; do
    tmp_file="$TMP_DIR/${lang}.txt"
    cat "$tmp_file" >> "$DEST_REPORT"
done

# cleanup
rm -rf "$TMP_DIR"

echo "Report written to $DEST_REPORT"
