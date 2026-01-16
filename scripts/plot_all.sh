#!/bin/bash

BASE_DIR=~/ai2-llm/classifiers/code-quality/data/the-stack-v2/spring2code_v2/minhash_v2_annotated/sample_1GB/countup_criteria_v2/gpt-5-mini/10k_trimmed

for lang_dir in "$BASE_DIR"/*/; do
    lang=$(basename "$lang_dir")
    echo "========================================"
    echo "Processing: $lang"
    echo "========================================"
    uv run bonepick label-distribution \
        -d "$lang_dir" \
        -l '.countup_criteria_v2.score' \
        -k '.text' \
        -t ordinal
    echo ""
done
