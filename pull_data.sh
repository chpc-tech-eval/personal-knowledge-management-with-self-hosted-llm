#!/bin/bash
set -euo pipefail

# ARG IS PATH TO SCRIPT
# Cloning Github repo markown
#
REPO_DIR=$(dirname "$0")
DATA_URL="https://github.com/chpc-tech-eval/scc"
OUTPUT_DIR="$REPO_DIR/data/sel"
TEMP_GITHUB="/tmp/sel_github"


mkdir -p "$OUTPUT_DIR"

[ -d "$TEMP_GITHUB/.git" ] && (cd "$TEMP_GITHUB" && git pull) || git clone "$DATA_URL" "$TEMP_GITHUB" 

echo "Extracting the markdown files"

# Deep Seek cooked a little here:
find "$TEMP_GITHUB" -name "*.md" -type f | while read file; do
    # Convert path to safe filename
    new_name=$(echo "$file" | sed "s|$TEMP_GITHUB/||" | tr '/' '_')
    cp "$file" "$OUTPUT_DIR/$new_name"
    echo "Copied: $file → $new_name"
done
