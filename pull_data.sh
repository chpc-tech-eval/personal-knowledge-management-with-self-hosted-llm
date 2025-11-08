#!/bin/bash
set -euo pipefail

DATA_URL="https://github.com/chpc-tech-eval/scc"
OUTPUT_DIR="/opt/shared/data/raw"
TEMP_GITHUB="/opt/shared/data/_selection_round_github"


mkdir -p "$OUTPUT_DIR"

[ -d "$TEMP_GITHUB/.git" ] && (cd "$TEMP_GITHUB" && git pull) || git clone "$DATA_URL" "$TEMP_GITHUB" 

echo "Extracting the markdown files"

# DeepSeek cooked a little here:
find "$TEMP_GITHUB" -name "*.md" -type f | while read file; do
    # Convert path to safe filename
    new_name=$(echo "$file" | sed "s|$TEMP_GITHUB/||" | tr '/' '_')
    cp "$file" "$OUTPUT_DIR/$new_name"
    echo "Copied: $file -> $OUTPUT_DIR/$new_name"
done
