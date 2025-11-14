#!/bin/bash
set -euo pipefail

PROJ_DIR=$(dirname $0)
DATA_DIR="$PROJ_DIR/data"
OUTPUT_DIR="/opt/shared/data/raw"

mkdir -p "$OUTPUT_DIR"

echo "Extracting the markdown files"

# DeepSeek cooked a little here:
find "$DATA_DIR" -name "*.md" -type f | while read file; do
    # Convert path to safe filename
    new_name=$(echo "$file" | sed "s|$DATA_DIR/||" | tr '/' '_')
    cp "$file" "$OUTPUT_DIR/$new_name"
    echo "Copied: $file -> $OUTPUT_DIR/$new_name"
done
