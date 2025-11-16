#!/bin/bash

set -euxo pipefail

git clone https://github.com/ggerganov/llama.cpp

uv run \
  --with numpy \
  --with torch \
  --with sentencepiece \
  --with transformers \
  --with gguf \
  llama.cpp/convert_hf_to_gguf.py \
  /opt/shared/lora/merged_model/ \
  --outfile /opt/shared/gguf/qwen2-5-7B-lora.gguf \
  --outtype f16
