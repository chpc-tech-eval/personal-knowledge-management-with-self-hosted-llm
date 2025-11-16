#!/bin/bash

set -euxo pipefail

# pull the base model for comparison
ollama pull "qwen2.5:7b"

# create the finetuned model from the GGUF files
ollama create "qwen2.5-7B-lora" -f "Modelfile.qwen2-5-7b-lora"
