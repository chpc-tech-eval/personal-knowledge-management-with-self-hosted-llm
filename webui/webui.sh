#!/bin/bash

set -euxo pipefail

docker run -d \
  --network host \
  -v open-webui:/app/backend/data \
  -e HOST=0.0.0.0 \
  -e OPENAI_API_BASE_URL=http://localhost:9099 \
  -e OPENAI_API_KEY=0p3n-w3bu! \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

# Default ports:
# Open WebUI: 8080
# Pipelines: 9099
# Ollama: 11434
