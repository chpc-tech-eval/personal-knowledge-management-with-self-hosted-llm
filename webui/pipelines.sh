#!/bin/bash

set -euxo pipefail

docker run -d \
    --network host \
    -e RESET_PIPELINES_DIR=true \
    -v pipelines:/app/pipelines \
    -v /opt/shared/data/chromadb:/data/chromadb:z \
	-v $(realpath $(dirname $0)/../src/retrieval.py):/deps/retrieval.py \
    --name pipelines \
    --restart always \
    ghcr.io/open-webui/pipelines:main
