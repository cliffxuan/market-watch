#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1
docker build -t market-watch .
mkdir -p data
docker network create warp_network || true
docker run --rm --network warp_network -v "$DIR/src":/app/src -p 8501:8501 market-watch
