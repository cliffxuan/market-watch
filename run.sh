#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1
docker build -t market-watch .
mkdir -p data
docker run --rm -p 8501:8501 market-watch
