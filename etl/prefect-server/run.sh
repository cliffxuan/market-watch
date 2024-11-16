#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1
docker build -t prefect-server .
mkdir -p data
docker run --rm -p 4200:4200 \
  -e PREFECT_API_URL="https://prefect.nuoya.co.uk/api" \
  -e PREFECT_SERVER_DATABASE_CONNECTION_URL="sqlite+aiosqlite:///data/prefect.db" \
  -v "$PWD/data":/data \
  prefect-server
