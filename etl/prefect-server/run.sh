#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1
docker build -t prefect-server .
docker run --rm -p 4200:4200 -e PREFECT_API_URL="https://prefect.nuoya.co.uk/api" prefect-server