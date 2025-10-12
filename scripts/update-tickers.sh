#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/.." || exit 1

precheck() {
  if ! git diff-index --quiet HEAD --; then
    echo "There are uncommitted changes. Please commit or stash them before running this script."
    exit 1
  fi
  if ! git pull; then
    echo "failed pulling remote."
    exit 1
  fi
}

fetch() {
  ./.venv/bin/python scripts/scrape_lists.py
  ./.venv/bin/python etl/__main__.py
}

commit() {
  git add data/
  git commit -m 'refetched tickers data'
}

push() {
  git push origin main -f
}

all() {
  precheck && fetch && commit && push
}

while getopts ":fhp" opt; do
  case "$opt" in
  f)
    fetch
    exit 0
    ;;
  p)
    precheck
    exit 0
    ;;
  h)
    echo Usage "$(basename "$0") [-c | -f]"
    exit 0
    ;;
  \?)
    echo "Invalid option: -$OPTARG" >&2
    exit 1
    ;;
  :)
    echo "Option -$OPTARG requires an argument." >&2
    exit 1
    ;;
  esac
done
all
